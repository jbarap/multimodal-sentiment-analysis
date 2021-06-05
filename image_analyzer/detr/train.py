import argparse
from collections import deque
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import yaml
from tqdm import tqdm

import detr.logs.logger as log

from detr import models
from detr.datasets import transforms
from detr.datasets.flickr_faces import FlickrFaces
from detr.eval import validation_loop
from detr.utils import data_utils
from detr.utils.checkpoints import CheckpointManager


def train(args):
    logger = log.get_logger(__name__)

    with open(Path(args.config_base_path, args.config).with_suffix(".yaml"), 'r') as f:
        config = yaml.safe_load(f)

    train_transforms = transforms.get_train_transforms()
    val_transforms = transforms.get_val_transforms()

    logger.info("Loading the dataset...")
    if config['dataset']['name'] == 'flickr_faces':
        train_dataset = FlickrFaces(config['dataset']['labels_file'],
                                    config['dataset']['images_path'],
                                    train_transforms,
                                    train=True,
                                    train_val_split=config['dataset']['train_val_split'])

        val_dataset = FlickrFaces(config['dataset']['labels_file'],
                                  config['dataset']['images_path'],
                                  val_transforms,
                                  train=False,
                                  train_val_split=config['dataset']['train_val_split'])

    else:
        raise ValueError("Dataset name not recognized or implemented")

    train_loader = DataLoader(train_dataset,
                              config['training']['batch_size'],
                              shuffle=True,
                              collate_fn=data_utils.collate_fn)

    val_loader = DataLoader(val_dataset,
                            config['training']['batch_size'],
                            shuffle=True,
                            collate_fn=data_utils.collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_manager = CheckpointManager(args.config, args.save_every, save_path=args.save_path)

    logger.info("Loading model...")
    model = models.DETR(config['dataset']['num_classes'],
                        config['model']['dim_model'],
                        config['model']['n_heads'],
                        n_queries=config['model']['n_queries'],
                        head_type=config['model']['head_type'])
    model.to(device)

    # TODO: implement scheduler, data parallel
    optim = AdamW(model.parameters(),
                  config['training']['lr'],
                  weight_decay=config['training']['weight_decay'])

    if args.mode == 'pretrained':
        model.load_demo_state_dict('data/state_dicts/detr_demo.pth')
    elif args.mode == 'checkpoint':
        state_dict, optim_dict = checkpoint_manager.load_checkpoint('latest')
        model.load_state_dict(state_dict)
        model.to(device)  # necessary for full compatibility of CPU and GPU?

        optim.load_state_dict(optim_dict)

    if args.train_section == 'head':
        to_train = ['ffn']
    elif args.train_section == 'backbone':
        to_train = ['backbone', 'conv']
    elif args.train_section == 'adapters':
        to_train = ['object_queries', 'ffn', 'transformer']
    elif args.train_section == 'no_backbone':
        to_train = ['ffn', 'transformer', 'row', 'col', 'object']
    elif args.train_section == 'all':
        to_train = ['ffn', 'backbone', 'conv', 'transformer', 'row', 'col', 'object']
    else:
        raise AttributeError("Invalid train_section option")

    # Freeze everything but the modules that are in to_train
    for name, param in model.named_parameters():
        if not any(map(name.startswith, to_train)):
            param.requires_grad = False

    matcher = models.HungarianMatcher(config['losses']['lambda_matcher_classes'],
                                      config['losses']['lambda_matcher_l1'])

    loss_fn = models.DETRLoss(config['losses']['lambda_loss_classes'],
                              config['losses']['lambda_loss_l1'],
                              config['dataset']['num_classes'],
                              config['losses']['no_class_weight'])

    # writer = SummaryWriter(log_dir=Path(__file__)/'logs/tensorboard')
    # maybe image with boxes every now and then
    # maybe look into add_hparams

    logger.info("Starting training...")
    loss_hist = deque()
    loss_desc = "Loss: n/a"

    update_every_n_steps = config['training']['effective_batch_size'] // config['training']['batch_size']
    steps = 1

    starting_epoch = checkpoint_manager.current_epoch

    for epoch in range(starting_epoch, config['training']['epochs']):
        epoch_desc = f"Epoch [{epoch}/{config['training']['epochs']}]"

        for images, labels in tqdm(train_loader, f"{epoch_desc} | {loss_desc}"):
            images = images.to(device)
            labels = data_utils.labels_to_device(labels, device)

            output = model(images)
            matching_indices = matcher(output, labels)
            matching_indices = data_utils.indices_to_device(matching_indices, device)

            loss = loss_fn(output, labels, matching_indices) / update_every_n_steps
            loss_hist.append(loss.item() * update_every_n_steps)
            loss.backward()

            if steps % update_every_n_steps == 0:
                optim.step()
                optim.zero_grad()

            steps += 1

        checkpoint_manager.step(model, optim, sum(loss_hist) / len(loss_hist))

        loss_desc = f"Loss: {sum(loss_hist)/len(loss_hist)}"
        loss_hist.clear()

        # NOTE: eval sometimes showing strange behavior when training the transformer
        if (epoch % args.eval_every == 0) and epoch != 0:
            validation_loop(model, matcher, val_loader, loss_fn, device)

    checkpoint_manager.save_checkpoint(model, optim, sum(loss_hist) / len(loss_hist))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('detr_train')

    parser.add_argument('--mode', default='pretrained', choices=['pretrained', 'checkpoint', 'from_scratch'])
    parser.add_argument('--train_section', default='head', choices=['head', 'backbone', 'all', 'adapters', 'no_backbone'])
    parser.add_argument('--config', default='flickr_faces')
    parser.add_argument('--config_base_path', default='configs/')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--save_path', default='data/state_dicts/')
    parser.add_argument('--eval_every', type=int, default=10)
    args = parser.parse_args()

    train(args)

