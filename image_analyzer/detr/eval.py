import torch
from tqdm import tqdm

from collections import deque

from detr.utils import data_utils


def validation_loop(model, matcher, val_loader, loss_fn, device):
    model.eval()
    loss_hist = deque()

    with torch.no_grad():
        for images, labels in tqdm(val_loader, 'Evaluating model'):
            images = images.to(device)
            labels = data_utils.labels_to_device(labels, device)

            output = model(images)
            matching_indices = matcher(output, labels)
            matching_indices = data_utils.indices_to_device(matching_indices, device)

            loss = loss_fn(output, labels, matching_indices)
            loss_hist.append(loss.item())

    model.train()

    # revert the previous line only for the model's backbone, freezing the batchnorm layers
    model.backbone.eval()

    val_loss = sum(loss_hist) / len(loss_hist)
    print("Validation loss: ", val_loss)

    return {'loss': val_loss}

