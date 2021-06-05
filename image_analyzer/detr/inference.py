import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

import albumentations as A
import albumentations.pytorch.transforms

from detr.models.detr import DETR
from detr.datasets import transforms
from detr.utils import data_utils


def inference_on_image(model, image_path, save_path=None, target_w=1333, target_h=800):
    image = Image.open(image_path)

    image_w_h_ratio = image.width / image.height
    target_w_h_ratio = target_w / target_h

    # Shrink the image down so that it fits inside of a (target_w, target_h) rectangle
    # while keeking the original aspect ratio, fill the remaining space with zeros
    if image_w_h_ratio >= target_w_h_ratio:
        shrink_ratio = target_w / image.width
        resized_w = int(image.width * shrink_ratio)
        resized_h = int(image.height * shrink_ratio)
        resized_image = image.resize((resized_w, resized_h))
    else:
        shrink_ratio = target_h / image.height
        resized_h = int(image.height * shrink_ratio)
        resized_w = int(image.width * shrink_ratio)
        resized_image = image.resize((resized_w, resized_h))

    inference_transform = A.Compose([
        A.PadIfNeeded(target_h, target_w, value=0, border_mode=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensor(),
    ])

    tensor_image = inference_transform(image=np.array(resized_image))['image']
    tensor_image = torch.unsqueeze(tensor_image, 0)

    # Compute the inference
    model.eval()
    with torch.no_grad():
        inference = model(tensor_image)

    inference = filter_inference_results(inference)

    # Grab the first element of the list because the inference is over a single image
    coords = inference['bboxes'][0]
    classes = inference['classes'][0]

    # The model outputs yolo format images, turn them into (x1, y1, x2, y2)
    coords = denormalize_coords(coords, target_w, target_h)

    # Shift coordinates so that the origin is with respect to the inner image, not the padding
    shifted_coords = shift_coords(coords, -(target_w-resized_w)/2, -(target_h-resized_h)/2)

    # Then return the coordinates to the scale of the original image
    scaled_coords = scale_coords(shifted_coords, 1/shrink_ratio, 1/shrink_ratio)

    if save_path is not None:
        canvas = np.array(image)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        for i, c in enumerate(scaled_coords):
            cv2.circle(canvas, (int(c[0]), int(c[1])), 3, (0, 0, 255), -1)

        save_path = Path(save_path)
        cv2.imwrite(str(save_path.with_suffix('.jpg')), canvas)

        print('Image saved at:', str(save_path.with_suffix('.jpg')))

    return {'coords': scaled_coords, 'classes': classes}


def filter_inference_results(inference):
    """Filter inference results to not contain (no_object) classes.

    Args:
        inference: Dictionary of results.
                   {'bboxes': Tensor[batch_size, n_queries, 2],
                    'logits': Tensor[batch_size, n_queries, n_classes]}

    Returns:
        Dicts of keys 'bboxes' and 'classes' containing lists of size batch_size with the inferences.
    """

    bboxes = inference['bboxes']  # [batch, 100, 2]

    classes = inference['logits']  # [batch, 100, 2]
    no_obj_index = classes.size(-1) - 1

    classes = classes.softmax(-1)
    _, classes = torch.max(classes, dim=-1)  # [batch, 100]

    filtered_classes = [c[c!=no_obj_index].numpy() for c in classes]
    filtered_bboxes = [b[c!=no_obj_index].numpy() for (c, b) in zip(classes, bboxes)]

    return {'bboxes': filtered_bboxes, 'classes': filtered_classes}


# TODO: Move to box ops
def denormalize_coords(coords, width, height):
    return [(x*width, y*height) for (x, y) in coords]


def normalize_coords(coords, width, height):
    return [(x/width, y/height, w/width, h/height) for (x, y, w, h) in coords]


def xywh_to_x1y1x2y2(coords):
    return [(x-w/2, y-h/2, x+w/2, y+w/2) for (x, y, w, h) in coords]


def shift_coords(coords, x_shift=0, y_shift=0):
    return [(x1+x_shift, y1+y_shift) for (x1, y1) in coords]


def scale_coords(coords, x_scale=1, y_scale=1):
    return [(x1*x_scale, y1*y_scale) for (x1, y1) in coords]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weights', required=True)
    parser.add_argument('--model_config', default='configs/flickr_faces.yaml')
    parser.add_argument('--image_path', default='data/examples/cat_dog.jpg')
    parser.add_argument('--image_save_path', default='data/examples/cat_dog_inference.jpg')

    args = parser.parse_args()

    with open(args.model_config, 'r') as f:
        config = yaml.safe_load(f)

    model = DETR(config['dataset']['num_classes'],
                 config['model']['dim_model'],
                 config['model']['n_heads'],
                 n_queries=config['model']['n_queries'],
                 head_type=config['model']['head_type'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_weights == 'demo':
        model.load_demo_state_dict('data/state_dicts/detr_demo.pth')
    else:
        state_dict = torch.load(args.model_weights, map_location=device)['state_dict']
        model.load_state_dict(state_dict)

    inference_on_image(model, args.image_path, args.image_save_path)

