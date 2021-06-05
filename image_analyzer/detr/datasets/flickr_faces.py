import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FlickrFaces(Dataset):
    """
    Note:
        class 0 represents a present point
        class 1 represents an absent point
        keypoints will have the name of 'bboxes' in the prediction for consistency
    """
    def __init__(self,
                 labels_file,
                 images_base_path,
                 transforms=None,
                 train=True,
                 train_val_split=0.9,
                 max_points=68):
        super().__init__()

        labels_file = Path(labels_file)
        images_path = Path(images_base_path)

        with open(labels_file) as f:
            data = json.load(f)

        n = len(data)

        indices = list(range(n))
        random.Random(42).shuffle(indices)

        if train:
            indices = set(indices[:int(train_val_split * n)])
        else:
            indices = set(indices[int(train_val_split * n):])

        sub_data = {int(key): value for (key, value) in data.items() if int(key) in indices}

        self.data = sub_data
        self.all_indices = list(indices)
        self.transforms = transforms
        self.images_path = images_path
        self.max_points = max_points

    def __getitem__(self, idx):
        feature_points = np.ones((self.max_points, 2), dtype=np.float32)
        feature_classes = np.ones((self.max_points, ), dtype=np.int64)

        item = self.data[self.all_indices[idx]]
        for i, point in enumerate(item['face_landmarks']):
            feature_points[i, 0] = np.clip(point[0], 1, 511)
            feature_points[i, 1] = np.clip(point[1], 1, 511)
            feature_classes[i] = 0

        image = Image.open(self.images_path / item['file_name'])

        if self.transforms is not None:
            transformed = self.transforms(image=np.array(image), keypoints=feature_points, class_labels=feature_classes)
            image = transformed['image']
            _, h, w = image.shape

            feature_points = torch.tensor(transformed['keypoints'], dtype=torch.float32)
            feature_points[:, 0] = feature_points[:, 0] / w
            feature_points[:, 1] = feature_points[:, 1] / h

            feature_classes = torch.tensor(transformed['class_labels'], dtype=torch.long)

        labels = {'bboxes': feature_points, 'classes': feature_classes}
        return image, labels

    def __len__(self):
        return len(self.data)

