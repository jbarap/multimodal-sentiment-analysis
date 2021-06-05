import random

import pytest

import torch
import numpy as np
import numpy.testing as np_test

import albumentations as A
import albumentations.pytorch.transforms

from detr.datasets.coco_subset import CocoSubset
from detr.models.matcher import HungarianMatcher


@pytest.fixture
def control_coco_subset():
    base_transforms = A.Compose([
        A.pytorch.ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    base_dataset = CocoSubset('tests/data/',
                              ['dog', 'cat'],
                              base_transforms,
                              'train',
                              1.0)

    return base_dataset


@pytest.fixture
def control_bboxes():
    return np.array([[0.14571428, 0.54320985, 0.23285714, 0.8302469],
                     [0.34142858, 0.66358024, 0.12857144, 0.5617284],
                     [0.4942857, 0.5123457, 0.22285715, 0.9259259],
                     [0.68142855, 0.65123457, 0.15857142, 0.60493827],
                     [0.8428571, 0.5092593, 0.22714286, 0.91049385]])


def test_coco_return_format(control_coco_subset):
    (image, labels) = control_coco_subset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(labels, dict)
    assert all(l in labels for l in ['bboxes', 'classes'])
    assert all(isinstance(value, torch.Tensor) for value in labels.values())


def test_coco_bboxes(control_coco_subset, control_bboxes):
    (image, labels) = control_coco_subset[0]
    np_test.assert_allclose(labels['bboxes'].numpy(), control_bboxes)


def test_matcher(control_bboxes):
    matcher = HungarianMatcher(1, 1, 1)

    predictions = {
        'bboxes': torch.rand((1, 100, 4), dtype=torch.float32),
        'logits': torch.rand((1, 100, 91), dtype=torch.float32)
    }

    labels = [
        {
            'bboxes': torch.tensor(control_bboxes, dtype=torch.float32),
            'classes': torch.tensor([18, 17, 18, 17, 18], dtype=torch.long)
        }
    ]

    indices = list(range(predictions['logits'].size(1)))
    indices = random.sample(indices, labels[0]['classes'].size(0))

    pred_classes = torch.zeros((len(indices), 91), dtype=torch.float32)
    pred_classes[range(len(indices)), labels[0]['classes'].numpy().astype('int')] = 1

    predictions['bboxes'][0, indices] = labels[0]['bboxes']
    predictions['logits'][0, indices] = pred_classes

    matching_indices = matcher(predictions, labels)
    matching_indices = matching_indices[0]

    assert torch.equal(torch.tensor(indices)[matching_indices[1]], matching_indices[0])

