import albumentations as A
import albumentations.pytorch.transforms


def get_train_transforms():
    transforms = A.Compose([
        A.SmallestMaxSize(800),
        A.PadIfNeeded(800, 1333, value=0, border_mode=0),
        A.RandomBrightnessContrast(p=0.2, contrast_limit=0.1, brightness_limit=0.1),
        A.GaussianBlur(blur_limit=(3, 5)),
        A.Rotate(limit=20),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels']))
    return transforms


def get_val_transforms():
    transforms = A.Compose([
        A.SmallestMaxSize(800),
        A.PadIfNeeded(800, 1333, value=0, border_mode=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels']))
    return transforms

