import torch
from PIL import Image


def denormalize_tensor_image(image, mean='imagenet', std='imagenet', pillow_output=True):
    if mean == 'imagenet':
        mean = torch.tensor([0.485, 0.456, 0.406])
    else:
        mean = torch.tensor(mean)

    if std == 'imagenet':
        std = torch.tensor([0.229, 0.224, 0.225])
    else:
        std = torch.tensor(mean)

    mean = mean.view(3, 1, 1)
    std = std.view(3, 1, 1)

    image = image.clone()
    image = (image*std) + mean
    image = image * 255
    image = image.permute(1, 2, 0)

    if pillow_output:
        image = Image.fromarray(image.numpy().astype('uint8'))

    return image


def collate_fn(batch):
    """Batches a dataset that returns instances of (image, {'label': Tensor, ...}).

    Due to images having different number of objects within them, labels can't simply be
    batched into a Tensor of shape [n_instances, n_objects, feature_size], but images can.
    This function batches images as a Tensor and labels as a dict of lists of Tensors.

    Args:
        batch: List of instances to be batched together. instances must be in the form:
            (Tensor, {'label_key': Tensor})
    Returns:
        Batched instances as a tuple (images, labels), where images is a Tensor of shape
        [n_instances, channels, height, width], and labels is a list of size n_instances
        of dicts with keys:
            'bboxes': Containing a Tensor of shape [n_objects_in_the_image, 4].
            'classes': Containing a Tensor of shape [n_objects_in_the_image]
    """
    images, labels = zip(*batch)
    images = torch.cat([img.unsqueeze(0) for img in images])
    return images, labels


def labels_to_device(labels, device):
    for label in labels:
        for key in label:
            label[key] = label[key].to(device)
    return labels


def indices_to_device(indices, device):
    for i, _ in enumerate(indices):
        indices[i][0] = indices[i][0].to(device)
        indices[i][1] = indices[i][1].to(device)
    return indices

