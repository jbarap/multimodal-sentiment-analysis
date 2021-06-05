import torch
import torch.nn.functional as F
from torch import nn


class DETRLoss(nn.Module):
    def __init__(self, lambda_classes, lambda_l1, num_classes, no_class_weight=0.1):
        super().__init__()
        self.lambda_classes = lambda_classes
        self.lambda_l1 = lambda_l1

        self.no_class_index = num_classes
        self.no_class_weight = no_class_weight

    def forward(self, predictions, labels, indices):
        """Compute the loss.

        Args:
            predictions: Dictionary of predictions of the model.
                         The two keys needed are "logits" and "bboxes".
                Logits:
                    Class predictions of the model. Tensor of shape [batch_size, n_object_queries, n_classes].
                Bboxes:
                    Boxes predictions of the model. Tensor of shape [batch_size, n_object_queries, 4].

            labels: Dictionary of labels corresponsing to the predictions.
                    The two keys needed are "classes" and "bboxes".
                Classes:
                    Class labels of the model. List of size [batch_size] that contains Tensors of shape [n_objects_in_image]
                Bboxes:
                    Boxes labels of the model. Tensor of shape [batch_size, n_object_queries, 4].

            indices: List of tuples corresponsing to the pairing indices.
        """
        class_loss = self.classification_loss(predictions, labels, indices)
        l1_loss = self.bbox_losses(predictions, labels, indices)

        # compute the box loss
        total_loss = class_loss * self.lambda_classes + l1_loss * self.lambda_l1

        return total_loss

    def classification_loss(self, predictions, labels, indices):
        # extract the indices of both: predictions and labels into separate Tensors
        # pred/labels_idx hold the indices of each corresponsing matched pair
        # pred/labels_batch_idx is just a Tensor with the batch that each index belongs to
        pred_all_indices = [(torch.full_like(pred, batch), pred) for batch, (pred, _) in enumerate(indices)]
        pred_batch_idx, pred_idx = map(torch.cat, zip(*pred_all_indices))

        # compute the classification loss: -log(P(Ci))
        classes_pred = predictions['logits']

        device = classes_pred.device

        target_classes = torch.cat([batch['classes'][J] for batch, (_, J) in zip(labels, indices)])
        all_target_classes = torch.full(
            (classes_pred.size(0), classes_pred.size(1)),
            self.no_class_index, dtype=torch.int64,
            device=device
        )
        all_target_classes[pred_batch_idx, pred_idx] = target_classes

        # down-weight the no-object class to account for class imbalance
        classes_weights = torch.ones((self.no_class_index + 1,), device=device)
        classes_weights[self.no_class_index] = self.no_class_weight

        class_loss = F.cross_entropy(
            classes_pred.transpose(1, 2),
            all_target_classes,
            weight=classes_weights
        )
        return class_loss

    @staticmethod
    def bbox_losses(predictions, labels, indices):
        """Returns (giou_loss, l1_loss) of predictions and labels bounding boxes."""

        pred_all_indices = [(torch.full_like(pred, batch), pred) for batch, (pred, _) in enumerate(indices)]
        pred_batch_idx, pred_idx = map(torch.cat, zip(*pred_all_indices))

        boxes_pred = predictions['bboxes']  # [batch_size, n_object_queries, 2]
        boxes_pred = boxes_pred[pred_batch_idx, pred_idx]

        boxes_labels = torch.cat([batch['bboxes'][J] for batch, (_, J) in zip(labels, indices)])

        flat_boxes_pred = boxes_pred
        flat_boxes_labels = boxes_labels

        # compute the L1 loss
        l1_loss = F.l1_loss(flat_boxes_pred, flat_boxes_labels)

        return l1_loss

