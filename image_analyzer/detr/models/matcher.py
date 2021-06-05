"""
Basically Facebook's original implementation of the hungarian matcher, commented.
"""

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    def __init__(self, lambda_classes, lambda_l1):
        super().__init__()
        self.lambda_classes = lambda_classes
        self.lambda_l1 = lambda_l1

    def forward(self, predictions, labels):
        """Calculate the matching indices between predictions and labels.

        Args:
            predictions: Dict with keys:
                bboxes: Tensor [batch_size, n_object_queries, 4].
                logits: Tensor [batch_size, n_object_queries, 92].

            labels: List [batch_size] of dicts of keys:
                bboxes: Tensor [n_objects_in_image, 4]
                classes: Tensor [n_objects_in_image]
        """
        batch_size, num_queries = predictions['logits'].shape[:-1]

        # We flatten the batch dimension to make the computations easier
        # Due to images containing a different number of boxes, this way you
        # generalize
        flat_boxes_pred = predictions['bboxes'].flatten(0, 1)
        flat_boxes_pred = flat_boxes_pred.detach()

        flat_boxes_labels = torch.cat([label['bboxes'] for label in labels])
        flat_boxes_labels = flat_boxes_labels.detach()

        pairwise_l1_loss = torch.cdist(flat_boxes_pred, flat_boxes_labels, p=1)
        boxes_loss = self.lambda_l1 * pairwise_l1_loss

        # They use the prediction directly as a loss instead of cross-entropy
        flat_classes_pred = predictions['logits'].flatten(0, 1)
        flat_classes_pred = flat_classes_pred.detach()

        flat_classes_labels = torch.cat([label['classes'] for label in labels])
        flat_classes_labels = flat_classes_labels.detach()

        class_loss = -flat_classes_pred[:, flat_classes_labels]

        hungarian_loss = self.lambda_classes * class_loss + boxes_loss

        # Now to that you have the pairwise match loss, split it into the original batches
        # computation has to be on cpu because of scipy
        hungarian_loss = hungarian_loss.view(batch_size, num_queries, -1).cpu()

        boxes_per_batch = [label['bboxes'].size(0) for label in labels]

        # Iterate over every batch and solve for the assignment of indices that minimize loss
        # eg. (src2, src3, src1) that matches (tgt1, tgt3, tgt2)
        indices = [linear_sum_assignment(cost_matrix[n_batch]) for n_batch, cost_matrix in
                   enumerate(hungarian_loss.split(boxes_per_batch, -1))]

        indices = [[torch.as_tensor(pred_idx, dtype=torch.int64), torch.as_tensor(tgt_idx, dtype=torch.int64)]
                   for pred_idx, tgt_idx in indices]

        return indices

