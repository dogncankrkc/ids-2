"""
Focal Loss (Multiclass)

This module implements the Focal Loss function for multiclass classification.
Focal Loss reduces the contribution of easy-to-classify samples and focuses
training on hard and minority-class examples, making it well-suited for
imbalanced intrusion detection datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss implementation.

    Motivation:
    Standard Cross Entropy loss is dominated by easy samples.
    Focal Loss down-weights well-classified examples and forces the model
    to focus on hard or minority-class samples.

    Parameters:
        alpha (Tensor, optional): Class weights for imbalance handling.
        gamma (float): Focusing parameter controlling down-weighting strength.
        reduction (str): 'mean', 'sum', or 'none'.
        device (str or torch.device): Target device for tensors.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean", device="cpu"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.device = device

        # Optional class weighting (alpha)
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha.to(device)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Compute the Focal Loss.

        Args:
            inputs (Tensor): Logits of shape [N, C] (no softmax applied).
            targets (Tensor): Ground-truth class indices of shape [N].

        Returns:
            Tensor: Computed focal loss value.
        """

        # Compute standard cross-entropy loss without reduction
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction="none",
        )

        # Compute pt = probability of the correct class
        pt = torch.exp(-ce_loss)

        # Apply focal loss modulation
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
