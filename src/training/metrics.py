"""
Evaluation Metrics (Scikit-Learn Optimized)

This module provides fast and reliable evaluation metrics for CNN-based models.
It wraps sklearn.metrics while remaining fully compatible with a PyTorch
training loop by handling tensor-to-numpy conversions internally.
"""

from typing import Optional
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score as sklearn_f1_score,
)


def _prepare_data(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Convert model outputs and targets to CPU-based NumPy arrays
    suitable for scikit-learn metric computation.
    """
    # Convert logits to class indices if needed
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)

    # Detach from graph, move to CPU, and convert to NumPy
    preds_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    return preds_np, targets_np


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy using scikit-learn.

    Returns:
        Accuracy as a percentage (0–100).
    """
    p, t = _prepare_data(predictions, targets)
    return accuracy_score(t, p) * 100.0


def precision(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    average: str = "macro",
) -> float:
    """
    Compute precision score using scikit-learn.

    Returns:
        Precision as a fraction (0–1).
    """
    p, t = _prepare_data(predictions, targets)
    return float(precision_score(t, p, average=average, zero_division=0))


def recall(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    average: str = "macro",
) -> float:
    """
    Compute recall score using scikit-learn.

    Returns:
        Recall as a fraction (0–1).
    """
    p, t = _prepare_data(predictions, targets)
    return float(recall_score(t, p, average=average, zero_division=0))


def f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    average: str = "macro",
) -> float:
    """
    Compute F1 score using scikit-learn.

    Returns:
        F1 score as a fraction (0–1).
    """
    p, t = _prepare_data(predictions, targets)
    return float(sklearn_f1_score(t, p, average=average, zero_division=0))


def get_predictions_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert raw model logits into discrete class predictions.

    - Binary case (single output): sigmoid threshold at 0.5
    - Multiclass case: argmax over class dimension
    """
    # Binary classification case
    if logits.dim() == 1 or logits.shape[1] == 1:
        probs = torch.sigmoid(logits)
        return (probs >= 0.5).long()

    # Multiclass classification case
    return torch.argmax(logits, dim=1)
