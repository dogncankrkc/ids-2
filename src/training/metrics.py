"""
Evaluation Metrics

This module provides various metrics for evaluating CNN model performance.
"""

from typing import Optional

import torch
import numpy as np


def accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Calculate accuracy.

    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth labels

    Returns:
        Accuracy as a percentage
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)

    correct = (predictions == targets).sum().item()
    total = targets.size(0)

    return 100.0 * correct / total


def precision(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    average: str = "macro",
) -> float:
    """
    Calculate precision.

    Args:
        predictions: Model predictions
        targets: Ground truth labels
        num_classes: Number of classes
        average: Averaging method ('macro', 'micro', 'weighted')

    Returns:
        Precision score
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)

    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    if num_classes is None:
        num_classes = max(predictions.max(), targets.max()) + 1

    precision_scores = []
    weights = []

    for cls in range(num_classes):
        tp = np.sum((predictions == cls) & (targets == cls))
        fp = np.sum((predictions == cls) & (targets != cls))

        if tp + fp > 0:
            precision_scores.append(tp / (tp + fp))
        else:
            precision_scores.append(0.0)

        weights.append(np.sum(targets == cls))

    if average == "macro":
        return float(np.mean(precision_scores))
    elif average == "micro":
        tp_total = np.sum(predictions == targets)
        return float(tp_total / len(targets))
    elif average == "weighted":
        weights = np.array(weights) / sum(weights)
        return float(np.sum(np.array(precision_scores) * weights))
    else:
        raise ValueError(f"Unknown average method: {average}")


def recall(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    average: str = "macro",
) -> float:
    """
    Calculate recall.

    Args:
        predictions: Model predictions
        targets: Ground truth labels
        num_classes: Number of classes
        average: Averaging method ('macro', 'micro', 'weighted')

    Returns:
        Recall score
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)

    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    if num_classes is None:
        num_classes = max(predictions.max(), targets.max()) + 1

    recall_scores = []
    weights = []

    for cls in range(num_classes):
        tp = np.sum((predictions == cls) & (targets == cls))
        fn = np.sum((predictions != cls) & (targets == cls))

        if tp + fn > 0:
            recall_scores.append(tp / (tp + fn))
        else:
            recall_scores.append(0.0)

        weights.append(np.sum(targets == cls))

    if average == "macro":
        return float(np.mean(recall_scores))
    elif average == "micro":
        tp_total = np.sum(predictions == targets)
        return float(tp_total / len(targets))
    elif average == "weighted":
        weights = np.array(weights) / sum(weights)
        return float(np.sum(np.array(recall_scores) * weights))
    else:
        raise ValueError(f"Unknown average method: {average}")


def f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    average: str = "macro",
) -> float:
    """
    Calculate F1 score.

    Args:
        predictions: Model predictions
        targets: Ground truth labels
        num_classes: Number of classes
        average: Averaging method

    Returns:
        F1 score
    """
    prec = precision(predictions, targets, num_classes, average)
    rec = recall(predictions, targets, num_classes, average)

    if prec + rec > 0:
        return 2 * (prec * rec) / (prec + rec)
    return 0.0


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Calculate confusion matrix.

    Args:
        predictions: Model predictions
        targets: Ground truth labels
        num_classes: Number of classes

    Returns:
        Confusion matrix as numpy array
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)

    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    if num_classes is None:
        num_classes = max(predictions.max(), targets.max()) + 1

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pred, target in zip(predictions, targets):
        matrix[target, pred] += 1

    return matrix

def get_predictions_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # Binary case → output shape: (batch, 1) veya (batch,)
    if logits.dim() == 1 or logits.shape[1] == 1:
        probs = torch.sigmoid(logits)
        return (probs >= 0.5).long()

    # Multiclass case
    return torch.argmax(logits, dim=1)
