"""
Evaluation Metrics (Optimized with Scikit-Learn)

This module provides high-performance metrics for evaluating CNN model performance.
It wraps sklearn.metrics to maintain compatibility with the existing trainer loop.
"""

from typing import Optional
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score as sklearn_f1_score,
    confusion_matrix as sklearn_confusion_matrix
)

def _prepare_data(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Helper to move tensors to CPU and convert to numpy for sklearn.
    """
    # Eğer logits geldiyse (batch, num_classes) ve argmax gerekiyorsa
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)
    
    # CPU'ya al ve numpy array yap (Gradient takibini kopararak)
    preds_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    return preds_np, targets_np


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate accuracy using sklearn.
    Returns percentage (0-100).
    """
    p, t = _prepare_data(predictions, targets)
    # Sklearn 0-1 arası döner, trainer.py yüzde beklediği için 100 ile çarpıyoruz.
    return accuracy_score(t, p) * 100.0


def precision(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None, # İmza uyumu için tutuyoruz
    average: str = "macro",
) -> float:
    """
    Calculate precision using sklearn.
    Returns fraction (0-1).
    """
    p, t = _prepare_data(predictions, targets)
    # zero_division=0: Tahmin edilmeyen sınıflar için hata vermek yerine 0 döner.
    return float(precision_score(t, p, average=average, zero_division=0))


def recall(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    average: str = "macro",
) -> float:
    """
    Calculate recall using sklearn.
    Returns fraction (0-1).
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
    Calculate F1 score using sklearn.
    Returns fraction (0-1).
    """
    p, t = _prepare_data(predictions, targets)
    return float(sklearn_f1_score(t, p, average=average, zero_division=0))


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Calculate confusion matrix using sklearn.
    Returns numpy array.
    """
    p, t = _prepare_data(predictions, targets)
    return sklearn_confusion_matrix(t, p)


def get_predictions_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Model çıktısını (logits) sınıf tahminlerine (index) çevirir.
    
    Binary case (1 output): Sigmoid > 0.5
    Multiclass case (N outputs): Argmax
    """
    # Binary case → output shape: (batch, 1) veya (batch,)
    if logits.dim() == 1 or logits.shape[1] == 1:
        probs = torch.sigmoid(logits)
        return (probs >= 0.5).long()

    # Multiclass case (batch, num_classes)
    # Senin modelin binary modda bile 2 output verdiği için burası çalışacak (Doğru olan bu)
    return torch.argmax(logits, dim=1)