"""
Training Module

Exports:
    - Trainer: Main class to handle training loop
    - Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
"""

from .trainer import Trainer
from .metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix,  # <-- YENİ EKLENDİ
)

__all__ = [
    "Trainer",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix",
]