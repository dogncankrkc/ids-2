"""
Training Module for IDS (CNN-based)

This package exports:
    - Trainer class
    - Evaluation metrics (accuracy, precision, recall, F1)
    - Loss function utilities for:
        * Binary classification (BCEWithLogitsLoss)
        * Multiclass classification (CrossEntropyLoss)
"""

from .trainer import Trainer
from .metrics import accuracy, precision, recall, f1_score
import torch.nn as nn

# LOSS SELECTION HELPERS
def get_loss(binary: bool = True):
    """
    Returns appropriate loss function for IDS CNN:
        - Binary   → BCEWithLogitsLoss
        - Multiclass → CrossEntropyLoss
    """
    return nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()


__all__ = [
    # Trainer
    "Trainer",

    # Metrics
    "accuracy",
    "precision",
    "recall",
    "f1_score",

    # Loss selector
    "get_loss",
]
