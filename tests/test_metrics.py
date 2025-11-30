"""
Tests for Training Metrics
"""

import pytest
import torch
import numpy as np

from src.training.metrics import accuracy, precision, recall, f1_score, confusion_matrix


class TestAccuracy:
    """Tests for accuracy metric."""

    def test_perfect_accuracy(self):
        """Test accuracy with perfect predictions."""
        predictions = torch.tensor([0, 1, 2, 3, 4])
        targets = torch.tensor([0, 1, 2, 3, 4])
        acc = accuracy(predictions, targets)
        assert acc == 100.0

    def test_zero_accuracy(self):
        """Test accuracy with all wrong predictions."""
        predictions = torch.tensor([1, 2, 3, 4, 0])
        targets = torch.tensor([0, 1, 2, 3, 4])
        acc = accuracy(predictions, targets)
        assert acc == 0.0

    def test_partial_accuracy(self):
        """Test accuracy with partial correct predictions."""
        predictions = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 1, 0])
        acc = accuracy(predictions, targets)
        assert acc == 50.0

    def test_accuracy_with_logits(self):
        """Test accuracy with logit inputs."""
        predictions = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
        targets = torch.tensor([0, 1])
        acc = accuracy(predictions, targets)
        assert acc == 100.0


class TestPrecisionRecall:
    """Tests for precision and recall metrics."""

    def test_precision_perfect(self):
        """Test precision with perfect predictions."""
        predictions = torch.tensor([0, 1, 2])
        targets = torch.tensor([0, 1, 2])
        prec = precision(predictions, targets)
        assert prec == 1.0

    def test_recall_perfect(self):
        """Test recall with perfect predictions."""
        predictions = torch.tensor([0, 1, 2])
        targets = torch.tensor([0, 1, 2])
        rec = recall(predictions, targets)
        assert rec == 1.0


class TestF1Score:
    """Tests for F1 score metric."""

    def test_f1_perfect(self):
        """Test F1 score with perfect predictions."""
        predictions = torch.tensor([0, 1, 2])
        targets = torch.tensor([0, 1, 2])
        f1 = f1_score(predictions, targets)
        assert f1 == 1.0


class TestConfusionMatrix:
    """Tests for confusion matrix."""

    def test_confusion_matrix_shape(self):
        """Test confusion matrix shape."""
        predictions = torch.tensor([0, 1, 2, 0, 1, 2])
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        cm = confusion_matrix(predictions, targets, num_classes=3)
        assert cm.shape == (3, 3)

    def test_confusion_matrix_diagonal(self):
        """Test confusion matrix with perfect predictions."""
        predictions = torch.tensor([0, 1, 2])
        targets = torch.tensor([0, 1, 2])
        cm = confusion_matrix(predictions, targets, num_classes=3)
        # Perfect predictions should have all values on diagonal
        assert cm[0, 0] == 1
        assert cm[1, 1] == 1
        assert cm[2, 2] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
