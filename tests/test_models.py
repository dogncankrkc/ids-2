"""
Tests for IDS CNN Model Architecture
"""

import pytest
import torch
from src.models.cnn_model import IDS_CNN, create_ids_model


class TestIDSModel:
    """Tests for IDS_CNN model."""

    def test_init_binary(self):
        model = IDS_CNN(num_classes=2)
        assert model.fc2.out_features == 2   # binary IDS

    def test_init_multiclass(self):
        model = IDS_CNN(num_classes=8)   
        assert model.fc2.out_features == 8

    def test_forward_shape(self):
        model = IDS_CNN(num_classes=7)
        x = torch.randn(8, 1, 7, 10)  # (batch, channels, height, width)
        out = model(x)
        assert out.shape == (8, 7)

    def test_count_parameters(self):
        model = IDS_CNN(num_classes=7)
        assert model.count_parameters() > 0

    def test_factory_binary(self):
        model = create_ids_model(mode="binary")
        assert model.fc2.out_features == 2

    def test_factory_multiclass(self):
        model = create_ids_model(mode="multiclass", num_classes=5)
        assert model.fc2.out_features == 5

    def test_invalid_factory(self):
        with pytest.raises(ValueError):
            create_ids_model(mode="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
