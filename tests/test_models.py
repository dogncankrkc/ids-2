"""
Tests for CNN Model Architectures
"""

import pytest
import torch

from src.models.cnn_model import SimpleCNN, VGGStyleCNN, create_model


class TestSimpleCNN:
    """Tests for SimpleCNN model."""

    def test_init_default(self):
        """Test default initialization."""
        model = SimpleCNN()
        assert model.num_classes == 10
        assert model.input_channels == 3
        assert model.input_size == (32, 32)

    def test_init_custom(self):
        """Test custom initialization."""
        model = SimpleCNN(num_classes=100, input_channels=1, input_size=(64, 64))
        assert model.num_classes == 100
        assert model.input_channels == 1
        assert model.input_size == (64, 64)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = SimpleCNN(num_classes=10)
        x = torch.randn(8, 3, 32, 32)
        output = model(x)
        assert output.shape == (8, 10)

    def test_forward_batch_size_1(self):
        """Test forward pass with batch size 1."""
        model = SimpleCNN(num_classes=10)
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        assert output.shape == (1, 10)

    def test_count_parameters(self):
        """Test parameter counting."""
        model = SimpleCNN()
        num_params = model.count_parameters()
        assert num_params > 0
        assert isinstance(num_params, int)

    def test_grayscale_input(self):
        """Test model with grayscale input."""
        model = SimpleCNN(input_channels=1)
        x = torch.randn(4, 1, 32, 32)
        output = model(x)
        assert output.shape == (4, 10)


class TestVGGStyleCNN:
    """Tests for VGGStyleCNN model."""

    def test_init_default(self):
        """Test default initialization."""
        model = VGGStyleCNN()
        assert model.num_classes == 10
        assert model.input_channels == 3

    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = VGGStyleCNN(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        output = model(x)
        assert output.shape == (4, 10)

    def test_different_num_classes(self):
        """Test model with different number of classes."""
        model = VGGStyleCNN(num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 100)


class TestCreateModel:
    """Tests for model factory function."""

    def test_create_simple_model(self):
        """Test creating simple CNN model."""
        model = create_model(model_type="simple", num_classes=5)
        assert isinstance(model, SimpleCNN)
        assert model.num_classes == 5

    def test_create_vgg_model(self):
        """Test creating VGG-style model."""
        model = create_model(model_type="vgg", num_classes=20)
        assert isinstance(model, VGGStyleCNN)
        assert model.num_classes == 20

    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        with pytest.raises(ValueError):
            create_model(model_type="invalid")

    def test_create_model_with_all_params(self):
        """Test creating model with all parameters."""
        model = create_model(
            model_type="simple",
            num_classes=50,
            input_channels=1,
            input_size=(64, 64),
        )
        assert isinstance(model, SimpleCNN)
        assert model.num_classes == 50
        assert model.input_channels == 1
        assert model.input_size == (64, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
