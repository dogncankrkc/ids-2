"""
CNN Model Architectures

This module defines various CNN model architectures that can be used
for image classification and other computer vision tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class CNNModel(nn.Module):
    """
    Base CNN Model class that provides common functionality
    for all CNN architectures.
    """

    def __init__(self):
        super(CNNModel, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleCNN(CNNModel):
    """
    A simple CNN architecture for basic image classification tasks.

    Architecture:
        - 2 Convolutional layers with ReLU activation and MaxPooling
        - 2 Fully connected layers
        - Dropout for regularization

    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        input_size: Input image size (height, width) - default (32, 32)
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        input_size: Tuple[int, int] = (32, 32),
    ):
        super(SimpleCNN, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_size = input_size

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size after convolutions and pooling
        conv_output_size = self._get_conv_output_size()

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def _get_conv_output_size(self) -> int:
        """Calculate the output size after convolutional layers."""
        # After 2 pooling operations, size is reduced by factor of 4
        h = self.input_size[0] // 4
        w = self.input_size[1] // 4
        return 64 * h * w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First convolutional block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Second convolutional block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


class VGGStyleCNN(CNNModel):
    """
    A VGG-style CNN architecture with deeper convolutional layers.

    Architecture:
        - Multiple convolutional blocks with increasing filters
        - Batch normalization after each conv layer
        - MaxPooling after each block
        - Fully connected classifier

    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels
        input_size: Input image size (height, width)
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        input_size: Tuple[int, int] = (32, 32),
    ):
        super(VGGStyleCNN, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_size = input_size

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate classifier input size
        feature_size = self._get_feature_size()

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _get_feature_size(self) -> int:
        """Calculate the feature size after convolutional layers."""
        # After 3 pooling operations, size is reduced by factor of 8
        h = self.input_size[0] // 8
        w = self.input_size[1] // 8
        return 256 * h * w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_model(
    model_type: str = "simple",
    num_classes: int = 10,
    input_channels: int = 3,
    input_size: Tuple[int, int] = (32, 32),
) -> CNNModel:
    """
    Factory function to create CNN models.

    Args:
        model_type: Type of model ("simple" or "vgg")
        num_classes: Number of output classes
        input_channels: Number of input channels
        input_size: Input image size

    Returns:
        CNNModel instance
    """
    models = {
        "simple": SimpleCNN,
        "vgg": VGGStyleCNN,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](
        num_classes=num_classes,
        input_channels=input_channels,
        input_size=input_size,
    )
