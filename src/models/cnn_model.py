"""
CNN-2D Architecture for IDS (Tabular → Spatial Mapping)

Input:
    - Shape: (N, 1, 5, 8)
    - 40 features mapped to a 2D grid

Design Goals:
✓ Lightweight (Edge-friendly, Raspberry Pi 4B)
✓ Stable training (BatchNorm + Dropout)
✓ CNN2D-aware (Preserves spatial correlations)
✓ Binary & Multiclass support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# BASIC CONV BLOCK
# =====================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# =====================================================
# CNN2D MODEL
# =====================================================
class IDS_CNN2D(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()

        # -----------------------------
        # Feature extractor
        # -----------------------------
        self.features = nn.Sequential(
            ConvBlock(1, 16),      # (N, 16, 5, 8)
            ConvBlock(16, 32),     # (N, 32, 5, 8)
            nn.MaxPool2d(2),       # (N, 32, 2, 4)

            ConvBlock(32, 64),     # (N, 64, 2, 4)
            nn.MaxPool2d(2),       # (N, 64, 1, 2)

            ConvBlock(64, 128),    # (N, 128, 1, 2)
        )

        # -----------------------------
        # Global pooling + classifier
        # -----------------------------
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # (N, 128, 1, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(128, num_classes)

        self.num_classes = num_classes

    def forward(self, x):
        """
        Expected input:
            x -> (N, 1, 5, 8)
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (N, 1, 5, 8), got {x.shape}"
            )

        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (N, 128)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =====================================================
# MODEL FACTORY
# =====================================================
def create_ids_model(mode: str = "binary", num_classes: int = None):
    """
    Factory function used by train.py
    """
    if mode == "binary":
        return IDS_CNN2D(num_classes=2)

    elif mode == "multiclass":
        assert num_classes is not None, "num_classes must be provided"
        return IDS_CNN2D(num_classes=num_classes)

    else:
        raise ValueError("Invalid mode (binary | multiclass)")
