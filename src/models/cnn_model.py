"""
ResNet1D Nano Prime (Dual-Pooling Hybrid Head)

This module defines a lightweight 1D ResNet architecture optimized for
multiclass intrusion detection on tabular/network traffic data.
It uses GroupNorm for training stability and a dual-pooling (Avg + Max)
classification head to capture both global traffic volume patterns
and short-lived spike-based attack behaviors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Standard Residual Block with Group Normalization
# --------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        # First convolution
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.GroupNorm(4, out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolution
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.GroupNorm(4, out_channels)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# --------------------------------------------------
# ResNet1D Nano Prime Model
# --------------------------------------------------
class ResNet1D_Nano_Prime(nn.Module):
    """
    Lightweight ResNet1D architecture (~28k parameters).

    Key idea:
    - Dual pooling head (AdaptiveAvgPool + AdaptiveMaxPool)
      to capture both distributed traffic patterns and sharp spikes.
    """

    def __init__(self, num_classes=7, input_dim=39):
        super(ResNet1D_Nano_Prime, self).__init__()

        # Initial channel width (kept small for efficiency)
        self.inplanes = 16

        # Input stem
        self.conv1 = nn.Conv1d(
            1,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.GroupNorm(4, 16)
        self.relu = nn.ReLU(inplace=True)

        # Residual stages: 16 -> 32 -> 64
        self.layer1 = self._make_layer(16, blocks=1, stride=1)
        self.layer2 = self._make_layer(32, blocks=1, stride=2)
        self.layer3 = self._make_layer(64, blocks=1, stride=2)

        # Dual pooling head
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.dropout = nn.Dropout(0.3)

        # Classification layer (Avg + Max pooled features)
        self.fc = nn.Linear(64 * 2, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.GroupNorm(4, planes),
            )

        layers = []
        layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Expand shape from (N, L) to (N, 1, L) if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Dual pooling
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)

        # Concatenate pooled features
        x = torch.cat([x_avg, x_max], dim=1)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------
# Model factory
# --------------------------------------------------
def create_ids_model(
    mode: str = "multiclass",
    num_classes: int = 7,
    input_dim: int = 39
):
    print("[FACTORY] Initializing ResNet1D Nano Prime (Dual-Pooling Head)")
    model = ResNet1D_Nano_Prime(
        num_classes=num_classes,
        input_dim=input_dim
    )
    print(f"[INFO] Trainable parameters: {model.count_parameters():,}")
    return model
