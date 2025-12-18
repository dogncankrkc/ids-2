"""
ResNet1D - NANO STABLE (+ECA Attention)
Goal: Better class separation (esp. Web/BruteForce) without losing speed.

- GroupNorm for stability
- ECA (very lightweight channel attention)
- Still tiny parameter count (<< 1M)
"""

import torch
import torch.nn as nn


class ECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA)
    Ultra-lightweight attention: no FC layers, just 1D conv on pooled channels.
    """
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        y = self.avg_pool(x)          # (B, C, 1)
        y = y.transpose(1, 2)         # (B, 1, C)
        y = self.conv(y)              # (B, 1, C)
        y = self.sigmoid(y).transpose(1, 2)  # (B, C, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_eca=True):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(4, out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(4, out_channels)

        self.eca = ECALayer(out_channels, k_size=3) if use_eca else nn.Identity()
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))

        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out + residual)
        return out


class ResNet1D_Nano(nn.Module):
    def __init__(self, num_classes=7, dropout=0.2, use_eca=True):
        super().__init__()

        self.inplanes = 16

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(4, 16)
        self.relu = nn.ReLU(inplace=True)

        # Nano: 1 block per layer
        self.layer1 = self._make_layer(16, blocks=1, stride=1, use_eca=use_eca)
        self.layer2 = self._make_layer(32, blocks=1, stride=2, use_eca=use_eca)
        self.layer3 = self._make_layer(64, blocks=1, stride=2, use_eca=use_eca)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, blocks, stride=1, use_eca=True):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(4, planes),
            )

        layers = [ResidualBlock(self.inplanes, planes, stride=stride, downsample=downsample, use_eca=use_eca)]
        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes, stride=1, downsample=None, use_eca=use_eca))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Accept (N, L) or (N, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.relu(self.gn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_ids_model(mode: str = "multiclass", num_classes: int = 7, input_dim: int = 39, dropout: float = 0.2, use_eca: bool = True):
    # input_dim is kept for compatibility; model works with any L.
    print("[FACTORY] Initializing ResNet1D-Nano (+ECA, GroupNorm).")
    return ResNet1D_Nano(num_classes=num_classes, dropout=dropout, use_eca=use_eca)
