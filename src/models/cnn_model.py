"""
ResNet-1D Architecture for IDS
Optimized for Tabular Data: Uses Residual Connections + Wide Kernels to capture feature interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.1) 
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # Skip Connection 
        out = self.relu(out)
        return out

class IDS_CNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU(0.1)

        self.layer1 = self._make_layer(64, 64,  blocks=2, kernel_size=5, padding=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, kernel_size=3, padding=1, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, kernel_size=3, padding=1, stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
        
        self.num_classes = num_classes

    def _make_layer(self, in_channels, out_channels, blocks, kernel_size, padding, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, kernel_size, stride, padding))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, kernel_size, 1, padding))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        elif x.dim() == 4: x = x.view(x.size(0), 1, -1)

        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1) 
        
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_ids_model(mode="binary", num_classes=None):
    if mode == "binary":
        return IDS_CNN(num_classes=2)
    elif mode == "multiclass":
        assert num_classes is not None, "num_classes required"
        return IDS_CNN(num_classes=num_classes)
    else:
        raise ValueError("Invalid mode")