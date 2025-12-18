"""
ResNet1D - NANO PLUS (PURE MUSCLE EDITION)
Target: Maximum Efficiency via Simplicity + Capacity
Specs: 
- Channels: 24 (High Capacity)
- Activation: ReLU (Sharp Decision)
- Norm: GroupNorm (Stability)
- Attention: None (Raw Speed)
Params: ~64k
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        # 1. Konvolüsyon
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        # GroupNorm (4 grup her zaman idealdir)
        self.bn1 = nn.GroupNorm(4, out_channels) 
        self.relu = nn.ReLU(inplace=True) # Klasik ve Keskin
        
        # 2. Konvolüsyon
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.GroupNorm(4, out_channels)
        
        # DİKKAT: Attention yok! Saf işlem gücü.
        
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

class ResNet1D_NanoPlus(nn.Module):
    def __init__(self, num_classes=7, input_dim=39):
        super(ResNet1D_NanoPlus, self).__init__()
        
        # KAPASİTE: 24 Kanal (16 yerine)
        self.inplanes = 24
        
        # Giriş Katmanı
        self.conv1 = nn.Conv1d(1, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(4, 24)
        self.relu = nn.ReLU(inplace=True)
        
        # Katmanlar: 24 -> 48 -> 96
        # Her katmanda 1 blok (Nano yapısı)
        self.layer1 = self._make_layer(24, blocks=1, stride=1)
        self.layer2 = self._make_layer(48, blocks=1, stride=2)
        self.layer3 = self._make_layer(96, blocks=1, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Dropout standart 0.2
        self.dropout = nn.Dropout(0.2) 
        
        # Çıkış: 96 özellik -> Sınıflar
        self.fc = nn.Linear(96, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(4, planes),
            )
        layers = []
        layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def create_ids_model(mode: str = "multiclass", num_classes: int = 7, input_dim: int = 39):
    print(f"[FACTORY] Initializing ResNet1D-NanoPlus (24Ch Pure Muscle).")
    model = ResNet1D_NanoPlus(num_classes=num_classes, input_dim=input_dim)
    return model