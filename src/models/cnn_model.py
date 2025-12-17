"""
Optimized ResNet1D Architecture for IDS (LITE VERSION)
Target: ~960k Parameters (Perfect Balance for Edge AI)
Design: 
  - Reduced Channel Width (Starts at 32 instead of 64)
  - Retains Depth (Intelligence) but reduces Width (Bloat)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
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

class ResNet1D(nn.Module):
    def __init__(self, num_classes=8, input_dim=39):
        super(ResNet1D, self).__init__()
        
        # --- DİYET KISMI BAŞLIYOR ---
        # Giriş Kanalı: 1 -> 32 (Eskiden 64'tü)
        self.inplanes = 32
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Katmanlar (Genişlikleri yarıya indirdik)
        # Layer 1: 32 Filtre
        self.layer1 = self._make_layer(32, 2, stride=1)
        # Layer 2: 64 Filtre
        self.layer2 = self._make_layer(64, 2, stride=2)
        # Layer 3: 128 Filtre
        self.layer3 = self._make_layer(128, 2, stride=2)
        # Layer 4: 256 Filtre (Eskiden 512 idi)
        self.layer4 = self._make_layer(256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # Çıkış katmanı da küçüldü (256 -> num_classes)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_ids_model(mode: str = "multiclass", num_classes: int = 8, input_dim: int = 39):
    print(f"[FACTORY] Initializing ResNet1D-Lite (Optimized ~960k Params).")
    model = ResNet1D(num_classes=num_classes, input_dim=input_dim)
    return model