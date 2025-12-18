"""
ResNet1D - NANO STABLE (GroupNorm Edition)
Target: Stability & Speed (~30k Parameters)
Fix: Replaced BatchNorm with GroupNorm to stop validation fluctuations.
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
        # STABILITY FIX: BatchNorm yerine GroupNorm
        # GroupNorm(GrupSayısı, KanalSayısı). 4 Grup genelde idealdir.
        self.bn1 = nn.GroupNorm(4, out_channels) 
        self.relu = nn.ReLU(inplace=True)
        
        # 2. Konvolüsyon
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, 
            stride=1, padding=1, bias=False
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

class ResNet1D_Nano(nn.Module):
    def __init__(self, num_classes=7, input_dim=39):
        super(ResNet1D_Nano, self).__init__()
        
        self.inplanes = 16
        
        # Giriş Katmanı
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(4, 16) # Fix
        self.relu = nn.ReLU(inplace=True)
        
        # Katmanlar (Nano: Her layerda 1 blok)
        self.layer1 = self._make_layer(16, blocks=1, stride=1)
        self.layer2 = self._make_layer(32, blocks=1, stride=2)
        self.layer3 = self._make_layer(64, blocks=1, stride=2)
        # Layer 4'ü iptal ettik (Parametre tasarrufu için)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2) # Biraz düşürdük, GroupNorm zaten düzenliyor.
        
        # Final (64 Kanal -> 7 Sınıf)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(4, planes), # Fix
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

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_ids_model(mode: str = "multiclass", num_classes: int = 7, input_dim: int = 39):
    print(f"[FACTORY] Initializing ResNet1D-Nano (GroupNorm Stable).")
    model = ResNet1D_Nano(num_classes=num_classes, input_dim=input_dim)
    return model