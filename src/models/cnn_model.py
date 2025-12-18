"""
ResNet1D - NANO PLUS (Enhanced Capacity)
Goal: Better separation of 'subtle' classes (Web vs Recon)
Change: Increased channel depth [32, 64, 128] -> ~100k Params
"""

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
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
        
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet1D_Nano(nn.Module):
    def __init__(self, num_classes=6, dropout=0.1): # Default 6 class
        super(ResNet1D_Nano, self).__init__()
        
        # --- KAPASİTE ARTIRIMI (NANO PLUS) ---
        # Giriş kanallarını ve katman genişliklerini artırıyoruz.
        self.inplanes = 32 # Eskiden 16 idi
        
        # Giriş: 1 -> 32
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(4, 32)
        self.relu = nn.ReLU(inplace=True)
        
        # Bloklar: [32, 64, 128] (Eskiden 16, 32, 64 idi)
        self.layer1 = self._make_layer(32, blocks=1, stride=1)
        self.layer2 = self._make_layer(64, blocks=1, stride=2)
        self.layer3 = self._make_layer(128, blocks=1, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        # Çıkış: 128 -> Num_Classes
        self.fc = nn.Linear(128, num_classes)

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

def create_ids_model(mode: str = "multiclass", num_classes: int = 6, input_dim: int = 39, dropout: float = 0.1, **kwargs):
    print(f"[FACTORY] Initializing ResNet1D-Nano PLUS (Enhanced Capacity).")
    model = ResNet1D_Nano(num_classes=num_classes, dropout=dropout)
    return model