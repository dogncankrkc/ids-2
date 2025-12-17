"""
Hybrid Architecture: ResNet1D + SE Attention + Dual Pooling
GOAL: High Accuracy (>90%) with Low Parameters (<300k) for Raspberry Pi
Feature:
  1. SE Blocks: Re-calibrates feature maps (Smart Focus).
  2. Dual Pooling: Concatenates AvgPool and MaxPool (Captures peaks and averages).
  3. Compact Design: Max 128 channels + Dropout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. SE Block (Squeeze-and-Excitation)
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        # Özellik haritasını global olarak sıkıştır ve önem skorlarını hesapla
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid() # 0-1 arası önem katsayısı
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        # Orijinal özellikleri önem katsayılarıyla çarp
        return x * y.expand_as(x)

# ==========================================
# 2. Residual Block with Attention
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        # Conv1
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Conv2
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # --- HYBRID DOKUNUŞ: ATTENTION ---
        self.se = SEBlock(out_channels)
        # ---------------------------------
        
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Attention Uygula (Önemliyi öne çıkar)
        out = self.se(out)
        
        if self.downsample:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out

# ==========================================
# 3. Main Hybrid Model
# ==========================================
class HybridResNet(nn.Module):
    def __init__(self, num_classes=8, input_dim=39):
        super(HybridResNet, self).__init__()
        
        # Başlangıç: 32 Kanal
        self.inplanes = 32
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Katmanlar (Derin ama Dar - 128 Kanal Maksimum)
        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        # Layer 4'ü kaldırdık çünkü 256 kanala gerek yok, Attention açığı kapatıyor.
        
        # Dropout (Overfitting İlacı)
        self.dropout = nn.Dropout(0.5)
        
        # Çıkış Katmanı: 128 (Avg) + 128 (Max) = 256 Giriş
        self.fc = nn.Linear(128 * 2, num_classes)

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
        # Format düzeltme (N, 39) -> (N, 1, 39)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # --- DUAL POOLING (Hibrit Havuzlama) ---
        # AvgPool: Genel gidişatı yakalar
        x_avg = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        # MaxPool: Ani saldırı sinyallerini (Peak) yakalar
        x_max = F.adaptive_max_pool1d(x, 1).squeeze(2)
        
        # İkisini birleştir
        x = torch.cat([x_avg, x_max], dim=1) # (Batch, 256)
        # ---------------------------------------
        
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Factory Fonksiyonu (train.py bunu çağıracak)
def create_ids_model(mode: str = "multiclass", num_classes: int = 8, input_dim: int = 39):
    print(f"[FACTORY] Initializing Hybrid-ResNet (SE-Attention + DualPool).")
    model = HybridResNet(num_classes=num_classes, input_dim=input_dim)
    return model