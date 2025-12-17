"""
ResNet-MLP with Squeeze-and-Excitation (SE) Attention
VERSION: FINAL PRO (Attention Mechanism)
Target: High F1-Score & Accuracy
Feature: Dynamic Feature Weighting (Model decides what's important per sample)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# SE BLOCK (ATTENTION MECHANISM)
# =====================================================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # Özellik sayısını önce daralt (Squeeze), sonra genişlet (Excitation)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid() # 0 ile 1 arası önem skoru üretir
        )

    def forward(self, x):
        # x: (Batch, Channel)
        y = self.fc(x)
        return x * y # Özellikleri önem skorlarıyla çarp

# =====================================================
# RESIDUAL BLOCK WITH ATTENTION
# =====================================================
class ResidualBlockFC(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        
        # YENİ: Attention Bloğu
        # reduction=16: Parametre tasarrufu için oranı küçültür
        self.se = SEBlock(out_features, reduction=16)
        
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.block(x)
        out = self.se(out) # Çıkışı Attention ile ağırlıklandır
        return F.gelu(out + residual)

# =====================================================
# MAIN MODEL
# =====================================================
class IDS_ResNet_MLP(nn.Module):
    def __init__(self, num_classes=8, input_dim=39):
        super().__init__()
        
        # 1. Stem (Genişletilmiş Giriş - Projection)
        self.stem = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4), # 39 -> 156
            nn.BatchNorm1d(input_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 4, 512),       # 156 -> 512
            nn.BatchNorm1d(512),
            nn.GELU()
        )
        
        # 2. Deep Body with Attention
        # Parametreleri çok şişirmemek için dropout'u biraz artırdık
        self.layer1 = ResidualBlockFC(512, 512, dropout=0.3) 
        self.layer2 = ResidualBlockFC(512, 256, dropout=0.3) 
        self.layer3 = ResidualBlockFC(256, 128, dropout=0.3)
        
        # 3. Head
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
            
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc_out(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_ids_model(mode: str = "multiclass", num_classes: int = 8, input_dim: int = 39):
    print(f"[FACTORY] Initializing IDS_ResNet_MLP (SE-Attention Edition).")
    model = IDS_ResNet_MLP(num_classes=num_classes, input_dim=input_dim)
    return model