"""
Lightweight 1D CNN Architectures for IDS
OPTIMIZED VERSION: Switched from 2D (Image) to 1D (Sequence) processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class IDS_CNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        # --- 1D CONVOLUTION LAYERS ---
        # Giriş verisini (Batch, 1, 49) olarak alacağız.
        
        # Katman 1: 1 kanal -> 32 kanal
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32) # Batch Normalization öğrenmeyi hızlandırır
        
        # Katman 2: 32 -> 64 kanal
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        # Katman 3: 64 -> 128 kanal
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.4) # Dropout artırıldı

        # --- FULLY CONNECTED ---
        # Boyut Hesabı:
        # Giriş: 49
        # Pool 1 sonrası: 49 / 2 = 24
        # Pool 2 sonrası: 24 / 2 = 12
        # Pool 3 sonrası: 12 / 2 = 6
        # Çıkış boyutu: 128 kanal * 6 uzunluk = 768
        
        self.fc1 = nn.Linear(128 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.num_classes = num_classes

    def forward(self, x):
        # KRİTİK ADIM: 2D Gelen veriyi (N, 1, 7, 7) -> 1D'ye (N, 1, 49) çeviriyoruz.
        # Bu sayede preprocess.py veya train.py değiştirmeye gerek kalmıyor.
        if x.dim() == 4:
            x = x.view(x.size(0), 1, -1) 

        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
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