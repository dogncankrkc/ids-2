"""
Lightweight CNN Architectures for IDS (Intrusion Detection System)

This module defines a compact CNN architecture optimized for:
    - Binary IDS classification  → using 'binary_label'
    - Multiclass IDS classification  → using 'label2'
    - Edge deployment (Raspberry Pi 4 via TFLite)

Input format expected:
    (batch_size, channels=1, height=7, width=10)

The model is intentionally lightweight:
    - < 1 million parameters
    - Supports INT8 quantization for TFLite
    - Fast inference on Raspberry Pi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------
# Base class (optional, keeps code structured & clean)
# -------------------------------------------------------
class CNNBase(nn.Module):
    """Base class with helper methods."""

    def __init__(self):
        super(CNNBase, self).__init__()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -------------------------------------------------------
# IDS CNN Model – Main Model to be Used
# -------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class IDS_CNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        # --- CONV LAYERS ---
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # --- INIT LATER ---
        self.fc1 = nn.Linear(64 * 1 * 2, 64) 
        self.fc2 = nn.Linear(64, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # → (batch, 16, 3,5)
        x = self.pool(F.relu(self.conv2(x)))   # → (batch, 32, 1,2)
        x = F.relu(self.conv3(x))              # → (batch, 64, 1,2)

        x = self.dropout(x)

        x = torch.flatten(x, 1)   # (batch, ???)

        if self.fc1 is None:
            in_features = x.size(1)            
            print(f"[INFO] Dynamically setting fc1 input size = {in_features}")

            self.fc1 = nn.Linear(in_features, 64).to(x.device)
            self.fc2 = nn.Linear(64, self.num_classes).to(x.device)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def count_parameters(self):
        """Trainable (update edilebilir) parametre sayısını döndürür."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# -------------------------------------------------------
# Model Factory Function
# -------------------------------------------------------
def create_ids_model(mode="binary", num_classes=None):
    """
    Factory for creating the IDS CNN model.

    Args:
        mode        : "binary" or "multiclass"
        num_classes : required if mode="multiclass"
    """

    if mode == "binary":
        return IDS_CNN(num_classes=2)

    elif mode == "multiclass":
        assert num_classes is not None, \
            "num_classes must be provided for multiclass mode"
        return IDS_CNN(num_classes=num_classes)

    else:
        raise ValueError("mode must be 'binary' or 'multiclass'")
