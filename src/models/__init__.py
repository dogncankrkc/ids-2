"""
Models Module

Exports:
    - IDS_CNN: The main CNN architecture class
    - create_ids_model: Factory function to instantiate models easily
"""

from .cnn_model import ResNet1D_Nano_Prime, create_ids_model
from .binary_models import create_binary_model

__all__ = [
    "ResNet1D_Nano_Prime",
    "create_ids_model",
    "create_binary_model",
]