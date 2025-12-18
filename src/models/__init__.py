"""
Models Module

Exports:
    - IDS_CNN: The main CNN architecture class
    - create_ids_model: Factory function to instantiate models easily
"""

from .cnn_model import ResidualBlockSE, create_ids_model

__all__ = [
    "ResidualBlockSE",
    "create_ids_model",
]