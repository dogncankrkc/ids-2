"""
IDS CNN Models

This package provides lightweight CNN architectures
for Intrusion Detection Systems (binary + multiclass).
"""

from .cnn_model import IDS_CNN, create_ids_model

__all__ = ["IDS_CNN", "create_ids_model"]
