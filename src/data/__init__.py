"""
Data Loading & Preprocessing Module (IDS - CSV Based)

This package provides:
1. Loading multiple CSV files from /data/raw
2. Preprocessing pipelines for:
    - Binary IDS classification
    - Multiclass IDS classification
3. CNN-ready tensor generation (reshaped, scaled, encoded)
"""

from .preprocess import (
    preprocess_multiclass,
    preprocess_single_sample, 
)

__all__ = [
    "preprocess_multiclass",
    "preprocess_single_sample",
]
