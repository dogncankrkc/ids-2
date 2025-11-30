"""
IDS-CNN Framework (Raspberry Pi Compatible)

This package provides modular components for:
    • CSV-based data preprocessing (PCAP → numeric features → tensors)
    • Lightweight CNN models (binary & multiclass IDS)
    • Training utilities (optimizer, scheduler, early stopping)
    • Evaluation metrics (precision, recall, F1, confusion matrix)
    • Edge deployment support (TFLite conversion for Raspberry Pi)

Structure:
    src/
    ├─ data/          → CSV loaders & preprocessing
    ├─ model/         → CNN architectures (lightweight)
    ├─ training/      → trainer, metrics, loss selector
    ├─ utils/         → reproducibility & visualization utilities

Suited for edge-level IDS on Raspberry Pi 4.
"""

__version__ = "0.2.0"
