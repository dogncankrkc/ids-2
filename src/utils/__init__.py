"""
Utility Module for IDS-CNN Project

Provides:
    - Visualization tools (training curves, confusion matrix)
    - Reproducibility helpers (seed, device)
    - Model IO (save/load)
"""

from .visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_learning_rate,     
)
from .helpers import (
    set_seed,
    get_device,
    save_model,
    load_model,
    prepare_for_training,   
)

__all__ = [
    # Visualization
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_learning_rate",

    # Helpers
    "set_seed",
    "get_device",
    "save_model",
    "load_model",
    "prepare_for_training",
]

# Package version (optional)
__version__ = "0.2.0"
