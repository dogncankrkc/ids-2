"""
Helper Utilities.

This module provides general helper functions for CNN model development:
- Seeding for reproducibility
- Device selection (CPU/CUDA/MPS)
- Model summarization
- Optimizer and Scheduler factories
"""

import os
import random
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn

# ------------------------
# UTILITY FUNCTIONS
# ------------------------

# Set random seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across numpy, torch, and python random.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device selection
def get_device() -> torch.device:
    """
    Automatically selects the best available device:
    - MPS (Apple Silicon)
    - CUDA (NVIDIA)
    - CPU (Fallback)
    
    Returns:
        torch.device: Selected device.
    """
    if torch.backends.mps.is_available():   # Apple Silicon GPU
        return torch.device("mps")
    elif torch.cuda.is_available():         # NVIDIA GPU 
        return torch.device("cuda")
    else:
        return torch.device("cpu")          # Fallback to CPU

# Model summary and parameter counting
def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.

    Args:
        model (nn.Module): PyTorch model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print model summary 
def print_model_summary(model: nn.Module) -> None:
    """
    Prints a formatted summary of the model architecture and parameter count.
    """
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 60)
    print(model)
    print("=" * 60)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("=" * 60)

# Optimizer factory
def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """
    Factory function to create an optimizer.

    Args:
        model: PyTorch model containing parameters to optimize.
        optimizer_name: 'adam', 'adamw', or 'sgd'.
        learning_rate: Learning rate.
        weight_decay: L2 regularization factor.
        momentum: Momentum factor (only for SGD).

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    optimizers = {
        "adam": lambda: torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        ),
        "adamw": lambda: torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        ),
        "sgd": lambda: torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum,
            weight_decay=weight_decay
        ),
    }

    name = optimizer_name.lower()
    if name not in optimizers:
        raise ValueError(
            f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}"
        )

    return optimizers[name]()

# Scheduler factory
def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "cosine",
    epochs: int = 100,
    **kwargs,
) -> Any:
    """
    Factory function to create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_name: 'step', 'cosine', 'plateau' (or 'reduce_lr_on_plateau'), 'exponential'.
        epochs: Total epochs (used for CosineAnnealing).
        **kwargs: Extra arguments like patience, factor, etc.

    Returns:
        LR Scheduler object.
    """
    # Common settings for ReduceLROnPlateau if not provided in kwargs
    patience = kwargs.get("patience", 10)
    factor = kwargs.get("factor", 0.1)

    schedulers = {
        "step": lambda: torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs.get("step_size", 30), gamma=kwargs.get("gamma", 0.1)
        ),
        "cosine": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        ),
       "reduce_lr_on_plateau": lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience
        ),
        "plateau": lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=factor, patience=patience
        ),
        "exponential": lambda: torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=kwargs.get("gamma", 0.95)
        ),
    }

    name = scheduler_name.lower()
    if name not in schedulers:
        raise ValueError(
            f"Unknown scheduler: {name}. Available: {list(schedulers.keys())}"
        )

    return schedulers[name]()


def prepare_for_training(seed: int = 42) -> torch.device:
    """
    Full setup wrapper for training scripts:
    1. Sets random seed.
    2. Selects the appropriate computation device.

    Returns:
        torch.device: The device to be used for training.
    """
    set_seed(seed)
    device = get_device()
    return device