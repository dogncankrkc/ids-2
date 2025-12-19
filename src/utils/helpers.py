"""
Helper Utilities for CNN Training Pipelines

This module provides reusable utility functions commonly required in CNN-based
training workflows, including:
- Reproducibility via global seeding
- Automatic device selection (MPS / CUDA / CPU)
- Model parameter counting
- Optimizer factory
- Learning rate scheduler factory
- One-call training environment preparation
"""

import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------
# Reproducibility utilities
# --------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set global random seeds to ensure experiment reproducibility.

    This function synchronizes random states across:
    - Python's built-in random module
    - NumPy
    - PyTorch (CPU and CUDA)

    It also enforces deterministic behavior in cuDNN at the cost of performance.

    Args:
        seed (int): Seed value used across all random generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------
# Device selection
# --------------------------------------------------

def get_device() -> torch.device:
    """
    Select the best available computation device automatically.

    Priority order:
    1. Apple Silicon GPU via MPS
    2. NVIDIA GPU via CUDA
    3. CPU as a fallback

    Returns:
        torch.device: Selected computation device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# --------------------------------------------------
# Model inspection
# --------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): Model instance to inspect.

    Returns:
        int: Number of parameters with requires_grad=True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------
# Optimizer factory
# --------------------------------------------------

def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """
    Create and configure a PyTorch optimizer.

    Supported optimizers:
    - Adam
    - AdamW
    - SGD (with momentum)

    Args:
        model (nn.Module): Model whose parameters will be optimized.
        optimizer_name (str): Optimizer identifier ('adam', 'adamw', 'sgd').
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay (L2 regularization).
        momentum (float): Momentum value (SGD only).

    Returns:
        torch.optim.Optimizer: Instantiated optimizer.
    """
    optimizers = {
        "adam": lambda: torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        ),
        "adamw": lambda: torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        ),
        "sgd": lambda: torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        ),
    }

    name = optimizer_name.lower()
    if name not in optimizers:
        raise ValueError(
            f"Unknown optimizer: {name}. Available options: {list(optimizers.keys())}"
        )

    return optimizers[name]()


# --------------------------------------------------
# Scheduler factory
# --------------------------------------------------

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "cosine",
    epochs: int = 100,
    **kwargs,
) -> Any:
    """
    Create a learning rate scheduler for a given optimizer.

    Supported schedulers:
    - StepLR
    - CosineAnnealingLR
    - ReduceLROnPlateau (min or max mode)
    - ExponentialLR

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        scheduler_name (str): Scheduler identifier.
        epochs (int): Total training epochs (used by cosine scheduler).
        **kwargs: Scheduler-specific parameters such as patience, factor, gamma.

    Returns:
        Learning rate scheduler instance.
    """
    patience = kwargs.get("patience", 10)
    factor = kwargs.get("factor", 0.1)

    schedulers = {
        "step": lambda: torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1),
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
            f"Unknown scheduler: {name}. Available options: {list(schedulers.keys())}"
        )

    return schedulers[name]()


# --------------------------------------------------
# Training environment setup
# --------------------------------------------------

def prepare_for_training(seed: int = 42) -> torch.device:
    """
    Convenience wrapper to prepare the training environment.

    This function:
    1. Sets global random seeds for reproducibility.
    2. Automatically selects the best available computation device.

    Args:
        seed (int): Random seed value.

    Returns:
        torch.device: Device to be used for training.
    """
    set_seed(seed)
    device = get_device()
    return device
