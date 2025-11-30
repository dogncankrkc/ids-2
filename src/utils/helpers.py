"""
Helper Utilities

This module provides general helper functions for CNN model development.
"""

import os
import random
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Automatically select GPU (Apple MPS / NVIDIA CUDA) if available,
    otherwise CPU.
    """
    if torch.backends.mps.is_available():  # Apple Silicon GPU (MPS backend)
        return torch.device("mps")
    elif torch.cuda.is_available():        # NVIDIA GPU 
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def save_model(
    model: nn.Module,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a trained model to disk.

    Args:
        model: PyTorch model to save
        path: Path to save the model
        metadata: Optional metadata to save with the model
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    save_dict = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }

    torch.save(save_dict, path)
    print(f"Model saved to {path}")


def load_model(
    model: nn.Module,
    path: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a trained model from disk.

    Args:
        model: PyTorch model instance to load weights into
        path: Path to the saved model
        device: Device to load the model to

    Returns:
        Dictionary containing metadata
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print(f"Model loaded from {path}")

    return checkpoint.get("metadata", {})


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, input_size: tuple) -> None:
    """
    Print a summary of the model architecture.

    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
    """
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 60)
    print(model)
    print("=" * 60)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("=" * 60)


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """
    Get an optimizer for training.

    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adam', 'sgd', 'adamw')
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        momentum: Momentum (for SGD)

    Returns:
        PyTorch optimizer
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

    if optimizer_name.lower() not in optimizers:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. Available: {list(optimizers.keys())}"
        )

    return optimizers[optimizer_name.lower()]()


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "cosine",
    epochs: int = 100,
    **kwargs,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get a learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        scheduler_name: Name of scheduler
        epochs: Total number of training epochs
        **kwargs: Additional scheduler arguments

    Returns:
        PyTorch learning rate scheduler
    """
    schedulers = {
        "step": lambda: torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs.get("step_size", 30), gamma=kwargs.get("gamma", 0.1)
        ),
        "cosine": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        ),
        "plateau": lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        ),
        "exponential": lambda: torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=kwargs.get("gamma", 0.95)
        ),
    }

    if scheduler_name.lower() not in schedulers:
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}. Available: {list(schedulers.keys())}"
        )

    return schedulers[scheduler_name.lower()]()

def prepare_for_training(seed: int = 42) -> torch.device:
    """
    Full setup for IDS training:
    - sets random seed
    - selects device (GPU/CPU)
    """
    set_seed(seed)
    device = get_device()
    return device
