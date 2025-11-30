"""
Main Training Script

This script provides a complete pipeline for training CNN models
on image classification tasks.
"""

import argparse
import os
import yaml
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.cnn_model import create_model
from src.training.trainer import Trainer
from src.utils.helpers import set_seed, get_device, get_optimizer, get_scheduler
from src.data.transforms import get_train_transforms, get_test_transforms


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_cifar10_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
    data_dir: str = "data",
) -> Dict[str, DataLoader]:
    """
    Create CIFAR-10 data loaders for quick experimentation.

    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        data_dir: Directory to download/store data

    Returns:
        Dictionary with train and test data loaders
    """
    # CIFAR-10 normalization values
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {"train": train_loader, "val": test_loader}


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train CNN Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "custom"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command line arguments
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    # Set seed for reproducibility
    set_seed(config.get("seed", 42))

    # Get device
    device = get_device()

    # Create data loaders
    if args.dataset == "cifar10":
        loaders = create_cifar10_loaders(
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
        )
    else:
        # Custom dataset loading would go here
        raise NotImplementedError("Custom dataset loading not implemented")

    # Create model
    model = create_model(
        model_type=config["model"]["type"],
        num_classes=config["model"]["num_classes"],
        input_channels=config["model"]["input_channels"],
        input_size=tuple(config["model"]["input_size"]),
    )

    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {model.count_parameters():,}")

    # Create loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(
        model=model,
        optimizer_name=config["training"]["optimizer"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=config["training"]["scheduler"],
        epochs=config["training"]["epochs"],
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=config["checkpoint"]["save_dir"],
    )

    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        epochs=config["training"]["epochs"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        verbose=config["logging"]["verbose"],
    )

    # Save final model
    final_model_path = os.path.join(config["checkpoint"]["save_dir"], "../final/model.pth")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "history": history,
    }, final_model_path)

    print(f"\nTraining complete! Final model saved to {final_model_path}")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
