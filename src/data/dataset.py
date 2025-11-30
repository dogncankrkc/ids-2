"""
Dataset Classes and Data Loading Utilities

This module provides custom dataset classes and utilities for
creating data loaders for CNN training.
"""

import os
from typing import Tuple, Optional, Callable, Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class ImageDataset(Dataset):
    """
    Custom Dataset class for loading images from directories.

    Expected directory structure:
        root/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
                img4.jpg

    Args:
        root_dir: Path to the root directory containing class folders
        transform: Optional transform to apply to images
        target_transform: Optional transform to apply to labels
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.samples: List[Tuple[str, int]] = []

        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load dataset from directory structure."""
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"Root directory not found: {self.root_dir}")

        # Get sorted list of classes
        self.classes = sorted(
            [d for d in os.listdir(self.root_dir)
             if os.path.isdir(os.path.join(self.root_dir, d))]
        )

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Load all image paths
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for filename in os.listdir(class_dir):
                if os.path.splitext(filename)[1].lower() in valid_extensions:
                    img_path = os.path.join(class_dir, filename)
                    self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image, label)
        """
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def create_data_loaders(
    train_dir: str,
    val_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory (optional)
        test_dir: Path to test data directory (optional)
        train_transform: Transforms for training data
        val_transform: Transforms for validation/test data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Dictionary containing data loaders
    """
    loaders: Dict[str, DataLoader] = {}

    # Training loader
    train_dataset = ImageDataset(train_dir, transform=train_transform)
    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Validation loader
    if val_dir:
        val_dataset = ImageDataset(val_dir, transform=val_transform)
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    # Test loader
    if test_dir:
        test_dataset = ImageDataset(test_dir, transform=val_transform)
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return loaders
