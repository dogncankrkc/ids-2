"""
Image Transformation Utilities

This module provides common image transformations for training
and testing CNN models.
"""

from typing import Tuple, List, Optional

import torchvision.transforms as T


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    horizontal_flip: bool = True,
    random_rotation: Optional[int] = 10,
    color_jitter: bool = True,
) -> T.Compose:
    """
    Get training data transformations with augmentation.

    Args:
        image_size: Target image size (height, width)
        mean: Mean values for normalization (ImageNet defaults)
        std: Standard deviation values for normalization
        horizontal_flip: Whether to apply random horizontal flip
        random_rotation: Degrees for random rotation (None to disable)
        color_jitter: Whether to apply color jitter augmentation

    Returns:
        Composed transforms for training
    """
    transforms_list: List[T.transforms.Transform] = [
        T.Resize(image_size),
    ]

    # Data augmentation
    if horizontal_flip:
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))

    if random_rotation:
        transforms_list.append(T.RandomRotation(degrees=random_rotation))

    if color_jitter:
        transforms_list.append(
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            )
        )

    # Random crop with padding for additional augmentation
    transforms_list.append(T.RandomCrop(image_size, padding=4))

    # Convert to tensor and normalize
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return T.Compose(transforms_list)


def get_test_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """
    Get test/validation data transformations (no augmentation).

    Args:
        image_size: Target image size (height, width)
        mean: Mean values for normalization
        std: Standard deviation values for normalization

    Returns:
        Composed transforms for testing/validation
    """
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_inference_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """
    Get transforms for inference on single images.

    Args:
        image_size: Target image size
        mean: Mean values for normalization
        std: Standard deviation values for normalization

    Returns:
        Composed transforms for inference
    """
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
