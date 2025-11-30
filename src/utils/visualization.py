"""
Visualization Utilities

This module provides functions for visualizing training progress,
model predictions, and evaluation metrics.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """
    Plot training history (loss and accuracy curves).

    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the figure (optional)
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    axes[0].plot(history.get("train_loss", []), label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Plot accuracy
    axes[1].plot(history.get("train_acc", []), label="Train Accuracy")
    if "val_acc" in history:
        axes[1].plot(history["val_acc"], label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
) -> None:
    """
    Plot a confusion matrix.

    Args:
        cm: Confusion matrix array
        class_names: Names of the classes
        save_path: Path to save the figure (optional)
        figsize: Figure size
        cmap: Colormap for the plot
    """
    num_classes = len(cm)

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Set labels
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def visualize_predictions(
    model: torch.nn.Module,
    data_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    num_images: int = 16,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 12),
) -> None:
    """
    Visualize model predictions on a batch of images.

    Args:
        model: Trained model
        data_loader: Data loader to get images from
        class_names: Names of the classes
        device: Device to run inference on
        num_images: Number of images to visualize
        save_path: Path to save the figure (optional)
        figsize: Figure size
    """
    model.eval()

    # Get a batch of images
    images, labels = next(iter(data_loader))
    images = images[:num_images]
    labels = labels[:num_images]

    # Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predictions = outputs.max(1)
        predictions = predictions.cpu()

    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_images)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()

    for idx in range(num_images):
        ax = axes[idx]

        # Convert to numpy and transpose for matplotlib
        img = images[idx].numpy().transpose(1, 2, 0)
        ax.imshow(img)

        true_label = class_names[labels[idx]]
        pred_label = class_names[predictions[idx]]
        correct = labels[idx] == predictions[idx]

        color = "green" if correct else "red"
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        ax.axis("off")

    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_learning_rate(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4),
) -> None:
    """
    Plot learning rate schedule.

    Args:
        history: Dictionary containing learning rate history
        save_path: Path to save the figure (optional)
        figsize: Figure size
    """
    if "lr" not in history:
        print("No learning rate history found")
        return

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(history["lr"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
