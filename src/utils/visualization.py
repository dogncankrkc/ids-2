"""
Visualization Utilities for IDS (Numerical Features)

This module provides visualization helpers specifically designed for
Intrusion Detection System (IDS) experiments on numerical feature sets.

Included visualizations:
- Training and validation loss / accuracy curves
- Learning rate evolution across epochs
- Confusion matrix visualization (raw or normalized)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """
    Plot training and validation performance curves.

    This function visualizes:
    - Training vs validation loss
    - Training vs validation accuracy

    It expects a history dictionary produced by the training loop.

    Args:
        history (Dict[str, List[float]]): Dictionary containing metric history
            (e.g., 'train_loss', 'val_loss', 'train_acc', 'val_acc').
        save_path (str, optional): File path to save the plot image.
        figsize (Tuple[int, int]): Size of the matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss curves
    axes[0].plot(history.get("train_loss", []), label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy curves
    axes[1].plot(history.get("train_acc", []), label="Train Accuracy")
    if "val_acc" in history:
        axes[1].plot(history["val_acc"], label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training vs Validation Accuracy")
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
    normalize: bool = False,
    show_percent: bool = False,
) -> None:
    """
    Visualize a confusion matrix as a heatmap.

    Supports both raw counts and row-normalized representations.

    Args:
        cm (np.ndarray): Confusion matrix of shape [N, N].
        class_names (List[str], optional): Names of the classes.
        save_path (str, optional): File path to save the figure.
        figsize (Tuple[int, int]): Size of the matplotlib figure.
        cmap (str): Colormap used for visualization.
        normalize (bool): If True, normalize rows to sum to 1.
        show_percent (bool): If True and normalize=True, display values as percentages.
    """
    num_classes = cm.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # Normalize confusion matrix if requested
    if normalize:
        cm_float = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        title = "Normalized Confusion Matrix"
        fmt = ".1f" if show_percent else ".2f"
        if show_percent:
            cm_float = cm_float * 100.0
    else:
        cm_float = cm
        title = "Confusion Matrix"
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_float, interpolation="nearest", cmap=cmap)

    if normalize:
        im.set_clim(0.0, 100.0 if show_percent else 1.0)

    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    # Annotate each cell with its value
    thresh = cm_float.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                format(cm_float[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm_float[i, j] > thresh else "black",
                fontsize=10,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Confusion matrix saved to: {save_path}")

    plt.close(fig)


def plot_learning_rate(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4),
) -> None:
    """
    Plot the learning rate evolution over training epochs.

    Useful for diagnosing scheduler behavior and learning dynamics.

    Args:
        history (Dict[str, List[float]]): Training history containing 'lr' values.
        save_path (str, optional): File path to save the plot image.
        figsize (Tuple[int, int]): Size of the matplotlib figure.
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
