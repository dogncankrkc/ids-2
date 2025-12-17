"""
Visualization Utilities (IDS Version – Numerical Features)

Only includes relevant functions for network intrusion detection:
    ✔ Training & validation curves
    ✔ Learning rate visualization
    ✔ Confusion matrix heatmap
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
    normalize: bool = False,
    show_percent: bool = False,
) -> None:
    """
    Plots a confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix (NxN).
        class_names (List[str], optional): Class labels.
        save_path (str, optional): Path to save the figure.
        figsize (Tuple[int, int]): Figure size.
        cmap (str): Matplotlib colormap.
        normalize (bool): If True, row-normalize values.
        show_percent (bool): If True and normalize=True, show percentages (0–100).
    """

    num_classes = cm.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # ---------------- Normalization ----------------
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
    # ------------------------------------------------

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm_float, interpolation="nearest", cmap=cmap)

    if normalize:
        im.set_clim(0., 100. if show_percent else 1.)

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

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_float.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, format(cm_float[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_float[i, j] > thresh else "black",
                fontsize=10
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
