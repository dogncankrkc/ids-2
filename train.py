"""
IDS – ResNet-MLP Training Script (Multiclass)

This script runs a full multiclass IDS training pipeline using a ResNet-based 1D CNN.

Architecture and data format:
- Numerical features are treated as a 1D signal: (N, 1, L)
- Designed for network traffic classification (IDS)
- StandardScaler + fixed LabelEncoder to prevent data leakage

Pipeline:
1. Load capped dataset
2. Preprocess (split, scaling, encoding)
3. Reshape features to ResNet format (N, 1, L)
4. Train with early stopping
5. Evaluate best checkpoint on test set
6. Save metrics, model bundle, and plots
"""

import argparse
import os
import yaml
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Model factory (defined in src/models/cnn_model.py)
from src.models.cnn_model import create_ids_model
from src.training.trainer import Trainer
from src.utils.helpers import (
    set_seed,
    get_device,
    get_optimizer,
    get_scheduler,
    count_parameters,
)
from src.data.preprocess import preprocess_multiclass
from src.utils.visualization import plot_training_history, plot_confusion_matrix

from src.utils.losses import FocalLoss
# Feature list imported from preprocess module for consistent test-split export
from src.data.preprocess import SELECTED_FEATURES


# ============================================================
# Paths and constants
# ============================================================

PROCESSED_DATA_PATH = "data/processed/CIC2023_SEPARATE_ATTACK_ONLY.csv"
TEST_DATA_SAVE_PATH = "data/processed/test_split_saved.csv"
ENCODER_PATH = "data/processed/label_encoder.pkl"


# ============================================================
# Utilities
# ============================================================

def load_config(config_path: str):
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def to_python_int(d):
    """
    Convert NumPy integer keys/values to native Python int.

    This is useful when dumping dictionaries to YAML.

    Args:
        d (dict): Dictionary potentially containing NumPy integer types.

    Returns:
        dict: Dictionary converted to Python int types.
    """
    return {int(k): int(v) for k, v in d.items()}


# ============================================================
# Data loader creation
# ============================================================

def create_ids_loaders(batch_size: int, num_workers: int = 0):
    """
    Load the dataset, apply preprocessing, and construct PyTorch DataLoaders.

    Output tensor format is compatible with ResNet1D models: (N, 1, L).

    Args:
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of DataLoader workers.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
            Train, validation, and test loaders plus y_train labels.
    """
    print(f"[INFO] Loading dataset from: {PROCESSED_DATA_PATH}")

    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {PROCESSED_DATA_PATH}. "
            f"Please run the capped dataset creator first."
        )

    df = pd.read_csv(PROCESSED_DATA_PATH)

    # --------------------------------------------------------
    # Preprocessing (split, scaling, encoding; optional SMOTE inside)
    # --------------------------------------------------------
    # This function performs all preprocessing steps in a leak-free manner.
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_multiclass(df, augmentation="smote")

    print("[INFO] Multiclass preprocessing completed.")
    print(f"[INFO] Feature count per sample (L): {X_train.shape[1]}")

    # Basic safety check: feature dimensionality must match across splits
    assert (
        X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    ), "Feature count mismatch across splits!"

    # --------------------------------------------------------
    # Save processed test split (for reproducibility)
    # --------------------------------------------------------
    print(f"[INFO] Saving test split to: {TEST_DATA_SAVE_PATH}")
    os.makedirs(os.path.dirname(TEST_DATA_SAVE_PATH), exist_ok=True)

    df_test = pd.DataFrame(X_test, columns=SELECTED_FEATURES)
    df_test["label"] = y_test
    df_test.to_csv(TEST_DATA_SAVE_PATH, index=False)

    print(f"[SUCCESS] Test set saved ({len(df_test)} samples)")

    # --------------------------------------------------------
    # Tensor conversion and reshape for ResNet: (N, 1, L)
    # --------------------------------------------------------
    # The model expects input as (N, 1, L), so we add a channel dimension via unsqueeze(1).
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32).unsqueeze(1)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32).unsqueeze(1)

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)

    print(f"[INFO] Model input shape: {X_train_t.shape} (train)")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"[INFO] Train samples: {len(train_loader.dataset)}")
    print(f"[INFO] Val samples  : {len(val_loader.dataset)}")
    print(f"[INFO] Test samples : {len(test_loader.dataset)}")

    return train_loader, val_loader, test_loader, y_train


# ============================================================
# Main training pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multiclass_config.yaml",
        help="Path to training config file",
    )
    args = parser.parse_args()

    # --------------------------------------------------------
    # Load config and initialize environment
    # --------------------------------------------------------
    if not os.path.exists(args.config):
        print(f"[WARNING] Config file not found at {args.config}. Make sure the path is correct.")

    try:
        config = load_config(args.config)
    except Exception:
        # Fallback configuration to avoid crashing when config is missing/invalid
        print("[INFO] Loading fallback default configuration.")
        config = {
            "model": {"type": "ids_resnet_mlp", "num_classes": 8},
            "data": {"batch_size": 128, "num_workers": 0},
            "training": {
                "epochs": 50,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "optimizer": "adamw",
                "scheduler": "plateau",
                "early_stopping_patience": 10,
            },
            "checkpoint": {"save_dir": "models/checkpoints/ids_multiclass_resnet1d"},
            "logging": {"verbose": True},
            "seed": 42,
        }

    set_seed(config.get("seed", 42))
    device = get_device()

    print(f"[INFO] Using device: {device}")
    num_workers = config["data"].get("num_workers", 0)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    train_loader, val_loader, test_loader, y_train = create_ids_loaders(
        batch_size=config["data"]["batch_size"],
        num_workers=num_workers,
    )

    # --------------------------------------------------------
    # Save class distribution (for reporting)
    # --------------------------------------------------------
    dist = {
        "train": to_python_int(
            dict(pd.Series(y_train).value_counts().sort_index())
        ),
        "val": to_python_int(
            dict(pd.Series(val_loader.dataset.tensors[1].numpy()).value_counts().sort_index())
        ),
        "test": to_python_int(
            dict(pd.Series(test_loader.dataset.tensors[1].numpy()).value_counts().sort_index())
        ),
    }

    os.makedirs(config["checkpoint"]["save_dir"], exist_ok=True)
    with open(
        os.path.join(config["checkpoint"]["save_dir"], "class_distribution.yaml"),
        "w"
    ) as f:
        yaml.safe_dump(dist, f)

    # --------------------------------------------------------
    # Load label encoder (for class names)
    # --------------------------------------------------------
    if os.path.exists(ENCODER_PATH):
        encoder = joblib.load(ENCODER_PATH)
        class_names = encoder.classes_.tolist()
        num_classes = len(class_names)
        print("\n[INFO] Class index mapping:")
        for i, cls in enumerate(class_names):
            print(f"  {i} -> {cls}")
    else:
        # Fallback if encoder is not found
        num_classes = 8
        class_names = [str(i) for i in range(8)]

    # --------------------------------------------------------
    # Model initialization
    # --------------------------------------------------------
    # Input dimension L is taken from the preprocessed tensor: (N, 1, L)
    input_dim = train_loader.dataset.tensors[0].shape[2]

    model = create_ids_model(
        mode="multiclass",
        num_classes=num_classes,
        input_dim=input_dim
    ).to(device)

    print(f"\n[INFO] Model: {model.__class__.__name__}")
    print(f"[INFO] Trainable parameters: {count_parameters(model):,}")

    # --------------------------------------------------------
    # Save model info (for experiment tracking)
    # --------------------------------------------------------
    model_info = {
        "model_name": model.__class__.__name__,
        "num_parameters": count_parameters(model),
        "input_shape": f"(N, 1, {input_dim})",
        "num_features": input_dim,
    }

    with open(os.path.join(config["checkpoint"]["save_dir"], "model_info.yaml"), "w") as f:
        yaml.safe_dump(model_info, f)

    # --------------------------------------------------------
    # Loss function with class weights
    # --------------------------------------------------------
    # Compute balanced class weights from the training labels
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"[INFO] Class Weights: {class_weights.cpu().numpy()}")

    # Use Focal Loss to emphasize hard and minority-class samples
    print("[INFO] Using Focal Loss (gamma=2.5) with class balancing")
    criterion = FocalLoss(alpha=class_weights, gamma=2.5, device=device)

    optimizer = get_optimizer(
        model=model,
        optimizer_name=config["training"]["optimizer"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # --------------------------------------------------------
    # Scheduler selection
    # --------------------------------------------------------
    if config["training"]["scheduler"] == "cosine":
        print("[INFO] Scheduler: CosineAnnealingWarmRestarts")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
    else:
        scheduler = get_scheduler(
            optimizer=optimizer,
            scheduler_name=config["training"]["scheduler"],
            epochs=config["training"]["epochs"],
            patience=config["training"].get("early_stopping_patience", 10),
        )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=config["checkpoint"]["save_dir"],
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    print("\n" + "=" * 40)
    print(f"STARTING TRAINING ({model.__class__.__name__})")
    print("=" * 40)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["training"]["epochs"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        verbose=config["logging"]["verbose"],
    )

    # --------------------------------------------------------
    # Load best checkpoint and evaluate on test set
    # --------------------------------------------------------
    best_model_path = os.path.join(config["checkpoint"]["save_dir"], "best_model.pth")

    if os.path.exists(best_model_path):
        print("\n[INFO] Loading best model from checkpoint...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------
    print("\n[INFO] Running final evaluation on test set...")
    test_results = trainer.test(test_loader)

    metrics_path = os.path.join(
        config["checkpoint"]["save_dir"],
        "test_metrics_multiclass.yaml",
    )
    with open(metrics_path, "w") as f:
        yaml.safe_dump(test_results, f)

    print(f"[INFO] Test metrics saved to: {metrics_path}")

    # --------------------------------------------------------
    # Save final bundle
    # --------------------------------------------------------
    final_model_path = os.path.join(
        config["checkpoint"]["save_dir"],
        "final_model_multiclass.pth",
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "history": history,
            "test_results": test_results,
            "class_names": class_names,
        },
        final_model_path,
    )

    print(f"[DONE] Final model saved to: {final_model_path}")
    if hasattr(trainer, 'best_val_acc'):
        print(f"[DONE] Best validation accuracy: {trainer.best_val_acc:.2f}%")

    # --------------------------------------------------------
    # Plot results
    # --------------------------------------------------------
    plot_training_history(
        history=history,
        save_path=os.path.join(
            config["checkpoint"]["save_dir"],
            "training_history_multiclass.png",
        ),
    )

    # Confusion matrix
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        save_path=os.path.join(
            config["checkpoint"]["save_dir"],
            "confusion_matrix_multiclass.png",
        ),
        normalize=True,
    )


if __name__ == "__main__":
    main()
