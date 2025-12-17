"""
IDS – CNN Training Script (Multiclass, CNN2D).

Architecture:
- Input features reshaped into 2D grid (1 x 5 x 8)
- Designed for network traffic classification (IDS)
- StandardScaler + fixed LabelEncoder (NO data leakage)

Pipeline:
1. Load preprocessed dataset
2. Preprocess (scaling + label encoding, NO resampling)
3. Reshape features → CNN2D format
4. Train with early stopping
5. Evaluate best model
6. Save metrics, model, plots

Author: IDS Research Pipeline
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

# ============================================================
# PATHS & CONSTANTS
# ============================================================

PROCESSED_DATA_PATH = "data/processed/CIC2023_CAPPED.csv"
TEST_DATA_SAVE_PATH = "data/processed/test_split_saved.csv"
ENCODER_PATH = "data/processed/label_encoder.pkl"

# CNN2D grid definition (40 features)
GRID_H = 5
GRID_W = 8
NUM_FEATURES = GRID_H * GRID_W


# ============================================================
# UTILITIES
# ============================================================

def load_config(config_path: str):
    """Loads YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# DATA LOADER CREATION
# ============================================================

def create_ids_loaders(batch_size: int, num_workers: int = 0):
    """
    Loads dataset, applies preprocessing and prepares PyTorch DataLoaders.
    CNN2D format: (N, 1, 5, 8)
    """

    print(f"[INFO] Loading dataset from: {PROCESSED_DATA_PATH}")

    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {PROCESSED_DATA_PATH}. "
            f"Please run create_balanced_dataset.py first."
        )

    df = pd.read_csv(PROCESSED_DATA_PATH)

    # --------------------------------------------------------
    # Preprocessing (NO resampling here)
    # --------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_multiclass(df)

    print("[INFO] Multiclass preprocessing completed.")
    print(f"[INFO] Feature count per sample: {X_train.shape[1]}")

    # Safety check
    assert X_train.shape[1] == NUM_FEATURES, "Feature count mismatch!"

    # --------------------------------------------------------
    # Save processed test split (for reproducibility)
    # --------------------------------------------------------
    print(f"[INFO] Saving test split to: {TEST_DATA_SAVE_PATH}")
    os.makedirs(os.path.dirname(TEST_DATA_SAVE_PATH), exist_ok=True)

    df_test = pd.DataFrame(X_test)
    df_test["label"] = y_test
    df_test.to_csv(TEST_DATA_SAVE_PATH, index=False)

    print(f"[SUCCESS] Test set saved ({len(df_test)} samples)")

    # --------------------------------------------------------
    # Tensor conversion & reshape for CNN2D
    # --------------------------------------------------------
    X_train_t = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, GRID_H, GRID_W)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32).view(-1, 1, GRID_H, GRID_W)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32).view(-1, 1, GRID_H, GRID_W)

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)

    print(f"[INFO] CNN2D input shape: {X_train_t.shape} (train)")

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
# MAIN TRAINING PIPELINE
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
    # Load config & setup
    # --------------------------------------------------------
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_config(args.config)

    set_seed(config.get("seed", 42))
    device = get_device()

    print(f"[INFO] Using device: {device}")

    num_workers = config["data"].get("num_workers", 0)
    print(f"[INFO] DataLoader num_workers: {num_workers}")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    train_loader, val_loader, test_loader, y_train = create_ids_loaders(
        batch_size=config["data"]["batch_size"],
        num_workers=num_workers,
    )

    # --------------------------------------------------------
    # Load label encoder (for class names)
    # --------------------------------------------------------
    encoder = joblib.load(ENCODER_PATH)
    class_names = encoder.classes_.tolist()
    num_classes = len(class_names)

    print("\n[INFO] Class index mapping:")
    for i, cls in enumerate(class_names):
        print(f"  {i} -> {cls}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = create_ids_model(mode="multiclass", num_classes=num_classes).to(device)

    print(f"\n[INFO] Model: {model.__class__.__name__}")
    print(f"[INFO] Trainable parameters: {count_parameters(model):,}")

    # --------------------------------------------------------
    # Loss with class weights (important for IDS!)
    # --------------------------------------------------------
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

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
    # Training
    # --------------------------------------------------------
    print("\n" + "=" * 40)
    print("STARTING TRAINING (MULTICLASS CNN2D)")
    print("=" * 40)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["training"]["epochs"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        verbose=config["logging"]["verbose"],
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------
    best_model_path = os.path.join(
        config["checkpoint"]["save_dir"],
        "best_model.pth",
    )

    print("\n[INFO] Loading best model from checkpoint...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------
    print("\n" + "=" * 40)
    print("FINAL TEST EVALUATION")
    print("=" * 40)

    test_results = trainer.test(test_loader)

    metrics_path = os.path.join(
        config["checkpoint"]["save_dir"],
        "test_metrics_multiclass.yaml",
    )

    with open(metrics_path, "w") as f:
        yaml.safe_dump(test_results, f)

    print(f"[INFO] Test metrics saved to: {metrics_path}")

    # --------------------------------------------------------
    # Save final model
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
    print(f"[DONE] Best validation accuracy: {trainer.best_val_acc:.2f}%")

    # --------------------------------------------------------
    # Plots
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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
