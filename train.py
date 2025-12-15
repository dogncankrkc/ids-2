"""
IDS – CNN Training Script (Binary + Multiclass).

This script handles the full training pipeline:
1. Loads the balanced CSV dataset.
2. Preprocesses data (Encoding, Scaling, Reshaping) via 'preprocess.py'.
3. Converts data to PyTorch Tensors and creates DataLoaders.
4. Initializes the CNN model, Optimizer, and Scheduler.
5. Runs the training loop using the Trainer class.
6. Saves the final model and training history.
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# --- CUSTOM MODULE IMPORTS ---
from src.models.cnn_model import create_ids_model      # CNN Model Class
from src.training.trainer import Trainer                # Trainer Class wrapper
from src.utils.helpers import (
    set_seed, get_device, get_optimizer, get_scheduler
)
from src.data.preprocess import (
     preprocess_binary, preprocess_multiclass
)

from src.utils.visualization import plot_training_history, plot_confusion_matrix 
from sklearn.metrics import confusion_matrix 
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# GLOBAL CONFIGURATION
# ------------------------
# Path to the clean, balanced dataset created by 'create_balanced_dataset.py'
PROCESSED_DATA_PATH = "data/processed/CIC2023_Balanced_50k.csv"
TEST_DATA_SAVE_PATH = "data/processed/test_split_saved.csv"   

def load_config(config_path: str):
    """
    Loads hyperparameters and settings from a YAML configuration file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_ids_loaders(mode: str, batch_size: int):
    """
    Orchestrates the data loading and preprocessing pipeline.

    Steps:
    1. Reads the balanced CSV file.
    2. Calls the appropriate preprocessing function (Binary or Multiclass).
    3. Converts Numpy arrays to PyTorch Tensors.
    4. Permutes dimensions to match CNN input: (Batch, Channel, Height, Width).
    5. Wraps tensors into DataLoaders.

    Args:
        mode (str): 'binary' or 'multiclass'.
        batch_size (int): Batch size for training.

    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    print(f"[INFO] Loading dataset from: {PROCESSED_DATA_PATH}")
    
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {PROCESSED_DATA_PATH}. Please run 'create_balanced_dataset.py' first.")

    print(f"[INFO] Raw Data Shape: {df.shape}")

    # ---------------------------------------------------------
    # 1. PREPROCESSING (Resampling, Scaling, Splitting)
    # ---------------------------------------------------------
    if mode == "binary":
        # Target: 0 (Benign) vs 1 (Attack)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_binary(df)
        print("[INFO] Mode: Binary Classification")
    
    elif mode == "multiclass":
        # Target: 8 Classes (DDoS, Web, Recon, etc.)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_multiclass(df)
        print("[INFO] Mode: Multiclass Classification")
    
    else:
        raise ValueError("mode must be 'binary' or 'multiclass'")
    
    # ---------------------------------------------------------
    # 2. SAVE TEST SET (CRITICAL STEP)
    # ---------------------------------------------------------
    print(f"[INFO] Saving processed Test Set to {TEST_DATA_SAVE_PATH}...")
    
    # X_test shape is likely (N, 7, 7, 1). Flatten it for CSV saving.
    # This automatically handles 7x7 (49 cols) or 7x10 (70 cols).
    X_test_flat = X_test.reshape(X_test.shape[0], -1) 
    
    df_test_save = pd.DataFrame(X_test_flat)
    df_test_save["label"] = y_test
    
    os.makedirs(os.path.dirname(TEST_DATA_SAVE_PATH), exist_ok=True)
    df_test_save.to_csv(TEST_DATA_SAVE_PATH, index=False)
    print(f"[SUCCESS] Test set saved! ({len(df_test_save)} samples)")

    # ---------------------------------------------------------
    # 3. TENSOR CONVERSION & RESHAPING
    # ---------------------------------------------------------
    # Current Shape: (N, 7, 7, 1) -> (Height, Width, Channel)
    # PyTorch Expects: (N, 1, 7, 7) -> (Channel, Height, Width)
    # We use .permute(0, 3, 1, 2) to fix this.
    
    # Train Set
    X_train_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    # Validation Set
    X_val_t   = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
    y_val_t   = torch.tensor(y_val, dtype=torch.long)

    # Test Set
    X_test_t  = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
    y_test_t  = torch.tensor(y_test, dtype=torch.long)

    # ---------------------------------------------------------
    # 4. CREATE DATALOADERS
    # ---------------------------------------------------------
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset   = TensorDataset(X_val_t,   y_val_t)
    test_dataset  = TensorDataset(X_test_t,  y_test_t)

    print(f"[INFO] Train Samples: {len(train_dataset)}")
    print(f"[INFO] Val Samples:   {len(val_dataset)}")
    print(f"[INFO] Test Samples:  {len(test_dataset)}")

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val":   DataLoader(val_dataset,   batch_size=batch_size, shuffle=False),
        "test":  DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    }


def main():
    """
    Main execution function.
    - Parses arguments.
    - Sets up the model, optimizer, and loss function.
    - Runs the training loop.
    - Saves the final model.
    """
    parser = argparse.ArgumentParser(description="Train IDS CNN Model")
    # Defaulting to multiclass config
    parser.add_argument("--config", type=str, default="configs/multiclass_config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, default="multiclass", choices=["binary", "multiclass"], help="Training mode")
    args = parser.parse_args()

    # Load Configuration
    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        return

    config = load_config(args.config)

    # Set Reproducibility Seed & Device (CPU/GPU)
    set_seed(config.get("seed", 42))
    device = get_device()
    print(f"Using device: {device}")

    # ----------- DATA PREPARATION ----------------
    loaders = create_ids_loaders(
        mode=args.mode,
        batch_size=config["data"]["batch_size"]
    )

    # ----------- MODEL INITIALIZATION ------------
    if args.mode == "binary":
        num_classes = 2
    else:
        # Use value from config or default to 8
        num_classes = config.get("model", {}).get("num_classes", 8)

    model = create_ids_model(
        mode=args.mode,
        num_classes=num_classes
    )
    
    # Move model to GPU if available
    model = model.to(device)

    print(f"Model Architecture: {model.__class__.__name__}")
    print(f"Trainable Parameters: {model.count_parameters():,}")

    # ----------- OPTIMIZER, LOSS & SCHEDULER -----
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

    # ----------- TRAINER SETUP -------------------
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=config["checkpoint"]["save_dir"],
    )

    # ----------- START TRAINING LOOP -------------
    print("\n" + "="*30)
    print(f"STARTING TRAINING ({args.mode.upper()})")
    print("="*30)
    
    history = trainer.train(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        epochs=config["training"]["epochs"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        verbose=config["logging"]["verbose"],
    )

    # ----------- SAVE FINAL MODEL ----------------
    final_filename = f"final_model_{args.mode}.pth"
    final_path = os.path.join(config["checkpoint"]["save_dir"], final_filename)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "history": history,
            "num_classes": num_classes,
            "mode": args.mode
        },
        final_path,
    )

    print(f"\n[DONE] Training finished! Model saved to: {final_path}")
    print(f"Best Validation Accuracy: {trainer.best_val_acc:.2f}%")

    # 1. (LOSS & ACCURACY)
    print("[INFO] Generating training history plots...")
    plot_save_path = os.path.join(config["checkpoint"]["save_dir"], f"history_{args.mode}.png")
    
    plot_training_history(
        history=history,
        save_path=plot_save_path
    )
    print(f"[INFO] History plot saved to {plot_save_path}")

    # 2. DRAW CONFUSION MATRIX (FOR TEST SET)
    print("[INFO] Generating Confusion Matrix on Test Set...")
    
    # Set model to evaluation mode
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # Make predictions on test data
    with torch.no_grad():
        for inputs, labels in loaders["test"]:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate confusion matrix using sklearn
    cm = confusion_matrix(all_labels, all_preds)

    # Specify class names
    if args.mode == "binary":
        class_names = ["Benign", "Attack"]
    else:
        # Convert numbers from 0 to num_classes to strings
        class_names = [str(i) for i in range(num_classes)]

    cm_save_path = os.path.join(config["checkpoint"]["save_dir"], f"confusion_matrix_{args.mode}.png")
    
    # Draw and save the matrix
    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        save_path=cm_save_path,
        normalize=True  # For report, show as percentage
    )
    print(f"[INFO] Confusion Matrix saved to {cm_save_path}")


if __name__ == "__main__":
    main()
    