"""
IDS – CNN Training Script (Binary + Multiclass).
Updated for 1D CNN with StandardScaler.
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from src.models.cnn_model import create_ids_model
from src.training.trainer import Trainer
from src.utils.helpers import set_seed, get_device, get_optimizer, get_scheduler, count_parameters
from src.data.preprocess import preprocess_binary, preprocess_multiclass
from src.utils.visualization import plot_training_history, plot_confusion_matrix

PROCESSED_DATA_PATH = "data/processed/CIC2023_CAPPED_SMOTE.csv"
TEST_DATA_SAVE_PATH = "data/processed/test_split_saved.csv"

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_ids_loaders(mode: str, batch_size: int):
    print(f"[INFO] Loading dataset from: {PROCESSED_DATA_PATH}")
    
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {PROCESSED_DATA_PATH}. Please run create_balanced_dataset.py first.")

    # 1. Preprocess
    if mode == "binary":
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_binary(df)
        print("[INFO] Mode: Binary Classification")
    elif mode == "multiclass":
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_multiclass(df)
        print("[INFO] Mode: Multiclass Classification")
    else:
        raise ValueError("mode must be 'binary' or 'multiclass'")

    # 2. Save Test Set
    print(f"[INFO] Saving processed Test Set to {TEST_DATA_SAVE_PATH}...")
    df_test_save = pd.DataFrame(X_test) # No reshape needed for CSV
    df_test_save["label"] = y_test
    os.makedirs(os.path.dirname(TEST_DATA_SAVE_PATH), exist_ok=True)
    df_test_save.to_csv(TEST_DATA_SAVE_PATH, index=False)
    print(f"[SUCCESS] Test set saved! ({len(df_test_save)} samples)")

    # 3. Tensor Conversion
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32)
    y_val_t   = torch.tensor(y_val, dtype=torch.long)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.long)

    # 4. DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset   = TensorDataset(X_val_t,   y_val_t)
    test_dataset  = TensorDataset(X_test_t,  y_test_t)

    print(f"[INFO] Train Samples: {len(train_dataset)}")
    print(f"[INFO] Val Samples:   {len(val_dataset)}")
    
    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val":   DataLoader(val_dataset,   batch_size=batch_size, shuffle=False),
        "test":  DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/multiclass_config.yaml")
    parser.add_argument("--mode", type=str, default="multiclass", choices=["binary", "multiclass"])
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        return

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device()
    print(f"Using device: {device}")

    loaders = create_ids_loaders(mode=args.mode, batch_size=config["data"]["batch_size"])

    if args.mode == "binary":
        num_classes = 2
    else:
        num_classes = config.get("model", {}).get("num_classes", 8)

    model = create_ids_model(mode=args.mode, num_classes=num_classes)
    model = model.to(device)

    print(f"Model Architecture: {model.__class__.__name__}")
    print(f"Trainable Parameters: {count_parameters(model):,}")

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

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=config["checkpoint"]["save_dir"],
    )

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

    print("\n" + "=" * 30)
    print("LOADING BEST MODEL FOR TEST")
    print("=" * 30)

    best_model_path = os.path.join(
        config["checkpoint"]["save_dir"],
        "best_model.pth"
    )

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found: {best_model_path}")

    model.load_state_dict(
        torch.load(best_model_path, map_location=device)
    )
    model.to(device)
    model.eval()

    print("[INFO] Best model loaded successfully.")


    print("\n" + "=" * 30)
    print("RUNNING FINAL TEST EVALUATION")
    print("=" * 30)

    test_results = trainer.test(loaders["test"])
    # Save test metrics separately
    test_metrics_path = os.path.join(
        config["checkpoint"]["save_dir"],
        f"test_metrics_{args.mode}.yaml"
    )

    with open(test_metrics_path, "w") as f:
        yaml.safe_dump(test_results, f)

    print(f"[INFO] Test metrics saved to: {test_metrics_path}")



    final_path = os.path.join(config["checkpoint"]["save_dir"], f"final_model_{args.mode}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "history": history,
        "test_results": test_results,
        "num_classes": num_classes, 
        "mode": args.mode
    }, final_path)

    print(f"\n[DONE] Training finished! Model saved to: {final_path}")
    print(f"Best Validation Accuracy: {trainer.best_val_acc:.2f}%")

    plot_training_history(history=history, save_path=os.path.join(config["checkpoint"]["save_dir"], f"history_{args.mode}.png"))

    # Confusion Matrix
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loaders["test"]:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    class_names = ["Benign", "Attack"] if args.mode == "binary" else [str(i) for i in range(num_classes)]
    
    plot_confusion_matrix(
        cm=cm, 
        class_names=class_names, 
        save_path=os.path.join(config["checkpoint"]["save_dir"], f"confusion_matrix_{args.mode}.png"),
        normalize=True
    )

if __name__ == "__main__":
    main()