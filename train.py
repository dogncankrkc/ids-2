"""
IDS – CNN Training Script (Binary + Multiclass)
Works with tabular CSV features → reshaped to (1,7,10) for CNN input.
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.cnn_model import create_ids_model      # CNN for IDS
from src.training.trainer import Trainer                # our trainer class
from src.utils.helpers import (
    set_seed, get_device, get_optimizer, get_scheduler
)
# preprocess fonksiyonları artık 6 değer döndürüyor
from src.data.preprocess import (
     preprocess_binary, preprocess_multiclass
)

from src.data.dataset import load_raw_csv

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_ids_loaders(
    mode: str,
    batch_size: int,
    data_dir: str = "data/raw"
):
    """
    Load CSV → preprocess → train/val/test split → DataLoader
    """
    df = load_raw_csv(data_dir=data_dir)

    # ---------------------------------------------------------
    # DÜZELTME 1: Her iki modda da 6 değişkeni karşılıyoruz.
    # (preprocess.py dosyasını güncellemiştik)
    # ---------------------------------------------------------
    if mode == "binary":
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_binary(df)
        # num_classes burada kullanılmıyor, model create ederken manuel 2 veriyoruz.
    
    elif mode == "multiclass":
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_multiclass(df)
    
    else:
        raise ValueError("mode must be 'binary' or 'multiclass'")

    # ---------------------------------------------------------
    # DÜZELTME 2: Tensor dönüşümleri (Train ve Val için)
    # ---------------------------------------------------------
    # Train
    X_train_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    # Validation (Eğitim sırasında başarım ölçmek için)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
    y_val_t   = torch.tensor(y_val, dtype=torch.long)

    # Test (Eğitim bittikten sonra kullanmak istersen diye hazır tutuyoruz)
    # Şimdilik loader'a eklemiyoruz ama istenirse eklenebilir.
    # X_test_t  = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
    # y_test_t  = torch.tensor(y_test, dtype=torch.long)

    # ---------------------------------------------------------
    # Dataset ve Loader Oluşturma
    # ---------------------------------------------------------
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset   = TensorDataset(X_val_t,   y_val_t)   # Artık gerçek Val seti!

    print(f"[INFO] Train Size: {len(train_dataset)} | Val Size: {len(val_dataset)}")

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val":   DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    }

def main():
    parser = argparse.ArgumentParser(description="Train IDS CNN Model")
    parser.add_argument("--config", type=str, default="configs/binary_config.yaml")
    parser.add_argument("--mode", type=str, default="binary", choices=["binary", "multiclass"])
    args = parser.parse_args()

    # Load YAML config
    config = load_config(args.config)

    # Set seed & device
    set_seed(config.get("seed", 42))
    device = get_device()
    print(f"Using device: {device}")

    # ----------- DATA LOADERS (CSV) -------------
    loaders = create_ids_loaders(
        mode=args.mode,
        batch_size=config["data"]["batch_size"],
        data_dir=config["data"]["raw_dir"]
    )

    # ----------- MODEL --------------------------
    if args.mode == "binary":
        num_classes = 2
    else:
        num_classes = config["model"]["num_classes"]   # örn: 8

    model = create_ids_model(
        mode=args.mode,
        num_classes=num_classes
    )

    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {model.count_parameters():,}")

    # ----------- TRAINING SETUP -----------------
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

    # -------------- TRAIN -------------------------
    print("\nStarting training...")
    history = trainer.train(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        epochs=config["training"]["epochs"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        verbose=config["logging"]["verbose"],
    )

    # -------------- SAVE MODEL --------------------
    final_path = os.path.join(config["checkpoint"]["save_dir"], "final_model.pth")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "history": history,
        },
        final_path,
    )

    print(f"\nTraining finished! Saved to: {final_path}")
    print(f"Best Validation Accuracy: {trainer.best_val_acc:.2f}%")


if __name__ == "__main__":
    main()