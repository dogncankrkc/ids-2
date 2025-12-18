"""
IDS – ResNet-MLP Training Script (Multiclass).

Architecture:
- Input features kept as 1D vector (N, 1, L)
- Designed for network traffic classification (IDS)
- StandardScaler + fixed LabelEncoder (NO data leakage)

Pipeline:
1. Load capped dataset
2. Preprocess (split + scaling + encoder, optional selective SMOTE inside preprocess)
3. Reshape features → ResNet-MLP format (N, 1, L)
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

# Model Factory (src/models/cnn_model.py içindeki fonksiyonu çağırır)
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

# Preprocess dosyasından özellik listesini alıyoruz
from src.data.preprocess import SELECTED_FEATURES

import torch.nn.functional as F
# train.py (veya ana scriptin) EN TEPESİNE, importların altına ekle:

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Cross Entropy Loss (log_softmax + nll_loss)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss) # Probability of true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
# ============================================================
# PATHS & CONSTANTS
# ============================================================

PROCESSED_DATA_PATH = "data/processed/CIC2023_ATTACK_ONLY_6CLASS.csv"
TEST_DATA_SAVE_PATH = "data/processed/test_split_saved.csv"
ENCODER_PATH = "data/processed/label_encoder.pkl"

# ============================================================
# UTILITIES
# ============================================================

def load_config(config_path: str):
    """Loads YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def to_python_int(d):
    """Helper to convert numpy types to python int for YAML dumping"""
    return {int(k): int(v) for k, v in d.items()}

# ============================================================
# DATA LOADER CREATION
# ============================================================

def create_ids_loaders(batch_size: int, num_workers: int = 0):
    """
    Loads dataset, applies preprocessing and prepares PyTorch DataLoaders.
    ResNet input format: (N, 1, L)
    """
    print(f"[INFO] Loading dataset from: {PROCESSED_DATA_PATH}")

    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {PROCESSED_DATA_PATH}. "
            f"Please run the capped dataset creator first."
        )

    df = pd.read_csv(PROCESSED_DATA_PATH)

    # --------------------------------------------------------
    # Preprocessing (split + scaling + encoder, optional SMOTE inside)
    # --------------------------------------------------------
    # Bu fonksiyon preprocess.py'den geliyor ve tüm ağır işi yapıyor
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_multiclass(df)

    print("[INFO] Multiclass preprocessing completed.")
    print(f"[INFO] Feature count per sample (L): {X_train.shape[1]}")

    # Basic safety checks
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Feature count mismatch across splits!"

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
    # Tensor conversion & reshape for ResNet: (N, 1, L)
    # --------------------------------------------------------
    # Model (N, 1, L) bekliyor, o yüzden unsqueeze(1) yapıyoruz.
    # Ancak yeni model (N, L) de kabul ediyor, yine de standart koruyalım.
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
        # Varsayılan konfig dosyası yoksa uyarı ver
        print(f"[WARNING] Config file not found at {args.config}. Make sure path is correct.")
    
    # Config dosyasını yükle
    try:
        config = load_config(args.config)
    except:
        # Fallback config (Eğer dosya yoksa kod patlamasın diye)
        print("[INFO] Loading fallback default configuration.")
        config = {
            "model": {"type": "ids_resnet_mlp", "num_classes": 8},
            "data": {"batch_size": 128, "num_workers": 0},
            "training": {
                "epochs": 50, "learning_rate": 0.001, "weight_decay": 0.0001,
                "optimizer": "adamw", "scheduler": "plateau", "early_stopping_patience": 10
            },
            "checkpoint": {"save_dir": "models/checkpoints/ids_multiclass_resnet1d"},
            "logging": {"verbose": True},
            "seed": 42
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
    # SAVE CLASS DISTRIBUTION (FOR REPORTING)
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
        # Fallback if encoder not found immediately (shouldn't happen)
        num_classes = 8
        class_names = [str(i) for i in range(8)]

    # --------------------------------------------------------
    # Model Initialization
    # --------------------------------------------------------
    # Preprocess adımından gelen input boyutunu (L) otomatik al
    input_dim = train_loader.dataset.tensors[0].shape[2] # (N, 1, L) -> L is index 2
    
    model = create_ids_model(
        mode="multiclass", 
        num_classes=num_classes,
        input_dim=input_dim
    ).to(device)

    print(f"\n[INFO] Model: {model.__class__.__name__}")
    print(f"[INFO] Trainable parameters: {count_parameters(model):,}")
    
    # --------------------------------------------------------
    # Save Model Info
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
    # Loss with Class Weights
    # --------------------------------------------------------
    # Dengeli eğitim için sınıf ağırlıklarını hesapla
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"[INFO] Class Weights: {class_weights.cpu().numpy()}")

    loss_type = config.get("training", {}).get("loss", {}).get("type", "cross_entropy")
    # Label Smoothing ile Loss
    if loss_type == "focal":
        print(f"[INFO] Using Focal Loss (gamma={config['training']['loss'].get('gamma', 2.0)})")
        # Focal Loss sınıfını yukarıda tanımlamıştık
        criterion = FocalLoss(alpha=class_weights, gamma=config['training']['loss'].get('gamma', 2.0))
    else:
        print("[INFO] Using CrossEntropy Loss")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

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
    # Training Loop
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
    # Load Best Model & Test
    # --------------------------------------------------------
    best_model_path = os.path.join(config["checkpoint"]["save_dir"], "best_model.pth")

    if os.path.exists(best_model_path):
        print("\n[INFO] Loading best model from checkpoint...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    model.eval()

    # --------------------------------------------------------
    # Final Evaluation
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
    # Save Final Bundle
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
    # Plot Results
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