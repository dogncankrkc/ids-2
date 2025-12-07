"""
Test Script for IDS - Generates CSV + Plots (Confusion Matrix & Curves)
"""

import os
import torch
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# === IMPORTS ===
from src.models.cnn_model import create_ids_model
from src.training.metrics import accuracy, precision, recall, f1_score
from src.data.preprocess import preprocess_multiclass

# Calling visualization functions
from src.utils.visualization import plot_training_history, plot_confusion_matrix

# ==============================
# CONFIGURATION
# ==============================
# This is our "large" dataset
TEST_CSV = "data/raw/datasense_MASTER_FULL.csv" 
MODEL_PATH = "models/checkpoints/ids_multiclass/best_model.pth"
ENCODER_PATH = "models/label_encoder.pkl"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "prediction_results_multiclass.csv")


# ==============================
# 1) LABEL MAP & MODEL LOAD
# ==============================
def load_label_map():
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder yok: {ENCODER_PATH}")
    encoder = joblib.load(ENCODER_PATH)
    return {i: label for i, label in enumerate(encoder.classes_)}

def load_model(num_classes):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Modeli oluştur
    model = create_ids_model(mode="multiclass", num_classes=num_classes)
    model.to(device)

    # Dummy forward (Lazy layers için)
    _ = model(torch.randn(1, 1, 7, 10, device=device))

    # Checkpoint yükle
    print(f"[INFO] Loading model weights from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # --- FIX HERE ---
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # final_model.pth format (Dictionary)
        model.load_state_dict(checkpoint["model_state_dict"])
        history = checkpoint.get("history", {})
    else:
        # best_model.pth format (Direct State Dict)
        model.load_state_dict(checkpoint)
        history = {} # Best model keeps only weights so that history is empty
    # -----------------------

    model.eval()
    return model, device, history


# ==============================
# 2) LOAD TEST DATA
# ==============================
def load_test_data():
    print(f"[INFO] Loading dataset: {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)
    
    # Only take the Test set (Preprocess returns 6 values)
    _, _, X_test, _, _, y_test = preprocess_multiclass(df)
    
    print(f"[INFO] Test Set Size: {len(y_test)} samples")

    X_test_t = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    return DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)


# ==============================
# 3) MAIN EVALUATION
# ==============================
def main():
    # A) Preparation
    inv_label_map = load_label_map()
    num_classes = len(inv_label_map)
    model, device, history = load_model(num_classes)
    loader = load_test_data()

    # B) Prediction Loop
    all_preds, all_targets, all_probs = [], [], []
    print("[INFO] Running predictions...")
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            
            probs = torch.softmax(logits, dim=1)
            conf, preds = torch.max(probs, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_probs.append(conf.cpu())

    # Concatenate tensors
    t_preds = torch.cat(all_preds)
    t_targets = torch.cat(all_targets)
    t_probs = torch.cat(all_probs)
    
    # Numpy conversion (for Pandas and Sklearn)
    np_preds = t_preds.numpy()
    np_targets = t_targets.numpy()
    np_probs = t_probs.numpy()

    # C) CSV OLLECT & SAVE
    df_results = pd.DataFrame({
        "real_label": [inv_label_map[i] for i in np_targets],
        "pred_label": [inv_label_map[i] for i in np_preds],
        "confidence": np_probs,
        "correct": (np_preds == np_targets).astype(int)
    })
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ CSV Saved: {OUTPUT_CSV}")

    # D) metrics 
    print("\n" + "="*30)
    print("      TEST METRICS      ")
    print("="*30)
    print(f"Accuracy  : {accuracy(t_preds, t_targets):.2f}%")
    print(f"Precision : {precision(t_preds, t_targets):.4f}")
    print(f"Recall    : {recall(t_preds, t_targets):.4f}")
    print(f"F1 Score  : {f1_score(t_preds, t_targets):.4f}")
    print("="*30)

    # ==============================
    # E) PLOT GRAPHS (VISUALIZATION)
    # ==============================
    
    # 1. Training History (Loss/Acc Curves)
    if history:
        print("[INFO] Plotting Training Curves...")
        plot_training_history(
            history, 
            save_path=os.path.join(OUTPUT_DIR, "training_curves.png")
        )
    else:
        print("[WARN] History not found, skipping curves.")

    # 2. Confusion Matrix
    print("[INFO] Plotting Confusion Matrix...")
    
    # Get class names (from label map)
    class_names = [inv_label_map[i] for i in range(num_classes)]
    
    # Calculate confusion matrix with sklearn
    cm = confusion_matrix(np_targets, np_preds)
    
    plot_confusion_matrix(
        cm,
        class_names=class_names,
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
        figsize=(10, 8)
    )

    plot_confusion_matrix(
        cm,
        class_names=class_names,
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.png"), # New file name
        figsize=(10, 8),
        normalize=True   # Shows percentages (0.0 - 1.0 range)
    )
    
    print(f"✅ All plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()