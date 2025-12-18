"""
IDS - Final Evaluation Script (Standalone - Elite Compatible)
------------------------------------------
Loads the BEST trained model and performs final testing.
Path Fix applied for 'scripts/' directory execution.
"""

import sys
import os
import yaml
import torch
import joblib
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

# ==========================================
# PATH SETUP (FIXED)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))

# KRİTİK DÜZELTME: Script 'scripts/' klasöründe olduğu için
# bir üst dizine (..) çıkarak proje kökünü buluyoruz.
project_root = os.path.abspath(os.path.join(current_dir, "..")) 

if project_root not in sys.path:
    sys.path.append(project_root)

# Internal Imports
# (Proje root eklendikten sonra importlar çalışır)
from src.models.cnn_model import create_ids_model
from src.training.trainer import Trainer
from src.utils.visualization import plot_confusion_matrix
from src.utils.helpers import get_device

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG_PATH = os.path.join(project_root, "configs/multiclass_config.yaml")

# Explorer görüntüsüne göre doğru dosya ismi:
TEST_DATA_PATH = os.path.join(project_root, "data/processed/test_split_saved.csv")

# Encoder yolu
ENCODER_PATH = os.path.join(project_root, "data/processed/label_encoder.pkl")
OUTPUT_DIR = os.path.join(project_root, "outputs/final_evaluation")

def load_config(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    print(f"[WARN] Config not found at {path}")
    return {}

def main():
    # 1. Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    config = load_config(CONFIG_PATH)
    
    # Config dosyasındaki kayıt klasörünü al
    checkpoint_dir = os.path.join(project_root, config.get("checkpoint", {}).get("save_dir", "models/checkpoints"))
    
    # EN İYİ MODELİ HEDEFLİYORUZ
    model_path = os.path.join(checkpoint_dir, "best_model.pth")
    
    batch_size = config.get("data", {}).get("batch_size", 256)
    num_workers = 0 

    print("\n" + "="*50)
    print("STARTING FINAL EVALUATION")
    print("="*50)
    print(f"[INFO] Device      : {device}")
    print(f"[INFO] Root Dir    : {project_root}")
    print(f"[INFO] Model Path  : {model_path}")
    print(f"[INFO] Test Data   : {TEST_DATA_PATH}")

    # 2. Load Label Encoder
    if os.path.exists(ENCODER_PATH):
        encoder = joblib.load(ENCODER_PATH)
        class_names = encoder.classes_
        num_classes = len(class_names)
        print(f"[INFO] Classes     : {class_names}")
    else:
        print(f"[WARN] Label encoder not found at {ENCODER_PATH}. Using numeric labels.")
        # Config'den okumaya çalış
        num_classes = config.get("model", {}).get("num_classes", 7)
        class_names = [str(i) for i in range(num_classes)]

    # 3. Load Test Data
    if not os.path.exists(TEST_DATA_PATH):
        # Hata olursa tam yolu göstersin ki debug edelim
        raise FileNotFoundError(f"CRITICAL: Test data NOT found at:\n{TEST_DATA_PATH}\nCheck if the file exists in 'data/processed'.")
    
    df_test = pd.read_csv(TEST_DATA_PATH)
    
    # Label sütunu kontrolü
    label_col = "label" if "label" in df_test.columns else "multiclass_label"
    print(f"[INFO] Using label column: {label_col}")
    
    X_np = df_test.drop(columns=[label_col]).values
    y_np = df_test[label_col].values

    X_test = torch.tensor(X_np, dtype=torch.float32)
    y_test = torch.tensor(y_np, dtype=torch.long)

    if X_test.ndim == 2:
        X_test = X_test.unsqueeze(1)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"[INFO] Test Samples: {len(test_dataset)}")

    # 4. Load Model
    # Factory fonksiyonu 'cnn_model.py' içindeki en son sınıfı (Elite) çağırır.
    model = create_ids_model(num_classes=num_classes)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Did you finish training?")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("[INFO] Model (Best Epoch) loaded successfully.")

    # 5. Run Prediction Loop
    all_preds = []
    all_targets = []

    print("\n[INFO] Running inference...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    # 6. Generate Confusion Matrix
    print("\n[INFO] Generating Confusion Matrix...")
    cm = confusion_matrix(all_targets, all_preds)
    
    # Save Report
    plot_confusion_matrix(
        cm, 
        class_names=class_names, 
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_elite.png"),
        normalize=True
    )
    
    # Calculate Final Accuracy
    acc = np.sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    print(f"\n🏆 FINAL TEST ACCURACY: {acc*100:.2f}%")
    print(f"[RESULT] Matrix saved to: {OUTPUT_DIR}/confusion_matrix_elite.png")

if __name__ == "__main__":
    main()