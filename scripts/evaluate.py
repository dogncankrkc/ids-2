"""
IDS - Final Evaluation Script (Standalone)
------------------------------------------
This script loads the best trained model and the reserved test dataset
to perform a final performance evaluation.

Features:
- Auto-detects project root directory to fix import errors.
- Loads 'best_model.pth' and 'test_split_saved.csv'.
- Generates:
    1. Metrics Report (YAML)
    2. Prediction Details (CSV)
    3. Confusion Matrix (PNG)
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
# PATH SETUP (CRITICAL FOR IMPORTS)
# ==========================================
# Resolves the project root directory regardless of where the script is run from.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Internal Imports
from src.models.cnn_model import create_ids_model
from src.training.trainer import Trainer
from src.utils.visualization import plot_confusion_matrix
from src.utils.helpers import get_device

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG_PATH = os.path.join(project_root, "configs/multiclass_config.yaml")
TEST_DATA_PATH = os.path.join(project_root, "data/processed/test_split_saved.csv")
ENCODER_PATH = os.path.join(project_root, "models/label_encoder.pkl")
OUTPUT_DIR = os.path.join(project_root, "outputs/final_evaluation")

# Default settings if config fails
DEFAULT_BATCH_SIZE = 256 # Safe for Mac/PC

def load_config(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    print(f"[WARN] Config not found at {path}. Using defaults.")
    return {}

def main():
    # 1. Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    config = load_config(CONFIG_PATH)
    
    checkpoint_dir = os.path.join(project_root, config.get("checkpoint", {}).get("save_dir", "models/checkpoints"))
    model_path = os.path.join(checkpoint_dir, "best_model.pth")
    batch_size = config.get("data", {}).get("batch_size", DEFAULT_BATCH_SIZE)
    # Force num_workers=0 for MacOS stability in standalone scripts
    num_workers = 0 

    print("\n" + "="*50)
    print("STARTING FINAL EVALUATION")
    print("="*50)
    print(f"[INFO] Device      : {device}")
    print(f"[INFO] Model Path  : {model_path}")
    print(f"[INFO] Test Data   : {TEST_DATA_PATH}")

    # 2. Load Label Encoder (for proper class names in Confusion Matrix)
    if os.path.exists(ENCODER_PATH):
        encoder = joblib.load(ENCODER_PATH)
        class_names = encoder.classes_
        num_classes = len(class_names)
        print(f"[INFO] Classes     : {class_names}")
    else:
        print("[WARN] Label encoder not found. Using numeric labels.")
        num_classes = config.get("model", {}).get("num_classes", 8)
        class_names = [str(i) for i in range(num_classes)]

    # 3. Load Test Data
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found at {TEST_DATA_PATH}. Run training first.")
    
    df_test = pd.read_csv(TEST_DATA_PATH)
    
    # Separate features and labels
    X_np = df_test.drop(columns=["label"]).values
    y_np = df_test["label"].values

    # Convert to Tensors
    X_test = torch.tensor(X_np, dtype=torch.float32)
    y_test = torch.tensor(y_np, dtype=torch.long)

    # Reshape for 1D CNN: (Batch, Features) -> (Batch, 1, Features)
    if X_test.ndim == 2:
        X_test = X_test.unsqueeze(1)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    print(f"[INFO] Test Samples: {len(test_dataset)}")

    # 4. Load Model
    model = create_ids_model(num_classes=num_classes)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully.")

    # 5. Run Evaluation (using Trainer.test method for consistency)
    trainer = Trainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters()), # Dummy optimizer
        device=device,
        checkpoint_dir=checkpoint_dir
    )

    print("\n[INFO] Running inference metrics...")
    results = trainer.test(test_loader)

    # Save metrics to YAML
    metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.yaml")
    with open(metrics_path, "w") as f:
        yaml.safe_dump(results, f)
    print(f"[SAVE] Metrics saved to: {metrics_path}")

    # 6. Detailed Predictions & Confusion Matrix
    print("\n[INFO] Generating Confusion Matrix...")
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    # Generate CSV Report
    results_df = pd.DataFrame({
        "True_Label_ID": all_targets,
        "Pred_Label_ID": all_preds,
        # Map IDs to Names if encoder exists
        "True_Label_Name": [class_names[i] if i < len(class_names) else str(i) for i in all_targets],
        "Pred_Label_Name": [class_names[i] if i < len(class_names) else str(i) for i in all_preds]
    })
    
    # Check for "Other" class predictions
    other_count = results_df[results_df["True_Label_Name"] == "Other"].shape[0]
    if other_count > 0:
        print(f"[WARN] Found {other_count} samples labeled as 'Other'. Data cleaning recommended.")

    csv_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"[SAVE] Predictions CSV: {csv_path}")

    # Generate Confusion Matrix Plots
    cm = confusion_matrix(all_targets, all_preds)
    
    plot_confusion_matrix(
        cm, 
        class_names=class_names, 
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
        normalize=True
    )
    
    print("\n EVALUATION COMPLETE.")

if __name__ == "__main__":
    main()