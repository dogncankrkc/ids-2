"""
IEEE REPORT GENERATOR - CONFUSION MATRIX (FINAL)
-----------------------------------------------
- Loads best_model.pth
- Uses reserved test set
- Generates IEEE-compliant confusion matrix
"""

import os
import sys
import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# --------------------------------------------------
# PATH FIX
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from src.models.cnn_model import create_ids_model
from src.utils.helpers import get_device

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH = os.path.join(
    project_root,
    "models/checkpoints/ids_nano_focal_groupNorm_gan/best_model.pth"
)
TEST_DATA_PATH = os.path.join(
    project_root,
    "data/processed/test_split_saved.csv"
)

ENCODER_PATH = os.path.join(
    project_root,
    "data/processed/label_encoder.pkl"
)

SAVE_DIR = os.path.join(project_root, "outputs/figures_ieee")
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_PATH = os.path.join(SAVE_DIR, "confusion_matrix_ieee.png")

BATCH_SIZE = 256
DEVICE = get_device()

# --------------------------------------------------
# IEEE STYLE
# --------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "figure.dpi": 300,
    "axes.grid": False
})

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df_test = pd.read_csv(TEST_DATA_PATH)
X = df_test.drop(columns=["label"]).values
y_true = df_test["label"].values

X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_t = torch.tensor(y_true, dtype=torch.long)

# --------------------------------------------------
# LOAD LABEL NAMES
# --------------------------------------------------
encoder = joblib.load(ENCODER_PATH)
class_names = encoder.classes_.tolist()
num_classes = len(class_names)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = create_ids_model(
    num_classes=num_classes,
    input_dim=X.shape[1]
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(
    checkpoint["model_state_dict"]
    if "model_state_dict" in checkpoint
    else checkpoint
)

model.eval()

# --------------------------------------------------
# INFERENCE
# --------------------------------------------------
all_preds = []

with torch.no_grad():
    for i in range(0, len(X_t), BATCH_SIZE):
        batch = X_t[i:i+BATCH_SIZE].to(DEVICE)
        outputs = model(batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
cm = confusion_matrix(y_true, all_preds)

# Normalize (row-wise)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

# --------------------------------------------------
# PLOT
# --------------------------------------------------
plt.figure(figsize=(6, 5))
plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
plt.colorbar(fraction=0.046)

plt.xticks(
    ticks=np.arange(num_classes),
    labels=class_names,
    rotation=45,
    ha="right"
)
plt.yticks(
    ticks=np.arange(num_classes),
    labels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Cell values
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(
            j, i,
            f"{cm_norm[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if cm_norm[i, j] > 0.5 else "black",
            fontsize=9
        )

plt.tight_layout()
plt.savefig(SAVE_PATH, bbox_inches="tight")
plt.close()

print(f"[SUCCESS] IEEE Confusion Matrix saved to:\n{SAVE_PATH}")
