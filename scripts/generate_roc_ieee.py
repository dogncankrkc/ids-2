"""
IEEE REPORT GENERATOR - ROC CURVES & AUC (FINAL)
-----------------------------------------------
- Per-class ROC
- Macro & Micro Average ROC
- IEEE-ready visualization
- Apple Silicon (MPS) compatible
"""

import sys
import os

# --------------------------------------------------
# PATH FIX (scripts/ -> project root)
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from torch.utils.data import TensorDataset, DataLoader

from src.models.cnn_model import create_ids_model

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH = os.path.join(
    parent_dir,
    "models/checkpoints/ids_nano_focal_groupNorm_gan/best_model.pth"
)

TEST_DATA_PATH = os.path.join(
    parent_dir,
    "data/processed/test_split_saved.csv"
)

ENCODER_PATH = os.path.join(
    parent_dir,
    "data/processed/label_encoder.pkl"
)

# OUTPUT (IEEE FIGURES)
FIGURE_DIR = os.path.join(parent_dir, "outputs/figures_ieee")
os.makedirs(FIGURE_DIR, exist_ok=True)

SAVE_PATH = os.path.join(FIGURE_DIR, "roc_multiclass_ieee.png")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 512

# --------------------------------------------------
# IEEE PLOT STYLE
# --------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300
})

# --------------------------------------------------
# LOAD DATA & MODEL
# --------------------------------------------------
def load_data_and_model():
    print(f"[INFO] Device: {DEVICE}")

    df_test = pd.read_csv(TEST_DATA_PATH)
    label_col = "label" if "label" in df_test.columns else "multiclass_label"

    X = df_test.drop(columns=[label_col]).values
    y = df_test[label_col].values

    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y, dtype=torch.long)

    test_loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    encoder = joblib.load(ENCODER_PATH)
    class_names = encoder.classes_.tolist()

    num_classes = len(class_names)
    input_dim = X.shape[1]

    model = create_ids_model(
        num_classes=num_classes,
        input_dim=input_dim
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    model.eval()
    return model, test_loader, class_names, num_classes

# --------------------------------------------------
# ROC & AUC
# --------------------------------------------------
def plot_roc_curve(model, loader, class_names, num_classes):
    y_true, y_score = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)

            y_true.extend(y.numpy())
            y_score.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    y_bin = label_binarize(y_true, classes=range(num_classes))

    fpr, tpr, roc_auc = {}, {}, {}

    print("\n[RESULTS] AUC SCORES")
    print("-" * 40)

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(f"{class_names[i]:<15}: {roc_auc[i]:.4f}")

    # -------- MICRO --------
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_bin.ravel(), y_score.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # -------- MACRO --------
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes

    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    print("-" * 40)
    print(f"Macro-average AUC : {roc_auc['macro']:.4f}")
    print(f"Micro-average AUC : {roc_auc['micro']:.4f}")

    # --------------------------------------------------
    # PLOT
    # --------------------------------------------------
    plt.figure(figsize=(8, 6))
    colors = cycle([
        "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b", "#e377c2"
    ])

    for i, color in zip(range(num_classes), colors):
        plt.plot(
            fpr[i], tpr[i],
            lw=2, color=color,
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})"
        )

    # Macro
    plt.plot(
        all_fpr, mean_tpr,
        linestyle=":",
        linewidth=3,
        color="navy",
        label=f"Macro-average (AUC = {roc_auc['macro']:.2f})"
    )

    # Random baseline
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5)

    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 − Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Multi-Class ROC Curves Test Set")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.savefig(SAVE_PATH, bbox_inches="tight")
    print(f"\n[SUCCESS] Figure saved to:\n{SAVE_PATH}")

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    model, loader, classes, n_classes = load_data_and_model()
    plot_roc_curve(model, loader, classes, n_classes)
