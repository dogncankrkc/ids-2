"""
TRUE HIERARCHICAL CASCADE INFERENCE – REAL-WORLD TEST (IEEE EDITION v3)
----------------------------------------------------------------------
Stage-1: XGBoost (Binary Gatekeeper)   -> predicts {Benign, Attack}
Stage-2: CNN (Attack Specialist)       -> predicts {BruteForce, DDoS, DoS, Mirai, Recon, Spoofing, Web}

IEEE-ready outputs:
- Metrics YAML + Predictions CSV
- Confusion Matrices:
  1) Stage-1 Binary (counts + row-normalized)
  2) Stage-2 Conditional (Attack-only forwarded) (counts + row-normalized)  [NO BENIGN]
  3) End-to-End Cascade (counts + row-normalized)

Author: Dogancan Karakoc
"""

import os
import sys
import time
import yaml
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
)

# =====================================================
# PATH SETUP
# =====================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.models.cnn_model import create_ids_model
from src.utils.helpers import get_device

# =====================================================
# CONFIGURATION
# =====================================================
TEST_PATH = "data/processed/real_world_test_v1/test_raw.csv"

OUT_DIR = "outputs/cascade_realworld_eval_ieee_v3"
os.makedirs(OUT_DIR, exist_ok=True)

METRICS_PATH = os.path.join(OUT_DIR, "cascade_metrics.yaml")
PRED_PATH    = os.path.join(OUT_DIR, "cascade_predictions.csv")

# Model Paths
BIN_MODEL_PATH  = "models/checkpoints/binary_ids/xgboost_binary_model.pkl"
BIN_SCALER_PATH = "data/processed/binary/binary_feature_scaler.pkl"
BIN_THRESH_PATH = "models/checkpoints/binary_ids/xgboost_threshold.yaml"

MULTI_DIR     = "models/checkpoints/ids_gan_focal"
MULTI_MODEL   = os.path.join(MULTI_DIR, "best_model.pth")
MULTI_SCALER  = os.path.join(MULTI_DIR, "feature_scaler.pkl")
LABEL_ENCODER = os.path.join(MULTI_DIR, "label_encoder.pkl")

FEATURES = [
    'Header_Length','Protocol Type','Time_To_Live','Rate',
    'fin_flag_number','syn_flag_number','rst_flag_number',
    'psh_flag_number','ack_flag_number','ece_flag_number',
    'cwr_flag_number','ack_count','syn_count','fin_count',
    'rst_count','HTTP','HTTPS','DNS','Telnet','SMTP','SSH',
    'IRC','TCP','UDP','DHCP','ARP','ICMP','IGMP','IPv',
    'LLC','Tot sum','Min','Max','AVG','Std','Tot size',
    'IAT','Number','Variance'
]

# Inference batch size for Stage-2
STAGE2_BATCH_SIZE = 4096

# IEEE-like plot defaults (Times-like serif)
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
})

# =====================================================
# UTILITIES
# =====================================================
def to_native(x):
    if isinstance(x, dict):
        return {k: to_native(v) for k, v in x.items()}
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, list):
        return [to_native(v) for v in x]
    return x

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def plot_cm_matplotlib(cm, labels, out_path, title, normalize=False):
    """
    IEEE-friendly confusion matrix plot (matplotlib only).
    - cm: numpy array
    - labels: list[str]
    - normalize: if True, expects cm in [0,1] floats (row-normalized)
    """
    n = len(labels)
    figsize = (7, 5) if n <= 8 else (10, 8)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap=plt.cm.Blues)

    ax.set_title(title, pad=12)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    thresh = cm.max() * (0.6 if not normalize else 0.5)

    for i in range(n):
        for j in range(n):
            value = cm[i, j]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if value > thresh else "black",
                fontsize=9
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_cm_pair(y_true, y_pred, labels, prefix_path, title_base):
    """
    Saves:
    - <prefix_path>_counts.png
    - <prefix_path>_normalized.png  (row-normalized: per true class)
    """
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    out_counts = f"{prefix_path}_counts.png"
    out_norm = f"{prefix_path}_normalized.png"
    _ensure_dir(out_counts)

    plot_cm_matplotlib(
        cm_counts, labels, out_counts,
        title=f"{title_base} (Counts)",
        normalize=False
    )
    plot_cm_matplotlib(
        cm_norm, labels, out_norm,
        title=f"{title_base} (Row-normalized)",
        normalize=True
    )
    print(f"[SAVE] CM -> {os.path.basename(out_counts)}, {os.path.basename(out_norm)}")

def load_threshold(path, default=0.5):
    if not os.path.exists(path):
        return float(default)
    with open(path, "r") as f:
        obj = yaml.safe_load(f) or {}
    return float(obj.get("threshold", default))

# =====================================================
# MAIN PIPELINE
# =====================================================
def main():
    device = get_device()
    print("=" * 70)
    print("TRUE HIERARCHICAL CASCADE EVALUATION – REAL-WORLD TEST (IEEE v3)")
    print("=" * 70)

    # -----------------------------
    # 1) LOAD TEST DATA
    # -----------------------------
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Missing test CSV: {TEST_PATH}")

    df = pd.read_csv(TEST_PATH)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Test CSV missing features: {missing[:10]} (and possibly more)")

    if "binary_label" not in df.columns or "multiclass_label" not in df.columns:
        raise ValueError("Test CSV must contain columns: binary_label, multiclass_label")

    X = df[FEATURES].values
    y_bin = df["binary_label"].astype(int).values                 # 0/1
    y_multi = df["multiclass_label"].astype(str).values           # "Benign" or attack class

    n_total = len(df)
    n_attack_true = int(np.sum(y_bin == 1))
    n_benign_true = int(np.sum(y_bin == 0))
    print(f"[DATA] N={n_total:,} | Benign={n_benign_true:,} | Attack={n_attack_true:,}")

    # -----------------------------
    # 2) LOAD MODELS
    # -----------------------------
    print("\n[INIT] Loading models...")

    # Stage-1
    if not os.path.exists(BIN_MODEL_PATH):
        raise FileNotFoundError(f"Missing: {BIN_MODEL_PATH}")
    if not os.path.exists(BIN_SCALER_PATH):
        raise FileNotFoundError(f"Missing: {BIN_SCALER_PATH}")

    bin_model = joblib.load(BIN_MODEL_PATH)
    bin_scaler = joblib.load(BIN_SCALER_PATH)
    threshold = load_threshold(BIN_THRESH_PATH, default=0.5)
    print(f"[STAGE-1] Fixed threshold = {threshold:.4f}")

    # Stage-2
    if not os.path.exists(MULTI_MODEL):
        raise FileNotFoundError(f"Missing: {MULTI_MODEL}")
    if not os.path.exists(MULTI_SCALER):
        raise FileNotFoundError(f"Missing: {MULTI_SCALER}")
    if not os.path.exists(LABEL_ENCODER):
        raise FileNotFoundError(f"Missing: {LABEL_ENCODER}")

    multi_scaler = joblib.load(MULTI_SCALER)
    encoder = joblib.load(LABEL_ENCODER)
    # IMPORTANT: Stage-2 classes must be attack-only (should NOT include Benign)
    stage2_classes = [str(c) for c in encoder.classes_]

    label_map = {i: stage2_classes[i] for i in range(len(stage2_classes))}
    multi_model = create_ids_model(num_classes=len(stage2_classes))

    ckpt = torch.load(MULTI_MODEL, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    multi_model.load_state_dict(state_dict)
    multi_model.to(device).eval()

    # -----------------------------
    # 3) STAGE-1 INFERENCE
    # -----------------------------
    print("\n[STAGE-1] Running binary gatekeeper...")
    t0 = time.perf_counter()
    Xb = bin_scaler.transform(X)
    bin_probs = bin_model.predict_proba(Xb)[:, 1]
    bin_pred = (bin_probs >= threshold).astype(int)
    t1 = time.perf_counter()

    # stage-1 metrics
    p1, r1, f1, _ = precision_recall_fscore_support(y_bin, bin_pred, average="binary", zero_division=0)
    auc1 = roc_auc_score(y_bin, bin_probs)

    # forwarded ALL predicted attacks (real-world behavior)
    forward_idx_all = np.where(bin_pred == 1)[0]
    forwarded_all = int(len(forward_idx_all))

    # forwarded TRUE attacks only (for Stage-2 conditional attack-only evaluation)
    forward_idx_attack_only = np.where((y_bin == 1) & (bin_pred == 1))[0]
    forwarded_attack_only = int(len(forward_idx_attack_only))

    print(f"[STAGE-1] Precision={p1:.4f} | Recall={r1:.4f} | F1={f1:.4f} | AUC={auc1:.4f}")
    print(f"[STAGE-1] Forwarded (all predicted attacks)      : {forwarded_all:,}  ({forwarded_all/n_total:.2%})")
    print(f"[STAGE-1] Forwarded (true attacks only, TP subset): {forwarded_attack_only:,}  ({(forwarded_attack_only/max(1,n_attack_true)):.2%} of attacks)")

    # Stage-1 Confusion Matrix (binary)
    cm1_prefix = os.path.join(OUT_DIR, "cm_1_stage1_binary")
    save_cm_pair(
        y_true=y_bin,
        y_pred=bin_pred,
        labels=[0, 1],
        prefix_path=cm1_prefix,
        title_base="Stage-1 Gatekeeper Confusion Matrix (Benign=0, Attack=1)"
    )

    # -----------------------------
    # 4) STAGE-2 INFERENCE
    # -----------------------------
    print("\n[STAGE-2] Running multiclass specialist...")

    # Initialize final predictions for end-to-end cascade
    final_pred = np.array(["Benign"] * n_total, dtype=object)

    stage2_pred_all = None  # predictions for all forwarded samples
    if forwarded_all > 0:
        Xm_all = multi_scaler.transform(X[forward_idx_all])

        preds_buf = []
        t2 = time.perf_counter()
        with torch.no_grad():
            for start in range(0, len(Xm_all), STAGE2_BATCH_SIZE):
                end = min(start + STAGE2_BATCH_SIZE, len(Xm_all))
                batch = torch.tensor(Xm_all[start:end], dtype=torch.float32).unsqueeze(1).to(device)
                logits = multi_model(batch)
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                preds_buf.append(preds)
        t3 = time.perf_counter()

        preds_all = np.concatenate(preds_buf, axis=0)
        stage2_pred_all = np.array([label_map[int(p)] for p in preds_all], dtype=object)

        # Put into cascade output (ALL forwarded samples get an attack type)
        final_pred[forward_idx_all] = stage2_pred_all

        print(f"[STAGE-2] Inference done on forwarded samples: {forwarded_all:,}")
    else:
        print("[STAGE-2] No samples forwarded by Stage-1 (bin_pred never == 1).")

    # ---- Stage-2 evaluation (IEEE-clean): ATTACK-ONLY forwarded subset
    # This avoids "Benign" in Stage-2 confusion matrix and matches "attack specialist" definition.
    stage2_conditional_metrics = {
        "evaluated_on": "True attacks that were forwarded by Stage-1 (TP subset).",
        "conditional_macro_f1": 0.0,
        "conditional_macro_precision": 0.0,
        "conditional_macro_recall": 0.0,
        "forwarded_true_attacks": forwarded_attack_only,
        "total_true_attacks": n_attack_true,
    }

    if forwarded_attack_only > 0:
        # Build Stage-2 predictions aligned with attack-only forwarded indices
        # We already predicted for all forwarded; pick the attack-only slice positions
        # Create mapping from forward_idx_all -> pred
        # Efficient way: index mask
        mask_attack_only_in_forwarded_all = np.isin(forward_idx_all, forward_idx_attack_only)
        y_s2_true = y_multi[forward_idx_attack_only]  # should be attack labels only
        y_s2_pred = stage2_pred_all[mask_attack_only_in_forwarded_all]

        # Ensure labels are attack-only for plotting (no Benign)
        labels_s2 = stage2_classes  # fixed order from label encoder

        # Metrics (macro over attack classes)
        p2, r2, f2, _ = precision_recall_fscore_support(
            y_s2_true, y_s2_pred, average="macro", zero_division=0
        )
        stage2_conditional_metrics.update({
            "conditional_macro_precision": float(p2),
            "conditional_macro_recall": float(r2),
            "conditional_macro_f1": float(f2),
        })

        # Stage-2 confusion matrices (attack-only)
        cm2_prefix = os.path.join(OUT_DIR, "cm_2_stage2_attack_only")
        save_cm_pair(
            y_true=y_s2_true,
            y_pred=y_s2_pred,
            labels=labels_s2,
            prefix_path=cm2_prefix,
            title_base="Stage-2 Specialist Confusion Matrix (Attack-only, Conditional on Stage-1)"
        )

        print(f"[STAGE-2] Conditional (attack-only) Macro-F1={f2:.4f}")
    else:
        print("[STAGE-2] No true-attack samples were forwarded (TP subset empty). Stage-2 conditional CM skipped.")

    # -----------------------------
    # 5) END-TO-END CASCADE METRICS
    # -----------------------------
    print("\n[CASCADE] Computing end-to-end metrics...")
    acc = accuracy_score(y_multi, final_pred)

    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_multi, final_pred, average="macro", zero_division=0
    )
    p_mic, r_mic, f_mic, _ = precision_recall_fscore_support(
        y_multi, final_pred, average="micro", zero_division=0
    )

    # False Negative Rate for attacks: predicted Benign while true is attack
    fn = int(np.sum((y_multi != "Benign") & (final_pred == "Benign")))
    fn_rate = float(fn / max(1, n_attack_true))

    print(f"[CASCADE] Accuracy={acc:.4f} | Macro-F1={f_mac:.4f} | Micro-F1={f_mic:.4f} | Attack-FNR={fn_rate:.4f}")

    # End-to-end confusion matrices (include Benign)
    labels_cascade = ["Benign"] + stage2_classes
    cm3_prefix = os.path.join(OUT_DIR, "cm_3_cascade_end_to_end")
    save_cm_pair(
        y_true=y_multi,
        y_pred=final_pred,
        labels=labels_cascade,
        prefix_path=cm3_prefix,
        title_base="End-to-End Cascade Confusion Matrix (Benign + Attack Classes)"
    )

    # -----------------------------
    # 6) PER-CLASS BREAKDOWN (for IEEE table)
    # -----------------------------
    cls_labels = labels_cascade
    p_cls, r_cls, f_cls, s_cls = precision_recall_fscore_support(
        y_multi, final_pred, labels=cls_labels, zero_division=0
    )
    per_class = {}
    for cls, p, r, f, s in zip(cls_labels, p_cls, r_cls, f_cls, s_cls):
        per_class[str(cls)] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }

    # -----------------------------
    # 7) SAVE METRICS + PREDICTIONS
    # -----------------------------
    metrics = {
        "dataset": {
            "test_path": TEST_PATH,
            "n_total": int(n_total),
            "n_benign": int(n_benign_true),
            "n_attack": int(n_attack_true),
        },
        "stage_1_gatekeeper": {
            "model": "XGBoost",
            "threshold": float(threshold),
            "precision": float(p1),
            "recall": float(r1),
            "f1": float(f1),
            "roc_auc": float(auc1),
            "forwarded_all_predicted_attacks": int(forwarded_all),
            "forwarded_true_attacks_only": int(forwarded_attack_only),
        },
        "stage_2_specialist": stage2_conditional_metrics,
        "cascade_end_to_end": {
            "accuracy": float(acc),
            "macro_f1": float(f_mac),
            "micro_f1": float(f_mic),
            "macro_precision": float(p_mac),
            "macro_recall": float(r_mac),
            "attack_false_negative_rate": float(fn_rate),
            "attack_false_negatives": int(fn),
        },
        "per_class_breakdown": per_class,
        "artifacts": {
            "cm_stage1_binary_counts": os.path.basename(cm1_prefix + "_counts.png"),
            "cm_stage1_binary_normalized": os.path.basename(cm1_prefix + "_normalized.png"),
            "cm_stage2_attack_only_counts": os.path.basename(os.path.join(OUT_DIR, "cm_2_stage2_attack_only_counts.png")),
            "cm_stage2_attack_only_normalized": os.path.basename(os.path.join(OUT_DIR, "cm_2_stage2_attack_only_normalized.png")),
            "cm_cascade_counts": os.path.basename(cm3_prefix + "_counts.png"),
            "cm_cascade_normalized": os.path.basename(cm3_prefix + "_normalized.png"),
        }
    }

    with open(METRICS_PATH, "w") as f:
        yaml.safe_dump(to_native(metrics), f, sort_keys=False)

    out_df = df.copy()
    out_df["stage1_prob_attack"] = bin_probs
    out_df["stage1_pred"] = bin_pred
    out_df["final_pred"] = final_pred
    out_df.to_csv(PRED_PATH, index=False)

    print("\n" + "=" * 70)
    print("[DONE] IEEE-ready cascade evaluation complete.")
    print(f"[SAVE] Metrics      : {METRICS_PATH}")
    print(f"[SAVE] Predictions  : {PRED_PATH}")
    print(f"[SAVE] Figures dir  : {OUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
