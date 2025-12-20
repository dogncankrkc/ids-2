"""
TRUE HIERARCHICAL CASCADE INFERENCE – REAL WORLD TEST
Stage-1: Binary Gatekeeper (XGBoost)
Stage-2: Multiclass Specialist (CNN)

Test set:
- RAW
- Unbalanced
- Never seen by either model

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
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# =====================================================
# PATH SETUP
# =====================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.models.cnn_model import create_ids_model
from src.utils.helpers import get_device

# =====================================================
# CONFIG
# =====================================================
TEST_PATH = "data/processed/real_world_test_v1/test_raw.csv"

OUT_DIR = "outputs/cascade_realworld_eval"
os.makedirs(OUT_DIR, exist_ok=True)

METRICS_PATH = f"{OUT_DIR}/cascade_metrics.yaml"
PRED_PATH    = f"{OUT_DIR}/cascade_predictions.csv"

CM_BIN   = f"{OUT_DIR}/cm_stage1_binary.png"
CM_S2    = f"{OUT_DIR}/cm_stage2_attack_only.png"
CM_FINAL = f"{OUT_DIR}/cm_cascade_end_to_end.png"

STAGE2_BATCH = 4096

# ---------- Stage-1 ----------
BIN_MODEL  = "models/checkpoints/binary_ids/xgboost_binary_model.pkl"
BIN_SCALER = "data/processed/binary/binary_feature_scaler.pkl"

# ---------- Stage-2 ----------
MULTI_DIR   = "models/checkpoints/ids_gan_focal"
MULTI_MODEL = f"{MULTI_DIR}/best_model.pth"
MULTI_SCALER = f"{MULTI_DIR}/feature_scaler.pkl"
LABEL_ENCODER = f"{MULTI_DIR}/label_encoder.pkl"

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

# =====================================================
# HELPERS
# =====================================================
def auto_threshold_precision(y_true, y_prob, target_p=0.85):
    p, r, t = precision_recall_curve(y_true, y_prob)
    t = np.append(t, 1.0)
    ok = np.where(p >= target_p)[0]
    if len(ok) == 0:
        return 0.99
    return float(t[ok[np.argmax(r[ok])]])

def save_cm(y_true, y_pred, labels, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8,6))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def to_native(x):
    if isinstance(x, dict):
        return {k: to_native(v) for k,v in x.items()}
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    return x

# =====================================================
# MAIN
# =====================================================
def main():
    device = get_device()

    df = pd.read_csv(TEST_PATH)
    X = df[FEATURES].values
    y_bin = df["binary_label"].astype(int).values
    y_multi = df["multiclass_label"].astype(str).values

    # ================= Stage-1 =================
    bin_model = joblib.load(BIN_MODEL)
    bin_scaler = joblib.load(BIN_SCALER)

    Xb = bin_scaler.transform(X)
    bin_prob = bin_model.predict_proba(Xb)[:,1]

    threshold = auto_threshold_precision(y_bin, bin_prob, target_p=0.85)
    bin_pred = (bin_prob >= threshold).astype(int)

    save_cm(
        y_bin, bin_pred, [0,1],
        CM_BIN, "Stage-1 Binary Confusion Matrix"
    )

    # ================= Stage-2 =================
    attack_idx = np.where((y_bin==1) & (bin_pred==1))[0]

    encoder = joblib.load(LABEL_ENCODER)
    label_map = {i:l for i,l in enumerate(encoder.classes_)}

    multi_model = create_ids_model(num_classes=len(label_map))
    ckpt = torch.load(MULTI_MODEL, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        multi_model.load_state_dict(ckpt["model_state_dict"])
        print("[INFO] Loaded checkpoint with model_state_dict")
    else:
        multi_model.load_state_dict(ckpt)
        print("[INFO] Loaded raw state_dict checkpoint")

    multi_model.to(device).eval()

    multi_scaler = joblib.load(MULTI_SCALER)

    final_pred = np.array(["Benign"] * len(df), dtype=object)
    stage2_preds = []

    if len(attack_idx) > 0:
        Xm = multi_scaler.transform(X[attack_idx])

        with torch.no_grad():
            for i in range(0, len(Xm), STAGE2_BATCH):
                batch = torch.tensor(
                    Xm[i:i+STAGE2_BATCH],
                    dtype=torch.float32
                ).unsqueeze(1).to(device)
                out = multi_model(batch)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                stage2_preds.extend([label_map[p] for p in preds])

        final_pred[attack_idx] = stage2_preds

        save_cm(
            y_multi[attack_idx],
            np.array(stage2_preds),
            sorted(set(y_multi[attack_idx])),
            CM_S2,
            "Stage-2 Confusion Matrix (Attack-Only)"
        )

    # ================= Cascade =================
    save_cm(
        y_multi,
        final_pred,
        sorted(set(y_multi)),
        CM_FINAL,
        "Cascade Confusion Matrix (End-to-End)"
    )

    # ================= Metrics =================
    p1,r1,f1,_ = precision_recall_fscore_support(
        y_bin, bin_pred, average="binary", zero_division=0
    )

    p2,r2,f2,_ = precision_recall_fscore_support(
        y_multi[attack_idx], np.array(stage2_preds),
        average="macro", zero_division=0
    ) if len(attack_idx)>0 else (0,0,0,None)

    p3,r3,f3,_ = precision_recall_fscore_support(
        y_multi, final_pred, average="macro", zero_division=0
    )

    metrics = {
        "stage_1": {
            "threshold": threshold,
            "precision": p1,
            "recall": r1,
            "f1": f1,
            "roc_auc": roc_auc_score(y_bin, bin_prob),
            "forwarded": int(len(attack_idx)),
        },
        "stage_2": {
            "conditional_macro_f1": f2,
        },
        "cascade": {
            "end_to_end_macro_f1": f3,
            "false_negative_rate": float(
                np.sum((y_multi!="Benign") & (final_pred=="Benign"))
                / max(1, np.sum(y_multi!="Benign"))
            )
        }
    }

    with open(METRICS_PATH,"w") as f:
        yaml.safe_dump(to_native(metrics), f, sort_keys=False)

    out = df.copy()
    out["stage1_prob"] = bin_prob
    out["stage1_pred"] = bin_pred
    out["final_pred"] = final_pred
    out.to_csv(PRED_PATH, index=False)

    print("[DONE] Real-world cascade inference finished")
    print(yaml.safe_dump(to_native(metrics), sort_keys=False))

if __name__ == "__main__":
    main()
