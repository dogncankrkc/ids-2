"""
RUN SINGLE INCIDENT EVENT (EDGE) - REALISTIC CASCADE INFERENCE
--------------------------------------------------------------
- Picks ONE sample from a CSV (e.g., a DDoS row)
- Runs Stage-1 XGBoost gatekeeper (binary)
- If Attack -> runs Stage-2 CNN specialist (multiclass attack-only)
- Produces ONE JSON security event (edge output)
- Optionally sends the event to a cloud LLM (Analyst Agent) and saves the report

Author: Dogancan Karakoc
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import argparse
from datetime import datetime, timezone

import yaml
import joblib
import numpy as np
import pandas as pd
import torch

# -------------------------
# PATH SETUP
# -------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
    
from scripts.send_event_to_agent import send_event_to_agent

from src.models.cnn_model import create_ids_model
from src.utils.helpers import get_device


# -------------------------
# CONFIG (defaults)
# -------------------------
DEFAULT_TEST_PATH = "data/processed/real_world_test_v1/test_raw.csv"

BIN_MODEL_PATH = "models/checkpoints/binary_ids/xgboost_binary_model.pkl"
BIN_SCALER_PATH = "data/processed/binary/binary_feature_scaler.pkl"
BIN_THRESH_PATH = "models/checkpoints/binary_ids/xgboost_threshold.yaml"

MULTI_DIR = "models/checkpoints/ids_gan_focal"
MULTI_MODEL = os.path.join(MULTI_DIR, "best_model.pth")
MULTI_SCALER = os.path.join(MULTI_DIR, "feature_scaler.pkl")
LABEL_ENCODER = os.path.join(MULTI_DIR, "label_encoder.pkl")

OUT_DIR = "outputs/single_incident_event"
EVENT_JSON_PATH = os.path.join(OUT_DIR, "incident_event.json")
AGENT_REPORT_PATH = os.path.join(OUT_DIR, "agent_report.txt")

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

STAGE2_BATCH_SIZE = 1  # single sample


def load_threshold(path: str, default: float = 0.5) -> float:
    if not os.path.exists(path):
        return float(default)
    with open(path, "r") as f:
        obj = yaml.safe_load(f) or {}
    return float(obj.get("threshold", default))


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def pick_single_row(df: pd.DataFrame, target_label: str | None, seed: int) -> pd.Series:
    """
    Select one row:
    - If target_label provided: filter multiclass_label == target_label
    - Else: take a random row
    """
    if target_label is not None:
        sub = df[df["multiclass_label"].astype(str) == str(target_label)]
        if len(sub) == 0:
            raise ValueError(f"No rows found for multiclass_label == '{target_label}'")
        # pick one deterministically (seed)
        sub = sub.sample(n=1, random_state=seed).reset_index(drop=True)
        return sub.iloc[0]
    else:
        df2 = df.sample(n=1, random_state=seed).reset_index(drop=True)
        return df2.iloc[0]


def build_event_json(
    device_id: str,
    stage1_pred: str,
    stage1_conf: float,
    stage2_class: str | None,
    stage2_conf: float | None,
    feature_snapshot: dict,
    meta: dict,
) -> dict:
    """
    This is the canonical 'edge security event' format.
    """
    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": now_utc_iso(),
        "device_id": device_id,
        "edge_decision": {
            "stage_1": {
                "model": "XGBoost",
                "prediction": stage1_pred,      # "Benign" or "Attack"
                "confidence": stage1_conf,      # probability for Attack (or 1 - prob)
            },
            "stage_2": None,
        },
        "features_snapshot": feature_snapshot,
        "meta": meta,
    }

    if stage1_pred == "Attack":
        event["edge_decision"]["stage_2"] = {
            "model": "ResNet1D-Nano",
            "attack_class": stage2_class,
            "confidence": stage2_conf,
        }

    return event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default=DEFAULT_TEST_PATH)
    parser.add_argument("--device_id", type=str, default="raspberrypi-01")
    parser.add_argument("--target_label", type=str, default="DDoS",
                        help="Pick one sample with multiclass_label == target_label. Use 'None' to pick random.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot_keys", type=str, default="Rate,Protocol Type,Time_To_Live",
                        help="Comma-separated feature names to include in features_snapshot.")
    # optional cloud
    parser.add_argument("--send_to_agent", action="store_true",
                        help="If set, will call cloud analyst agent (Gemini) using env vars.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # -------------------------
    # LOAD DATA
    # -------------------------
    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"Missing test CSV: {args.test_path}")

    df = pd.read_csv(args.test_path)

    required_cols = FEATURES + ["binary_label", "multiclass_label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Test CSV missing required columns: {missing[:10]}")

    # choose row
    target = None if str(args.target_label).lower() == "none" else args.target_label
    row = pick_single_row(df, target_label=target, seed=args.seed)

    # single sample arrays
    x = row[FEATURES].values.astype(np.float32).reshape(1, -1)
    y_bin_true = int(row["binary_label"])
    y_multi_true = str(row["multiclass_label"])

    # -------------------------
    # LOAD STAGE-1
    # -------------------------
    if not os.path.exists(BIN_MODEL_PATH):
        raise FileNotFoundError(f"Missing: {BIN_MODEL_PATH}")
    if not os.path.exists(BIN_SCALER_PATH):
        raise FileNotFoundError(f"Missing: {BIN_SCALER_PATH}")

    bin_model = joblib.load(BIN_MODEL_PATH)
    bin_scaler = joblib.load(BIN_SCALER_PATH)
    threshold = load_threshold(BIN_THRESH_PATH, default=0.5)

    # Stage-1 inference
    xb = bin_scaler.transform(x)
    prob_attack = float(bin_model.predict_proba(xb)[:, 1][0])
    pred_attack = int(prob_attack >= threshold)

    stage1_pred = "Attack" if pred_attack == 1 else "Benign"
    stage1_conf = prob_attack if stage1_pred == "Attack" else (1.0 - prob_attack)

    # -------------------------
    # LOAD STAGE-2 (only if needed)
    # -------------------------
    stage2_class = None
    stage2_conf = None

    if stage1_pred == "Attack":
        device = get_device()

        if not os.path.exists(MULTI_MODEL):
            raise FileNotFoundError(f"Missing: {MULTI_MODEL}")
        if not os.path.exists(MULTI_SCALER):
            raise FileNotFoundError(f"Missing: {MULTI_SCALER}")
        if not os.path.exists(LABEL_ENCODER):
            raise FileNotFoundError(f"Missing: {LABEL_ENCODER}")

        multi_scaler = joblib.load(MULTI_SCALER)
        encoder = joblib.load(LABEL_ENCODER)
        classes = [str(c) for c in encoder.classes_]
        label_map = {i: classes[i] for i in range(len(classes))}

        # model
        model = create_ids_model(num_classes=len(classes))
        ckpt = torch.load(MULTI_MODEL, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.to(device).eval()

        xm = multi_scaler.transform(x).astype(np.float32)
        with torch.no_grad():
            batch = torch.tensor(xm, dtype=torch.float32).unsqueeze(1).to(device)  # [1,1,39]
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            stage2_class = label_map[pred_idx]
            stage2_conf = float(probs[pred_idx])

    # -------------------------
    # BUILD FEATURE SNAPSHOT
    # -------------------------
    snapshot_keys = [k.strip() for k in args.snapshot_keys.split(",") if k.strip()]
    snapshot = {}
    for k in snapshot_keys:
        if k in row.index:
            v = row[k]
            # make JSON friendly
            if isinstance(v, (np.integer,)):
                snapshot[k] = int(v)
            elif isinstance(v, (np.floating,)):
                snapshot[k] = float(v)
            else:
                snapshot[k] = str(v)

    # -------------------------
    # BUILD EVENT JSON
    # -------------------------
    meta = {
        "true_labels": {
            "binary_label": y_bin_true,
            "multiclass_label": y_multi_true,
        },
        "threshold": threshold,
        "source_test_path": args.test_path,
        "selection": {
            "target_label": target,
            "seed": args.seed,
        }
    }

    event = build_event_json(
        device_id=args.device_id,
        stage1_pred=stage1_pred,
        stage1_conf=stage1_conf,
        stage2_class=stage2_class,
        stage2_conf=stage2_conf,
        feature_snapshot=snapshot,
        meta=meta,
    )

    # save JSON
    with open(EVENT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(event, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("[EDGE] Single incident event generated")
    print(f"[SAVE] {EVENT_JSON_PATH}")
    print("-" * 60)
    print(f"True multiclass: {y_multi_true}")
    print(f"Stage-1        : {stage1_pred} (conf={stage1_conf:.4f}, thr={threshold:.3f})")
    if stage1_pred == "Attack":
        print(f"Stage-2        : {stage2_class} (conf={stage2_conf:.4f})")
    print("=" * 60)

    # optional cloud agent call
    report = send_event_to_agent(event)

    print("\n[CLOUD REPORT]\n")
    print(report)
    
    with open("incident_analysis_report.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
