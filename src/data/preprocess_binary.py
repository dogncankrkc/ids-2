"""
Binary Preprocessing Utilities for CIC-IoT-2023 (Hierarchical IDS – Stage 1)

Purpose:
- Prepare a realistic binary dataset for edge IDS usage
- Binary task: Benign (0) vs Attack (1)
- Designed as a fast pre-filter before multiclass inference

Design principles:
- Realistic class imbalance (Benign-dominant)
- Attack diversity over attack volume
- Leak-free split (aligned with multiclass pipeline)
- GAN usage strictly limited to minority attack classes
- Dataset snapshots saved for research / thesis reproducibility
"""

import os
import joblib
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
SELECTED_FEATURES = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH',
    'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv',
    'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size',
    'IAT', 'Number', 'Variance'
]

SAVE_DIR = "data/processed/binary"
SCALER_PATH = f"{SAVE_DIR}/binary_feature_scaler.pkl"
GAN_DATA_PATH = "data/processed/GAN_SYNTHETIC_ONLY.csv"

TRAIN_PATH = f"{SAVE_DIR}/train_binary.csv"
VAL_PATH   = f"{SAVE_DIR}/val_binary.csv"
TEST_PATH  = f"{SAVE_DIR}/test_binary.csv"
STATS_PATH = f"{SAVE_DIR}/binary_dataset_stats.yaml"

# Sampling strategy
MAX_BENIGN_SAMPLES = 1_000_000
MAX_ATTACK_TOTAL = 250_000
MAX_GAN_RATIO_PER_CLASS = 0.30   # ≤30% GAN contribution


# --------------------------------------------------
# MAIN PREPROCESS FUNCTION
# --------------------------------------------------
def preprocess_binary(df_real: pd.DataFrame, use_gan: bool = True):
    """
    Prepare binary dataset for hierarchical IDS.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """

    print("\n" + "=" * 70)
    print("BINARY PREPROCESSING STARTED (HIERARCHICAL IDS – RESEARCH MODE)")
    print("=" * 70)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # --------------------------------------------------
    # 1. LABEL CREATION
    # --------------------------------------------------
    label_col = "multiclass_label" if "multiclass_label" in df_real.columns else "label"

    df = df_real.copy()
    df["binary_label"] = (df[label_col] != "Benign").astype(int)

    benign_df = df[df["binary_label"] == 0]
    attack_df = df[df["binary_label"] == 1]

    print(f"[INFO] Total benign samples  : {len(benign_df):,}")
    print(f"[INFO] Total attack samples  : {len(attack_df):,}")

    # --------------------------------------------------
    # 2. BENIGN SAMPLING
    # --------------------------------------------------
    if len(benign_df) > MAX_BENIGN_SAMPLES:
        benign_df = benign_df.sample(
            n=MAX_BENIGN_SAMPLES,
            random_state=42
        )
        print(f"[INFO] Benign downsampled to {len(benign_df):,}")

    # --------------------------------------------------
    # 3. ATTACK SAMPLING (DIVERSITY-FOCUSED)
    # --------------------------------------------------
    attack_groups = attack_df.groupby(label_col)
    per_class_target = MAX_ATTACK_TOTAL // attack_groups.ngroups

    attack_samples = []
    attack_stats = {}

    print("\n[INFO] Attack class sampling strategy:")
    for cls_name, cls_df in attack_groups:
        real_count = len(cls_df)
        take = min(real_count, per_class_target)

        attack_samples.append(
            cls_df.sample(n=take, random_state=42)
        )

        attack_stats[cls_name] = {
            "real_available": int(real_count),
            "real_used": int(take),
            "gan_added": 0
        }

        print(f"  {cls_name:<15} | Real used: {take:<8} / {real_count:<8}")

    attack_df_final = pd.concat(attack_samples, ignore_index=True)

    # --------------------------------------------------
    # 4. LIMITED GAN AUGMENTATION
    # --------------------------------------------------
    if use_gan and os.path.exists(GAN_DATA_PATH):
        print("\n[INFO] Applying limited GAN augmentation")

        df_gan = pd.read_csv(GAN_DATA_PATH)
        df_gan["binary_label"] = 1

        gan_added = []

        for cls_name, cls_df in attack_df_final.groupby(label_col):
            real_count = len(cls_df)
            max_gan = int(real_count * MAX_GAN_RATIO_PER_CLASS)

            gan_candidates = df_gan[df_gan[label_col] == cls_name]

            if len(gan_candidates) == 0 or max_gan == 0:
                continue

            take = min(len(gan_candidates), max_gan)
            gan_added.append(
                gan_candidates.sample(n=take, random_state=42)
            )

            attack_stats[cls_name]["gan_added"] = int(take)

            print(f"  {cls_name:<15} | GAN added: {take:<6}")

        if gan_added:
            attack_df_final = pd.concat(
                [attack_df_final] + gan_added,
                ignore_index=True
            )

    print(f"\n[INFO] Final attack samples: {len(attack_df_final):,}")

    # --------------------------------------------------
    # 5. FINAL DATASET MERGE
    # --------------------------------------------------
    final_df = pd.concat([benign_df, attack_df_final], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42)

    X = final_df[SELECTED_FEATURES].values
    y = final_df["binary_label"].values

    # --------------------------------------------------
    # 6. SPLITTING (LEAK-FREE)
    # --------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    # --------------------------------------------------
    # 7. SCALING (TRAIN-ONLY FIT)
    # --------------------------------------------------
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)
    print(f"[INFO] Binary scaler saved to {SCALER_PATH}")

    # --------------------------------------------------
    # 8. SAVE DATASET SNAPSHOTS
    # --------------------------------------------------
    def _save_csv(X, y, path):
        df_out = pd.DataFrame(X, columns=SELECTED_FEATURES)
        df_out["binary_label"] = y
        df_out.to_csv(path, index=False)

    _save_csv(X_train, y_train, TRAIN_PATH)
    _save_csv(X_val,   y_val,   VAL_PATH)
    _save_csv(X_test,  y_test,  TEST_PATH)

    print(f"[SAVE] Train CSV -> {TRAIN_PATH}")
    print(f"[SAVE] Val   CSV -> {VAL_PATH}")
    print(f"[SAVE] Test  CSV -> {TEST_PATH}")

    # --------------------------------------------------
    # 9. SAVE DATASET METADATA
    # --------------------------------------------------
    stats = {
        "sampling_policy": {
            "max_benign": MAX_BENIGN_SAMPLES,
            "max_attack_total": MAX_ATTACK_TOTAL,
            "max_gan_ratio_per_class": MAX_GAN_RATIO_PER_CLASS,
        },
        "final_distribution": {
            "train": {
                "benign": int((y_train == 0).sum()),
                "attack": int((y_train == 1).sum()),
            },
            "val": {
                "benign": int((y_val == 0).sum()),
                "attack": int((y_val == 1).sum()),
            },
            "test": {
                "benign": int((y_test == 0).sum()),
                "attack": int((y_test == 1).sum()),
            },
        },
        "attack_class_breakdown": attack_stats,
    }

    with open(STATS_PATH, "w") as f:
        yaml.safe_dump(stats, f)

    print(f"[SAVE] Dataset stats -> {STATS_PATH}")

    print("\n" + "=" * 70)
    print("BINARY PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return X_train, X_val, X_test, y_train, y_val, y_test
