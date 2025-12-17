"""
FINAL DATASET CREATOR – CIC-IoT-2023 (CAPPED, NO LEAKAGE)

OUTPUT:
- data/processed/CIC2023_CAPPED.csv

IMPORTANT:
- NO SMOTE
- NO SCALER
- NO ENCODER
- NO TRAIN/VAL/TEST SPLIT

This file ONLY:
✓ maps raw labels → 8-class multiclass
✓ caps samples safely
✓ cleans NaN / Inf
✓ adds PAD feature
"""

import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = "data/raw/CIC2023_FULL_MERGED.csv"
OUTPUT_PATH = "data/processed/CIC2023_CAPPED.csv"

CHUNK_SIZE = 1_000_000
BENIGN_CAP = 1_000_000
ATTACK_CAP = 100_000
PAD_FEATURE_NAME = "PAD_0"

SELECTED_FEATURES = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH',
    'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv',
    'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size',
    'IAT', 'Number', 'Variance',
    PAD_FEATURE_NAME
]

TARGET_CLASSES = [
    "Benign", "DDoS", "DoS", "Recon",
    "Web", "BruteForce", "Spoofing", "Mirai"
]

# ============================================================
# LABEL MAPPING
# ============================================================

def map_to_multiclass(label: str) -> str:
    label = str(label).strip().upper()

    if label == 'NAN' or label == '':
        return 'Other'

    if 'BENIGN' in label:
        return 'Benign'

    elif 'DDOS' in label:
        return 'DDoS'

    elif 'DOS' in label:
        return 'DoS'

    elif (
        'RECON' in label or
        'VULNERABILITYSCAN' in label or
        'PING' in label or
        'PORTSCAN' in label or
        'OSSCAN' in label or
        'HOSTDISCOVERY' in label
    ):
        return 'Recon'

    elif (
        'XSS' in label or
        'SQL' in label or
        'UPLOAD' in label or
        'BROWSER' in label or
        'COMMAND' in label or
        'BACKDOOR' in label or
        'MALWARE' in label
    ):
        return 'Web'

    elif 'BRUTEFORCE' in label or 'DICTIONARY' in label:
        return 'BruteForce'

    elif 'SPOOFING' in label or 'MITM' in label:
        return 'Spoofing'

    elif 'MIRAI' in label:
        return 'Mirai'

    else:
        return 'Other'


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    collected = {cls: 0 for cls in TARGET_CLASSES}
    chunks = []

    print("[INFO] Starting capped dataset creation")

    for i, chunk in enumerate(pd.read_csv(INPUT_PATH, chunksize=CHUNK_SIZE)):
        print(f"[INFO] Processing chunk {i+1}")

        # -----------------------------------------
        # LABEL SOURCE (CRITICAL FIX)
        # -----------------------------------------
        if "multiclass_label" in chunk.columns:
            raw_labels = chunk["multiclass_label"]
        elif "label" in chunk.columns:
            raw_labels = chunk["label"]
        else:
            raise ValueError("No label column found in dataset")

        # 1️⃣ Label mapping
        chunk["multiclass_label"] = raw_labels.apply(map_to_multiclass)

        # 2️⃣ DROP Other (KRİTİK SATIR)
        chunk = chunk[chunk["multiclass_label"] != "Other"]

        # -----------------------------------------
        # FEATURE CLEANING
        # -----------------------------------------
        chunk[PAD_FEATURE_NAME] = 0.0
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(subset=SELECTED_FEATURES, inplace=True)

        # -----------------------------------------
        # CLASS CAPPING
        # -----------------------------------------
        for cls, group in chunk.groupby("multiclass_label"):
            cap = BENIGN_CAP if cls == "Benign" else ATTACK_CAP
            remaining = cap - collected[cls]

            if remaining <= 0:
                continue

            take = group.sample(
                n=min(len(group), remaining),
                random_state=42
            )

            collected[cls] += len(take)
            chunks.append(take)

        # -----------------------------------------
        # EARLY STOP
        # -----------------------------------------
        if all(
            collected[c] >= (BENIGN_CAP if c == "Benign" else ATTACK_CAP)
            for c in collected
        ):
            print("[INFO] All caps reached. Stopping early.")
            break

    # -----------------------------------------
    # FINAL DATASET
    # -----------------------------------------
    df = pd.concat(chunks, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df["binary_label"] = (df["multiclass_label"] != "Benign").astype(int)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\n✅ CAPPED DATASET SAVED:", OUTPUT_PATH)
    print("\n[FINAL DISTRIBUTION]")
    print(df["multiclass_label"].value_counts())
    print("\n[BINARY DISTRIBUTION]")
    print(df["binary_label"].value_counts())


if __name__ == "__main__":
    main()
