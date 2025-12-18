"""
FINAL DATASET CREATOR – CIC-IoT-2023 (CAPPED, CLEAN)

Purpose:
1. Read huge raw CSV in chunks
2. Map raw labels to 8 IDS classes
3. Cap dataset size (NO resampling, NO scaling here)
4. Save clean capped dataset for preprocess.py
"""

import os
import numpy as np
import pandas as pd

# ============================
# CONFIGURATION
# ============================

INPUT_PATH = "data/raw/CIC2023_FULL_MERGED.csv"
OUTPUT_PATH = "data/processed/CIC2023_CAPPED.csv"

CHUNK_SIZE = 1_000_000

# Keep Benign realistic but manageable
BENIGN_CAP = 250_000
ATTACK_CAP = 100_000

# ------------------------------------------------------------
# SELECTED FEATURES
# ------------------------------------------------------------
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

# ============================
# LABEL MAPPING
# ============================
def map_to_multiclass(label: str) -> str:
    label = str(label).strip().upper()

    if label == 'NAN' or label == '':
        return 'Other'
    if 'BENIGN' in label:
        return 'Benign'
    if 'DDOS' in label:
        return 'DDoS'
    if 'DOS' in label:
        return 'DoS'
    if any(x in label for x in ['RECON', 'VULNERABILITY', 'PING', 'PORTSCAN', 'OSSCAN', 'HOSTDISCOVERY']):
        return 'Recon'
    if any(x in label for x in ['XSS', 'SQL', 'UPLOAD', 'BROWSER', 'COMMAND', 'BACKDOOR', 'MALWARE']):
        return 'Web'
    if 'BRUTEFORCE' in label or 'DICTIONARY' in label:
        return 'BruteForce'
    if 'SPOOFING' in label or 'MITM' in label:
        return 'Spoofing'
    if 'MIRAI' in label:
        return 'Mirai'
    return 'Other'

# ============================
# MAIN
# ============================
def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    collected = {
        'Benign': 0, 'DDoS': 0, 'DoS': 0, 'Recon': 0,
        'Web': 0, 'BruteForce': 0, 'Spoofing': 0, 'Mirai': 0
    }

    chunks_to_save = []

    print(f"[INFO] Creating capped dataset...")

    for i, chunk in enumerate(pd.read_csv(INPUT_PATH, chunksize=CHUNK_SIZE)):
        print(f" -> Processing chunk {i+1}", end="\r")

        label_col = "multiclass_label" if "multiclass_label" in chunk.columns else "label"
        chunk["multiclass_label"] = chunk[label_col].apply(map_to_multiclass)

        chunk = chunk[chunk["multiclass_label"] != "Other"]
        chunk = chunk[SELECTED_FEATURES + ["multiclass_label"]]

        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(inplace=True)

        for cls, group in chunk.groupby("multiclass_label"):
            cap = BENIGN_CAP if cls == "Benign" else ATTACK_CAP
            remaining = cap - collected[cls]
            if remaining <= 0:
                continue

            take = group.head(remaining)
            chunks_to_save.append(take)
            collected[cls] += len(take)

        if all(collected[k] >= (BENIGN_CAP if k == "Benign" else ATTACK_CAP) for k in collected):
            print("\n[INFO] Caps reached. Stopping early.")
            break

    df_final = pd.concat(chunks_to_save, ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n[FINAL CLASS DISTRIBUTION]")
    print(df_final["multiclass_label"].value_counts())

    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SAVED] {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
