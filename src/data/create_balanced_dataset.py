"""
FINAL DATASET CREATOR – CIC-IoT-2023 (BENIGN + SEPARATE ATTACKS)

Purpose:
1. Read huge raw CSV (chunk-based).
2. Map raw labels to 7 DISTINCT ATTACK CLASSES.
3. KEEP BENIGN traffic (for Binary IDS).
4. Create:
   - binary_label   : 0 = Benign, 1 = Attack
   - multiclass_label: Benign or specific attack type
5. Cap large attack classes, keep benign realistic.
"""

import os
import numpy as np
import pandas as pd
import argparse

# ============================
# CONFIGURATION
# ============================

DEFAULT_INPUT = "data/raw/CIC2023_FULL_MERGED.csv"
DEFAULT_OUTPUT = "data/processed/CIC2023_BINARY_AND_ATTACK_MAPPED.csv"

CHUNK_SIZE = 1_000_000

ATTACK_CAP = 100_000      # per attack class
BENIGN_CAP = None         # None = keep all benign (binary preprocess will downsample)

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
def map_to_separate_attacks(label: str) -> str:
    label = str(label).strip().upper()

    if label == 'NAN' or label == '':
        return 'Benign'

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

    return 'Benign'


# ============================
# MAIN GENERATOR
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    collected_attacks = {
        'DDoS': 0,
        'DoS': 0,
        'Recon': 0,
        'Web': 0,
        'BruteForce': 0,
        'Spoofing': 0,
        'Mirai': 0
    }

    benign_collected = 0
    chunks_to_save = []

    print("[INFO] Creating BENIGN + SEPARATE ATTACK dataset")
    print(f"[INFO] Attack cap per class: {ATTACK_CAP}")
    print("[INFO] Benign kept realistic (no cap here)")

    for i, chunk in enumerate(pd.read_csv(args.input, chunksize=CHUNK_SIZE)):
        print(f" -> Processing chunk {i+1}", end="\r")

        label_col = None
        for col in ["multiclass_label", "label", "Label"]:
            if col in chunk.columns:
                label_col = col
                break
        if label_col is None:
            continue

        chunk["multiclass_label"] = chunk[label_col].apply(map_to_separate_attacks)
        chunk["binary_label"] = (chunk["multiclass_label"] != "Benign").astype(int)

        for feat in SELECTED_FEATURES:
            if feat not in chunk.columns:
                chunk[feat] = 0

        chunk = chunk[SELECTED_FEATURES + ["binary_label", "multiclass_label"]]
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(inplace=True)

        # ---------------------------
        # BENIGN
        # ---------------------------
        benign_chunk = chunk[chunk["binary_label"] == 0]
        if BENIGN_CAP is None or benign_collected < BENIGN_CAP:
            chunks_to_save.append(benign_chunk)
            benign_collected += len(benign_chunk)

        # ---------------------------
        # ATTACKS
        # ---------------------------
        attack_chunk = chunk[chunk["binary_label"] == 1]
        for cls, group in attack_chunk.groupby("multiclass_label"):
            if cls == "Benign":
                continue

            current = collected_attacks.get(cls, 0)
            if current >= ATTACK_CAP:
                continue

            needed = ATTACK_CAP - current
            take = group.head(needed)

            chunks_to_save.append(take)
            collected_attacks[cls] += len(take)

        if all(v >= ATTACK_CAP for v in collected_attacks.values()):
            print(f"\n[INFO] All attack caps reached at chunk {i+1}.")
            break

    print("\n[INFO] Concatenating and saving final dataset...")

    if not chunks_to_save:
        print("[ERROR] No data collected.")
        return

    df_final = pd.concat(chunks_to_save, ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n[FINAL DISTRIBUTION]")
    print("Binary:")
    print(df_final["binary_label"].value_counts())
    print("\nMulticlass:")
    print(df_final["multiclass_label"].value_counts())

    df_final.to_csv(args.output, index=False)
    print(f"\n[SUCCESS] Dataset saved to: {args.output}")


if __name__ == "__main__":
    main()
