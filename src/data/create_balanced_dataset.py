"""
FINAL DATASET CREATOR – CIC-IoT-2023 (SEPARATE ATTACKS ONLY)

Purpose:
1. Read huge raw CSV.
2. Map raw labels to 7 DISTINCT ATTACK CLASSES (DoS & DDoS separated).
3. DROP BENIGN traffic completely.
4. Cap large attacks at 100k, keep small attacks as is.
"""

import os
import numpy as np
import pandas as pd
import argparse

# ============================
# CONFIGURATION
# ============================

DEFAULT_INPUT = "data/raw/CIC2023_FULL_MERGED.csv"
DEFAULT_OUTPUT = "data/processed/CIC2023_SEPARATE_ATTACK_ONLY.csv"

CHUNK_SIZE = 1_000_000

# Sadece saldırı limiti var. Benign yok.
ATTACK_CAP = 100_000 

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
# LABEL MAPPING (ATTACKS ONLY)
# ============================
def map_to_separate_attacks(label: str) -> str:
    label = str(label).strip().upper()

    if label == 'NAN' or label == '':
        return 'Other'
    
    if 'BENIGN' in label:
        return 'Other' 
    
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
# MAIN GENERATOR
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    collected = {
        'DDoS': 0,
        'DoS': 0,
        'Recon': 0,
        'Web': 0, 
        'BruteForce': 0, 
        'Spoofing': 0, 
        'Mirai': 0
    }
    
    chunks_to_save = []

    print(f"[INFO] Creating SEPARATE ATTACK-ONLY dataset...")
    print(f"[INFO] Attack Cap: {ATTACK_CAP}")

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
        
        chunk = chunk[chunk["multiclass_label"] != "Other"]
        
        for feat in SELECTED_FEATURES:
            if feat not in chunk.columns:
                chunk[feat] = 0
                
        chunk = chunk[SELECTED_FEATURES + ["multiclass_label"]]
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(inplace=True)

        for cls, group in chunk.groupby("multiclass_label"):
            cap = ATTACK_CAP
            
            current_count = collected.get(cls, 0)
            if current_count >= cap:
                continue

            needed = cap - current_count
            take = group.head(needed)
            
            chunks_to_save.append(take)
            collected[cls] += len(take)

        if all(val >= ATTACK_CAP for key, val in collected.items()):
            print(f"\n[INFO] All caps reached at chunk {i+1}. Stopping early.")
            break

    print("\n[INFO] Concatenating and saving...")
    if not chunks_to_save:
        print("[ERROR] No data collected!")
        return

    df_final = pd.concat(chunks_to_save, ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n[FINAL DISTRIBUTION - ATTACK ONLY (SEPARATE)]")
    print(df_final["multiclass_label"].value_counts())

    df_final.to_csv(args.output, index=False)
    print(f"\n[SUCCESS] Dataset saved to: {args.output}")

if __name__ == "__main__":
    main()