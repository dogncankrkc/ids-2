"""
ATTACK-ONLY DATASET CREATOR – CIC-IoT-2023 (HIERARCHICAL STAGE 2)

Purpose:
1. Filter out BENIGN traffic completely.
2. Keep DoS and DDoS separate (7 Attack Classes).
3. Create a clean dataset for the 'Specialist' Attack Classifier.
"""

import os
import numpy as np
import pandas as pd

# ============================
# CONFIGURATION
# ============================

INPUT_PATH = "data/raw/CIC2023_FULL_MERGED.csv"
OUTPUT_PATH = "data/processed/CIC2023_ATTACK_ONLY.csv"

CHUNK_SIZE = 1_000_000

# Saldırıları bol bol alalım, Benign olmadığı için yerimiz var.
ATTACK_CAP = 75_000 

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
# LABEL MAPPING (NO MERGING, NO BENIGN)
# ============================
def map_to_attack_class(label: str) -> str:
    label = str(label).strip().upper()

    if label == 'NAN' or label == '': return 'Ignore'
    
    # Benign'i GÖRMEZDEN GEL (Ignore)
    if 'BENIGN' in label:
        return 'Ignore'
    
    # DoS ve DDoS AYRI
    if 'DDOS' in label: return 'DDoS'
    if 'DOS' in label: return 'DoS'
    
    # Diğerleri
    if any(x in label for x in ['RECON', 'VULNERABILITY', 'PING', 'PORTSCAN', 'OSSCAN', 'HOSTDISCOVERY']): return 'Recon'
    if any(x in label for x in ['XSS', 'SQL', 'UPLOAD', 'BROWSER', 'COMMAND', 'BACKDOOR', 'MALWARE']): return 'Web'
    if 'BRUTEFORCE' in label or 'DICTIONARY' in label: return 'BruteForce'
    if 'SPOOFING' in label or 'MITM' in label: return 'Spoofing'
    if 'MIRAI' in label: return 'Mirai'
        
    return 'Ignore'

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    collected = {
        'DDoS': 0, 'DoS': 0, 'Recon': 0, 'Web': 0, 
        'BruteForce': 0, 'Spoofing': 0, 'Mirai': 0
    }
    
    chunks_to_save = []

    print(f"[INFO] Creating ATTACK-ONLY dataset (No Benign)...")
    print(f"[INFO] Target Cap per Attack: {ATTACK_CAP}")

    for i, chunk in enumerate(pd.read_csv(INPUT_PATH, chunksize=CHUNK_SIZE)):
        print(f" -> Processing chunk {i+1}", end="\r")

        if "multiclass_label" in chunk.columns: label_col = "multiclass_label"
        elif "label" in chunk.columns: label_col = "label"
        else: continue

        chunk["multiclass_label"] = chunk[label_col].apply(map_to_attack_class)

        # 'Ignore' olanları (Benign ve Tanımsız) at
        chunk = chunk[chunk["multiclass_label"] != "Ignore"]
        chunk = chunk[SELECTED_FEATURES + ["multiclass_label"]]
        
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(inplace=True)

        for cls, group in chunk.groupby("multiclass_label"):
            if collected[cls] >= ATTACK_CAP: continue
            
            needed = ATTACK_CAP - collected[cls]
            take = group.head(needed)
            chunks_to_save.append(take)
            collected[cls] += len(take)

        if all(val >= ATTACK_CAP for val in collected.values()):
            print(f"\n[INFO] Caps reached at chunk {i+1}.")
            break

    if not chunks_to_save:
        print("\n[ERROR] No attack data found!")
        return

    df_final = pd.concat(chunks_to_save, ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n[FINAL ATTACK DISTRIBUTION]")
    print(df_final["multiclass_label"].value_counts())

    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SUCCESS] Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()