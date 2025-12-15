"""
Dataset Generator Script for CIC-IoT-2023 (CAPPED + SMOTE).

Pipeline:
1. Read massive raw CSV in chunks (RAM-safe)
2. Map raw labels → 8 attack categories
3. Drop raw label columns
4. Cap dataset:
   - Benign: EXACT 1,000,000
   - Each attack class: max 100,000
5. Clean features:
   - Inf  -> removed
   - NaN  -> filled with feature mean (GLOBAL)
6. Apply SMOTE ONLY on attack classes that have < 100k samples
7. Create:
   - multiclass_label (8-class)
   - binary_label (Benign=0, Attack=1)
8. Save final dataset
"""

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# ============================
# CONFIGURATION
# ============================

INPUT_PATH = "data/raw/CIC2023_FULL_MERGED.csv"
OUTPUT_PATH = "data/processed/CIC2023_CAPPED_SMOTE.csv"

BENIGN_CAP = 1_000_000
ATTACK_CAP = 100_000
CHUNK_SIZE = 1_000_000

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
        return 'DROP'

    if 'BENIGN' in label:
        return 'Benign'
    elif 'DDOS' in label:
        return 'DDoS'
    elif 'DOS' in label:
        return 'DoS'
    elif any(x in label for x in ['RECON', 'VULNERABILITYSCAN', 'PING',
                                  'PORTSCAN', 'OSSCAN', 'HOSTDISCOVERY']):
        return 'Recon'
    elif any(x in label for x in ['XSS', 'SQL', 'UPLOAD', 'BROWSER',
                                  'COMMAND', 'BACKDOOR', 'MALWARE']):
        return 'Web'
    elif any(x in label for x in ['BRUTEFORCE', 'DICTIONARY']):
        return 'BruteForce'
    elif any(x in label for x in ['SPOOFING', 'MITM']):
        return 'Spoofing'
    elif 'MIRAI' in label:
        return 'Mirai'
    else:
        return 'Other'

# ============================
# FEATURE CLEANING
# ============================

def clean_features_mean_impute(df, feature_cols):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    means = df[feature_cols].mean()
    df[feature_cols] = df[feature_cols].fillna(means)

    return df

# ============================
# MAIN GENERATOR
# ============================

def create_dataset_with_smote():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    collected = {
        'Benign': 0, 'DDoS': 0, 'DoS': 0, 'Recon': 0,
        'Web': 0, 'BruteForce': 0, 'Spoofing': 0, 'Mirai': 0
    }

    sampled_chunks = []

    print(f"[INFO] Reading raw dataset: {INPUT_PATH}")

    for i, chunk in enumerate(pd.read_csv(INPUT_PATH, chunksize=CHUNK_SIZE)):
        print(f" → Chunk {i+1}")

        chunk.columns = chunk.columns.str.strip()

        label_col = (
            'multiclass_label' if 'multiclass_label' in chunk.columns
            else 'label' if 'label' in chunk.columns
            else None
        )
        if label_col is None:
            continue

        chunk = chunk.dropna(subset=[label_col])

        # Map → multiclass_label
        chunk['multiclass_label'] = chunk[label_col].apply(map_to_multiclass)
        chunk = chunk[~chunk['multiclass_label'].isin(['DROP', 'Other'])]

        # Drop raw label column
        if label_col != 'multiclass_label':
            chunk.drop(columns=[label_col], inplace=True)

        for cls, group in chunk.groupby('multiclass_label'):
            cap = BENIGN_CAP if cls == 'Benign' else ATTACK_CAP
            current = collected[cls]

            if current >= cap:
                continue

            needed = cap - current
            taken = group.sample(n=min(len(group), needed), random_state=42)

            sampled_chunks.append(taken)
            collected[cls] += len(taken)

        if (
            collected['Benign'] >= BENIGN_CAP and
            all(collected[c] >= ATTACK_CAP for c in collected if c != 'Benign')
        ):
            print("[INFO] All caps reached, stopping early.")
            break

    print("[INFO] Concatenating capped dataset...")
    df = pd.concat(sampled_chunks, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ============================
    # FEATURE CLEANING
    # ============================
    df = clean_features_mean_impute(df, SELECTED_FEATURES)

    # Ensure Benign EXACT 1M
    benign_df = df[df['multiclass_label'] == 'Benign']
    if len(benign_df) < BENIGN_CAP:
        benign_df = benign_df.sample(BENIGN_CAP, replace=True, random_state=42)

    attack_df = df[df['multiclass_label'] != 'Benign']
    df = pd.concat([benign_df, attack_df], axis=0)

    print("\n[DISTRIBUTION BEFORE SMOTE]")
    print(df['multiclass_label'].value_counts())

    # ============================
    # SMOTE (ATTACK ONLY)
    # ============================
    df_benign = df[df['multiclass_label'] == 'Benign']
    df_attack = df[df['multiclass_label'] != 'Benign']

    encoder = LabelEncoder()
    y_attack = encoder.fit_transform(df_attack['multiclass_label'])
    X_attack = df_attack[SELECTED_FEATURES].values

    scaler = StandardScaler()
    X_attack_scaled = scaler.fit_transform(X_attack)

    class_counts = np.bincount(y_attack)
    smote_strategy = {
        cls: ATTACK_CAP
        for cls, cnt in enumerate(class_counts)
        if cnt < ATTACK_CAP
    }

    smote = SMOTE(
        sampling_strategy=smote_strategy,
        random_state=42,
        n_jobs=-1
    )

    X_res, y_res = smote.fit_resample(X_attack_scaled, y_attack)
    X_res = scaler.inverse_transform(X_res)

    df_attack_smote = pd.DataFrame(X_res, columns=SELECTED_FEATURES)
    df_attack_smote['multiclass_label'] = encoder.inverse_transform(y_res)

    # Merge + shuffle
    df_final = pd.concat(
        [df_benign, df_attack_smote],
        axis=0
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    # ============================
    # BINARY LABEL
    # ============================
    df_final['binary_label'] = (
        df_final['multiclass_label'].apply(lambda x: 0 if x == 'Benign' else 1)
    )

    print("\n[DISTRIBUTION AFTER SMOTE]")
    print(df_final['multiclass_label'].value_counts())
    print("\n[BINARY LABEL DISTRIBUTION]")
    print(df_final['binary_label'].value_counts())

    print(f"\n[SAVING] {OUTPUT_PATH}")
    df_final.to_csv(OUTPUT_PATH, index=False)
    print("[DONE] Dataset ready.")

# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":
    create_dataset_with_smote()
