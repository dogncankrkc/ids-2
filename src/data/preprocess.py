"""
Preprocessing Utilities for CIC-IoT-2023 Dataset.
FINAL FIXED VERSION:
- Uses StandardScaler (Critical for network data with outliers).
- REMOVED Padding (Dynamic input size).
- Handles Infinity/NaN cleanup.
- INCLUDES 'preprocess_single_sample' to fix ImportError.
"""

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
import torch
from imblearn.over_sampling import SMOTE

# ------------------------
# CONFIGURATION
# ------------------------
SELECTED_FEATURES = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate', 
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 
    'ece_flag_number', 'cwr_flag_number', 
    'ack_count', 'syn_count', 'fin_count', 'rst_count', 
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC', 
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Variance'
]

SAMPLES_PER_CLASS = 50000 
BENIGN_TARGET = 1_000_000
ATTACK_TARGET = 100_000

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def map_attack_label(label: str) -> str:
    label = str(label).strip().upper()
    if label == 'NAN' or label == '': return 'Other'
    if 'BENIGN' in label: return 'Benign'
    elif 'DDOS' in label: return 'DDoS'
    elif 'DOS' in label: return 'DoS'
    elif 'RECON' in label or 'VULNERABILITYSCAN' in label or 'PING' in label or 'PORTSCAN' in label or 'OSSCAN' in label or 'HOSTDISCOVERY' in label: return 'Recon'
    elif 'XSS' in label or 'SQL' in label or 'UPLOAD' in label or 'BROWSER' in label or 'COMMAND' in label or 'BACKDOOR' in label or 'MALWARE' in label: return 'Web'
    elif 'BRUTEFORCE' in label or 'DICTIONARY' in label: return 'BruteForce'
    elif 'SPOOFING' in label or 'MITM' in label: return 'Spoofing'
    elif 'MIRAI' in label: return 'Mirai'
    else: return 'Other'

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Removes Infinity and NaN values."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(subset=SELECTED_FEATURES, inplace=True)
    after = len(df)
    if before != after:
        print(f"[WARN] Dropped {before - after} rows containing Infinity or NaN values.")
    return df

def balance_dataset_smote_capped(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: list,
    attack_target: int = 100_000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Balancing strategy:
    - Benign: untouched
    - Attack classes:
        > 100k  -> random downsample to 100k
        < 100k  -> SMOTE up to 100k
    """

    print("[INFO] Balancing dataset with capped SMOTE (attack ≤ 100k)")

    # 1️⃣ Drop 'Other'
    df = df[df[label_col] != "Other"].reset_index(drop=True)

    # 2️⃣ Split Benign / Attack
    df_benign = df[df[label_col] == "Benign"]
    df_attack = df[df[label_col] != "Benign"]

    print("[INFO] Original distribution:")
    print(df[label_col].value_counts())

    # 3️⃣ Containers
    balanced_parts = []

    # 4️⃣ Process each attack class independently
    for label in df_attack[label_col].unique():
        df_cls = df_attack[df_attack[label_col] == label]
        n = len(df_cls)

        print(f"[INFO] {label}: {n} samples")

        # Case A — Too many → RANDOM DOWNSAMPLE
        if n > attack_target:
            df_cls_bal = df_cls.sample(
                n=attack_target,
                random_state=random_state
            )

        # Case B — Too few → SMOTE
        elif n < attack_target:
            X = df_cls[feature_cols].values
            y = df_cls[label_col].values

            # Scale before SMOTE (MANDATORY)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            smote = SMOTE(
                sampling_strategy={label: attack_target},
                k_neighbors=min(5, n - 1),
                random_state=random_state,
                n_jobs=-1
            )

            X_res, y_res = smote.fit_resample(X_scaled, y)

            X_res = scaler.inverse_transform(X_res)

            df_cls_bal = pd.DataFrame(X_res, columns=feature_cols)
            df_cls_bal[label_col] = y_res

        # Case C — Exactly 100k
        else:
            df_cls_bal = df_cls

        balanced_parts.append(df_cls_bal)

    # 5️⃣ Merge attacks + benign
    df_attack_balanced = pd.concat(balanced_parts, axis=0)
    df_final = pd.concat(
        [df_benign, df_attack_balanced],
        axis=0
    ).sample(frac=1, random_state=random_state).reset_index(drop=True)

    print("[INFO] Final distribution:")
    print(df_final[label_col].value_counts())

    return df_final


def balance_dataset(df: pd.DataFrame, label_col: str, n_samples: int = SAMPLES_PER_CLASS) -> pd.DataFrame:
    print(f"[INFO] Balancing dataset to ~{n_samples} samples per class...")
    df_balanced = pd.DataFrame()
    unique_classes = df[label_col].unique()
    
    for label in unique_classes:
        if label == 'Other': continue
            
        df_class = df[df[label_col] == label]
        count = len(df_class)
        if count == 0: continue
            
        if count > n_samples:
            df_resampled = resample(df_class, replace=False, n_samples=n_samples, random_state=42)
        else:
            df_resampled = resample(df_class, replace=True, n_samples=n_samples, random_state=42)
        
        df_balanced = pd.concat([df_balanced, df_resampled])
        
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_balanced

def split_train_val_test(X, y, test_size=0.15, val_size=0.15):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# ------------------------
# MULTICLASS PREPROCESSING
# ------------------------
def preprocess_multiclass(
    df: pd.DataFrame,
    scaler_path: str = "models/scaler_multi.pkl",
    encoder_path: str = "models/label_encoder.pkl",
) -> Tuple:

    print("[INFO] Preprocessing multiclass dataset (NO RESAMPLING)")

    if 'binary_label' in df.columns:
        df = df.drop(columns=['binary_label'])

    # 1️⃣ Label column
    if 'Mapped_Label' in df.columns:
        df['target_label'] = df['Mapped_Label']
    elif 'multiclass_label' in df.columns:
        df['target_label'] = df['multiclass_label'].apply(map_attack_label)
    else:
        raise ValueError("No label column found")

    # 2️⃣ Defensive cleaning (CSV zaten temiz ama safety net)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[SELECTED_FEATURES] = df[SELECTED_FEATURES].fillna(
        df[SELECTED_FEATURES].mean()
    )

    # 3️⃣ Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['target_label'])

    _ensure_dir(encoder_path)
    joblib.dump(encoder, encoder_path)

    # 4️⃣ Feature matrix
    X = df[SELECTED_FEATURES].values

    # 5️⃣ Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _ensure_dir(scaler_path)
    joblib.dump(scaler, scaler_path)

    print("[INFO] Final dataset shape:", X_scaled.shape)
    print("[INFO] Class distribution:")
    print(pd.Series(y).value_counts())

    return split_train_val_test(X_scaled, y)


# ------------------------
# BINARY PREPROCESSING
# ------------------------
def preprocess_binary(
    df: pd.DataFrame,
    scaler_path: str = "models/scaler_binary.pkl",
) -> Tuple:
    print("[INFO] Starting Binary Preprocessing (StandardScaler)...")
    
    if 'binary_label' in df.columns:
        df['label_bin'] = df['binary_label']
    else:
        target_col = 'Mapped_Label' if 'Mapped_Label' in df.columns else 'multiclass_label'
        df['label_bin'] = df[target_col].apply(lambda x: 0 if 'Benign' in str(x) else 1)

    df = clean_dataset(df)
    df = balance_dataset(df, 'label_bin', n_samples=SAMPLES_PER_CLASS)
    y = df['label_bin'].values
    
    X = df[SELECTED_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _ensure_dir(scaler_path)
    joblib.dump(scaler, scaler_path)

    return split_train_val_test(X_scaled, y)

# ------------------------
# SINGLE SAMPLE PREPROCESSING
# ------------------------
def preprocess_single_sample(df_row: pd.DataFrame) -> torch.Tensor:
    """
    Prepares a single row for inference (Prediction).
    Updated to use StandardScaler and handle no-padding logic.
    """
    if isinstance(df_row, pd.Series):
        df_row = df_row.to_frame().T
        
    # Ensure all features exist
    for col in SELECTED_FEATURES:
        if col not in df_row.columns:
            df_row[col] = 0.0     
            
    # Clean possible Inf in single sample
    df_row.replace([np.inf, -np.inf], 0, inplace=True)
    df_row.fillna(0, inplace=True)

    row_values = df_row[SELECTED_FEATURES].values
    
    # Load correct scaler
    try:
        scaler = joblib.load("models/scaler_multi.pkl")
    except FileNotFoundError:
        try:
            scaler = joblib.load("models/scaler_binary.pkl")
        except FileNotFoundError:
            raise FileNotFoundError("Scaler not found! You must train the model first.")
        
    x_scaled = scaler.transform(row_values)
    
    # Convert to Tensor (N, 1, Features) for 1D CNN
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    x_tensor = x_tensor.unsqueeze(1) # Add channel dim
    
    return x_tensor