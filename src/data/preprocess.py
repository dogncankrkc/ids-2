"""
Preprocessing Utilities for CIC-IoT-2023 (ATTACK ONLY EDITION)
VERSION: ROBUST HYBRID (RobustScaler + Selective SMOTE)
Goal: Handle outliers and balance attack classes (e.g. Web vs DDoS).
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from imblearn.over_sampling import SMOTE

# --------------------------------------------------
# FEATURES
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

SAVE_DIR = "data/processed"
ENCODER_PATH = f"{SAVE_DIR}/label_encoder.pkl"
SCALER_PATH  = f"{SAVE_DIR}/feature_scaler.pkl"

def _save_split(X, y, name):
    df = pd.DataFrame(X, columns=SELECTED_FEATURES)
    df["label"] = y
    path = f"{SAVE_DIR}/{name}_preprocessed.csv"
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}")

def preprocess_multiclass(df: pd.DataFrame):

    print("[INFO] Multiclass preprocessing (ROBUST SCALER + SELECTIVE SMOTE)")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1️⃣ FEATURES & LABELS
    X = df[SELECTED_FEATURES].values
    
    # Label sütunu kontrolü
    label_col = "multiclass_label" if "multiclass_label" in df.columns else "label"
    y = df[label_col].values

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    joblib.dump(encoder, ENCODER_PATH)

    class_map = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print("[INFO] Class mapping:", class_map)
    
    # KONTROL: Eğer yanlışlıkla Benign geldiyse uyaralım (ama işlem yapmayalım)
    if 'Benign' in class_map:
        print("[WARN] 'Benign' class detected in Attack-Only dataset! It will be treated as just another class.")

    # 2️⃣ STRATIFIED SPLIT
    # Train (%70), Val (%15), Test (%15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_enc, test_size=0.30, stratify=y_enc, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # (3. ADIM - BENIGN CAPPING ÇIKARILDI)

    # --------------------------------------------------
    # 4️⃣ ROBUST SCALING (FIT TRAIN ONLY)
    # --------------------------------------------------
    # StandardScaler YERİNE RobustScaler.
    # Bu, devasa DDoS paket boyutlarının (outlier) Web/Recon trafiğini ezmesini engeller.
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)

    # --------------------------------------------------
    # 5️⃣ SELECTIVE SMOTE
    # --------------------------------------------------
    # Saldırı sınıfları arasında dengesizlik varsa (örn. çok az Spoofing, çok fazla DoS)
    # az olanları çoğaltır.
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    max_count = max(class_counts.values())
    target_count = int(max_count * 0.5) # En büyüğün yarısı kadar olsun en az.

    smote_targets = {}
    for cls, cnt in class_counts.items():
        if cnt < target_count:
            smote_targets[cls] = target_count

    if smote_targets:
        print(f"[INFO] Applying SMOTE to minorities (Target: {target_count}): {list(smote_targets.keys())}")
        smote = SMOTE(
            sampling_strategy=smote_targets,
            random_state=42,
            k_neighbors=5,
            n_jobs=-1
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # --------------------------------------------------
    # 6️⃣ DISTRIBUTION CHECK & SAVE
    # --------------------------------------------------
    print("\n[FINAL TRAIN DISTRIBUTION]")
    print(pd.Series(y_train).value_counts().sort_index())
    
    _save_split(X_train, y_train, "train")
    _save_split(X_val,   y_val,   "val")
    _save_split(X_test,  y_test,  "test")

    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_single_sample(df_row: pd.DataFrame) -> torch.Tensor:
    if isinstance(df_row, pd.Series):
        df_row = df_row.to_frame().T
    for col in SELECTED_FEATURES:
        if col not in df_row.columns:
            df_row[col] = 0.0
    df_row.replace([np.inf, -np.inf], 0, inplace=True)
    df_row.fillna(0, inplace=True)

    scaler = joblib.load(SCALER_PATH)
    x = scaler.transform(df_row[SELECTED_FEATURES].values)

    return torch.tensor(x, dtype=torch.float32).unsqueeze(1)