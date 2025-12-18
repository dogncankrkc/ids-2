"""
Preprocessing Utilities for CIC-IoT-2023 (BALANCED EDITION)
VERSION: CLEAN ROBUST (RobustScaler Only)
Goal: Prepare balanced data for training (Split + Scale). No SMOTE needed.
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

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

    print("[INFO] Multiclass preprocessing (ROBUST SCALER ONLY)")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1️⃣ FEATURES & LABELS
    X = df[SELECTED_FEATURES].values
    
    # Label sütunu kontrolü
    label_col = "multiclass_label" if "multiclass_label" in df.columns else "label"
    y = df[label_col].values

    # Label Encoding
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    joblib.dump(encoder, ENCODER_PATH)

    class_map = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print("[INFO] Class mapping:", class_map)

    # 2️⃣ STRATIFIED SPLIT
    # Train (%70), Val (%15), Test (%15)
    # stratify=y_enc -> Sınıf oranlarını koruyarak böler
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_enc, test_size=0.30, stratify=y_enc, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # 3️⃣ ROBUST SCALING
    # RobustScaler, DDoS gibi ataklardaki uç değerleri (outliers) yönetmek için harikadır.
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)

    # 4️⃣ DISTRIBUTION CHECK & SAVE
    print("\n[FINAL TRAIN DISTRIBUTION]")
    print(pd.Series(y_train).value_counts().sort_index())
    
    _save_split(X_train, y_train, "train")
    _save_split(X_val,   y_val,   "val")
    _save_split(X_test,  y_test,  "test")

    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_single_sample(df_row: pd.DataFrame) -> torch.Tensor:
    # Inference (Tahmin) sırasında tekil veriyi işlemek için kullanılır
    if isinstance(df_row, pd.Series):
        df_row = df_row.to_frame().T
    
    # Eksik sütun tamamlama
    for col in SELECTED_FEATURES:
        if col not in df_row.columns:
            df_row[col] = 0.0
            
    df_row.replace([np.inf, -np.inf], 0, inplace=True)
    df_row.fillna(0, inplace=True)

    # Kaydedilen scaler'ı yükle ve dönüştür
    scaler = joblib.load(SCALER_PATH)
    x = scaler.transform(df_row[SELECTED_FEATURES].values)

    return torch.tensor(x, dtype=torch.float32).unsqueeze(1)