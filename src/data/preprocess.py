"""
Preprocessing Utilities for CIC-IoT-2023 (ACADEMIC STANDARD)
VERSION: SPLIT FIRST -> AUGMENT TRAIN
Goal: Keep Test/Val pure (Real data only). Augment ONLY Train set with GAN.
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

# --------------------------------------------------
# CONFIG & PATHS
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

# GAN Dosyasının Yeri (Bunu kodun bulabilmesi lazım)
GAN_DATA_PATH = "data/processed/GAN_SYNTHETIC_ONLY.csv"

def _save_split(X, y, name):
    df = pd.DataFrame(X, columns=SELECTED_FEATURES)
    df["label"] = y
    path = f"{SAVE_DIR}/{name}_preprocessed.csv"
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}")

def preprocess_multiclass(df_real: pd.DataFrame):
    """
    df_real: Sadece gerçek veriyi (CIC2023_SEPARATE_ATTACK_ONLY.csv) alır.
    """
    print("[INFO] Multiclass preprocessing (SPLIT FIRST -> AUGMENT TRAIN)")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1️⃣ FEATURES & LABELS (GERÇEK VERİ)
    X = df_real[SELECTED_FEATURES].values
    
    label_col = "multiclass_label" if "multiclass_label" in df_real.columns else "label"
    y = df_real[label_col].values

    # Label Encoding
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    joblib.dump(encoder, ENCODER_PATH)

    class_map = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print("[INFO] Class mapping:", class_map)

    # 2️⃣ STRATIFIED SPLIT (ÖNCE BÖLÜYORUZ - SAF VERİ)
    print("[INFO] Splitting Real Data (Train: %70, Val: %15, Test: %15)...")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_enc, test_size=0.30, stratify=y_enc, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # 3️⃣ GAN AUGMENTATION (SADECE TRAIN İÇİN)
    if os.path.exists(GAN_DATA_PATH):
        print(f"[INFO] Loading GAN data for training augmentation: {GAN_DATA_PATH}")
        df_gan = pd.read_csv(GAN_DATA_PATH)
        
        X_gan = df_gan[SELECTED_FEATURES].values
        y_gan_raw = df_gan[label_col].values
        y_gan = encoder.transform(y_gan_raw)

        # HEDEF: ILIMLI AUGMENTATION (40.000)
        TARGET_AUGMENT_LIMIT = 40000 
        print(f"[INFO] Target Train Size per Class: Capped at {TARGET_AUGMENT_LIMIT}")
        
        X_gan_to_add = []
        y_gan_to_add = []

        unique, counts = np.unique(y_train, return_counts=True)

        for cls_idx in unique:
            cls_name = encoder.inverse_transform([cls_idx])[0]
            current_count = counts[np.where(unique == cls_idx)[0][0]]
            
            # Sadece 40k'nın altındakileri tamamla (Web, BruteForce)
            if current_count < TARGET_AUGMENT_LIMIT:
                max_gan_add = int(current_count * 1.0)   # 1.0 -> en fazla 100% GAN
                needed = min(TARGET_AUGMENT_LIMIT - current_count, max_gan_add)
                
                indices = np.where(y_gan == cls_idx)[0]
                
                if len(indices) > 0:
                    selected_indices = np.random.choice(indices, needed, replace=True)
                    X_gan_to_add.append(X_gan[selected_indices])
                    y_gan_to_add.append(y_gan[selected_indices])
                    print(f"   -> Augmenting {cls_name}: +{needed} GAN samples added.")
        
        if X_gan_to_add:
            X_gan_add = np.vstack(X_gan_to_add)
            y_gan_add = np.concatenate(y_gan_to_add)
            
            X_train = np.vstack([X_train, X_gan_add])
            y_train = np.concatenate([y_train, y_gan_add])
            
            # Shuffle
            perm = np.random.permutation(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]
            print("[INFO] Moderate GAN augmentation applied.")
    else:
        print("[WARN] GAN data file not found! Training will proceed with imbalanced real data.")

    # 4️⃣ ROBUST SCALING
    # Fit'i SADECE Augmented Train üzerinde yapıyoruz
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    # Val ve Test'i dönüştürüyoruz (Data Leakage yok)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)

    # 5️⃣ SAVE & RETURN
    print("\n[FINAL TRAIN DISTRIBUTION (Augmented)]")
    print(pd.Series(y_train).value_counts().sort_index())
    
    print("\n[FINAL TEST DISTRIBUTION (Pure Real)]")
    print(pd.Series(y_test).value_counts().sort_index())
    
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