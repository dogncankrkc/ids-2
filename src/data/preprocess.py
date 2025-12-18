"""
Preprocessing Utilities for CIC-IoT-2023
VERSION: UNIVERSAL (Smart Check for Benign + RobustScaler)
Goal: Works for both 'Binary' and 'Attack-Only' datasets automatically.
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

def preprocess_multiclass(df: pd.DataFrame):

    print("[INFO] Multiclass preprocessing (ROBUST SCALER + SELECTIVE SMOTE)")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1️⃣ FEATURES & LABELS
    X = df[SELECTED_FEATURES].values
    y = df["multiclass_label"].values

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    joblib.dump(encoder, ENCODER_PATH)

    class_map = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print("[INFO] Class mapping:", class_map)

    # 2️⃣ STRATIFIED SPLIT
    # Train (%70), Val (%15), Test (%15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_enc, test_size=0.30, stratify=y_enc, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # --------------------------------------------------
    # 3️⃣ UNDERSAMPLE BENIGN (SMART CHECK) - GÜNCELLENDİ
    # --------------------------------------------------
    # Artık 'Benign' var mı diye kontrol ediyoruz.
    # Attack-Only setinde Benign olmayacağı için hata vermemeli.
    
    if 'Benign' in class_map:
        print("[INFO] Benign class detected. Checking for capping...")
        benign_id = class_map['Benign']
        indices_benign = np.where(y_train == benign_id)[0]
        indices_others = np.where(y_train != benign_id)[0]
        
        # Binary veya Mixed modda Benign çok fazlaysa kısıyoruz.
        TARGET_BENIGN = 100_000 
        
        if len(indices_benign) > TARGET_BENIGN:
            print(f"[INFO] Capping Training Benign samples to {TARGET_BENIGN}")
            indices_benign = np.random.choice(indices_benign, TARGET_BENIGN, replace=False)
            
            indices_keep = np.concatenate([indices_others, indices_benign])
            np.random.shuffle(indices_keep)
            
            X_train = X_train[indices_keep]
            y_train = y_train[indices_keep]
    else:
        print("[INFO] No Benign class found (Attack-Only Mode). Skipping undersampling.")

    # --------------------------------------------------
    # 4️⃣ ROBUST SCALING
    # --------------------------------------------------
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)

    # --------------------------------------------------
    # 5️⃣ SELECTIVE SMOTE
    # --------------------------------------------------
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    # Hedef: En kalabalık sınıfın %50'si kadar olsun.
    if len(class_counts) > 1: # Sadece tek sınıf varsa SMOTE patlar
        max_count = max(class_counts.values())
        target_count = int(max_count * 0.5) 

        smote_targets = {}
        for cls, cnt in class_counts.items():
            if cnt < target_count:
                smote_targets[cls] = target_count

        if smote_targets:
            print(f"[INFO] Applying SMOTE to minorities (Target: {target_count}): {list(smote_targets.keys())}")
            try:
                smote = SMOTE(
                    sampling_strategy=smote_targets,
                    random_state=42,
                    k_neighbors=5,
                    n_jobs=-1
                )
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except Exception as e:
                print(f"[WARNING] SMOTE failed (likely due to very small class size): {e}")
                print("[INFO] Continuing without SMOTE for this run.")

    # --------------------------------------------------
    # 6️⃣ SAVE & FINISH
    # --------------------------------------------------
    print("\n[FINAL TRAIN DISTRIBUTION]")
    print(pd.Series(y_train).value_counts().sort_index())
    
    _save_split(X_train, y_train, "train")
    _save_split(X_val,   y_val,   "val")
    _save_split(X_test,  y_test,  "test")

    return X_train, X_val, X_test, y_train, y_val, y_test


def _save_split(X, y, name):
    df = pd.DataFrame(X, columns=SELECTED_FEATURES)
    df["label"] = y
    path = f"{SAVE_DIR}/{name}_preprocessed.csv"
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}")

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