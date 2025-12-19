"""
Preprocessing Utilities for CIC-IoT-2023 (REPORTING EDITION)
VERSION: SPLIT FIRST -> FIT REAL -> LOG EVERYTHING -> AUGMENT
Goal: Detailed console logs for thesis/reports. Save intermediate steps.
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
GAN_DATA_PATH = "data/processed/GAN_SYNTHETIC_ONLY.csv"
_SCALER_CACHE = None

def _save_split(X, y, name):
    """Verilen veri setini CSV olarak kaydeder ve bilgi basar."""
    df = pd.DataFrame(X, columns=SELECTED_FEATURES)
    df["label"] = y
    path = f"{SAVE_DIR}/{name}_preprocessed.csv"
    df.to_csv(path, index=False)
    print(f"   [SAVE] {name:<15} -> {path} | Rows: {len(df):,}")

def preprocess_multiclass(df_real: pd.DataFrame):
    print("\n" + "="*60)
    print("🚀 PREPROCESSING STARTED (REPORTING MODE)")
    print("="*60)
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # -------------------------------------------------------
    # 1. INITIAL ANALYSIS
    # -------------------------------------------------------
    label_col = "multiclass_label" if "multiclass_label" in df_real.columns else "label"
    print(f"\n[STEP 1] Initial Real Data Analysis")
    print(f"   Total Samples: {len(df_real):,}")
    print(f"   Feature Count: {len(SELECTED_FEATURES)}")
    
    # Label Encoding
    X = df_real[SELECTED_FEATURES].values
    y = df_real[label_col].values
    
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    joblib.dump(encoder, ENCODER_PATH)
    
    # Class Mapping Tablosu
    print("\n   --- Class ID Mapping ---")
    for cls_name, cls_id in zip(encoder.classes_, encoder.transform(encoder.classes_)):
        count = (y_enc == cls_id).sum()
        print(f"   ID {cls_id}: {cls_name:<20} | Count: {count:,}")

    # -------------------------------------------------------
    # 2. SPLITTING (TRAIN/VAL/TEST)
    # -------------------------------------------------------
    print(f"\n[STEP 2] Splitting Data (Train: 70%, Val: 15%, Test: 15%)")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_enc, test_size=0.30, stratify=y_enc, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    
    print(f"   Train Set (Real) : {len(X_train):,} samples")
    print(f"   Val Set          : {len(X_val):,} samples")
    print(f"   Test Set         : {len(X_test):,} samples")

    # -------------------------------------------------------
    # 3. SCALING
    # -------------------------------------------------------
    print(f"\n[STEP 3] Robust Scaling (Fitting on Real Train Only)")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    
    # ARA KAYIT: Saf eğitim verisini kaydet (Rapor kıyaslaması için şart)
    print("\n   [INFO] Saving INTERMEDIATE datasets (Pure Real)...")
    _save_split(X_train, y_train, "train_PURE_REAL")
    _save_split(X_val,   y_val,   "val")
    _save_split(X_test,  y_test,  "test")

    # -------------------------------------------------------
    # 4. GAN AUGMENTATION (SMART STRATEGY)
    # -------------------------------------------------------
    print(f"\n[STEP 4] GAN Augmentation Strategy (k=2 Ratio)")
    
    if os.path.exists(GAN_DATA_PATH):
        df_gan = pd.read_csv(GAN_DATA_PATH)
        X_gan = scaler.transform(df_gan[SELECTED_FEATURES].values)
        y_gan = encoder.transform(df_gan[label_col].values)
        
        unique, counts = np.unique(y_train, return_counts=True)
        max_real_count = max(counts) 

        # AYARLAR
        HARD_CAP = 45000 
        RATIO_CAP = 2     
        
        print(f"   HARD CAP  : {HARD_CAP:,} (Max samples per class)")
        print(f"   RATIO CAP : {RATIO_CAP}x (Max {RATIO_CAP}x of real data)")
        
        X_add, y_add = [], []
        
        print("\n   --- Augmentation Details ---")
        print(f"   {'Class Name':<20} | {'Real (Train)':<12} | {'Max Allowed':<12} | {'GAN Added':<10} | {'Final Total':<12} | {'Status'}")
        print("-" * 95)

        for cls_idx in unique:
            cls_name = encoder.inverse_transform([cls_idx])[0]
            real_count = counts[np.where(unique == cls_idx)[0][0]]
            
            # Hedef Hesapla
            target_max = min(max_real_count, HARD_CAP)
            ratio_limit = int(real_count * (1 + RATIO_CAP))
            final_target = min(target_max, ratio_limit)
            
            gan_added = 0
            status = "No Change"
            
            if real_count < final_target:
                needed = int(final_target - real_count)
                indices = np.where(y_gan == cls_idx)[0]
                
                if len(indices) > 0:
                    selected = np.random.choice(indices, needed, replace=(len(indices) < needed))
                    X_add.append(X_gan[selected])
                    y_add.append(y_gan[selected])
                    gan_added = needed
                    status = "AUGMENTED"
                else:
                    status = "NO GAN DATA"
            else:
                status = "SUFFICIENT"
            
            final_total = real_count + gan_added
            print(f"   {cls_name:<20} | {real_count:<12,} | {final_target:<12,} | {gan_added:<10,} | {final_total:<12,} | {status}")

        if X_add:
            X_train = np.vstack([X_train, np.vstack(X_add)])
            y_train = np.concatenate([y_train, np.concatenate(y_add)])
            
            # Shuffle
            perm = np.random.permutation(len(X_train))
            X_train, y_train = X_train[perm], y_train[perm]
            print(f"\n   [RESULT] Augmentation Complete. Train Set Size: {len(X_train):,}")
        else:
            print("\n   [RESULT] No Augmentation performed.")

    else:
        print(f"\n   [WARN] GAN file not found at {GAN_DATA_PATH}. Skipping augmentation.")

    # -------------------------------------------------------
    # 5. FINAL SAVE
    # -------------------------------------------------------
    print(f"\n[STEP 5] Saving Final Augmented Datasets")
    _save_split(X_train, y_train, "train") # Bu artık Augmented Train
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")

    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_single_sample(df_row: pd.DataFrame) -> torch.Tensor:
    global _SCALER_CACHE
    if isinstance(df_row, pd.Series): df_row = df_row.to_frame().T
    for col in SELECTED_FEATURES:
        if col not in df_row.columns: df_row[col] = 0.0
    df_row.replace([np.inf, -np.inf], 0, inplace=True)
    df_row.fillna(0, inplace=True)
    if _SCALER_CACHE is None: _SCALER_CACHE = joblib.load(SCALER_PATH)
    x = _SCALER_CACHE.transform(df_row[SELECTED_FEATURES].values)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(1)