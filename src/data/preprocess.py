"""
Preprocessing Utilities for CIC-IoT-2023 Dataset.
FINAL BALANCED VERSION (Undersample Benign + SMOTE Minority)

✔ Train / Val / Test split FIRST
✔ Undersample BENIGN only in TRAIN (to match Attack Count)
✔ SMOTE Minority only in TRAIN
✔ StandardScaler fit ONLY on TRAIN
✔ Saves preprocessed splits
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# Hedeflenen sınıf başı örnek sayısı (Eğitim için)
# Saldırı sınıfların (DDoS, Mirai vs.) ortalama 70k civarında, hepsini buna eşitleyeceğiz.
TARGET_SAMPLES_PER_CLASS = 70_000 

SAVE_DIR = "data/processed"
ENCODER_PATH = f"{SAVE_DIR}/label_encoder.pkl"
SCALER_PATH  = f"{SAVE_DIR}/feature_scaler.pkl"


# --------------------------------------------------
# MULTICLASS PREPROCESS
# --------------------------------------------------
def preprocess_multiclass(df: pd.DataFrame):

    print("[INFO] Multiclass preprocessing (UNDERSAMPLE BENIGN + SMOTE)")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1️⃣ FEATURES & LABELS
    X = df[SELECTED_FEATURES].values
    y = df["multiclass_label"].values

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    joblib.dump(encoder, ENCODER_PATH)

    class_map = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print("[INFO] Class mapping:", class_map)

    # 2️⃣ SPLIT (STRATIFIED)
    # Önce Test/Val ayıralım ki gerçek dünya verisine dokunmayalım
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_enc, test_size=0.30, stratify=y_enc, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    # --------------------------------------------------
    # 3️⃣ UNDERSAMPLING BENIGN (ONLY TRAIN) - YENİ KISIM
    # --------------------------------------------------
    print(f"[INFO] Balancing Train Set... (Target: {TARGET_SAMPLES_PER_CLASS} per class)")
    
    benign_id = class_map['Benign']
    
    # Train setindeki Benign indekslerini bul
    indices_benign = np.where(y_train == benign_id)[0]
    indices_others = np.where(y_train != benign_id)[0]
    
    # Benign sayısı çoksa rastgele azalt (Undersample)
    if len(indices_benign) > TARGET_SAMPLES_PER_CLASS:
        indices_benign = np.random.choice(
            indices_benign, TARGET_SAMPLES_PER_CLASS, replace=False
        )
        print(f"[INFO] Benign undersampled to {TARGET_SAMPLES_PER_CLASS}")
    
    # İndeksleri birleştir ve karıştır
    indices_keep = np.concatenate([indices_others, indices_benign])
    np.random.shuffle(indices_keep)
    
    X_train = X_train[indices_keep]
    y_train = y_train[indices_keep]

    # --------------------------------------------------
    # 4️⃣ SCALE (FIT TRAIN ONLY)
    # --------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)

    # --------------------------------------------------
    # 5️⃣ SMOTE FOR MINORITY (ONLY TRAIN)
    # --------------------------------------------------
    # Artık hedefimiz sabit: TARGET_SAMPLES_PER_CLASS (70k)
    smote_targets = {}
    
    # Mevcut sayılara bak, 70k'dan az olanları 70k'ya tamamla
    unique, counts = np.unique(y_train, return_counts=True)
    current_counts = dict(zip(unique, counts))
    
    for cls_id, count in current_counts.items():
        if count < TARGET_SAMPLES_PER_CLASS:
            smote_targets[cls_id] = TARGET_SAMPLES_PER_CLASS
            
    if smote_targets:
        print(f"[INFO] Applying SMOTE to classes: {smote_targets.keys()}")
        smote = SMOTE(
            sampling_strategy=smote_targets,
            random_state=42,
            k_neighbors=5,
            n_jobs=-1
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # --------------------------------------------------
    # 6️⃣ DISTRIBUTION CHECK
    # --------------------------------------------------
    print("\n[FINAL TRAIN DISTRIBUTION - BALANCED]")
    print(pd.Series(y_train).value_counts().sort_index())
    
    print("\n[TEST DISTRIBUTION - REALISTIC]")
    print(pd.Series(y_test).value_counts().sort_index())

    # --------------------------------------------------
    # 7️⃣ SAVE SPLITS
    # --------------------------------------------------
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


# --------------------------------------------------
# SINGLE SAMPLE (INFERENCE)
# --------------------------------------------------
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

    # ResNet-MLP için (1, 39) yeterli, ama model kodumuz 
    # 3D gelirse (1, 1, 39) otomatik düzeltiyor.
    # Uyumluluk için eski formatı koruyorum:
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)

    return x