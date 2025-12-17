"""
Preprocessing Utilities for CIC-IoT-2023 Dataset.
FINAL STABLE VERSION (NO LEAKAGE)

✔ Train / Val / Test split FIRST
✔ StandardScaler fit ONLY on TRAIN
✔ SMOTE ONLY on TRAIN
✔ Selective SMOTE: Web + BruteForce → 100k
✔ Benign untouched
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

PAD_FEATURE_NAME = "PAD_0"

# --------------------------------------------------
# FEATURES (40 = 5x8 GRID)
# --------------------------------------------------
SELECTED_FEATURES = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH',
    'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv',
    'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size',
    'IAT', 'Number', 'Variance',
    PAD_FEATURE_NAME
]

ATTACK_TARGET = 100_000
TARGET_TRAIN_COUNT = int(0.7 * ATTACK_TARGET) 

SAVE_DIR = "data/processed"
ENCODER_PATH = f"{SAVE_DIR}/label_encoder.pkl"
SCALER_PATH  = f"{SAVE_DIR}/feature_scaler.pkl"


# --------------------------------------------------
# MULTICLASS PREPROCESS
# --------------------------------------------------
def preprocess_multiclass(df: pd.DataFrame):

    print("[INFO] Multiclass preprocessing (SELECTIVE SMOTE)")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # --------------------------------------------------
    # 1️⃣ FEATURES & LABELS
    # --------------------------------------------------
    X = df[SELECTED_FEATURES].values
    y = df["multiclass_label"].values

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    joblib.dump(encoder, ENCODER_PATH)

    class_map = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print("[INFO] Class mapping:", class_map)

    benign_id = class_map["Benign"]

    # --------------------------------------------------
    # 2️⃣ SPLIT (STRATIFIED)
    # --------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_enc, test_size=0.30, stratify=y_enc, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # --------------------------------------------------
    # 3️⃣ SCALE (FIT TRAIN ONLY)
    # --------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)

    # --------------------------------------------------
    # 4️⃣ SELECTIVE SMOTE (TRAIN ONLY)
    # --------------------------------------------------
    smote_targets = {}

    for cls_name in ["Web", "BruteForce"]:
        if cls_name in class_map:
            cls_id = class_map[cls_name]
            current = int((y_train == cls_id).sum())
            if current < TARGET_TRAIN_COUNT:
                smote_targets[cls_id] = TARGET_TRAIN_COUNT
    print("[INFO] SMOTE targets:", smote_targets)

    if smote_targets:
        smote = SMOTE(
            sampling_strategy=smote_targets,
            random_state=42,
            k_neighbors=5,
            n_jobs=-1
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # --------------------------------------------------
    # 5️⃣ DISTRIBUTION CHECK
    # --------------------------------------------------
    print("\n[TRAIN DISTRIBUTION AFTER SMOTE]")
    print(pd.Series(y_train).value_counts().sort_index())

    # --------------------------------------------------
    # 6️⃣ SAVE SPLITS
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
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)

    return x

def preprocess_binary(df: pd.DataFrame):
    """
    Preprocesses the dataset for binary classification.

    Args:
        df (pd.DataFrame): Input dataframe with raw data.  
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """