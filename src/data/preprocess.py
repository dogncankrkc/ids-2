"""
Preprocessing Utilities for CIC-IoT-2023 (6-CLASS MERGED EDITION)

Key features:
- Dynamic SMOTE: Finds the majority class (likely DoS-DDoS) and boosts others to match it.
- Fix: Accepts 'smote_target_ratio' dummy arg to prevent train.py errors.
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None
from typing import Optional, List

# ----------------------------
# FEATURES (39)
# ----------------------------
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
SCALER_PATH = f"{SAVE_DIR}/feature_scaler.pkl"


def _save_split(X, y, name: str):
    os.makedirs(SAVE_DIR, exist_ok=True)
    df = pd.DataFrame(X, columns=SELECTED_FEATURES)
    df["label"] = y
    path = f"{SAVE_DIR}/{name}_preprocessed.csv"
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}")


def _clean_df(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df = df.copy()
    for col in SELECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    df = df[SELECTED_FEATURES + [label_col]]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    for col in SELECTED_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)

    return df


def _apply_log1p(X: np.ndarray, feature_names: list[str], log_features: list[str]) -> np.ndarray:
    if not log_features:
        return X
    idxs = [feature_names.index(f) for f in log_features if f in feature_names]
    if not idxs:
        return X
    X2 = X.copy()
    X2[:, idxs] = np.log1p(np.clip(X2[:, idxs], 0.0, None))
    return X2


def preprocess_multiclass(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    log1p_features: Optional[List[str]] = None,
    use_smote: bool = True,
    smote_target_ratio: float = 0.5, # <--- BU PARAMETRE KRİTİK (Hata vermemesi için)
    smote_k_neighbors: int = 5,
    random_state: int = 42,
    save_splits: bool = True,
):
    print("[INFO] Preprocessing: Split -> RobustScaler -> Dynamic Max-SMOTE")

    os.makedirs(SAVE_DIR, exist_ok=True)

    if label_col is None:
        label_col = "multiclass_label" if "multiclass_label" in df.columns else "label"
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")

    df = _clean_df(df, label_col=label_col)

    X = df[SELECTED_FEATURES].values.astype(np.float32)
    y = df[label_col].astype(str).values

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    joblib.dump(encoder, ENCODER_PATH)

    class_map = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print("[INFO] Class mapping:", class_map)

    # Split: 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_enc, test_size=0.30, stratify=y_enc, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )

    # Optional Log1p
    log1p_features = log1p_features or []
    X_train = _apply_log1p(X_train, SELECTED_FEATURES, log1p_features)
    X_val = _apply_log1p(X_val, SELECTED_FEATURES, log1p_features)
    X_test = _apply_log1p(X_test, SELECTED_FEATURES, log1p_features)

    # Robust Scaling
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    # --------------------------------------------------
    # DYNAMIC SMOTE (TRAIN ONLY)
    # --------------------------------------------------
    if use_smote:
        if SMOTE is None:
            raise RuntimeError("imblearn is not installed but use_smote=True was requested.")

        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        # En kalabalık sınıfı hedef alıyoruz
        max_count = max(class_counts.values())
        print(f"[INFO] Majority Class Count: {max_count}")

        smote_targets = {}
        for cls, cnt in class_counts.items():
            if cnt < max_count:
                smote_targets[int(cls)] = max_count

        if smote_targets:
            print(f"[INFO] Boosting minorities to {max_count}: {list(smote_targets.keys())}")
            smote = SMOTE(
                sampling_strategy=smote_targets,
                random_state=random_state,
                k_neighbors=smote_k_neighbors,
                n_jobs=-1
            )
            X_train, y_train = smote.fit_resample(X_train, y_train)

    # Distribution Check
    print("\n[TRAIN DISTRIBUTION (encoded)]")
    print(pd.Series(y_train).value_counts().sort_index())

    if save_splits:
        _save_split(X_train, y_train, "train")
        _save_split(X_val, y_val, "val")
        _save_split(X_test, y_test, "test")

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
    x = scaler.transform(df_row[SELECTED_FEATURES].values.astype(np.float32))
    return torch.tensor(x, dtype=torch.float32).unsqueeze(1)