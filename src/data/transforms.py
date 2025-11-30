"""
Preprocessing Utilities for IDS Data (CNN Input)

This module provides two different pipelines:
1) Binary Classification  → uses 'binary_label'
2) Multiclass IDS         → uses 'label2' (main attack types)

All other label columns are dropped.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple


DROP_COLS_COMMON = [
    "device_name", "device_mac", "label_full",
    "timestamp", "timestamp_start", "timestamp_end",
]

# ---- BINARY MODEL ----
def preprocess_binary(df: pd.DataFrame, reshape_to=(7, 10, 1)):
    """
    Preprocess for Binary IDS Classification.
    Uses 'binary_label' column as target.
    All other label columns are dropped.
    """
    df = df.drop(columns=DROP_COLS_COMMON, errors="ignore")

    # Keep only numeric features + target
    y = df["binary_label"].values  # Already binary (0/1)
    X = df.select_dtypes(include=["float64", "int64"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_reshaped = X_scaled.reshape(-1, reshape_to[0], reshape_to[1], reshape_to[2])
    return X_reshaped, y, scaler


# ---- MULTICLASS MODEL ----
def preprocess_multiclass(df: pd.DataFrame, reshape_to=(7, 10, 1)):
    """
    Preprocess for MULTICLASS IDS.
    Uses 'label2' → attack group (e.g., mitm, dos, recon...)
    All other label columns are dropped.
    """
    df = df.drop(columns=DROP_COLS_COMMON, errors="ignore")

    # Encode 'label2'
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label2"])

    X = df.select_dtypes(include=["float64", "int64"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_reshaped = X_scaled.reshape(-1, reshape_to[0], reshape_to[1], reshape_to[2])
    return X_reshaped, y, scaler, encoder
