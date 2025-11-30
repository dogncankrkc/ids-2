"""
Preprocessing Utilities for IDS Data (CNN Input)

This module prepares the data for:
1) Binary Classification  → uses 'binary_label'
2) Multiclass IDS         → uses 'label2' (main attack types)

The output is ready for CNN input:
    shape → (N, 1, 7, 10)
"""

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch

# ------------------------
# CONFIG
# ------------------------
FEATURE_SHAPE = (7, 10, 1)  # CNN expects → 1 x 7 x 10
TEST_SIZE = 0.2             # 80 train / 20 test

# Labels to drop (not used as features)
DROP_COLS_COMMON = [
    "device_name",
    "device_mac",
    "label_full",
    "timestamp",
    "timestamp_start",
    "timestamp_end",
]

# ------------------------
# ELLE SEÇİLMİŞ 70 FEATURE
# (7x10 grid ile tam uyumlu)
# ------------------------
SELECTED_FEATURES = [
    "log_data-ranges_avg",
    "log_data-ranges_max",
    "log_data-ranges_min",
    # "log_data-ranges_std_deviation"  # <- BUNU BİLEREK ÇIKARDIK
    "log_data-types_count",
    "log_interval-messages",
    "log_messages_count",
    "network_fragmentation-score",
    "network_fragmented-packets",
    "network_header-length_avg",
    "network_header-length_max",
    "network_header-length_min",
    "network_header-length_std_deviation",
    "network_interval-packets",
    "network_ip-flags_avg",
    "network_ip-flags_max",
    "network_ip-flags_min",
    "network_ip-flags_std_deviation",
    "network_ip-length_avg",
    "network_ip-length_max",
    "network_ip-length_min",
    "network_ip-length_std_deviation",
    "network_ips_all_count",
    "network_ips_dst_count",
    "network_ips_src_count",
    "network_macs_all_count",
    "network_macs_dst_count",
    "network_macs_src_count",
    "network_mss_avg",
    "network_mss_max",
    "network_mss_min",
    "network_mss_std_deviation",
    "network_packet-size_avg",
    "network_packet-size_max",
    "network_packet-size_min",
    "network_packet-size_std_deviation",
    "network_packets_all_count",
    "network_packets_dst_count",
    "network_packets_src_count",
    "network_payload-length_avg",
    "network_payload-length_max",
    "network_payload-length_min",
    "network_payload-length_std_deviation",
    "network_ports_all_count",
    "network_ports_dst_count",
    "network_ports_src_count",
    "network_protocols_all_count",
    "network_protocols_dst_count",
    "network_protocols_src_count",
    "network_tcp-flags-ack_count",
    "network_tcp-flags-fin_count",
    "network_tcp-flags-psh_count",
    "network_tcp-flags-rst_count",
    "network_tcp-flags-syn_count",
    "network_tcp-flags-urg_count",
    "network_tcp-flags_avg",
    "network_tcp-flags_max",
    "network_tcp-flags_min",
    "network_tcp-flags_std_deviation",
    "network_time-delta_avg",
    "network_time-delta_max",
    "network_time-delta_min",
    "network_time-delta_std_deviation",
    "network_ttl_avg",
    "network_ttl_max",
    "network_ttl_min",
    "network_ttl_std_deviation",
    "network_window-size_avg",
    "network_window-size_max",
    "network_window-size_min",
    "network_window-size_std_deviation",
]
assert len(SELECTED_FEATURES) == 70, "SELECTED_FEATURES tam 70 olmalı!"


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ------------------------
# BINARY CLASSIFICATION
# ------------------------
def preprocess_binary(
    df: pd.DataFrame,
    scaler_path: str = "models/scaler_binary.pkl",
    test_size: float = TEST_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess data for **binary classification** (benign vs attack).

    Returns:
        X_train, X_test, y_train, y_test  → ready for CNN
    """
    df = df.drop(columns=DROP_COLS_COMMON, errors="ignore")

    # target
    y = df["binary_label"].values  # 0 / 1

    # sadece seçili 70 feature
    X = df[SELECTED_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _ensure_dir(scaler_path)
    joblib.dump(scaler, scaler_path)

    # reshape → (N, 7, 10, 1)
    X_scaled = X_scaled.reshape(-1, FEATURE_SHAPE[0], FEATURE_SHAPE[1], FEATURE_SHAPE[2])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ------------------------
# MULTICLASS CLASSIFICATION
# ------------------------
def preprocess_multiclass(
    df: pd.DataFrame,
    scaler_path: str = "models/scaler_multi.pkl",
    encoder_path: str = "models/label_encoder.pkl",
    test_size: float = TEST_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess data for **multiclass IDS** (dos, recon, malware, ...).

    Returns:
        X_train, X_test, y_train, y_test (reshaped for CNN)
    """
    df = df.drop(columns=DROP_COLS_COMMON, errors="ignore")

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label2"])

    _ensure_dir(encoder_path)
    joblib.dump(encoder, encoder_path)

    X = df[SELECTED_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _ensure_dir(scaler_path)
    joblib.dump(scaler, scaler_path)

    X_scaled = X_scaled.reshape(-1, FEATURE_SHAPE[0], FEATURE_SHAPE[1], FEATURE_SHAPE[2])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ------------------------
# SINGLE SAMPLE PREPROCESSING (INFERENCE)
# ------------------------
def preprocess_single_sample(df_row: pd.DataFrame) -> torch.Tensor:
    """
    Preprocess a SINGLE CSV row for inference.

    Input:
        df_row: DataFrame (1 row) or Series
    Output:
        torch tensor → shape (1, 1, 7, 10)
    """
    if isinstance(df_row, pd.DataFrame):
        row = df_row[SELECTED_FEATURES].values  # (1, 70)
    else:
        # pd.Series ise
        row = df_row[SELECTED_FEATURES].to_frame().T.values  # (1, 70)

    scaler = joblib.load("models/scaler_multi.pkl")
    x_scaled = scaler.transform(row)

    x_reshaped = x_scaled.reshape(1, 1, 7, 10)
    return torch.tensor(x_reshaped, dtype=torch.float32)
