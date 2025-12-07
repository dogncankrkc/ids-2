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

def split_train_val_test(X, y, test_size=0.15, val_size=0.15):
    """
    Veriyi 3 parçaya böler:
    1. Önce %15 Test setini ayırır (Kenara kilitleriz).
    2. Kalan parçadan %15 Validation ayırır.
    3. Geriye kalan en büyük parça Train olur.
    """
    # 1. Adım: Test setini ayır (Stratify: Sınıf dengesini koru)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # 2. Adım: Kalanı (X_temp) Train ve Validation olarak ayır
    # val_size oranını X_temp üzerinden alırız.
    # Örneğin X_temp %85 ise, bunun %15'i val olur.
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    
    # Sırayla 6 parça döndürüyoruz
    return X_train, X_val, X_test, y_train, y_val, y_test

# ------------------------
# BINARY CLASSIFICATION
# ------------------------
def preprocess_binary(
    df: pd.DataFrame,
    scaler_path: str = "models/scaler_binary.pkl",
) -> Tuple:
    """
    Binary (Normal vs Attack) verisi hazırlar.
    Dönüş: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Gereksiz sütunları at
    df = df.drop(columns=DROP_COLS_COMMON, errors="ignore")

    # Hedef ve Özellikler
    y = df["binary_label"].values  # 0 veya 1
    X = df[SELECTED_FEATURES].values

    # Ölçeklendirme (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scaler'ı kaydet (Inference için lazım olacak)
    _ensure_dir(scaler_path)
    joblib.dump(scaler, scaler_path)

    # CNN için Reshape (Örn: 7x10)
    # Boyut: (Sample_Sayısı, 7, 10, 1) -> Sondaki 1 kanal sayısı
    X_reshaped = X_scaled.reshape(-1, FEATURE_SHAPE[0], FEATURE_SHAPE[1], FEATURE_SHAPE[2])

    # 3'lü ayrıma gönder
    return split_train_val_test(X_reshaped, y)


# ------------------------
# MULTICLASS CLASSIFICATION
# ------------------------
def preprocess_multiclass(
    df: pd.DataFrame,
    scaler_path: str = "models/scaler_multi.pkl",
    encoder_path: str = "models/label_encoder.pkl",
) -> Tuple:
    """
    Çok sınıflı (DoS, Probe, vb.) veri hazırlar.
    Dönüş: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    df = df.drop(columns=DROP_COLS_COMMON, errors="ignore")

    # Etiketleri Sayısal Hale Getir (Label Encoding)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label2"])

    # --- ÖNEMLİ: Hangi sayı hangi atağa denk geliyor görelim ---
    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print(f"\n[INFO] Multiclass Label Mapping: {mapping}")
    # -----------------------------------------------------------

    # Encoder'ı kaydet (Inference'da tersine çevirmek için)
    _ensure_dir(encoder_path)
    joblib.dump(encoder, encoder_path)

    # Özellikleri seç ve ölçeklendir
    X = df[SELECTED_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _ensure_dir(scaler_path)
    joblib.dump(scaler, scaler_path)

    # CNN için Reshape
    X_reshaped = X_scaled.reshape(-1, FEATURE_SHAPE[0], FEATURE_SHAPE[1], FEATURE_SHAPE[2])

    # 3'lü ayrıma gönder
    return split_train_val_test(X_reshaped, y)

# ------------------------
# SINGLE SAMPLE PREPROCESSING (INFERENCE)
# ------------------------
def preprocess_single_sample(df_row: pd.DataFrame) -> torch.Tensor:
    """
    Canlı sistemde tek bir satır veri geldiğinde kullanılır.
    Çıktı: (1, 1, 7, 10) boyutunda Tensor.
    """
    # Tek satır mı, seri mi kontrol et
    if isinstance(df_row, pd.DataFrame):
        row = df_row[SELECTED_FEATURES].values
    else:
        row = df_row[SELECTED_FEATURES].to_frame().T.values

    # Kaydedilmiş Scaler'ı yükle (Test verisini EĞİTİM scaler'ı ile dönüştürmeliyiz)
    # Not: Binary mi Multiclass mı kullandığına göre buradaki path değişebilir.
    # Varsayılan olarak multiclass scaler yüklüyoruz.
    try:
        scaler = joblib.load("models/scaler_multi.pkl")
    except FileNotFoundError:
        # Eğer henüz multi yoksa binary dene
        scaler = joblib.load("models/scaler_binary.pkl")
        
    x_scaled = scaler.transform(row)

    # Reshape: (1 örnek, 1 kanal, 7 yükseklik, 10 genişlik)
    # PyTorch formatına uygun hale getirdik.
    x_reshaped = x_scaled.reshape(1, 1, 7, 10)
    
    return torch.tensor(x_reshaped, dtype=torch.float32)