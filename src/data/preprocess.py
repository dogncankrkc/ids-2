"""
Preprocessing Utilities for CIC-IoT-2023 Dataset (CNN Input).

FIXES IN THIS VERSION:
- Added `clean_dataset` function to handle Infinity and NaN values.
- Prevents 'ValueError: Input X contains infinity' during Scaling.

This module prepares the data for the model on-the-fly during training:
1) Loading: Reads the balanced CSV.
2) Cleaning: Removes Infinity/NaN values that break the scaler.
3) Balancing: Upsamples minority classes (like Web) to 50k in memory.
4) Encoding: Converts 'DDoS', 'Benign' labels to integers (0, 1, 2...).
5) Scaling: Normalizes features to 0-1 range.
6) Reshaping: Converts 1D data to 7x7 image format for CNN.
"""

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import resample
import torch

# ------------------------
# CONFIGURATION
# ------------------------
FEATURE_SHAPE = (7, 7, 1)  # Target shape for CNN input: 1 Channel, 7x7 Grid

# List of 39 Numerical Features present in the CIC-IoT-2023 dataset
SELECTED_FEATURES = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate', 
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 
    'ece_flag_number', 'cwr_flag_number', 
    'ack_count', 'syn_count', 'fin_count', 'rst_count', 
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC', 
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Variance'
]

# We pad the 39 features to 49 to create a perfect 7x7 grid
TARGET_FEATURE_COUNT = 49
SAMPLES_PER_CLASS = 50000 

def _ensure_dir(path: str):
    """Creates the directory if it does not exist."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def map_attack_label(label: str) -> str:
    """
    Fallback mapping function. 
    Only used if 'Mapped_Label' is missing in the CSV.
    """
    label = str(label).strip().upper()
    if label == 'NAN' or label == '': return 'Other'
    if 'BENIGN' in label: return 'Benign'
    elif 'DDOS' in label: return 'DDoS'
    elif 'DOS' in label: return 'DoS'
    elif 'RECON' in label or 'VULNERABILITYSCAN' in label or 'PING' in label or 'PORTSCAN' in label or 'OSSCAN' in label or 'HOSTDISCOVERY' in label: return 'Recon'
    elif 'XSS' in label or 'SQL' in label or 'UPLOAD' in label or 'BROWSER' in label or 'COMMAND' in label or 'BACKDOOR' in label or 'MALWARE' in label: return 'Web'
    elif 'BRUTEFORCE' in label or 'DICTIONARY' in label: return 'BruteForce'
    elif 'SPOOFING' in label or 'MITM' in label: return 'Spoofing'
    elif 'MIRAI' in label: return 'Mirai'
    else: return 'Other'

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL FIX: Removes Infinity and NaN values.
    
    Why?
    - Network features like 'Rate' can be Infinity (division by zero).
    - MinMaxScaler throws a ValueError if it encounters Infinity.
    
    Steps:
    1. Replace 'inf' and '-inf' with NaN.
    2. Drop rows containing NaN in the selected features columns.
    """
    # 1. Replace Infinity with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. Check for NaNs only in the features we care about
    before = len(df)
    df.dropna(subset=SELECTED_FEATURES, inplace=True)
    after = len(df)
    
    if before != after:
        print(f"[WARN] Dropped {before - after} rows containing Infinity or NaN values.")
    
    return df

def balance_dataset(df: pd.DataFrame, label_col: str, n_samples: int = SAMPLES_PER_CLASS) -> pd.DataFrame:
    """
    Balances the dataset in RAM.
    This ensures every class has exactly 'n_samples' (e.g., 50k).
    """
    print(f"[INFO] Balancing dataset to ~{n_samples} samples per class...")
    df_balanced = pd.DataFrame()
    unique_classes = df[label_col].unique()
    
    for label in unique_classes:
        if label == 'Other': continue
            
        df_class = df[df[label_col] == label]
        count = len(df_class)
        
        if count == 0: continue
            
        if count > n_samples:
            # Undersample majority classes
            df_resampled = resample(df_class, replace=False, n_samples=n_samples, random_state=42)
        else:
            # Oversample minority classes
            df_resampled = resample(df_class, replace=True, n_samples=n_samples, random_state=42)
        
        df_balanced = pd.concat([df_balanced, df_resampled])
        
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[INFO] Balanced Counts:\n{df_balanced[label_col].value_counts()}")
    return df_balanced

def pad_features(X: np.ndarray, target_count: int = TARGET_FEATURE_COUNT) -> np.ndarray:
    """
    Pads the feature vector with zeros to match the target count (49).
    Required to reshape into a 7x7 grid.
    """
    current_count = X.shape[1]
    if current_count < target_count:
        padding_size = target_count - current_count
        padding = np.zeros((X.shape[0], padding_size))
        X_padded = np.hstack((X, padding))
        return X_padded
    return X

def split_train_val_test(X, y, test_size=0.15, val_size=0.15):
    """
    Splits data into Train (70%), Validation (15%), and Test (15%).
    Uses Stratified Split to maintain class distribution.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# ------------------------
# MULTICLASS CLASSIFICATION
# ------------------------
def preprocess_multiclass(
    df: pd.DataFrame,
    scaler_path: str = "models/scaler_multi.pkl",
    encoder_path: str = "models/label_encoder.pkl",
) -> Tuple:
    """
    Main pipeline for Multiclass training.
    """
    print("[INFO] Starting Multiclass Preprocessing...")

    # OPTIMIZATION: Use existing 'Mapped_Label' if available
    if 'Mapped_Label' in df.columns:
        print("[INFO] Using existing 'Mapped_Label' column.")
        df['target_label'] = df['Mapped_Label']
    elif 'multiclass_label' in df.columns:
        print("[INFO] Mapping 'multiclass_label' to categories...")
        df['target_label'] = df['multiclass_label'].apply(map_attack_label)
    else:
        raise ValueError("Critical Error: No label column found (Mapped_Label or multiclass_label missing).")

    # 1. Clean 'Other' classes
    df = df[df['target_label'] != 'Other']
    
    # 2. CLEAN DATASET (Remove Inf/NaN) - NEW STEP
    df = clean_dataset(df)

    # 3. Balance Dataset (Web 23k -> 50k happens here)
    df = balance_dataset(df, 'target_label', n_samples=SAMPLES_PER_CLASS)

    # 4. Encode Labels (String -> Integer)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['target_label'])
    
    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print(f"[INFO] Class Mapping: {mapping}")
    _ensure_dir(encoder_path)
    joblib.dump(encoder, encoder_path)

    # 5. Extract, Pad, Scale Features
    X = df[SELECTED_FEATURES].values
    X = pad_features(X, TARGET_FEATURE_COUNT)

    scaler = MinMaxScaler()
    # Now X is clean, so fit_transform will not crash
    X_scaled = scaler.fit_transform(X)
    _ensure_dir(scaler_path)
    joblib.dump(scaler, scaler_path)

    # 6. Reshape for CNN (N, 1, 7, 7)
    X_reshaped = X_scaled.reshape(-1, FEATURE_SHAPE[0], FEATURE_SHAPE[1], FEATURE_SHAPE[2])

    print(f"[INFO] Final Data Shape: {X_reshaped.shape}")
    return split_train_val_test(X_reshaped, y)

# ------------------------
# BINARY CLASSIFICATION
# ------------------------
def preprocess_binary(
    df: pd.DataFrame,
    scaler_path: str = "models/scaler_binary.pkl",
) -> Tuple:
    """
    Main pipeline for Binary training (Benign vs Attack).
    """
    print("[INFO] Starting Binary Preprocessing...")
    
    # Use binary_label if exists, else derive it
    if 'binary_label' in df.columns:
        df['label_bin'] = df['binary_label']
    else:
        # Fallback
        target_col = 'Mapped_Label' if 'Mapped_Label' in df.columns else 'multiclass_label'
        df['label_bin'] = df[target_col].apply(lambda x: 0 if 'Benign' in str(x) else 1)

    # 1. CLEAN DATASET (Remove Inf/NaN) - NEW STEP
    df = clean_dataset(df)

    # 2. Balance Dataset
    df = balance_dataset(df, 'label_bin', n_samples=SAMPLES_PER_CLASS)
    y = df['label_bin'].values
    
    # 3. Extract & Pad
    X = df[SELECTED_FEATURES].values
    X = pad_features(X, TARGET_FEATURE_COUNT)

    # 4. Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    _ensure_dir(scaler_path)
    joblib.dump(scaler, scaler_path)

    # 5. Reshape
    X_reshaped = X_scaled.reshape(-1, FEATURE_SHAPE[0], FEATURE_SHAPE[1], FEATURE_SHAPE[2])
    return split_train_val_test(X_reshaped, y)

def preprocess_single_sample(df_row: pd.DataFrame) -> torch.Tensor:
    """
    Prepares a single row for inference (Prediction).
    Also handles NaN/Inf for safety.
    """
    if isinstance(df_row, pd.Series):
        df_row = df_row.to_frame().T
        
    for col in SELECTED_FEATURES:
        if col not in df_row.columns:
            df_row[col] = 0.0     
            
    # Clean possible Inf in single sample
    df_row.replace([np.inf, -np.inf], 0, inplace=True)
    df_row.fillna(0, inplace=True)

    row_values = df_row[SELECTED_FEATURES].values
    
    # Pad
    if row_values.shape[1] < TARGET_FEATURE_COUNT:
        padding = np.zeros((row_values.shape[0], TARGET_FEATURE_COUNT - row_values.shape[1]))
        row_values = np.hstack((row_values, padding))
        
    # Scale
    try:
        scaler = joblib.load("models/scaler_multi.pkl")
    except FileNotFoundError:
        scaler = joblib.load("models/scaler_binary.pkl")
        
    x_scaled = scaler.transform(row_values)
    x_reshaped = x_scaled.reshape(1, 1, 7, 7)
    
    return torch.tensor(x_reshaped, dtype=torch.float32)