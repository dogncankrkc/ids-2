"""
Create REAL-WORLD Test Set (Leak-Free Version)
Change: RANDOM_STATE changed to avoid overlap with training set generation.
"""
import os
import pandas as pd
import numpy as np

# ======================================
# CONFIG
# ======================================
SOURCE_PATH = "data/processed/CIC2023_BINARY_AND_ATTACK_MAPPED.csv"
OUT_DIR = "data/processed/real_world_test_v1"
OUT_FILE = "test_raw.csv"

# KRİTİK DEĞİŞİKLİK: Eğitimde 42 kullandık, burada çakışmaması için farklı bir seed kullanıyoruz.
RANDOM_STATE = 9999 
TEST_RATIO = 0.15 # Biraz daha geniş alalım, garanti olsun

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

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[INFO] Loading source: {SOURCE_PATH}")
    df = pd.read_csv(SOURCE_PATH)

    # Basic sanity
    required = SELECTED_FEATURES + ["binary_label", "multiclass_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Etiket düzeltme (Eğer multiclass_label yoksa label'dan üret)
        if "label" in df.columns and "multiclass_label" not in df.columns:
            df["multiclass_label"] = df["label"]
        if "binary_label" not in df.columns:
            df["binary_label"] = (df["multiclass_label"] != "Benign").astype(int)

    # Shuffle with NEW SEED
    print(f"[INFO] Shuffling with seed {RANDOM_STATE} to avoid training overlap...")
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # Take RAW test slice
    n_test = int(len(df) * TEST_RATIO)
    df_test = df.iloc[:n_test].copy()
    
    # Save
    out_path = os.path.join(OUT_DIR, OUT_FILE)
    df_test[required].to_csv(out_path, index=False)

    print("[DONE] Real-world test set created")
    print(f"[PATH] {out_path}")
    print(f"[SIZE] {len(df_test)} samples")

if __name__ == "__main__":
    main()