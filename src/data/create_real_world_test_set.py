"""
Create REAL-WORLD Test Set for Hierarchical IDS (Stage-1 + Stage-2)

Goal:
- Produce a RAW, untouched test set
- No GAN
- No balancing
- No scaling
- Not used in ANY training or validation
- Closest to real deployment traffic

Author: Dogancan Karakoc
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

TEST_RATIO = 0.10        # %10 tamamen dış test
RANDOM_STATE = 42

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

# ======================================
# MAIN
# ======================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(SOURCE_PATH)

    # Basic sanity
    required = SELECTED_FEATURES + ["binary_label", "multiclass_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Shuffle once
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # Take RAW test slice
    n_test = int(len(df) * TEST_RATIO)
    df_test = df.iloc[:n_test].copy()

    # Keep only what inference needs
    df_test = df_test[required]

    # Save
    out_path = os.path.join(OUT_DIR, OUT_FILE)
    df_test.to_csv(out_path, index=False)

    # Print stats
    print("[DONE] Real-world test set created")
    print(f"[PATH] {out_path}")
    print("---- Distribution ----")
    print(df_test["binary_label"].value_counts())
    print(df_test["multiclass_label"].value_counts())

if __name__ == "__main__":
    main()
