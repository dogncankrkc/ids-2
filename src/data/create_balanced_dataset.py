"""
FINAL DATASET CREATOR – CIC-IoT-2023 (6-CLASS MERGED EDITION)

Changes:
- Merged 'DoS' and 'DDoS' into a single class: 'DoS-DDoS'.
- Fixed SettingWithCopyWarning by using explicit .copy()
"""

import os
import argparse
import numpy as np
from typing import Optional
import pandas as pd

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

# ARTIK 6 SALDIRI SINIFI VAR
ATTACK_CLASSES = ["BruteForce", "DoS-DDoS", "Mirai", "Recon", "Spoofing", "Web"]
ALL_CLASSES_WITH_BENIGN = ["Benign"] + ATTACK_CLASSES

# ----------------------------
# LABEL MAPPING (6 ATTACKS)
# ----------------------------
def map_to_6attacks(label: str) -> str:
    label = str(label).strip().upper()

    if label in ("NAN", "", "NONE"):
        return "Other"

    # Benign
    if "BENIGN" in label:
        return "Benign"

    # --- KRİTİK BİRLEŞTİRME ---
    # Hem DoS hem DDoS aynı etikete gidiyor
    if "DDOS" in label or "DOS" in label:
        return "DoS-DDoS"
    # --------------------------

    # Recon
    if any(x in label for x in ["RECON", "VULNERABILITY", "PING", "PORTSCAN", "OSSCAN", "HOSTDISCOVERY"]):
        return "Recon"

    # Web attacks
    if any(x in label for x in ["XSS", "SQL", "UPLOAD", "BROWSER", "COMMAND", "BACKDOOR", "MALWARE", "WEB"]):
        return "Web"

    # BruteForce
    if "BRUTEFORCE" in label or "DICTIONARY" in label:
        return "BruteForce"

    # Spoofing / MITM
    if "SPOOFING" in label or "MITM" in label:
        return "Spoofing"

    # Mirai
    if "MIRAI" in label:
        return "Mirai"

    return "Other"


def _find_label_column(df: pd.DataFrame) -> Optional[str]:
    for cand in ["multiclass_label", "label", "Label"]:
        if cand in df.columns:
            return cand
    return None


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in SELECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    keep = SELECTED_FEATURES + ["mapped_label"]
    
    # --- UYARI ÇÖZÜMÜ: .copy() eklendi ---
    df = df[keep].copy() 
    # -------------------------------------
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    for col in SELECTED_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/CIC2023_FULL_MERGED.csv")
    parser.add_argument("--chunk_size", type=int, default=1_000_000)

    # Dosya ismini değiştirdik ki eski 7'li ile karışmasın
    parser.add_argument("--out_attack", type=str, default="data/processed/CIC2023_ATTACK_ONLY_6CLASS.csv")
    parser.add_argument("--out_with_benign", type=str, default="data/processed/CIC2023_WITH_BENIGN_6CLASS.csv")

    parser.add_argument("--mode", type=str, default="both", choices=["attack_only", "with_benign", "both"])

    # Caps (Limitler)
    parser.add_argument("--benign_cap", type=int, default=250_000)
    parser.add_argument("--attack_cap", type=int, default=100_000)
    parser.add_argument("--web_cap", type=int, default=150_000)      # Web için yüksek limit
    parser.add_argument("--bruteforce_cap", type=int, default=150_000) # BruteForce için yüksek limit

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    os.makedirs(os.path.dirname(args.out_attack), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_with_benign), exist_ok=True)

    caps_with_benign = {
        "Benign": args.benign_cap,
        "Web": args.web_cap,
        "BruteForce": args.bruteforce_cap,
        "DoS-DDoS": args.attack_cap, # Birleşik sınıf limiti
        "Mirai": args.attack_cap,
        "Recon": args.attack_cap,
        "Spoofing": args.attack_cap,
    }

    caps_attack_only = {k: v for k, v in caps_with_benign.items() if k != "Benign"}

    collected_attack = {k: 0 for k in ATTACK_CLASSES}
    collected_with_benign = {k: 0 for k in ALL_CLASSES_WITH_BENIGN}

    chunks_attack = []
    chunks_with_benign = []

    print("[INFO] Creating capped datasets (6-CLASS MERGED)...")
    print(f"[INFO] Mode: {args.mode}")

    for i, chunk in enumerate(pd.read_csv(args.input, chunksize=args.chunk_size)):
        print(f" -> Processing chunk {i+1}", end="\r")

        label_col = _find_label_column(chunk)
        if label_col is None:
            continue

        chunk["mapped_label"] = chunk[label_col].apply(map_to_6attacks)
        chunk = chunk[chunk["mapped_label"] != "Other"]

        # Attack Only
        if args.mode in ("attack_only", "both"):
            sub = chunk[chunk["mapped_label"].isin(ATTACK_CLASSES)].copy()
            if not sub.empty:
                sub = _ensure_features(sub)
                for cls, group in sub.groupby("mapped_label"):
                    cap = caps_attack_only.get(cls, args.attack_cap)
                    if collected_attack[cls] >= cap:
                        continue
                    needed = cap - collected_attack[cls]
                    take = group.sample(n=min(needed, len(group)), random_state=args.seed)
                    chunks_attack.append(take)
                    collected_attack[cls] += len(take)

        # With Benign
        if args.mode in ("with_benign", "both"):
            sub2 = chunk[chunk["mapped_label"].isin(ALL_CLASSES_WITH_BENIGN)].copy()
            if not sub2.empty:
                sub2 = _ensure_features(sub2)
                for cls, group in sub2.groupby("mapped_label"):
                    cap = caps_with_benign.get(cls, args.attack_cap)
                    if collected_with_benign[cls] >= cap:
                        continue
                    needed = cap - collected_with_benign[cls]
                    take = group.sample(n=min(needed, len(group)), random_state=args.seed)
                    chunks_with_benign.append(take)
                    collected_with_benign[cls] += len(take)

        stop_attack = all(collected_attack[c] >= caps_attack_only[c] for c in ATTACK_CLASSES)
        stop_with_benign = all(collected_with_benign[c] >= caps_with_benign[c] for c in ALL_CLASSES_WITH_BENIGN)

        if (args.mode == "both" and stop_attack and stop_with_benign) or \
           (args.mode == "attack_only" and stop_attack) or \
           (args.mode == "with_benign" and stop_with_benign):
            print(f"\n[INFO] Caps reached at chunk {i+1}. Stopping early.")
            break

    if args.mode in ("attack_only", "both"):
        df_attack = pd.concat(chunks_attack, ignore_index=True)
        df_attack = df_attack.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        df_attack.rename(columns={"mapped_label": "multiclass_label"}, inplace=True)
        print("\n[ATTACK-ONLY FINAL DISTRIBUTION]")
        print(df_attack["multiclass_label"].value_counts())
        df_attack.to_csv(args.out_attack, index=False)
        print(f"[SUCCESS] Saved: {args.out_attack}")

    if args.mode in ("with_benign", "both"):
        df_wb = pd.concat(chunks_with_benign, ignore_index=True)
        df_wb = df_wb.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        df_wb.rename(columns={"mapped_label": "multiclass_label"}, inplace=True)
        df_wb.to_csv(args.out_with_benign, index=False)
        print(f"[SUCCESS] Saved: {args.out_with_benign}")

if __name__ == "__main__":
    main()