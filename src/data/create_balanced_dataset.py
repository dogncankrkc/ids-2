"""
FINAL DATASET CREATOR – CIC-IoT-2023 (MERGED EDITION)

Purpose:
1. Read huge raw CSV in chunks.
2. Map raw labels to 7 IDS classes (DoS & DDoS merged).
3. Cap dataset size (Strategic limits).
4. Save clean capped dataset for training.
"""

import os
import numpy as np
import pandas as pd

# ============================
# CONFIGURATION
# ============================

INPUT_PATH = "data/raw/CIC2023_FULL_MERGED.csv"
OUTPUT_PATH = "data/processed/CIC2023_CAPPED.csv"

CHUNK_SIZE = 1_000_000

# Stratejik Limitler:
# Benign'i gerçekçi tutuyoruz (250k).
# Saldırıları 100k ile sınırlıyoruz ki eğitim çok uzamasın.
BENIGN_CAP = 250_000
ATTACK_CAP = 100_000

# ------------------------------------------------------------
# SELECTED FEATURES (AVAILABLE IN CSV)
# ------------------------------------------------------------
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

# ============================
# LABEL MAPPING (STRATEGIC MERGE)
# ============================
def map_to_multiclass(label: str) -> str:
    label = str(label).strip().upper()

    if label == 'NAN' or label == '':
        return 'Other'
    
    # 1. Benign
    if 'BENIGN' in label:
        return 'Benign'
    
    # 2. DoS ve DDoS BİRLEŞİYOR -> 'DoS-DDoS'
    # Teknik Açıklama: Elimizdeki özellik setinde 'Source Rate' olmadığı için
    # model bu ikisini ayıramıyor. Performans için birleştirildi.
    if 'DDOS' in label or 'DOS' in label:
        return 'DoS-DDoS'
        
    # 3. Recon (Keşif)
    if any(x in label for x in ['RECON', 'VULNERABILITY', 'PING', 'PORTSCAN', 'OSSCAN', 'HOSTDISCOVERY']):
        return 'Recon'
        
    # 4. Web Saldırıları
    if any(x in label for x in ['XSS', 'SQL', 'UPLOAD', 'BROWSER', 'COMMAND', 'BACKDOOR', 'MALWARE']):
        return 'Web'
        
    # 5. Kaba Kuvvet
    if 'BRUTEFORCE' in label or 'DICTIONARY' in label:
        return 'BruteForce'
        
    # 6. Spoofing
    if 'SPOOFING' in label or 'MITM' in label:
        return 'Spoofing'
        
    # 7. Mirai (IoT Botnet)
    if 'MIRAI' in label:
        return 'Mirai'
        
    return 'Other'

# ============================
# MAIN GENERATOR
# ============================
def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Sayaçları yeni sınıflara göre güncelle
    collected = {
        'Benign': 0, 
        'DoS-DDoS': 0, # Birleşmiş sınıf
        'Recon': 0,
        'Web': 0, 
        'BruteForce': 0, 
        'Spoofing': 0, 
        'Mirai': 0
    }
    
    chunks_to_save = []

    print(f"[INFO] Creating capped dataset (Merged Edition)...")
    print(f"[INFO] Target Caps -> Benign: {BENIGN_CAP}, Attacks: {ATTACK_CAP}")

    for i, chunk in enumerate(pd.read_csv(INPUT_PATH, chunksize=CHUNK_SIZE)):
        print(f" -> Processing chunk {i+1}", end="\r")

        # Etiket sütununu bul
        if "multiclass_label" in chunk.columns:
            label_col = "multiclass_label"
        elif "label" in chunk.columns:
            label_col = "label"
        else:
            continue

        # Dönüştür
        chunk["multiclass_label"] = chunk[label_col].apply(map_to_multiclass)

        # Temizle
        chunk = chunk[chunk["multiclass_label"] != "Other"]
        chunk = chunk[SELECTED_FEATURES + ["multiclass_label"]]
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(inplace=True)

        # Topla (Capping)
        for cls, group in chunk.groupby("multiclass_label"):
            cap = BENIGN_CAP if cls == "Benign" else ATTACK_CAP
            
            # Eğer 'DoS-DDoS' sınıfıysa, DoS ve DDoS toplamını kontrol et
            current_count = collected.get(cls, 0)
            if current_count >= cap:
                continue

            needed = cap - current_count
            take = group.head(needed)
            
            chunks_to_save.append(take)
            collected[cls] += len(take)

        # Erken Durdurma
        if all(val >= (BENIGN_CAP if key == "Benign" else ATTACK_CAP) for key, val in collected.items()):
            print(f"\n[INFO] Caps reached at chunk {i+1}. Stopping early.")
            break

    # Kaydet
    print("\n[INFO] Concatenating and saving...")
    if not chunks_to_save:
        print("[ERROR] No data collected!")
        return

    df_final = pd.concat(chunks_to_save, ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n[FINAL CLASS DISTRIBUTION]")
    print(df_final["multiclass_label"].value_counts())

    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SUCCESS] Dataset saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()