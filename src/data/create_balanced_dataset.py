"""
FINAL DATASET CREATOR – CIC-IoT-2023 (CAPPED, CLEAN)

Görevi:
1. Dev boyuttaki ham veriyi parça parça okur.
2. Etiketleri sadeleştirir (Map to 8 classes).
3. Belirlenen sayılarda (Cap) veriyi alıp 'data/processed' klasörüne kaydeder.
4. SMOTE veya Scaler BURADA YAPILMAZ (Onlar preprocess.py işi).
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

# Strateji: Benign'i makul seviyede tut, Atakları maksimum al.
# preprocess.py aşamasında Benign daha da azaltılacak (Undersample).
BENIGN_CAP = 250_000  
ATTACK_CAP = 100_000  # Varsa hepsini al, yoksa 100k'da dur.

# ------------------------------------------------------------
# SELECTED FEATURES
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
# LABEL MAPPING
# ============================
def map_to_multiclass(label: str) -> str:
    label = str(label).strip().upper()

    if label == 'NAN' or label == '': return 'Other'
    if 'BENIGN' in label: return 'Benign'
    if 'DDOS' in label: return 'DDoS'
    if 'DOS' in label: return 'DoS'
    if any(x in label for x in ['RECON', 'VULNERABILITY', 'PING', 'PORTSCAN', 'OSSCAN', 'HOSTDISCOVERY']): return 'Recon'
    if any(x in label for x in ['XSS', 'SQL', 'UPLOAD', 'BROWSER', 'COMMAND', 'BACKDOOR', 'MALWARE']): return 'Web'
    if 'BRUTEFORCE' in label or 'DICTIONARY' in label: return 'BruteForce'
    if 'SPOOFING' in label or 'MITM' in label: return 'Spoofing'
    if 'MIRAI' in label: return 'Mirai'
    return 'Other'

# ============================
# MAIN GENERATOR
# ============================
def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Sayaçlar
    collected = {
        'Benign': 0, 'DDoS': 0, 'DoS': 0, 'Recon': 0,
        'Web': 0, 'BruteForce': 0, 'Spoofing': 0, 'Mirai': 0
    }
    
    chunks_to_save = [] # Verileri burada toplayacağız

    print(f"[INFO] Dataset Creation Started. Caps -> Benign: {BENIGN_CAP}, Attack: {ATTACK_CAP}")

    # CSV'yi parça parça oku
    for i, chunk in enumerate(pd.read_csv(INPUT_PATH, chunksize=CHUNK_SIZE)):
        print(f" -> Processing Chunk {i+1}...", end="\r")

        # 1. Etiket Sütununu Bul
        if "multiclass_label" in chunk.columns:
            label_col = "multiclass_label"
        elif "label" in chunk.columns:
            label_col = "label"
        else:
            continue # Etiket yoksa bu parçayı geç

        # 2. Etiketleri Dönüştür (Mapping)
        chunk["multiclass_label"] = chunk[label_col].apply(map_to_multiclass)

        # 3. 'Other' olanları at
        chunk = chunk[chunk["multiclass_label"] != "Other"]

        # 4. Özellikleri Seç ve Temizle
        # Sadece seçili featureları al
        chunk = chunk[SELECTED_FEATURES + ["multiclass_label"]]
        
        # Sonsuz sayıları (inf) temizle
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(inplace=True)

        # 5. Capping (Limit koyma)
        for cls, group in chunk.groupby("multiclass_label"):
            target_cap = BENIGN_CAP if cls == "Benign" else ATTACK_CAP
            
            current_count = collected.get(cls, 0)
            if current_count >= target_cap:
                continue # Bu sınıf dolduysa alma

            needed = target_cap - current_count
            
            # İhtiyaç kadarını al
            take = group.head(needed) # veya .sample(n=min(len(group), needed))
            
            chunks_to_save.append(take)
            collected[cls] += len(take)

        # 6. Erken Durdurma (Her şey dolduysa boşuna okuma)
        if all(val >= (BENIGN_CAP if key == "Benign" else ATTACK_CAP) for key, val in collected.items()):
            print(f"\n[INFO] All caps reached at chunk {i+1}. Stopping.")
            break

    # ============================
    # SAVE
    # ============================
    print("\n[INFO] Concatenating and saving...")
    if not chunks_to_save:
        print("[ERROR] No data collected!")
        return

    df_final = pd.concat(chunks_to_save, ignore_index=True)
    
    # Karıştır
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n[FINAL DISTRIBUTION]")
    print(df_final["multiclass_label"].value_counts())

    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SUCCESS] Dataset saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()