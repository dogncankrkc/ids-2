import pandas as pd
import os
import sys

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Dosya İsimleri
REAL_FILE = "CIC2023_SEPARATE_ATTACK_ONLY.csv"
SYN_FILE = "GAN_SYNTHETIC_ONLY.csv"
OUTPUT_FILE = "CIC2023_BALANCED_100K.csv"

# --- DÜZELTME BURADA ---
# Senin CSV dosyanın sütun başlığı 'multiclass_label' olduğu için burayı güncelledik.
LABEL_COL = 'multiclass_label' 

TARGET_COUNT = 100000 

def finalize_dataset():
    print("--- DATASET BİRLEŞTİRME (HEDEF: 100K) ---")
    
    # 1. Dosyaları Yükle
    print(f"[INFO] Gerçek veri yükleniyor: {REAL_FILE}")
    try:
        df_real = pd.read_csv(os.path.join(DATA_DIR, REAL_FILE))
    except FileNotFoundError:
        print(f"[HATA] Dosya bulunamadı: {REAL_FILE}")
        return

    print(f"[INFO] GAN verisi yükleniyor: {SYN_FILE}")
    try:
        df_syn = pd.read_csv(os.path.join(DATA_DIR, SYN_FILE))
    except FileNotFoundError:
        print(f"[HATA] Dosya bulunamadı: {SYN_FILE}")
        return

    # Sütun Kontrolü (Hata almamak için önlem)
    if LABEL_COL not in df_real.columns:
        print(f"[KRİTİK HATA] '{LABEL_COL}' sütunu CSV içinde bulunamadı!")
        print(f"Mevcut sütunlar: {list(df_real.columns)}")
        return

    final_dfs = []
    classes = df_real[LABEL_COL].unique()

    print("-" * 80)
    print(f"{'Sınıf':<15} | {'Gerçek':<10} | {'Eklenecek (GAN)':<15} | {'Son Durum':<10}")
    print("-" * 80)

    for cls in classes:
        # Gerçek veriyi al
        real_subset = df_real[df_real[LABEL_COL] == cls]
        real_count = len(real_subset)
        
        final_dfs.append(real_subset)

        # Eğer 100.000'den azsa (Web veya BruteForce)
        if real_count < TARGET_COUNT:
            needed = TARGET_COUNT - real_count
            
            # GAN verisinden o sınıfı bul
            # NOT: GAN verisinde de sütun adının aynı (multiclass_label) olduğunu varsayıyoruz.
            # Eğer GAN verisinde sütun adı 'label' ise aşağıyı df_syn['label'] yapman gerekebilir.
            syn_subset = df_syn[df_syn[LABEL_COL] == cls]
            
            if len(syn_subset) > 0:
                syn_to_add = syn_subset.sample(n=needed, replace=True, random_state=42)
                final_dfs.append(syn_to_add)
                print(f"{cls:<15} | {real_count:<10} | {needed:<15} | {TARGET_COUNT:<10}")
            else:
                print(f"{cls:<15} | {real_count:<10} | {'GAN YOK!':<15} | {real_count:<10}")
        
        else:
            print(f"{cls:<15} | {real_count:<10} | {'0 (Tamam)':<15} | {real_count:<10}")

    # 3. Birleştir ve Karıştır
    print("-" * 80)
    df_final = pd.concat(final_dfs, ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"[SONUÇ] Final Tablo Boyutu: {len(df_final)}")
    
    save_path = os.path.join(DATA_DIR, OUTPUT_FILE)
    df_final.to_csv(save_path, index=False)
    print(f"[BAŞARILI] Kaydedildi: {save_path}")

if __name__ == "__main__":
    finalize_dataset()