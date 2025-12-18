"""
GAN DATA GENERATOR (Scientific & Leak-Free)

Goal:
1. Load the SEPARATE dataset.
2. Split into Train/Test (to ensure GAN never sees Test data).
3. Train CTGAN only on the TRAINING portion of 'Web' and 'BruteForce'.
4. Generate synthetic samples to reach ~100k total per class.
5. Save ONLY the synthetic data to a separate file.
"""

import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# ============================
# CONFIGURATION
# ============================
INPUT_PATH = "data/processed/CIC2023_SEPARATE_ATTACK_ONLY.csv"
OUTPUT_PATH = "data/processed/GAN_SYNTHETIC_ONLY.csv"

# Hedef Sınıflar ve Üretilecek Miktarlar (100k'ya tamamlamak için)
# Web (23k var) -> Train'e ~16k düşer. Hedef 70k eklemek.
# BF (12k var) -> Train'e ~8k düşer. Hedef 85k eklemek.
TARGETS = {
    "Web": 75000,       
    "BruteForce": 88000 
}

EPOCHS = 300  # Kaliteli üretim için ideal

def main():
    print(f"[INFO] Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    
    # -------------------------------------------------------
    # 1. LEAKAGE PREVENTION (Sızıntı Önleme)
    # -------------------------------------------------------
    # GAN'ı tüm veriyle eğitirsek, test setindeki verileri de görmüş olur.
    # Bu bilimsel bir hatadır. O yüzden önce split yapıyoruz.
    print("[INFO] Splitting data to isolate Training set for GAN...")
    
    # Stratified Split (%70 Train) - preprocess.py ile aynı mantık
    train_df, _ = train_test_split(
        df, test_size=0.30, stratify=df["multiclass_label"], random_state=42
    )
    
    generated_batches = []

    for cls, amount in TARGETS.items():
        print(f"\n" + "="*40)
        print(f"[GAN] Training for Class: {cls}")
        print("="*40)
        
        # Sadece TRAIN setindeki o sınıfı al
        subset = train_df[train_df["multiclass_label"] == cls].copy()
        print(f"[INFO] Training samples available: {len(subset)}")
        
        # Metadata oluştur
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(subset)
        
        # Modeli Başlat
        # M1/M2 Mac'lerde 'cuda' bazen sorun çıkarabilir, gerekirse 'cpu'ya dönebilirsin.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Device: {device}")
        
        model = CTGANSynthesizer(
            metadata, 
            epochs=EPOCHS, 
            verbose=True,
            cuda=(device == "cuda")
        )
        
        # Eğit
        model.fit(subset)
        
        # Üret
        print(f"[INFO] Generating {amount} synthetic samples...")
        syn_data = model.sample(num_rows=amount)
        
        # Etiketleri Garantiye Al
        syn_data["multiclass_label"] = cls
        
        generated_batches.append(syn_data)
        print(f"[SUCCESS] Generated {len(syn_data)} rows for {cls}")

    # Kaydet
    if generated_batches:
        print(f"\n[INFO] Saving synthetic data to {OUTPUT_PATH}...")
        df_gan = pd.concat(generated_batches, ignore_index=True)
        df_gan.to_csv(OUTPUT_PATH, index=False)
        print("[DONE] GAN synthetic data ready. Now you can merge it during training.")
    else:
        print("[ERROR] No data generated.")

if __name__ == "__main__":
    main()