"""
IDS – Standalone Test Script
----------------------------
Bu dosya eğitim yapmaz. Sadece diske kaydedilmiş "best_model.pth"
ve "test_split_saved.csv" dosyalarını yükleyerek final performans
ve hız (latency) testlerini gerçekleştirir.
"""

import os
import sys

# ==========================================
# PATH DÜZELTME (IMPORT HATASINI ÇÖZER)
# ==========================================
# Scriptin bulunduğu klasörden iki adım geriye (proje kök dizinine) git
# ve Python'un arama yollarına ekle.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

# ==========================================
# IMPORTLAR (PATH EKLENDİKTEN SONRA)
# ==========================================
import torch
import joblib
import yaml
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

# Kendi modüllerini import et
from src.models.cnn_model import create_ids_model
from src.training.trainer import Trainer
from src.utils.visualization import plot_confusion_matrix
from src.utils.helpers import get_device

# ==============================
# AYARLAR (CONFIG)
# ==============================
# Dosya yollarını proje kök dizinine göre dinamik yapıyoruz
CONFIG_PATH = os.path.join(project_root, "configs/multiclass_config.yaml")
TEST_DATA_PATH = os.path.join(project_root, "data/processed/test_split_saved.csv")
ENCODER_PATH = os.path.join(project_root, "models/label_encoder.pkl")
OUTPUT_DIR = os.path.join(project_root, "outputs/test_results")
CHECKPOINT_DIR = os.path.join(project_root, "models/checkpoints/ids_multiclass")

# Eğer config dosyasından okuyamazsa default değerler
DEFAULT_BATCH_SIZE = 128
MODE = "multiclass"  # 'binary' veya 'multiclass'

def load_config(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    print(f"[WARN] Config bulunamadı: {path}, varsayılanlar kullanılacak.")
    return {}

def main():
    # 1. Hazırlık
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    config = load_config(CONFIG_PATH)
    
    # Config'den değerleri al veya default kullan
    batch_size = config.get("data", {}).get("batch_size", DEFAULT_BATCH_SIZE)
    # Config dosyasındaki path göreceli olabilir, biz yukarıda tanımladığımız CHECKPOINT_DIR'i kullanalım
    model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    print(f"\n[INFO] Device: {device}")
    print(f"[INFO] Model Path: {model_path}")
    print(f"[INFO] Test Data: {TEST_DATA_PATH}")

    # 2. Label Encoder Yükle (Sınıf isimleri için)
    if os.path.exists(ENCODER_PATH):
        encoder = joblib.load(ENCODER_PATH)
        classes = encoder.classes_
        num_classes = len(classes)
        inv_label_map = {i: label for i, label in enumerate(classes)}
        print(f"[INFO] Sınıflar yüklendi: {num_classes} adet")
    else:
        # Encoder yoksa manuel fallback
        print("[WARN] Label encoder bulunamadı, varsayılan sınıflar atanıyor.")
        num_classes = 2 if MODE == "binary" else 8
        inv_label_map = {i: str(i) for i in range(num_classes)}
        classes = list(inv_label_map.values())

    # 3. Test Verisini Yükle
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test verisi bulunamadı: {TEST_DATA_PATH}")
    
    df_test = pd.read_csv(TEST_DATA_PATH)
    
    # Veriyi Tensor'a çevir
    X_np = df_test.drop(columns=["label"]).values
    y_np = df_test["label"].values

    X_test = torch.tensor(X_np, dtype=torch.float32)
    y_test = torch.tensor(y_np, dtype=torch.long)

    # Model 1D CNN ise boyut ekle: (Batch, Features) -> (Batch, 1, Features)
    if X_test.ndim == 2:
        X_test = X_test.unsqueeze(1) 

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"[INFO] Test veri seti hazır: {len(test_dataset)} örnek")

    # 4. Modeli Oluştur ve Ağırlıkları Yükle
    model = create_ids_model(mode=MODE, num_classes=num_classes)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası yok: {model_path} - Önce eğitim yapın.")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("[INFO] Model başarıyla yüklendi.")

    # 5. Trainer ile Test (Trainer sınıfındaki test fonksiyonunu kullanıyoruz)
    trainer = Trainer(
        model=model,
        criterion=nn.CrossEntropyLoss(), # Test için zorunlu değil ama bulunsun
        optimizer=torch.optim.Adam(model.parameters()), # Dummy
        device=device,
        checkpoint_dir=CHECKPOINT_DIR
    )

    print("\n" + "="*40)
    print("🚀 BAŞLATILIYOR: LATENCY VE PERFORMANS TESTİ")
    print("="*40)
    
    # Trainer içindeki test metodunu çağır
    results = trainer.test(test_loader)

    # Sonuçları YAML olarak kaydet
    results_path = os.path.join(OUTPUT_DIR, "final_metrics.yaml")
    with open(results_path, "w") as f:
        yaml.safe_dump(results, f)
    print(f"[SAVE] Metrikler kaydedildi: {results_path}")

    # 6. Detaylı Analiz: Confusion Matrix ve CSV Çıktısı
    print("\n[INFO] Detaylı tahminler ve Confusion Matrix hazırlanıyor...")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    # CSV Kaydet
    results_df = pd.DataFrame({
        "True_Label_ID": all_targets,
        "Pred_Label_ID": all_preds,
        "True_Label_Name": [inv_label_map.get(i, str(i)) for i in all_targets],
        "Pred_Label_Name": [inv_label_map.get(i, str(i)) for i in all_preds]
    })
    results_df["Is_Correct"] = results_df["True_Label_ID"] == results_df["Pred_Label_ID"]
    
    csv_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"[SAVE] Tahmin detayları CSV: {csv_path}")

    # Confusion Matrix Çiz
    cm = confusion_matrix(all_targets, all_preds)
    
    plot_confusion_matrix(
        cm, 
        class_names=classes, 
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_norm.png"),
        normalize=True
    )
    
    plot_confusion_matrix(
        cm, 
        class_names=classes, 
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_count.png"),
        normalize=False
    )
    
    print(f"[SAVE] Confusion matrix görselleri kaydedildi: {OUTPUT_DIR}")
    print("\n✅ TEST İŞLEMİ TAMAMLANDI.")

if __name__ == "__main__":
    main()