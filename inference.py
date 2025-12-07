"""
IDS Inference Script – Works with trained IDS_CNN model
Dynamic Label Loading Version
"""

import torch
import pandas as pd
import joblib  # <-- EKLENDİ
import os

from src.models.cnn_model import IDS_CNN
from src.utils.helpers import get_device
from src.data.preprocess import preprocess_single_sample

# Global değişkeni boş bırakıyoruz, dinamik dolduracağız
INV_LABEL_MAP = {}

def load_label_map(encoder_path: str):
    """
    Eğitim sırasında kaydedilen encoder'ı yükler ve haritayı oluşturur.
    Böylece sıralama hatası (alphabetical mismatch) olmaz.
    """
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder bulunamadı: {encoder_path}")
    
    encoder = joblib.load(encoder_path)
    # classes_ özelliği ['benign', 'bruteforce', ...] şeklinde sıralı gelir.
    # index 0 -> benign, index 1 -> bruteforce...
    
    global INV_LABEL_MAP
    INV_LABEL_MAP = {i: label for i, label in enumerate(encoder.classes_)}
    print(f"[INFO] Label Map Yüklendi: {INV_LABEL_MAP}")

def load_model(model_path: str, num_classes: int):
    device = get_device()
    model = IDS_CNN(num_classes=num_classes).to(device)

    # map_location, GPU'da eğitilip CPU'da çalıştırılırsa hata vermemesi için önemli
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model, device

def predict(model: IDS_CNN, x_tensor: torch.Tensor, device: torch.device):
    with torch.no_grad():
        outputs = model(x_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = probs.max(dim=1)

    # Dinamik haritadan çekiyoruz
    label = INV_LABEL_MAP.get(pred_idx.item(), "Unknown")
    return label, conf.item() * 100.0

if __name__ == "__main__":
    # ==== Dosya Yolları ====
    sample_path  = "data/inference/sample.csv"
    model_path   = "models/best_ids_model.pth"
    encoder_path = "models/label_encoder.pkl"  # preprocess.py'nin kaydettiği dosya

    # ==== 1) Label Map'i Yükle ====
    load_label_map(encoder_path)
    num_classes = len(INV_LABEL_MAP)

    # ==== 2) Load Data & Preprocess ====
    df = pd.read_csv(sample_path)
    x = preprocess_single_sample(df)

    # ==== 3) Load Model ====
    model, device = load_model(model_path, num_classes=num_classes)

    # ==== 4) Predict ====
    label, conf = predict(model, x, device)
    print(f"Prediction → {label} | Confidence: {conf:.2f}%")