"""
IDS Inference Script – Works with trained IDS_CNN model
"""

import torch
import pandas as pd
from pathlib import Path

from src.models.cnn_model import IDS_CNN
from src.utils.helpers import get_device
from src.data.preprocess import preprocess_single_sample  


# -------- LABEL MAP (Multiclass) --------
INV_LABEL_MAP = {
    0: "benign",
    1: "dos",
    2: "ddos",
    3: "recon",
    4: "mitm",
    5: "bruteforce",
    6: "web",
    7: "malware",
}
# ----------------------------------------


def load_model(model_path: str, num_classes: int):
    device = get_device()
    model = IDS_CNN(num_classes=num_classes).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model, device


def predict(model: IDS_CNN, x_tensor: torch.Tensor, device: torch.device):
    """
    Forward pass & get prediction
    """
    with torch.no_grad():
        outputs = model(x_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = probs.max(dim=1)

    label = INV_LABEL_MAP[pred_idx.item()]
    return label, conf.item() * 100.0


if __name__ == "__main__":
    # ==== 1) Example Input (single-row CSV) ====
    sample_path = "data/inference/sample.csv"  # must be 1 row!
    model_path  = "models/best_ids_model.pth"

    df = pd.read_csv(sample_path)

    # ==== 2) Preprocess ====
    x = preprocess_single_sample(df)    # → (1,1,7,10) tensor

    # ==== 3) Load Model ====
    model, device = load_model(model_path, num_classes=8)

    # ==== 4) Predict ====
    label, conf = predict(model, x, device)
    print(f"Prediction → {label} | Confidence: {conf:.2f}%")
