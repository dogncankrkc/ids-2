# 📌 test_ids_generate_csv.py  (src/testing içine kaydet)

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# === MODEL & METRICS ===
from src.models.cnn_model import create_ids_model
from src.training.metrics import accuracy, precision, recall, f1_score
from src.data.preprocess import preprocess_multiclass

# ==============================
# LABEL MAP  (SON HALİ 🔽)
# ==============================
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

# ==============================
# CONFIG PATHS
# ==============================
MODEL_PATH = "models/checkpoints/ids_multiclass/final_model.pth"
TEST_CSV   = "data/raw/datasense_MASTER_FULL-6.csv"
OUTPUT_CSV = "outputs/prediction_results_multiclass.csv"


# ==============================
# 1) LOAD MODEL
# ==============================
def load_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = create_ids_model(mode="multiclass", num_classes=len(INV_LABEL_MAP))
    model.to(device)

    # 🔥 DUMMY SAFE INIT
    dummy = torch.randn(1, 1, 7, 10, device=device)
    _ = model(dummy)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device

# ==============================
# 2) LOAD TEST DATA
# ==============================
def load_test_data():
    df = pd.read_csv(TEST_CSV)
    _, X_test, _, y_test = preprocess_multiclass(df)

    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
    y_test = torch.tensor(y_test, dtype=torch.long)

    dataset = TensorDataset(X_test, y_test)
    return DataLoader(dataset, batch_size=32, shuffle=False), df


# ==============================
# 3) EVALUATE + SAVE CSV
# ==============================
def evaluate_and_save():
    model, device = load_model()
    loader, df_raw = load_test_data()

    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)

            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            max_conf = torch.max(probs, dim=1).values

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_probs.append(max_conf.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()

    # ==============================
    # KOLONLARI LABEL MAP İLE OLUŞTUR
    # ==============================
    df_results = pd.DataFrame({
        "row_index": range(len(all_targets)),
        "real_label_id": all_targets,
        "real_label_name": [INV_LABEL_MAP[i] for i in all_targets],
        "pred_label_id": all_preds,
        "pred_label_name": [INV_LABEL_MAP[i] for i in all_preds],
        "confidence": all_probs,
        "is_correct": (all_preds == all_targets).astype(int),
    })

    os.makedirs("outputs", exist_ok=True)
    df_results.to_csv(OUTPUT_CSV, index=False)

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy:  {accuracy(torch.tensor(all_preds), torch.tensor(all_targets)):.2f}%")
    print(f"Precision: {precision(torch.tensor(all_preds), torch.tensor(all_targets)):.4f}")
    print(f"Recall:    {recall(torch.tensor(all_preds), torch.tensor(all_targets)):.4f}")
    print(f"F1 Score:  {f1_score(torch.tensor(all_preds), torch.tensor(all_targets)):.4f}")
    print(f"\n📁 CSV KAYDEDİLDİ: {OUTPUT_CSV} ✓")


if __name__ == "__main__":
    evaluate_and_save()
