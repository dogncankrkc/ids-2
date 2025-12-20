"""
Dataset Visualization Script (IEEE-Compliant)

Figures:
Fig.1 - Original Raw Dataset Distribution (Fine-grained labels)
Fig.2 - Raw Dataset Mapped to 7 Classes
Fig.3 - Training Distribution with GAN Augmentation (Real vs GAN)

IEEE Rules Applied:
- NO titles (captions used in paper)
- High DPI (300)
- Plain background (no seaborn styling artifacts)
- Colorblind-friendly colors
- Log-scale for imbalanced distributions
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# =============================
# GLOBAL IEEE STYLE
# =============================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "axes.grid": False
})

# =============================
# PATHS
# =============================
RAW_DATA_PATH = "data/raw/CIC2023_FULL_MERGED.csv"
PURE_REAL_TRAIN_PATH = "data/processed/train_PURE_REAL_preprocessed.csv"
FINAL_TRAIN_PATH = "data/processed/train_preprocessed.csv"

SAVE_DIR = "outputs/figures_ieee"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================
# CLASS MAPPING (7-Class)
# =============================
CLASS_ORDER = [
    "BruteForce", "DDoS", "DoS",
    "Mirai", "Recon", "Spoofing", "Web"
]

def map_to_7class(label: str):
    label = str(label).upper()
    if "BRUTE" in label:
        return "BruteForce"
    if "DDOS" in label:
        return "DDoS"
    if "DOS" in label:
        return "DoS"
    if "MIRAI" in label:
        return "Mirai"
    if "RECON" in label or "SCAN" in label:
        return "Recon"
    if "SPOOF" in label or "MITM" in label:
        return "Spoofing"
    if "WEB" in label or "SQL" in label or "XSS" in label:
        return "Web"
    return None

# =============================
# FIGURE 1 — RAW DISTRIBUTION
# =============================
def plot_raw_distribution():
    print("[FIG1] Processing raw dataset...")
    counts = Counter()

    for chunk in pd.read_csv(RAW_DATA_PATH, chunksize=1_000_000):
        label_col = "label" if "label" in chunk.columns else "multiclass_label"
        counts.update(chunk[label_col].value_counts().to_dict())

    df = (
        pd.DataFrame(counts.items(), columns=["Attack Type", "Count"])
        .sort_values("Count", ascending=False)
    )

    plt.figure(figsize=(8, 6))
    plt.barh(
        df["Attack Type"],
        df["Count"],
        color="#4C72B0"
    )
    plt.xscale("log")
    plt.xlabel("Sample Count (log scale)")
    plt.ylabel("Attack Type")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/fig1_raw_full_distribution.png", dpi=300)
    plt.close()

# =============================
# FIGURE 2 — RAW → 7 CLASS
# =============================
def plot_raw_7class_distribution():
    print("[FIG2] Mapping raw dataset to 7 classes...")
    counts = Counter()

    for chunk in pd.read_csv(RAW_DATA_PATH, chunksize=1_000_000):
        label_col = "label" if "label" in chunk.columns else "multiclass_label"
        mapped = chunk[label_col].apply(map_to_7class).dropna()
        counts.update(mapped.value_counts().to_dict())

    df = pd.DataFrame(counts.items(), columns=["Class", "Count"])
    df["Class"] = pd.Categorical(df["Class"], CLASS_ORDER)
    df = df.sort_values("Class")

    plt.figure(figsize=(6, 4))
    plt.barh(
        df["Class"],
        df["Count"],
        color="#55A868"
    )
    plt.xscale("log")
    plt.xlabel("Sample Count (log scale)")
    plt.ylabel("Attack Category")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/fig2_raw_7class_distribution.png", dpi=300)
    plt.close()

# =============================
# FIGURE 3 — GAN AUGMENTATION
# =============================
def plot_gan_augmentation():
    print("[FIG3] Visualizing GAN augmentation...")

    df_real = pd.read_csv(PURE_REAL_TRAIN_PATH)
    df_final = pd.read_csv(FINAL_TRAIN_PATH)

    real_counts = df_real["label"].value_counts().sort_index()
    final_counts = df_final["label"].value_counts().sort_index()

    all_classes = sorted(set(real_counts.index) | set(final_counts.index))
    real_counts = real_counts.reindex(all_classes, fill_value=0)
    final_counts = final_counts.reindex(all_classes, fill_value=0)

    gan_counts = final_counts - real_counts
    gan_counts[gan_counts < 0] = 0

    idx_to_class = {
        0: "BruteForce",
        1: "DDoS",
        2: "DoS",
        3: "Mirai",
        4: "Recon",
        5: "Spoofing",
        6: "Web",
    }

    class_names = [idx_to_class[i] for i in all_classes]

    plt.figure(figsize=(7, 4))
    plt.barh(
        class_names,
        real_counts.values,
        label="Real",
        color="#4C72B0"
    )
    plt.barh(
        class_names,
        gan_counts.values,
        left=real_counts.values,
        label="GAN",
        color="#DD8452"
    )

    plt.xlabel("Number of Training Samples (Real + GAN)")
    plt.ylabel("Attack Category")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/fig3_gan_augmentation.png", dpi=300)
    plt.close()

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    # Uncomment figures as needed
    plot_raw_distribution()
    plot_raw_7class_distribution()
    plot_gan_augmentation()

    print("[DONE] IEEE-compliant dataset figures generated.")
