"""
GAN Data Generator (Leak-Free & Scientific)

This script trains a CTGAN model on the training split of selected attack classes
(Web and BruteForce) to generate synthetic samples without test data leakage.
The generated data is saved separately and can later be merged during model training
to balance class distributions.
"""

import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# --------------------------------------------------
# Configuration
# --------------------------------------------------
INPUT_PATH = "data/processed/CIC2023_SEPARATE_ATTACK_ONLY.csv"
OUTPUT_PATH = "data/processed/GAN_SYNTHETIC_ONLY.csv"

# Target synthetic sample counts per class
TARGETS = {
    "Web": 75000,
    "BruteForce": 88000
}

# Number of CTGAN training epochs
EPOCHS = 300


def main():
    print(f"[INFO] Loading dataset from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    # --------------------------------------------------
    # Data leakage prevention
    # --------------------------------------------------
    # The GAN must never see test data.
    # A stratified split is applied before GAN training.
    print("[INFO] Performing stratified train split for GAN training")

    train_df, _ = train_test_split(
        df,
        test_size=0.30,
        stratify=df["multiclass_label"],
        random_state=42
    )

    generated_batches = []

    for cls, amount in TARGETS.items():
        print("\n" + "=" * 40)
        print(f"[GAN] Processing class: {cls}")
        print("=" * 40)

        # Select only training samples of the target class
        subset = train_df[train_df["multiclass_label"] == cls].copy()
        print(f"[INFO] Training samples available: {len(subset)}")

        # Create metadata for CTGAN
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(subset)

        # Select device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Training device: {device}")

        # Initialize CTGAN
        model = CTGANSynthesizer(
            metadata,
            epochs=EPOCHS,
            verbose=True,
            cuda=(device == "cuda")
        )

        # Train GAN
        model.fit(subset)

        # Generate synthetic samples
        print(f"[INFO] Generating {amount} synthetic samples")
        syn_data = model.sample(num_rows=amount)

        # Ensure correct class label
        syn_data["multiclass_label"] = cls

        generated_batches.append(syn_data)
        print(f"[SUCCESS] Generated {len(syn_data)} samples for class: {cls}")

    # Save synthetic dataset
    if generated_batches:
        print(f"\n[INFO] Saving synthetic dataset to: {OUTPUT_PATH}")
        df_gan = pd.concat(generated_batches, ignore_index=True)
        df_gan.to_csv(OUTPUT_PATH, index=False)
        print("[DONE] Synthetic GAN dataset successfully created")
    else:
        print("[ERROR] No synthetic data was generated")


if __name__ == "__main__":
    main()
