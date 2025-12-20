"""
Binary IDS Training Script (Hierarchical IDS – Stage 1).

This script runs the complete binary IDS pipeline:
1. Load binary configuration
2. Load and preprocess dataset (benign vs attack)
3. Train and benchmark multiple ML models
4. Tune thresholds (recall-oriented)
5. Save models and evaluation results

Purpose:
- Provide a fast, edge-ready binary IDS gatekeeper
- Act as the first stage in a hierarchical IDS (Binary → Multiclass)
"""

import os
import yaml
import argparse
import pandas as pd

from src.data.preprocess_binary import preprocess_binary
from src.training.binary_trainer import run_binary_benchmark

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG LOADER
# ============================================================

def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to YAML config

    Returns:
        dict: Parsed configuration
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Binary IDS Trainer")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/binary_config.yaml",
        help="Path to binary IDS config file",
    )
    args = parser.parse_args()

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_config(args.config)

    print("\n" + "=" * 60)
    print("STARTING BINARY IDS PIPELINE")
    print("=" * 60)
    print(f"[INFO] Experiment: {config['experiment']['name']}")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    data_path = config["data"]["input_path"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"[INFO] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------
    print("\n[STEP] Binary preprocessing")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_binary(
        df_real=df,
        use_gan=config["data"]["augmentation"]["enabled"],
    )

    # --------------------------------------------------------
    # Benchmark training
    # --------------------------------------------------------
    print("\n[STEP] Running binary model benchmark")
    results = run_binary_benchmark(
        config=config,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("BINARY IDS PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    best_model = max(
        results.items(),
        key=lambda x: x[1]["recall"]
    )

    print(f"[BEST MODEL] {best_model[0].upper()}")
    print(f"[BEST RECALL] {best_model[1]['recall']:.4f}")
    print(f"[THRESHOLD ] {best_model[1]['threshold']:.2f}")


if __name__ == "__main__":
    main()
