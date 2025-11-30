"""
CSV Dataset Loader for IDS (Intrusion Detection System)

This module handles loading and merging CSV files from the raw data folder.
It provides a clean DataFrame as input for preprocessing.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_raw_csv(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load and concatenate all CSV files inside data/raw directory.

    Args:
        data_dir (str): Path to raw CSV folder.

    Returns:
        pd.DataFrame: Combined dataset with all CSV rows.
    """
    files = list(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    # Read and concatenate all CSV files
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    print(f"[INFO] Loaded {len(files)} CSV files → Total rows: {len(df)}")
    return df
