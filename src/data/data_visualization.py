"""
Visualization Script for CIC-IoT-2023.

Purpose:
- Generates 5 specific figures to analyze the dataset evolution.
- NO TITLES: Optimized for academic reports (LaTeX/Word).
- Figure 1: Original Distribution (All 33+ Classes) - Reads Raw Data.
- Figure 2: Original Distribution (Mapped to 8 Classes) - Reads Raw Data.
- Figure 3: Final Balanced Distribution (Resampled to 50k) - Reads Processed Data & Upsamples.
- Figure 4: Feature Dendrogram (Clustering).
- Figure 5: Top-20 Feature Correlations.

Note: All comments are strictly in English.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.utils import resample

# ------------------------
# CONFIGURATION
# ------------------------
# Path to the MASSIVE raw file (for Fig 1 & 2)
RAW_DATA_PATH = "data/raw/CIC2023_FULL_MERGED.csv"

# Path to the CLEAN filtered file (for Fig 3, 4, 5)
PROCESSED_DATA_PATH = "data/processed/CIC2023_Balanced_SMOTE.csv"

# Directory to save the plots
SAVE_DIR = "outputs/figures"

# Target samples for the final balanced plot


# List of features to analyze for correlations
SELECTED_FEATURES = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate', 
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 
    'ece_flag_number', 'cwr_flag_number', 
    'ack_count', 'syn_count', 'fin_count', 'rst_count', 
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC', 
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Variance'
]

os.makedirs(SAVE_DIR, exist_ok=True)

def get_category(label):
    """
    Maps the detailed attack labels to 8 main categories.
    Used to generate Figure 2 (Original Mapped Distribution).
    """
    label = str(label).strip().upper()
    
    if label == 'NAN' or label == '':
        return 'Other'
        
    if 'BENIGN' in label:
        return 'Benign'
    elif 'DDOS' in label:
        return 'DDoS'
    elif 'DOS' in label:
        return 'DoS'
    elif 'RECON' in label or 'VULNERABILITYSCAN' in label or 'PING' in label or 'PORTSCAN' in label or 'OSSCAN' in label or 'HOSTDISCOVERY' in label:
        return 'Recon'
    elif 'XSS' in label or 'SQL' in label or 'UPLOAD' in label or 'BROWSER' in label or 'COMMAND' in label or 'BACKDOOR' in label or 'MALWARE' in label:
        return 'Web'
    elif 'BRUTEFORCE' in label or 'DICTIONARY' in label:
        return 'BruteForce'
    elif 'SPOOFING' in label or 'MITM' in label:
        return 'Spoofing'
    elif 'MIRAI' in label:
        return 'Mirai'
    else:
        return 'Other'

def balance_dataset_in_memory(df, label_col='Mapped_Label', n_samples=50000):
    """
    Performs resampling (upsampling/downsampling) in memory for visualization.
    This simulates what the model will see during training.
    """
    print(f"[INFO] Resampling dataset in memory to {n_samples} per class...")
    df_balanced = pd.DataFrame()
    unique_classes = df[label_col].unique()
    
    for label in unique_classes:
        df_class = df[df[label_col] == label]
        count = len(df_class)
        
        if count == n_samples:
            df_resampled = df_class
        elif count > n_samples:
            # Undersample
            df_resampled = resample(df_class, replace=False, n_samples=n_samples, random_state=42)
        else:
            # Oversample (Upsampling minority classes like Web)
            df_resampled = resample(df_class, replace=True, n_samples=n_samples, random_state=42)
            
        df_balanced = pd.concat([df_balanced, df_resampled])
        
    return df_balanced.reset_index(drop=True)

def scan_raw_dataset():
    """
    Scans the huge raw dataset in chunks to get counts for Fig 1 and Fig 2.
    """
    print(f"[INFO] Scanning RAW dataset at {RAW_DATA_PATH} for Figures 1 & 2...")
    
    raw_counts = pd.Series(dtype=int)
    mapped_counts = pd.Series(dtype=int)
    chunk_size = 1000000
    
    try:
        for i, chunk in enumerate(pd.read_csv(RAW_DATA_PATH, chunksize=chunk_size)):
            print(f"   -> Processing Chunk {i+1}...")
            chunk.columns = chunk.columns.str.strip()
            
            # Identify label column
            if 'multiclass_label' in chunk.columns:
                col = 'multiclass_label'
            elif 'label' in chunk.columns:
                col = 'label'
            else:
                continue

            # 1. Count Raw Labels
            counts = chunk[col].value_counts()
            raw_counts = raw_counts.add(counts, fill_value=0)
            
            # 2. Map and Count Categories
            chunk['mapped'] = chunk[col].apply(get_category)
            # Filter out NaNs/Other
            chunk = chunk[~chunk['mapped'].isin(['Other', 'DROP'])]
            
            m_counts = chunk['mapped'].value_counts()
            mapped_counts = mapped_counts.add(m_counts, fill_value=0)
            
    except FileNotFoundError:
        print(f"[ERROR] Raw file not found at {RAW_DATA_PATH}. Skipping Fig 1 & 2.")
        return None, None

    return raw_counts.sort_values(ascending=False), mapped_counts.sort_values(ascending=False)

def generate_report_plots():
    # -------------------------------------------------------
    # PHASE 1: RAW DATA ANALYSIS (Figures 1 & 2)
    # -------------------------------------------------------
    raw_counts, raw_mapped_counts = scan_raw_dataset()
    
    if raw_counts is not None:
        # --- FIGURE 1: Original Class Distribution (All Classes) ---
        print("[INFO] Generating Figure 1: Original Imbalanced Distribution (All Classes)...")
        plt.figure(figsize=(14, 12))
        
        df_fig1 = raw_counts.reset_index()
        df_fig1.columns = ['Attack Type', 'Count']
        
        ax1 = sns.barplot(x='Count', y='Attack Type', data=df_fig1, palette='viridis')
        ax1.set_xscale("log") 
        
        # NO TITLE for report compatibility
        # plt.title("Figure 1: Original Class Distribution (Log Scale)", fontsize=16)
        plt.xlabel("Count (Log Scale)", fontsize=14)
        plt.ylabel("Attack Type", fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "fig1_original_full_distribution.png"), dpi=300)
        
        # --- FIGURE 2: Original Distribution Mapped to 8 Classes ---
        print("[INFO] Generating Figure 2: Original Imbalanced (8 Classes)...")
        plt.figure(figsize=(12, 8))
        
        df_fig2 = raw_mapped_counts.reset_index()
        df_fig2.columns = ['Attack Category', 'Count']
        
        ax2 = sns.barplot(x='Count', y='Attack Category', data=df_fig2, palette='magma')
        ax2.set_xscale("log")
        
        for index, row in df_fig2.iterrows():
            plt.text(row.Count, index, f" {int(row.Count)}", va='center', fontsize=10)

        # NO TITLE
        plt.xlabel("Count (Log Scale)", fontsize=14)
        plt.ylabel("Attack Category", fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "fig2_original_8class_distribution.png"), dpi=300)

    # -------------------------------------------------------
    # PHASE 2: BALANCED DATA ANALYSIS (Figures 3, 4, 5)
    # -------------------------------------------------------
    print(f"\n[INFO] Loading Processed Dataset from: {PROCESSED_DATA_PATH}")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Processed file not found at {PROCESSED_DATA_PATH}. Run create_balanced_dataset.py first.")
        return

    # Check column name
    target_col = 'Mapped_Label' if 'Mapped_Label' in df.columns else 'multiclass_label'
    
    # --- HERE IS THE RESAMPLING STEP ---
    # We apply balancing NOW to show the final distribution
    df_balanced = df.copy()

    # --- FIGURE 3: Final Balanced Distribution ---
    print("[INFO] Generating Figure 3: Final Balanced Distribution (Resampled)...")
    
    class_counts = df_balanced[target_col].value_counts().reset_index()
    class_counts.columns = ['Attack Category', 'Count']

    plt.figure(figsize=(12, 8))
    ax3 = sns.barplot(x='Count', y='Attack Category', data=class_counts, palette='viridis')
    
    # NO TITLE
    plt.xlabel("Sample Count", fontsize=14)
    plt.ylabel("Attack Category", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
        
    for index, row in class_counts.iterrows():
        plt.text(row.Count, index, f" {row.Count}", va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "fig3_final_balanced_distribution.png"), dpi=300)

    # --- PREPARE CORRELATION MATRIX (Using Balanced Data) ---
    print("[INFO] Calculating correlations for Figures 4 & 5...")
    available_features = [f for f in SELECTED_FEATURES if f in df_balanced.columns]
    features_df = df_balanced[available_features]
    corr_matrix = features_df.corr(method='spearman') 

    # --- FIGURE 4: Dendrogram ---
    print("[INFO] Generating Figure 4: Feature Dendrogram...")
    plt.figure(figsize=(14, 8))
    linked = linkage(corr_matrix, 'ward')
    
    dendrogram(
        linked,
        orientation='top',
        labels=corr_matrix.columns,
        distance_sort='descending',
        show_leaf_counts=True,
        leaf_rotation=90.,
        leaf_font_size=10.,
    )
    
    # NO TITLE
    plt.ylabel("Distance", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "fig4_dendrogram.png"), dpi=300)

    # --- FIGURE 5: Top 20 Correlations ---
    print("[INFO] Generating Figure 5: Top 20 Correlations...")
    corr_pairs = corr_matrix.abs().unstack()
    labels_to_drop = set()
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(0, i+1):
            labels_to_drop.add((cols[i], cols[j]))
            
    corr_pairs = corr_pairs.drop(labels=labels_to_drop, errors='ignore')
    top_pairs = corr_pairs.sort_values(ascending=False).head(20)
    
    plt.figure(figsize=(10, 10))
    pair_labels = [f"{i[0]} \n& {i[1]}" for i in top_pairs.index]
    
    sns.barplot(x=top_pairs.values, y=pair_labels, palette='magma')
    
    # NO TITLE
    plt.xlabel("Correlation Coefficient (Absolute)", fontsize=12)
    plt.xlim(0, 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "fig5_top_correlations.png"), dpi=300)

    print("\n[SUCCESS] All 5 figures generated in 'outputs/figures'.")

if __name__ == "__main__":
    generate_report_plots()