import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')

# File Names
REAL_DATA_FILE = "CIC2023_SEPARATE_ATTACK_ONLY.csv"
SYN_DATA_FILE = "GAN_SYNTHETIC_ONLY.csv"

# Label Column Name (Ensure this matches your CSV)
LABEL_COL = 'multiclass_label' 

def load_and_prep_data():
    print(f"[INFO] Loading: {REAL_DATA_FILE}...")
    try:
        df_real = pd.read_csv(os.path.join(DATA_DIR, REAL_DATA_FILE))
        # Label for the legend
        df_real['Dataset'] = 'Original' 
    except FileNotFoundError:
        print(f"[ERROR] File not found: {REAL_DATA_FILE}")
        sys.exit(1)

    print(f"[INFO] Loading: {SYN_DATA_FILE}...")
    try:
        df_syn = pd.read_csv(os.path.join(DATA_DIR, SYN_DATA_FILE))
        # Label for the legend
        df_syn['Dataset'] = 'Synthetic (GAN)'
    except FileNotFoundError:
        print(f"[ERROR] File not found: {SYN_DATA_FILE}")
        sys.exit(1)

    return df_real, df_syn

def plot_comparison(df_real, df_syn):
    # Combine datasets
    df_combined = pd.concat([df_real[[LABEL_COL, 'Dataset']], df_syn[[LABEL_COL, 'Dataset']]])

    # Count occurrences per class and dataset type
    data_counts = df_combined.groupby([LABEL_COL, 'Dataset']).size().reset_index(name='Count')
    
    # Sort order based on Original data counts (High to Low)
    order = df_real[LABEL_COL].value_counts().index

    # --- PLOTTING ---
    # Set seaborn theme for academic papers (cleaner look, larger fonts)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

    plt.figure(figsize=(14, 8))

    # Create Bar Plot
    ax = sns.barplot(
        data=data_counts,
        y=LABEL_COL,
        x='Count',
        hue='Dataset',
        order=order,
        palette="viridis",  # 'viridis' or 'mako' are often preferred in papers for clarity
        edgecolor="black",  # Add border to bars for better visibility
        linewidth=0.5
    )

    # Set Log Scale
    ax.set_xscale('log')

    # Titles and Labels (English)
    plt.title('Distribution of Attack Classes: Original vs. Synthetic Data', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Number of Samples (Log Scale)', fontsize=14, fontweight='bold')
    plt.ylabel('Attack Category', fontsize=14, fontweight='bold')
    
    # Legend adjustments
    plt.legend(title='Data Source', title_fontsize='13', fontsize='12', loc='lower right')

    # Add count labels to the end of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=5, fontsize=10)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save as PNG (High Res)
    png_path = os.path.join(OUTPUT_DIR, 'fig_gan_vs_real_distribution.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    # Save as PDF (Vector format - Best for Papers/LaTeX)
    pdf_path = os.path.join(OUTPUT_DIR, 'fig_gan_vs_real_distribution.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

    print(f"[SUCCESS] Plot saved to:\n  - {png_path}\n  - {pdf_path}")
    plt.show()

if __name__ == "__main__":
    real, syn = load_and_prep_data()
    
    # Column check
    if LABEL_COL not in real.columns:
        print(f"[ERROR] Column '{LABEL_COL}' not found in CSV.")
        print(f"Available columns: {list(real.columns)}")
    else:
        plot_comparison(real, syn)