"""
Binary Model Evaluation Script for IEEE Conference Paper (v2 - Automated).

Features:
1. Confusion Matrix (Standard Counts + Normalized Percentages)
2. ROC-AUC Curves (Comparison)
3. Latency vs F2-Score (Dynamic Data from CSV Benchmark)
4. Feature Importance (Selected Model: XGBoost)

Author: Dogan Can Karakoc
"""

import os
import yaml
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data/processed/binary/test_binary.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models/checkpoints/binary_ids")
BENCHMARK_CSV = os.path.join(MODELS_DIR, "binary_benchmark_summary.csv") # Benchmark sonuçları
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/figures_ieee/binary")

SELECTED_MODEL_KEY = "xgboost"

# IEEE Plot Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'legend.fontsize': 10,
})

# ============================================================
# UTILITIES
# ============================================================

def load_data():
    print(f"[INFO] Loading test data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Test data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    if "binary_label" in df.columns:
        y_test = df["binary_label"].values
        X_test = df.drop(columns=["binary_label"])
    else:
        y_test = df.iloc[:, -1].values
        X_test = df.iloc[:, :-1]
        
    return X_test, y_test

def load_model_and_threshold(model_name):
    model_path = os.path.join(MODELS_DIR, f"{model_name}_binary_model.pkl")
    config_path = os.path.join(MODELS_DIR, f"{model_name}_threshold.yaml")
    
    if not os.path.exists(model_path):
        print(f"[WARN] Model file not found: {model_path}")
        return None, 0.5

    model = joblib.load(model_path)
    
    threshold = 0.5
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            threshold = float(cfg.get("threshold", 0.5))
            
    return model, threshold

def format_model_name(name):
    """Converts 'logistic_regression' to 'Logistic Regression'"""
    return name.replace("_", " ").title().replace("Xgboost", "XGBoost").replace("Lightgbm", "LightGBM").replace("Catboost", "CatBoost")

# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_confusion_matrix_heatmap(y_true, probs, threshold, model_name, normalize=False):
    """Generates Standard or Normalized Confusion Matrix."""
    y_pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    title_suffix = ""
    fmt = ',d'
    filename_suffix = "counts"
    
    if normalize:
        # Normalize by True Labels (Rows)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title_suffix = " (Normalized)"
        fmt = '.2%'
        filename_suffix = "normalized"

    labels = ['Benign', 'Attack']
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
    
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.title(f'Confusion Matrix{title_suffix}: {model_name.upper()}')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, f"1_{model_name}_confusion_matrix_{filename_suffix}.png")
    plt.savefig(save_path)
    print(f"[PLOT] Saved Confusion Matrix ({filename_suffix}) -> {save_path}")
    plt.close()

def plot_roc_comparison(results, y_true):
    plt.figure(figsize=(8, 6))
    colors = sns.color_palette("husl", len(results))
    
    for i, (name, data) in enumerate(results.items()):
        probs = data['probs']
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        
        lw = 2.5 if name == SELECTED_MODEL_KEY else 1.5
        linestyle = '-' if name == SELECTED_MODEL_KEY else '--'
        label = f'{format_model_name(name)} (AUC = {roc_auc:.3f})'
        
        plt.plot(fpr, tpr, color=colors[i], lw=lw, linestyle=linestyle, label=label)

    plt.plot([0, 1], [0, 1], 'k:', lw=1)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "2_roc_comparison.png")
    plt.savefig(save_path)
    print(f"[PLOT] Saved ROC Comparison -> {save_path}")
    plt.close()

def plot_latency_tradeoff_dynamic():
    """
    Plots Latency vs F2-Score reading directly from benchmark CSV.
    Calculates Latency (us) from samples_per_sec.
    """
    if not os.path.exists(BENCHMARK_CSV):
        print(f"[ERROR] Benchmark CSV not found at {BENCHMARK_CSV}. Cannot plot latency tradeoff.")
        return

    df = pd.read_csv(BENCHMARK_CSV)
    
    # Gerekli sütunları kontrol et
    required_cols = ["model", "f2_score", "samples_per_sec"]
    if not all(col in df.columns for col in required_cols):
        print(f"[ERROR] CSV is missing columns. Found: {df.columns}. Expected: {required_cols}")
        return

    # Verileri hazırla
    models = df["model"].apply(format_model_name).tolist()
    f2_scores = df["f2_score"].values
    # Latency (mikrosaniye) = 1,000,000 / samples_per_sec
    latencies = (1_000_000 / df["samples_per_sec"]).values 

    plt.figure(figsize=(9, 6))
    
    # Tüm modelleri çiz
    plt.scatter(latencies, f2_scores, s=150, c='steelblue', alpha=0.8, edgecolors='k')
    
    # Seçili modeli (XGBoost) bul ve vurgula
    sel_name_formatted = format_model_name(SELECTED_MODEL_KEY)
    
    try:
        idx_sel = models.index(sel_name_formatted)
        plt.scatter([latencies[idx_sel]], [f2_scores[idx_sel]], s=250, c='crimson', 
                    edgecolors='black', linewidth=2, label=f'Selected ({sel_name_formatted})', zorder=10)
    except ValueError:
        print(f"[WARN] Selected model {SELECTED_MODEL_KEY} not found in CSV.")

    # İsimleri yazdır
    for i, txt in enumerate(models):
        offset = (5, 5)
        # Çakışmaları önlemek için basit manuel kaydırmalar
        if "LightGBM" in txt: offset = (5, -15)
        if "CatBoost" in txt: offset = (-30, 5)
            
        plt.annotate(txt, (latencies[i], f2_scores[i]), 
                     xytext=offset, textcoords='offset points', fontsize=10, fontweight='bold')

    plt.xlabel('Inference Latency (μs) [Log Scale] - Lower is Better', fontweight='bold')
    plt.ylabel('F2-Score - Higher is Better', fontweight='bold')
    plt.title('Performance vs. Latency Trade-off (Dynamic Data)')
    plt.xscale('log')
    
    from matplotlib.ticker import ScalarFormatter
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    
    plt.grid(True, linestyle='--', alpha=0.5, which="both")
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "3_latency_vs_f2.png")
    plt.savefig(save_path)
    print(f"[PLOT] Saved Latency Analysis (From CSV) -> {save_path}")
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette="viridis", hue=top_features, legend=False)
    
    plt.title(f'Top 15 Features: {format_model_name(model_name)}', fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, f"4_{model_name}_feature_importance.png")
    plt.savefig(save_path)
    print(f"[PLOT] Saved Feature Importance -> {save_path}")
    plt.close()

# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("STARTING IEEE FIGURE GENERATION (AUTOMATED)")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Test Data
    X_test, y_test = load_data()
    feature_names = list(X_test.columns)
    
    models_to_eval = ["logistic_regression", "decision_tree", "random_forest", "catboost", "xgboost", "lightgbm"]
    results = {}
    
    # 2. Process Models
    for m_name in models_to_eval:
        print(f"Processing {m_name}...")
        model, threshold = load_model_and_threshold(m_name)
        
        if model is None:
            continue
            
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.predict(X_test)
            
        results[m_name] = {"probs": probs, "threshold": threshold}
        
        # Sadece seçilen model için Confusion Matrix (Normal ve Normalized)
        if m_name == SELECTED_MODEL_KEY:
            # 1a. Counts
            plot_confusion_matrix_heatmap(y_test, probs, threshold, m_name, normalize=False)
            # 1b. Normalized (%)
            plot_confusion_matrix_heatmap(y_test, probs, threshold, m_name, normalize=True)
            # 4. Feature Importance
            plot_feature_importance(model, feature_names, m_name)

    # 3. Comparative Plots
    if results:
        plot_roc_comparison(results, y_test)
        plot_latency_tradeoff_dynamic() # CSV'den okuyarak çizer

    print("="*60)
    print(f"DONE. Figures saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()