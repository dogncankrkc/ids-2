"""
Binary IDS Trainer and Benchmark Runner.

This module orchestrates training, validation, threshold tuning,
and final evaluation of multiple classical ML models for
Stage-1 Binary IDS (Benign vs Attack).

Key responsibilities:
- Train multiple binary models sequentially
- Tune decision threshold on validation set (recall-oriented)
- Evaluate on test set with latency benchmarks
- Save comparable metrics for academic and production analysis
"""

import os
import time
import yaml
import csv
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.models.binary_models import (
    create_binary_model,
    train_binary_model,
    evaluate_binary_model,
)

def _to_native(obj):
    """Recursively convert numpy types to native Python types for YAML/JSON."""
    import numpy as _np

    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    return obj


# ============================================================
# THRESHOLD TUNING
# ============================================================

def tune_threshold(y_true, probs, thresholds, optimize_for="f1"):
    """
    Tune classification threshold on validation set.
    Supports: 'recall', 'precision', 'f1', 'f2' (recall-weighted)
    """
    best_score = -1.0
    best_threshold = 0.5
    best_metrics = {}

    # optimize_for 'recall' ise bile precision kontrolü koymak gerekir
    # ama şimdilik metric değişimine odaklanalım.
    
    for t in thresholds:
        preds = (probs >= t).astype(int)

        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        
        # F2 Score: Recall'a Precision'dan 2 kat daha fazla önem verir (IDS için ideal)
        denom = (4 * precision + recall)
        f2 = (5 * precision * recall) / denom if denom > 0 else 0.0

        if optimize_for == "recall":
            score = recall
        elif optimize_for == "precision":
            score = precision
        elif optimize_for == "f2":
            score = f2
        else: # default f1
            score = f1

        # NOT: Eğer skorlar eşitse, daha yüksek threshold'u (daha az False Positive) 
        # tercih etmek genellikle daha güvenlidir, o yüzden > yerine >= kullanılabilir
        # ama en yüksek skoru arıyoruz.
        if score > best_score:
            best_score = score
            best_threshold = t
            best_metrics = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "f2_score": f2
            }

    return best_threshold, best_metrics

# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def run_binary_benchmark(
    config: dict,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
):
    """
    Run binary IDS benchmark for all enabled models.
    """

    output_dir = config["output"]["base_dir"]
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    csv_rows = []

    # 🔹 NEW: Track best model globally
    best_model_name = None
    best_model_recall = -1.0

    threshold_cfg = config["evaluation"]["threshold"]
    thresholds = np.arange(
        threshold_cfg["range"][0],
        threshold_cfg["range"][1] + 1e-6,
        threshold_cfg["step"]
    )

    for model_name, model_cfg in config["models"].items():
        if not model_cfg.get("enabled", False):
            continue

        print("\n" + "=" * 60)
        print(f"TRAINING BINARY MODEL: {model_name.upper()}")
        print("=" * 60)

        # --------------------------------------------------
        # 1. Model initialization
        # --------------------------------------------------
        model = create_binary_model(
            model_name=model_name,
            params=model_cfg["params"]
        )

        # --------------------------------------------------
        # 2. Training
        # --------------------------------------------------
        start_train = time.time()
        model = train_binary_model(model, X_train, y_train)
        train_time = time.time() - start_train

        # --------------------------------------------------
        # 3. Validation (threshold tuning)
        # --------------------------------------------------
        val_eval = evaluate_binary_model(model, X_val, y_val)
        val_probs = val_eval["probabilities"]

        best_threshold, val_metrics = tune_threshold(
            y_true=y_val,
            probs=val_probs,
            thresholds=thresholds,
            optimize_for=threshold_cfg["optimize_for"],
        )

        print(f"[INFO] Best threshold: {best_threshold:.2f}")
        print(f"[INFO] Val Recall   : {val_metrics['recall']:.4f}")

        # 🔹 NEW: Save threshold separately
        threshold_path = os.path.join(output_dir, f"{model_name}_threshold.yaml")
        with open(threshold_path, "w") as f:
            yaml.safe_dump(
                {
                    "model": model_name,
                    "threshold": float(best_threshold),
                    "optimized_for": threshold_cfg["optimize_for"],
                },
                f
            )

        # --------------------------------------------------
        # 4. Test evaluation
        # --------------------------------------------------
        test_eval = evaluate_binary_model(model, X_test, y_test)
        test_probs = test_eval["probabilities"]
        test_preds = (test_probs >= best_threshold).astype(int)

        acc = accuracy_score(y_test, test_preds)
        prec = precision_score(y_test, test_preds, zero_division=0)
        rec = recall_score(y_test, test_preds, zero_division=0)
        f1 = f1_score(y_test, test_preds, zero_division=0)
        roc = roc_auc_score(y_test, test_probs)

        cm = confusion_matrix(y_test, test_preds)

        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn + 1e-9)
        fnr = fn / (fn + tp + 1e-9)

        # --------------------------------------------------
        # 5. Aggregate results (GÜNCELLENMİŞ VERSİYON)
        # --------------------------------------------------
        
        # Test seti için F2 Score hesapla
        denom_f2 = (4 * prec + rec)
        f2_score_test = (5 * prec * rec) / denom_f2 if denom_f2 > 0 else 0.0

        model_result = {
            "model": model_name,
            "threshold": best_threshold,
            "train_time_sec": train_time,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "f2_score": f2_score_test,    # <--- Rapora F2 eklendi
            "roc_auc": roc,
            "false_positive_rate": fpr,   
            "false_negative_rate": fnr,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "confusion_matrix": cm.tolist(),
            "latency": {
                "inference_time_sec": test_eval["inference_time_sec"],
                "time_per_sample_ms": test_eval["time_per_sample_ms"],
                "samples_per_sec": test_eval["samples_per_sec"],
            },
        }

        results[model_name] = model_result

        csv_rows.append({
            "model": model_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "f2_score": f2_score_test,  # <--- CSV'ye de eklendi
            "roc_auc": roc,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "threshold": best_threshold,
            "samples_per_sec": test_eval["samples_per_sec"],
        })

        # --------------------------------------------------
        # BEST MODEL SEÇİM MANTIĞI (DÜZELTİLDİ)
        # --------------------------------------------------
        # Config'deki 'optimize_for' neyse ona göre kıyasla
        target_metric = threshold_cfg.get("optimize_for", "recall")
        
        current_score = 0.0
        if target_metric == "f2":
            current_score = f2_score_test
        elif target_metric == "f1":
            current_score = f1
        elif target_metric == "precision":
            current_score = prec
        else:
            current_score = rec 

        # En iyi skoru güncelliyoruz
        # (Değişken adı best_model_recall kalsa da içine artık F2 yazıyoruz)
        if current_score > best_model_recall:
            best_model_recall = current_score
            best_model_name = model_name

        # --------------------------------------------------
        # 6. Save individual model (PRİNTLER GERİ GELDİ)
        # --------------------------------------------------
        if config["output"].get("save_models", True):
            model_path = os.path.join(output_dir, f"{model_name}_binary_model.pkl")
            import joblib
            joblib.dump(model, model_path)
            
            # BURAYA F2 DE EKLENDİ
            print(f"[TEST] Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} F2={f2_score_test:.4f} ROC-AUC={roc:.4f}")
            print(f"[TEST] FPR={fpr:.4f} FNR={fnr:.4f} | TN={tn} FP={fp} FN={fn} TP={tp}")
            print(f"[LAT ] {test_eval['samples_per_sec']:.0f} samples/sec | {test_eval['time_per_sample_ms']:.6f} ms/sample")

            print(f"[SAVE] Model saved to {model_path}")

    # ------------------------------------------------------
    # SAVE AGGREGATED RESULTS
    # ------------------------------------------------------
    yaml_path = os.path.join(output_dir, "binary_benchmark_results.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(_to_native(results), f, sort_keys=False)

    csv_path = os.path.join(output_dir, "binary_benchmark_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=csv_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    # 🔹 NEW: Save best model summary
    best_model_path = os.path.join(output_dir, "best_binary_model.yaml")
    with open(best_model_path, "w") as f:
        yaml.safe_dump(
            {
                "best_model": best_model_name,
                "criterion": threshold_cfg["optimize_for"],
                "recall": float(best_model_recall),
            },
            f
        )

    print("\n" + "=" * 60)
    print("BINARY BENCHMARK COMPLETED")
    print("=" * 60)
    print(f"[BEST MODEL] {best_model_name.upper()} | Recall: {best_model_recall:.4f}")
    print(f"[RESULTS] YAML: {yaml_path}")
    print(f"[RESULTS] CSV : {csv_path}")

    return results
