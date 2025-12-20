"""
Binary IDS Model Factory and Evaluation Utilities.

This module defines and initializes classical machine learning models
for Stage-1 Binary IDS (Benign vs Attack) in a hierarchical IDS pipeline.

Purpose:
- Provide fast, lightweight, and interpretable binary classifiers
- Serve as a gatekeeper before multiclass IDS inference
- Enable fair benchmarking under identical preprocessing and splits

Models included:
- Logistic Regression (ultra-fast baseline)
- Decision Tree (explainability reference)
- Random Forest (robust ensemble baseline)
- XGBoost (literature benchmark)
- LightGBM (edge-optimized boosting)
"""

from typing import Dict, Any
import time

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Optional heavy models (imported lazily in factory)
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None



# ============================================================
# MODEL FACTORY
# ============================================================

def create_binary_model(model_name: str, params: Dict[str, Any]):
    """
    Factory function to create a binary classification model.

    Args:
        model_name (str): Model identifier
        params (dict): Hyperparameters defined in config

    Returns:
        Instantiated model object
    """

    name = model_name.lower()
    
    if name == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("CatBoost is not installed. pip install catboost")
        return CatBoostClassifier(**params)

    if name == "logistic_regression":
        return LogisticRegression(**params)

    if name == "decision_tree":
        return DecisionTreeClassifier(**params)

    if name == "random_forest":
        return RandomForestClassifier(**params)

    if name == "xgboost":
        if xgb is None:
            raise ImportError("XGBoost is not installed.")
        return xgb.XGBClassifier(**params)

    if name == "lightgbm":
        if lgb is None:
            raise ImportError("LightGBM is not installed.")
        return lgb.LGBMClassifier(**params)

    raise ValueError(f"Unknown binary model type: {model_name}")


# ============================================================
# TRAINING UTILITY
# ============================================================

def train_binary_model(model, X_train, y_train):
    """
    Train a binary classification model.

    Args:
        model: Initialized model
        X_train (ndarray): Training features
        y_train (ndarray): Training labels

    Returns:
        Trained model
    """
    model.fit(X_train, y_train)
    return model


# ============================================================
# EVALUATION UTILITY
# ============================================================

def evaluate_binary_model(model, X, y):
    """
    Evaluate a binary model with latency measurement.

    Args:
        model: Trained model
        X (ndarray): Input features
        y (ndarray): Ground truth labels

    Returns:
        dict: Predictions, probabilities, and inference timing
    """

    start = time.time()

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
    else:
        preds = model.predict(X)
        probs = preds

    elapsed = time.time() - start

    return {
        "predictions": preds,
        "probabilities": probs,
        "inference_time_sec": elapsed,
        "samples": len(X),
        "time_per_sample_ms": (elapsed / len(X)) * 1000.0,
        "samples_per_sec": len(X) / elapsed if elapsed > 0 else 0.0,
    }
