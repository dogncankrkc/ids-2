"""
IDS Inference Script (Live Prediction)
--------------------------------------
This script demonstrates how to load the trained model and make a prediction 
on a single data sample (simulating a real-time network packet).

Features:
- Loads the trained model and label encoder.
- Creates a dummy network packet (or loads from CSV).
- Preprocesses the single sample (Scaling).
- Returns the predicted class and confidence score.
"""

import sys
import os
import torch
import pandas as pd
import joblib

# ==========================================
# PATH SETUP
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.cnn_model import create_ids_model
from src.utils.helpers import get_device
from src.data.preprocess import preprocess_single_sample

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = os.path.join(project_root, "models/checkpoints/ids_multiclass/best_model.pth")
ENCODER_PATH = os.path.join(project_root, "models/label_encoder.pkl")
# Assuming the scaler is saved where preprocess.py expects it (usually models/scaler_multi.pkl)

def load_label_map(encoder_path: str):
    """Loads the LabelEncoder to map indices back to class names."""
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder not found at: {encoder_path}")
    
    encoder = joblib.load(encoder_path)
    # Create a dictionary: {0: 'Benign', 1: 'DDoS', ...}
    label_map = {i: label for i, label in enumerate(encoder.classes_)}
    print(f"[INFO] Loaded Label Map: {len(label_map)} classes")
    return label_map

def load_model(model_path: str, num_classes: int, device: torch.device):
    """Loads the trained CNN model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first.")

    model = create_ids_model(num_classes=num_classes)
    
    # Handle map_location to ensure it works on CPU if trained on GPU
    checkpoint = torch.load(model_path, map_location=device)
    
    # Support both full checkpoint dicts and direct state dicts
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model

def predict_single_packet(model, sample_tensor, device, label_map):
    """Performs inference on a single tensor."""
    with torch.no_grad():
        # Move input to device
        inputs = sample_tensor.to(device)
        
        # Forward pass
        logits = model(inputs)
        
        # Softmax for probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Get max probability and class index
        confidence, pred_idx = probs.max(dim=1)
        
        # Map index to label
        idx = pred_idx.item()
        label = label_map.get(idx, "Unknown")
        
        return label, confidence.item() * 100.0

def get_dummy_sample():
    """
    Creates a fake dataframe row simulating a network packet.
    Values are arbitrary placeholders to test the pipeline.
    """
    data = {
        'Header_Length': [54.0], 
        'Protocol Type': [6.0], 
        'Time_To_Live': [64.0], 
        'Rate': [10.5], 
        'fin_flag_number': [0], 
        'syn_flag_number': [1], # Suspicious SYN flag?
        'rst_flag_number': [0], 
        'psh_flag_number': [0], 
        'ack_flag_number': [0], 
        'ece_flag_number': [0], 
        'cwr_flag_number': [0], 
        'ack_count': [0], 
        'syn_count': [1], 
        'fin_count': [0], 
        'rst_count': [0], 
        'HTTP': [0], 'HTTPS': [0], 'DNS': [0], 'Telnet': [0], 'SMTP': [0], 'SSH': [0], 
        'IRC': [0], 'TCP': [1], 'UDP': [0], 'DHCP': [0], 'ARP': [0], 'ICMP': [0], 'IGMP': [0], 'IPv': [1], 'LLC': [1], 
        'Tot sum': [500.0], 'Min': [40.0], 'Max': [1500.0], 'AVG': [800.0], 'Std': [100.0], 'Tot size': [500.0], 
        'IAT': [0.1], 'Number': [1], 'Variance': [0.5]
    }
    return pd.DataFrame(data)

def main():
    device = get_device()
    print(f"[INFO] Using Device: {device}")

    # 1. Load Labels
    try:
        label_map = load_label_map(ENCODER_PATH)
        num_classes = len(label_map)
    except Exception as e:
        print(f"[ERROR] Could not load encoder: {e}")
        return

    # 2. Load Model
    try:
        model = load_model(MODEL_PATH, num_classes, device)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return

    # 3. Prepare Data (Simulated Packet)
    print("[INFO] Simulating a network packet...")
    sample_df = get_dummy_sample()
    
    # Preprocess (Scaling + Tensor conversion)
    # Note: This relies on 'models/scaler_multi.pkl' existing!
    try:
        sample_tensor = preprocess_single_sample(sample_df)
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return

    # 4. Predict
    print("[INFO] Running Inference...")
    label, confidence = predict_single_packet(model, sample_tensor, device, label_map)

    print("\n" + "="*30)
    print(f"PREDICTION RESULT")
    print("="*30)
    print(f"Pred Class : {label}")
    print(f"Confidence : {confidence:.2f}%")
    print("="*30)

if __name__ == "__main__":
    main()