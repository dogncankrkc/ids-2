"""
CNN Feature Map Visualizer.
---------------------------
This script visualizes the internal activation maps of the CNN layers
to understand what features the model is extracting from the network packets.

Usage:
    python scripts/visualize_layers.py
"""

import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

# ==========================================
# PATH SETUP
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Internal Imports
from src.models.cnn_model import create_ids_model
from src.data.preprocess import preprocess_multiclass
from src.utils.helpers import get_device

# ==========================================
# CONFIGURATION
# ==========================================
# Using the saved test split to ensure we visualize valid data
TEST_DATA_PATH = os.path.join(project_root, "data/processed/test_split_saved.csv")
MODEL_PATH = os.path.join(project_root, "models/checkpoints/ids_multiclass/best_model.pth")
ENCODER_PATH = os.path.join(project_root, "models/label_encoder.pkl")
OUTPUT_DIR = os.path.join(project_root, "outputs/feature_maps")

# Layers to visualize (must match names in cnn_model.py)
LAYER_NAMES = ["features.0", "features.4", "features.8"] 
# features.0 -> Conv1, features.4 -> Conv2, features.8 -> Conv3 (Indices based on nn.Sequential)

def load_essentials():
    device = get_device()
    
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder not found at {ENCODER_PATH}")

    encoder = joblib.load(ENCODER_PATH)
    classes = encoder.classes_
    num_classes = len(classes)
    
    model = create_ids_model(num_classes=num_classes)
    model.to(device)
    
    print(f"[INFO] Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
         raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model, device, encoder

def get_single_sample_by_label(target_label="DDoS"):
    """
    Finds a single sample of a specific attack type from the test dataset.
    """
    print(f"[INFO] Searching for a sample of class '{target_label}'...")
    
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found at {TEST_DATA_PATH}")

    df = pd.read_csv(TEST_DATA_PATH)
    
    # Check if label is string or int
    # Assuming test_split_saved.csv has numeric labels (from training)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    
    # Load encoder to find the numeric ID of the target label
    encoder = joblib.load(ENCODER_PATH)
    
    # Handle case sensitivity
    available_classes = [c.upper() for c in encoder.classes_]
    target_upper = target_label.upper()
    
    if target_upper not in available_classes:
        print(f"[WARN] Class '{target_label}' not found. Available: {encoder.classes_}")
        return None, None

    # Find numeric index
    real_label_name = encoder.classes_[available_classes.index(target_upper)]
    target_id = encoder.transform([real_label_name])[0]

    # Find indices
    indices = np.where(y == target_id)[0]
    if len(indices) == 0:
        raise ValueError(f"No samples found for class '{real_label_name}'")
        
    # Pick the first one
    idx = indices[0] 
    sample_x = X[idx] 
    
    # Convert to Tensor: (1, 1, Features)
    tensor_x = torch.tensor(sample_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    return tensor_x, real_label_name

def visualize_layers():
    model, device, _ = load_essentials()
    
    # Choose an attack type to visualize
    target_attack = "DDoS" 
    
    inputs, label_name = get_single_sample_by_label(target_attack)
    if inputs is None: return

    inputs = inputs.to(device)
    
    # --- HOOK MECHANISM ---
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks to specific layers in the Sequential block
    # Accessing layers via index because they are inside nn.Sequential 'features'
    # Conv1
    model.features[0].register_forward_hook(get_activation('Conv1'))
    # Conv2
    model.features[4].register_forward_hook(get_activation('Conv2'))
    # Conv3
    model.features[8].register_forward_hook(get_activation('Conv3'))
    
    print(f"[INFO] Running inference on input -> {label_name}")
    _ = model(inputs)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Plot Feature Maps
    for layer_name, act in activations.items():
        # act shape: (1, Channels, Length) -> Squeeze batch: (Channels, Length)
        act = act.cpu().squeeze(0) 
        
        num_channels = act.shape[0]
        limit = min(num_channels, 16) # Show max 16 filters
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 6))
        fig.suptitle(f"Layer: {layer_name} Activations (First {limit} Filters)", fontsize=16)
        
        for i in range(16):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            
            if i < limit:
                # 1D Signal Visualization (Heatmap style)
                # Expand dims to make it look like a bar/image: (1, Length)
                signal = act[i].unsqueeze(0).numpy()
                ax.imshow(signal, cmap='viridis', aspect='auto') 
                ax.axis('off')
                ax.set_title(f"Filter {i}", fontsize=8)
            else:
                ax.axis('off')
                
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"{layer_name}_maps.png")
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
        plt.close()

if __name__ == "__main__":
    visualize_layers()