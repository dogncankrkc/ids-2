"""
CNN Feature Map Visualizer for IDS (Fixed for small dimensions)
Visualizes what the internal layers (Conv1, Conv2, Conv3) 'see' during inference.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import joblib

from src.models.cnn_model import create_ids_model
from src.data.preprocess import preprocess_multiclass

# --- CONFIGURATION ---
TEST_CSV     = "data/raw/datasense_MASTER_FULL.csv"  
MODEL_PATH   = "models/checkpoints/ids_multiclass/best_model.pth"
ENCODER_PATH = "models/label_encoder.pkl"
OUTPUT_DIR   = "outputs/feature_maps"

# Which layers to visualize?
LAYER_NAMES = ["conv1", "conv2", "conv3"]

def load_essentials():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    encoder = joblib.load(ENCODER_PATH)
    classes = encoder.classes_
    
    model = create_ids_model(mode="multiclass", num_classes=len(classes))
    model.to(device)
    
    print(f"[INFO] Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Handle both full checkpoint dict and direct state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model, device, classes

def get_single_sample_by_label(target_label="dos"):
    print(f"[INFO] Searching for a sample of class '{target_label}'...")
    df = pd.read_csv(TEST_CSV)
    
    # Preprocess (We only need the test split)
    # The function returns 6 values, we need the last ones (X_test, y_test)
    _, _, X_test, _, _, y_test = preprocess_multiclass(df)
    
    encoder = joblib.load(ENCODER_PATH)
    try:
        target_id = encoder.transform([target_label])[0]
    except ValueError:
        print(f"[WARN] '{target_label}' not found in dataset. Available classes: {encoder.classes_}")
        return None, None

    # Find indices for the target class
    indices = np.where(y_test == target_id)[0]
    if len(indices) == 0:
        raise ValueError(f"Class '{target_label}' not found in dataset!")
        
    # Pick the first one found
    idx = indices[0] 
    sample_x = X_test[idx] # Shape: (7, 10, 1)
    
    # Convert to Tensor: (1, 1, 7, 10)
    tensor_x = torch.tensor(sample_x, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    
    return tensor_x, target_label

def visualize_layers():
    model, device, classes = load_essentials()
    
    # Target attack to visualize (dos, mitm, benign, etc.)
    target_attack = "dos" 
    
    inputs, label_name = get_single_sample_by_label(target_attack)
    if inputs is None: return

    inputs = inputs.to(device)
    
    # --- HOOK MECHANISM ---
    # Captures the output (activation) of a specific layer
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.conv2.register_forward_hook(get_activation('conv2'))
    model.conv3.register_forward_hook(get_activation('conv3'))
    
    print(f"[INFO] Running model: Input -> {target_attack.upper()}")
    _ = model(inputs)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 0. Plot Input Data
    plt.figure(figsize=(5, 4))
    input_img = inputs.cpu().squeeze().numpy()
    plt.imshow(input_img, cmap='viridis', aspect='auto')
    plt.title(f"Input Data (7x10) - {label_name}")
    plt.colorbar()
    plt.savefig(f"{OUTPUT_DIR}/0_input.png")
    plt.close()

    # Plot Feature Maps for each layer
    for layer_name in LAYER_NAMES:
        # FIX: Only squeeze the Batch dimension (index 0).
        # This transforms (1, 64, 1, 2) -> (64, 1, 2).
        # It preserves dimensions even if they are 1 (e.g., Height=1).
        act = activations[layer_name].cpu().squeeze(0) 
        
        num_channels = act.shape[0]
        # Limit to first 16 channels to keep the plot readable
        limit = min(num_channels, 16)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 10))
        fig.suptitle(f"Layer: {layer_name} Feature Maps (First {limit} Channels)", fontsize=16)
        
        for i in range(16):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            
            if i < limit:
                # aspect='auto' is crucial for small feature maps (e.g. 1x2 pixels)
                # otherwise they might look like thin lines or be invisible.
                ax.imshow(act[i], cmap='plasma', aspect='auto') 
                ax.axis('off')
                ax.set_title(f"Filter {i}")
            else:
                ax.axis('off') # Hide unused subplots
                
        plt.tight_layout()
        save_path = f"{OUTPUT_DIR}/{layer_name}_maps.png"
        plt.savefig(save_path)
        print(f"✅ Saved: {save_path}")
        plt.close()

if __name__ == "__main__":
    visualize_layers()