"""
Model Efficiency Report (IEEE-Compliant)
---------------------------------------
Reports:
- Number of trainable parameters
- Model memory footprint (MB)
- Approximate FLOPs (Conv1D + Linear only)

Notes (IEEE):
- FLOPs are approximated
- Normalization, activation, and pooling layers are excluded
- Batch size = 1 (inference / edge scenario)
- Fixed input length L = 39
"""

import sys
import os
import torch
import torch.nn as nn

# --------------------------------------------------
# PATH FIX (scripts/ -> project root)
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models.cnn_model import create_ids_model

import time

def measure_latency_and_pps(model, input_dim=39, runs=500):
    model.eval()
    device = next(model.parameters()).device

    x = torch.randn(1, 1, input_dim).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(50):
            model(x)

    # Timing
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            model(x)
    end = time.time()

    avg_latency = (end - start) / runs  # seconds
    latency_ms = avg_latency * 1000
    pps = 1.0 / avg_latency

    return latency_ms, pps


# --------------------------------------------------
# PARAMETER COUNT
# --------------------------------------------------
def count_parameters(model):
    """Counts trainable parameters only."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --------------------------------------------------
# FLOPs ESTIMATION (APPROXIMATE)
# --------------------------------------------------
def estimate_flops(model, input_dim=39):
    """
    Approximate FLOPs calculation.

    Included layers:
      - Conv1D
      - Linear

    Excluded layers:
      - Normalization (BatchNorm / GroupNorm)
      - Activation (ReLU, GELU, etc.)
      - Pooling

    Assumptions:
      - Batch size = 1
      - Fixed input length L = input_dim
    """
    flops = 0
    x = torch.randn(1, 1, input_dim)

    layer_flops = []

    def conv1d_flops(m, x_in, x_out):
        # FLOPs = 2 * K * Cin * Cout * Lout
        _, out_channels, length_out = x_out.shape
        kernel_size = m.kernel_size[0]
        in_channels = m.in_channels
        return 2 * kernel_size * in_channels * out_channels * length_out

    def linear_flops(m, x_in, x_out):
        # FLOPs = 2 * In_Features * Out_Features
        return 2 * m.in_features * m.out_features

    def hook_fn(module, inputs, output):
        if isinstance(module, nn.Conv1d):
            layer_flops.append(conv1d_flops(module, inputs[0], output))
        elif isinstance(module, nn.Linear):
            layer_flops.append(linear_flops(module, inputs[0], output))

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    flops = sum(layer_flops)
    return flops

# --------------------------------------------------
# MODEL SIZE (MEMORY)
# --------------------------------------------------
def estimate_model_size_mb(model):
    """Estimates model size in MB (parameters + buffers)."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    INPUT_DIM = 39
    NUM_CLASSES = 7

    model = create_ids_model(
        num_classes=NUM_CLASSES,
        input_dim=INPUT_DIM
    )
    model.eval()

    params = count_parameters(model)
    flops = estimate_flops(model, input_dim=INPUT_DIM)
    size_mb = estimate_model_size_mb(model)
    latency_ms, pps = measure_latency_and_pps(model, input_dim=INPUT_DIM)


    print("\n" + "=" * 44)
    print("        MODEL EFFICIENCY REPORT")
    print("=" * 44)
    print(f"Model Name        : {model.__class__.__name__}")
    print(f"Trainable Params  : {params:,}")
    print(f"Model Size        : {size_mb:.3f} MB")
    print(f"Total FLOPs       : {flops:,} (approx)")
    print(f"Total MFLOPs      : {flops / 1e6:.2f} MFLOPs")
    print("=" * 44)
    print(
        "Note: FLOPs are approximated for Conv1D and Linear layers only "
        "(batch size = 1, input length L = 39)."
    )
