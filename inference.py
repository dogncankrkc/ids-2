"""
Inference Script

This script provides utilities for running inference with trained CNN models.
"""

import argparse
import os
from typing import List, Tuple

import torch
from PIL import Image

from src.models.cnn_model import create_model
from src.data.transforms import get_inference_transforms
from src.utils.helpers import get_device


# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_model_for_inference(
    model_path: str,
    model_type: str = "simple",
    num_classes: int = 10,
    device: torch.device = None,
) -> torch.nn.Module:
    """
    Load a trained model for inference.

    Args:
        model_path: Path to the saved model
        model_type: Type of model architecture
        num_classes: Number of classes
        device: Device to load model to

    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = get_device()

    model = create_model(
        model_type=model_type,
        num_classes=num_classes,
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def predict_single_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str],
    device: torch.device,
    image_size: Tuple[int, int] = (32, 32),
) -> Tuple[str, float]:
    """
    Predict class for a single image.

    Args:
        model: Trained model
        image_path: Path to the image
        class_names: List of class names
        device: Device for inference
        image_size: Expected image size

    Returns:
        Tuple of (predicted_class, confidence)
    """
    # Load and preprocess image
    transform = get_inference_transforms(image_size=image_size)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(1)

    predicted_class = class_names[predicted_idx.item()]
    confidence_value = confidence.item() * 100

    return predicted_class, confidence_value


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run inference with trained CNN model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="simple",
        choices=["simple", "vgg"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes",
    )
    args = parser.parse_args()

    # Get device
    device = get_device()

    # Load model
    model = load_model_for_inference(
        model_path=args.model_path,
        model_type=args.model_type,
        num_classes=args.num_classes,
        device=device,
    )

    # Run prediction
    predicted_class, confidence = predict_single_image(
        model=model,
        image_path=args.image,
        class_names=CIFAR10_CLASSES,
        device=device,
    )

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()
