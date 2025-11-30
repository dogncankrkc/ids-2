# CNN Model Development Framework

A comprehensive Python project structure for developing Convolutional Neural Network (CNN) models for image classification tasks.

## Project Structure

```
ids-1/
├── configs/                    # Configuration files
│   ├── default_config.yaml     # Default training configuration
│   └── cifar10_config.yaml     # CIFAR-10 specific configuration
│
├── data/                       # Data directory
│   ├── raw/                    # Raw, unprocessed data
│   ├── processed/              # Preprocessed data
│   └── external/               # External datasets
│
├── models/                     # Saved models
│   ├── checkpoints/            # Training checkpoints
│   └── final/                  # Final trained models
│
├── notebooks/                  # Jupyter notebooks for experimentation
│   └── 01_training_example.ipynb
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── models/                 # Model architectures
│   │   ├── __init__.py
│   │   └── cnn_model.py        # CNN model definitions
│   ├── data/                   # Data loading and processing
│   │   ├── __init__.py
│   │   ├── dataset.py          # Custom dataset classes
│   │   └── transforms.py       # Data transformations
│   ├── training/               # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop
│   │   └── metrics.py          # Evaluation metrics
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── helpers.py          # General helpers
│       └── visualization.py    # Plotting utilities
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_models.py
│   └── test_metrics.py
│
├── logs/                       # Training logs
├── train.py                    # Main training script
├── inference.py                # Inference script
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
└── .gitignore                  # Git ignore rules
```

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for models, data, training, and utilities
- **Multiple CNN Architectures**: Includes SimpleCNN and VGG-style CNN implementations
- **Data Augmentation**: Built-in support for common image augmentation techniques
- **Training Utilities**: Complete training loop with early stopping, learning rate scheduling, and checkpointing
- **Evaluation Metrics**: Accuracy, precision, recall, F1 score, and confusion matrix
- **Visualization**: Training history plots, confusion matrices, and prediction visualization
- **Configuration Management**: YAML-based configuration for easy experiment management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ids-1.git
cd ids-1
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Training with CIFAR-10

```bash
python train.py --dataset cifar10 --epochs 50
```

### Training with Custom Configuration

```bash
python train.py --config configs/cifar10_config.yaml
```

### Running Inference

```bash
python inference.py --model-path models/final/model.pth --image path/to/image.jpg
```

## Usage Examples

### Creating a Model

```python
from src.models.cnn_model import create_model

# Create a simple CNN
model = create_model(
    model_type="simple",
    num_classes=10,
    input_channels=3,
    input_size=(32, 32)
)

# Create a VGG-style CNN
model = create_model(
    model_type="vgg",
    num_classes=100,
    input_channels=3,
    input_size=(32, 32)
)
```

### Training a Model

```python
from src.models.cnn_model import create_model
from src.training.trainer import Trainer
from src.utils.helpers import get_device, get_optimizer

device = get_device()
model = create_model(model_type="simple", num_classes=10)
optimizer = get_optimizer(model, optimizer_name="adam", learning_rate=0.001)
criterion = torch.nn.CrossEntropyLoss()

trainer = Trainer(model, criterion, optimizer, device)
history = trainer.train(train_loader, val_loader, epochs=100)
```

### Evaluating Performance

```python
from src.training.metrics import accuracy, confusion_matrix

# Calculate accuracy
acc = accuracy(predictions, targets)

# Generate confusion matrix
cm = confusion_matrix(predictions, targets, num_classes=10)
```

## Model Architectures

### SimpleCNN
A lightweight CNN with 2 convolutional layers followed by 2 fully connected layers. Suitable for quick experimentation and simpler datasets.

### VGGStyleCNN
A deeper VGG-inspired architecture with 3 convolutional blocks, each containing 2 conv layers. Better suited for more complex image classification tasks.

## Configuration

Training parameters can be configured via YAML files:

```yaml
model:
  type: "simple"
  num_classes: 10
  input_channels: 3
  input_size: [32, 32]

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "cosine"
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision 0.13+
- NumPy, Pandas, Matplotlib
- See `requirements.txt` for complete list

## License

MIT License
