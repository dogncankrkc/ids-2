# IDS-CNN: Network Intrusion Detection System

A lightweight CNN-based intrusion detection system that transforms tabular network traffic features into a 2D representation for classification. The system supports both binary (benign vs attack) and multiclass (attack type) detection.

---

## 📋 Overview

This project uses a Convolutional Neural Network (CNN) to detect network intrusions from tabular data. 70 carefully selected network features are reshaped into a `(1, 7, 10)` format, allowing the CNN to learn spatial patterns in network traffic behavior.

**Key Features:**
- Binary classification: Benign vs Attack
- Multiclass classification: 8 attack categories (DoS, DDoS, Recon, MITM, Bruteforce, Web, Malware)
- Automatic train/validation/test split
- Model checkpointing and early stopping
- Comprehensive metrics (accuracy, precision, recall, F1-score)
- Easy-to-use configuration files

---

## 📁 Project Structure

```bash
ids-1/
├── configs/
│   ├── binary_config.yaml       # Binary classification settings
│   └── multiclass_config.yaml   # Multiclass classification settings
│
├── data/
│   └── raw/                     # Place your CSV files here
│
├── models/
│   ├── checkpoints/             # Training checkpoints
│   ├── scaler_binary.pkl        # Feature scaler (binary)
│   ├── scaler_multi.pkl         # Feature scaler (multiclass)
│   └── label_encoder.pkl        # Label encoder (multiclass)
│
├── outputs/                     # Test results and visualizations
│
├── src/
│   ├── data/                    # Data loading and preprocessing
│   ├── models/                  # CNN model architecture
│   ├── training/                # Training loop and metrics
│   ├── utils/                   # Helper functions
│   └── testing/                 # Evaluation scripts
│
├── notebooks/
│   └── 01_training_example.ipynb  # Training example
│
├── tests/                       # Unit tests
├── train.py                     # Training script
├── inference.py                 # Inference script
└── requirements.txt             # Dependencies
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/dogncankrkc/ids-1.git
cd ids-1

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare Your Data

Place your CSV files in the `data/raw/` folder. The system will automatically load and concatenate all CSV files.

**Required columns:**
- `binary_label`: For binary classification (0 = benign, 1 = attack)
- `label2`: For multiclass classification (attack type names)
- 70 network features (see `src/data/preprocess.py` for the complete list)

### 3. Train a Model

**Binary Classification:**
```bash
python train.py --config configs/binary_config.yaml --mode binary
```

**Multiclass Classification:**
```bash
python train.py --config configs/multiclass_config.yaml --mode multiclass
```

The training script will:
- Load and preprocess your data
- Split into train/validation/test sets (60/20/20)
- Train the CNN model
- Save checkpoints to `models/checkpoints/`
- Save the final model and training history

### 4. Run Inference

```bash
python inference.py
```

Edit the script to specify:
- `sample_path`: Path to a single-row CSV file
- `model_path`: Path to your trained model checkpoint

---

## ⚙️ Configuration

Training parameters can be customized in YAML config files:

```yaml
model:
  type: "ids_cnn"
  num_classes: 2          # 2 for binary, 8 for multiclass
  input_channels: 1
  input_size: [7, 10]

training:
  epochs: 40
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "cosine"
  early_stopping_patience: 6
```

---

## 📊 Model Architecture

The CNN model (`IDS_CNN`) consists of:
- 3 convolutional layers (16 → 32 → 64 filters)
- MaxPooling and Dropout for regularization
- 2 fully connected layers
- Adaptive to both binary and multiclass tasks

**Model size:** < 1M parameters (suitable for edge deployment)

---

## 🧪 Testing

Run unit tests:
```bash
pytest tests/ -v
```

Run evaluation on test set:
```bash
python -m src.testing.test_ids
```

---

## 📓 Jupyter Notebook

Explore the training process interactively:
```bash
jupyter notebook notebooks/01_training_example.ipynb
```

---

## 🔍 Key Files

| File | Description |
|------|-------------|
| `src/data/preprocess.py` | Feature selection and preprocessing |
| `src/models/cnn_model.py` | CNN architecture |
| `src/training/trainer.py` | Training loop with metrics |
| `train.py` | Main training script |
| `inference.py` | Prediction on new samples |

---

## 📈 Results

The model outputs are saved in `outputs/` including:
- Confusion matrices (normalized and counts)
- Prediction results CSV
- Training history plots

---

## 🤝 Contributing

1. Keep commits small and focused
2. Update tests when adding features
3. Document configuration changes
4. Use formatters: `black`, `isort`, `flake8`

---

## 📧 Contact

**Maintainer:** Doğancan Karakoç ([@dogncankrkc](https://github.com/dogncankrkc))

---

## 📄 License

MIT License - see LICENSE file for details

