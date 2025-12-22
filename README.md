## 🚀 Quick Start

### 1. Installation

```bash
# IDS (CIC2023) — Intrusion Detection System

Lightweight project that provides preprocessing, training and inference utilities for network intrusion detection using tabular features and a compact CNN (binary and multiclass workflows).

This repository contains data preparation tools (including a capped + SMOTE pipeline for CIC datasets), training scripts, model code and inference helpers used in the experiments.

---

## Quick summary

- Data: scripts to generate a processed dataset from large raw CSV(s) (see `src/data/create_balanced_dataset.py`).
- Training: `train.py` (configurable via YAML files in `configs/`).
- Inference: `scripts/inference.py` demonstrates loading a saved model and running a single-sample prediction.
- Models and checkpoints are stored under `models/checkpoints/`.

---

## Requirements

- Python 3.8+
- See `requirements.txt` for required packages. Note that PyTorch is expected but not pinned in the file — install a compatible `torch`/`torchvision` for your platform (CUDA vs CPU / MPS).

Example (macOS / CPU):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you intend to use GPU acceleration, install a matching `torch` build before `pip install -r requirements.txt`.

---

## Preparing data

Recommended: generate a processed dataset using the provided generator which reads large CSVs in chunks and creates a capped + SMOTE balanced dataset.

Run:

```bash
python src/data/create_balanced_dataset.py
```

This writes `data/processed/CIC2023_CAPPED_SMOTE.csv` by default. `train.py` expects a processed CSV at `data/processed/CIC2023_CAPPED_SMOTE.csv` (see `PROCESSED_DATA_PATH` in `train.py`) — either keep that path or edit `train.py` / the data pipeline to point to another processed CSV in `data/processed/`.

The generator also creates `multiclass_label` (8 attack categories) and `binary_label` (Benign=0, Attack=1) columns.

Note: smaller example processed files may already exist under `data/processed/` for testing (inspect that folder before running the full generator).

---

## Training

Training is driven by `train.py` and configurable via YAML config files in `configs/`.

Examples:

```bash
# Multiclass (default config)
python train.py --config configs/multiclass_config.yaml --mode multiclass

# Binary (uses binary config)
python train.py --config configs/binary_config.yaml --mode binary
```

What happens during training:
- Loads the processed CSV from `data/processed/CIC2023_CAPPED_SMOTE.csv` (or the path you set).
- Applies preprocessing (`src/data/preprocess.py`) and creates PyTorch DataLoaders.
- Trains the model defined by `src/models/cnn_model.py` and saves checkpoints to `models/checkpoints/` (checkpoint dir is also configurable in each YAML file).

Configuration highlights:
- `configs/multiclass_config.yaml` — model hyperparameters, `data.batch_size`, `training.epochs`, `training.learning_rate`, `checkpoint.save_dir`, `data.num_workers`.
- `configs/binary_config.yaml` — example binary-oriented config (CatBoost settings shown in that file). `train.py` supports both `binary` and `multiclass` modes.

---

## Inference

Use `scripts/inference.py` as a reference for loading a saved model, label encoder and scaler and running a single-sample prediction. Edit the constants at the top of the script to point to your model/scaler/encoder locations.

Run:

```bash
python scripts/inference.py
```

The script demonstrates how to create a dummy packet, preprocess it (via `src/data/preprocess.py`), load a trained model checkpoint and map predicted indices back to class names.

---

## Useful files and locations

- `train.py` — training entry point (CLI: `--config`, `--mode`).
- `src/data/create_balanced_dataset.py` — builds the capped + SMOTE processed CSV from large raw files.
- `src/data/preprocess.py` — preprocessing utilities and single-sample preprocessing.
- `src/models/cnn_model.py` — CNN architecture and model factory.
- `src/training/trainer.py` — training loop, validation, testing utilities.
- `scripts/inference.py` — example inference flow for a single sample.
- `configs/` — YAML files for binary and multiclass experiments.
- `models/checkpoints/` — where checkpoints and final models are saved by default.

---

## Running quick checks

To verify basic environment and script availability, useful commands:

```bash
python scripts/print_env_info.py
python train.py --config configs/multiclass_config.yaml --mode multiclass --help
python -m pytest -q
```

---

## Contributing

- Open issues for bugs or feature requests.
- Keep changes focused and add tests where appropriate.
- Use `black`, `isort`, `flake8` for style if modifying code.

---

## License & Contact

MIT License. Maintainer: Doğancan Karakoç — dogncankrkc@gmail.com


