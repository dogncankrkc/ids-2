# IDS — CNN-based Intrusion Detection (Team README)

Lightweight pipeline for network Intrusion Detection (IDS) using a compact
CNN. Input is tabular CSV data (70 selected features) reshaped to
`(1, 7, 10)` for the model. The project supports both binary and
multiclass classification.

Keep this README short, actionable and team-oriented — include high-level
purpose, how to run things locally, contribution guidelines and where to
find important project pieces.

## Project Structure

ids-1/
├── configs/                # YAML configs for training (binary + multiclass)
├── data/                   # Raw and optional processed CSV data
│   ├── raw/                # Put raw CSV files here (project consumes *.csv)
│   └── processed/          # Optional: processed CSVs
├── models/                 # Saved scalers/encoders and checkpoints
│   ├── checkpoints/
│   └── final/
├── notebooks/              # Analysis / experiments (optional)
├── src/                    # Source package: models, data, training, utils
├── tests/                  # Unit tests
├── train.py                # CLI training entrypoint
├── inference.py            # Simple inference example
├── requirements.txt        # Dependencies
└── setup.py                # Packaging metadata
```

**Purpose & Scope**
- Build and evaluate a compact CNN-based IDS from tabular features.
- Provide reusable preprocessing, training and inference utilities
  so team members can iterate on models and experiments.

**Quick Start (local)**

- Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

- Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

**Prepare data**
- Put one or more CSV files in `data/raw/`. The loader concatenates all
  `*.csv` files found there (`src/data/dataset.py`).
- Required label columns:
  - `binary_label` — binary classification (0 = benign, 1 = attack)
  - `label2` — multiclass label for attack category
- Feature selection (70 features) and preprocessing rules are in
  `src/data/preprocess.py` (`SELECTED_FEATURES`). Preprocessing saves
  scalers/encoders to `models/` (e.g. `models/scaler_multi.pkl`).

**Run training**
- Binary example:

```bash
python train.py --config configs/ids_binary.yaml --mode binary
```

- Multiclass example:

```bash
python train.py --config configs/ids_multiclass.yaml --mode multiclass
```

Notes:
- `train.py` reads the config YAML for hyperparameters, paths and logging.
- Checkpoints and the final model are stored under `models/checkpoints`.

**Inference**
- Use `inference.py` for a simple example that expects a one-row CSV
  (see `preprocess_single_sample` in `src/data/preprocess.py`).

```bash
python inference.py
```

Programmatic usage: preprocess with `preprocess_single_sample`, load the
model with `src.utils.helpers.load_model` or `IDS_CNN` and run a forward pass.

**Configuration**
- Configs live in `configs/` and follow this structure: `model`, `data`,
  `training`, `checkpoint`, `logging`, `seed`. Edit to change hyperparams
  or file paths.

**Tests & Code Quality**
- Run tests:

```bash
pytest tests/ -v
```

- For development, use the `dev` extras in `setup.py` or run the formatters
  and linters locally (black, flake8, isort, mypy).

**For Contributors / Team**
- Keep changes small and focused. Prefer descriptive commits.
- Update `SELECTED_FEATURES` and docs if you change feature selection.
- When adding new experiments, add a short notebook or script under
  `notebooks/` and include which config you used.
- If you add new data columns, update `src/data/preprocess.py` and add
  unit tests under `tests/`.

**Where to look next**
- `src/data/preprocess.py` — preprocessing, scaler/encoder saving
- `src/models/cnn_model.py` — model implementation and factory
- `src/training/trainer.py` — training loop, metrics, checkpointing

**Contact / Maintainers**
- Primary: `dogncankrkc` (repo owner). Add maintainers in GitHub settings
  or list emails here as the team grows.

License: MIT

