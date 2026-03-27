# IEEE-CIS Fraud Detection

This repository contains a machine learning pipeline for credit card fraud detection based on the Kaggle competition:
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/overview).

## What this project does

- Loads and merges IEEE-CIS transaction and identity data.
- Uses a time-based split to mimic the real competition setting.
- Builds an end-to-end preprocessing + modeling pipeline.
- Trains and tunes an XGBoost model for fraud probability prediction.
- Creates a Kaggle-ready submission file.

## Dataset setup (manual)

The competition data is **not included** in this repository.

1. Download the dataset manually from Kaggle.
2. Place the CSV files in `data/raw/`.
3. Expected files:
   - `train_transaction.csv`
   - `train_identity.csv`
   - `test_transaction.csv`
   - `test_identity.csv`

Without these files, training and prediction scripts will fail.

## Project structure

```text
fraud-detection/
├── data/
│   └── raw/                      # Kaggle CSV files (manual download)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_error_analysis.ipynb
├── src/
│   ├── data/                     # Data loading and time-based split
│   ├── pipeline/                 # Custom transformers + pipeline builder
│   ├── models/
│   │   ├── xgb/                  # Tuning, training, prediction, configs
│   │   ├── random_guessing.py    # Baseline (random)
│   │   └── zero_guessing.py      # Baseline (all zeros)
│   └── evaluation/               # Metrics and threshold analysis
├── utils/
│   └── log_experiment.py         # Logs Optuna runs to CSV
├── experiment_log.csv            # CSV keeping track of all experiments conducted
└── pyproject.toml
```

## Quick start

Use Python 3.12+.

```bash
uv sync
```

If you do not use `uv`, install dependencies from `pyproject.toml` with your preferred package manager.

## Main scripts

- Hyperparameter tuning (Optuna):  
  `python -m src.models.xgb.tune`
- Train model pipeline:  
  `python -m src.models.xgb.train`
- Create Kaggle submission:  
  `python -m src.models.xgb.predict`

Generated artifacts are stored in `src/models/xgb/` (for example `pipeline.pkl` and `submission.csv`).