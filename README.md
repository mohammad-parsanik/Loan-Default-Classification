# Loan Default Classification Pipeline

This repository contains an end-to-end Machine Learning pipeline to predict the worst future delinquency state of a customer's loan portfolio. It utilizes a **Set-Transformer** architecture to learn embeddings for a variable number of loans, followed by a robust **XGBoost Meta-Learner**.

## Features
- **Portfolio-Level Context:** Utilizes a PyTorch Set-Transformer without positional encodings, naturally handling unordered sets of loans per customer.
- **Cost-Sensitive Loss:** Penalizes missing high-risk customers heavily (Cost of 4) compared to over-flagging (Cost of 1).
- **Oracle Native Integration:** Built-in connection layer using `oracledb` via direct `host:port/service_name` strings.
- **Advanced Preprocessing:** Custom `DomainAwareImputer` handles time-since missing values accurately, combined with `RobustScaler` and `OutlierClipper`.

## Installation

This project utilizes `uv` to manage the virtual environment.

```bash
# Activate your environment
source /Users/mohammad/.venv/bin/activate

# Install requirements
uv pip install -r requirements.txt
```

> **Note:** The `oracledb` package defaults to a Thin mode which does not require Oracle Instant Client to be installed.

## Project Structure
- `project_config.py`: Core hyperparameters, table names, and paths.
- `src/db/`: Oracle `oracledb` connector.
- `src/data/`: Data loading, chronological splitting, data exploration, preprocessing pipeline, and PyTorch dataset.
- `src/model/`: PyTorch Set-Transformer, Cost-Sensitive Focal Loss, Trainer, and XGBoost Meta-Learner.
- `src/evaluation/`: Brier score, Macro F1, QWK, UMAP plots, ROC curves.
- `src/inference/`: Production scoring pipeline pulling from saved artifacts.
- `run.py`: Primary CLI entry point.

## Usage

### 1. Data Exploration & Profiling
Before training, explore your data to ensure the Oracle connection works, compute the 99th percentile of loans per customer (`MAX_LOANS_PER_CUSTOMER`), and generate distribution plots.

```bash
python run.py explore
```
Outputs are saved to `artifacts/data_exploration_report.json` and `artifacts/*.png`.

### 2. Training the Pipeline
Trains the Set-Transformer (to learn embeddings) and tunes XGBoost (using Optuna) sequentially.

```bash
python run.py train
```
Artifacts (models, scalers, metadata, plots) will be saved under `artifacts/<YYYYMMDD_HHMMSS>/`.

### 3. Production Inference
Score a new snapshot using the compiled artifacts.

```bash
python run.py predict \
    --artifact_dir ./artifacts/<TIMESTAMP> \
    --snapshot_date 14040531 \
    --output ./predictions.csv
```
