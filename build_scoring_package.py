"""
build_scoring_package.py
=========================
Package the minimal code + a trained bundle into a standalone folder that
another team/system can drop in WITHOUT this repo, torch, optuna, or a DB
driver. See the generated package's README_SCORING.md for usage.

Usage:
  python build_scoring_package.py --bundle artifacts/<ts>_final/fold_01/model_bundle.pkl \
                                  --output scoring_package/
"""

import argparse
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Everything score_dataframe()/Predictor need. Deliberately excludes
# src/model/* (DeepSets/losses/trainer/meta_learner — torch+optuna, legacy
# path only), src/evaluation/{metrics,ranking,visualization,fold_aggregator}.py
# and run.py/explore_*.py (training/evaluation-only).
SCORING_FILES = [
    "project_config.py",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/data_loader.py",
    "src/data/temporal_split.py",
    "src/data/preprocessing.py",
    "src/db/__init__.py",
    "src/db/mssql_connection.py",
    "src/baselines/__init__.py",
    "src/baselines/aggregated_xgboost.py",
    "src/evaluation/__init__.py",
    "src/evaluation/calibration.py",
    "src/evaluation/decision.py",
    "src/inference/__init__.py",
    "src/inference/model_loader.py",
    "src/inference/predictor.py",
]

REQUIREMENTS = """\
# Minimal dependency set for scoring only — no torch, no optuna, no umap.
numpy
pandas
scikit-learn
xgboost
joblib
# pyodbc only needed if you want Predictor's DB-backed predict() path
# (score_dataframe() does not need it).
"""

README = """\
# Scoring Package

Self-contained scoring code for the loan-default risk model — no training
pipeline, no torch/optuna, no database driver required.

## Install

```
pip install -r requirements-scoring.txt
```

## Score a DataFrame

```python
import pandas as pd
from src.inference.predictor import score_dataframe

df = pd.read_csv("customers_to_score.csv")   # same columns as TRAIN_TABLE,
                                              # minus WORST_FUTURE_CAT/DPD
queue = score_dataframe(df, "model_bundle.pkl")
queue.to_csv("queue.csv", index=False)
```

`df` needs one row per loan with the same columns as the source ETL table
(`NATIONAL_CODE`, `SNAPSHOT_DATE`, `DPD_DAYS`, `LOAN_CATEGORY`, ... — see
this project's `column_changes.md`). Multiple loans per customer and
multiple snapshots are handled automatically (grouped, deduped to the
newest snapshot per customer).

### Optional: refresh calibration

If you have a matured, labeled snapshot (a DataFrame with
`WORST_FUTURE_CAT` populated), pass it as `calibration_df=` to refit the
probability calibrator before scoring — otherwise the calibrator shipped
in the bundle is used unchanged:

```python
queue = score_dataframe(df, "model_bundle.pkl", calibration_df=matured_df)
```

### Optional: call-ledger freshness

```python
queue = score_dataframe(df, "model_bundle.pkl", called_log_path="calls.csv")
```

`calls.csv` needs `NATIONAL_CODE, CALLED_AT` columns — customers called
within the freshness window (`API_DATA_TTL_DAYS` in `project_config.py`)
are flagged `RECENTLY_CALLED` and excluded from the ranked queue.

## Output columns

| Column | Meaning |
|---|---|
| `RISK_RANK` | 1…N queue position; call the API in this order. `NaN` = not in the queue. |
| `RISK_SCORE` | Calibrated, masked P(entering the severe class) — the sort key. |
| `RULE_FLAG` | `""` = in queue; otherwise handled by rule, not ranked (`ALREADY_SEVERE`, `SUPERSEDED`, `RECENTLY_CALLED`, `PREDICTED_SEVERE`). |
| `P_NO_DELAY` … `P_SEVERE_PAST_DUE` | Full calibrated class distribution per customer. |
| `PREDICTED_CLASS`, `EXPECTED_COST` | Secondary/diagnostic. |

## Model file

`model_bundle.pkl` (ship alongside this code) is self-contained: fitted
scaler + XGBoost model + probability calibrator + feature metadata.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="Path to the trained model_bundle.pkl to include.")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "scoring_package",
                        help="Destination directory (default: ./scoring_package/)")
    args = parser.parse_args()

    if not args.bundle.exists():
        raise SystemExit(f"Bundle not found: {args.bundle}")

    out = args.output
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for rel in SCORING_FILES:
        src = BASE_DIR / rel
        if not src.exists():
            raise SystemExit(f"Missing expected source file: {src}")
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    shutil.copy2(args.bundle, out / "model_bundle.pkl")
    (out / "requirements-scoring.txt").write_text(REQUIREMENTS)
    (out / "README_SCORING.md").write_text(README)

    print(f"Scoring package written to {out.resolve()}")
    print(f"  {len(SCORING_FILES)} source files + model_bundle.pkl "
          f"(from {args.bundle}) + requirements-scoring.txt + README_SCORING.md")


if __name__ == "__main__":
    main()
