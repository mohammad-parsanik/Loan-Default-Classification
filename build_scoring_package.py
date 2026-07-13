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
    "src/inference/scoring_params.py",
    "src/inference/scoring.py",
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

## Two ways to score

Neither entry point below takes a snapshot-date argument — you control
which snapshot(s) get scored entirely by which rows are in `df` (filter
your own source data to the `SNAPSHOT_DATE` you want before calling).
`SNAPSHOT_DATE` should be a **float** formatted `YYYYMMDD` (e.g.
`20260621.0`). Mixing several snapshots in one `df` is fine — `df` is one
row per loan, and `pred_dedup_latest` (default `True`) keeps only each
customer's newest row in the ranked queue; older rows are still returned,
flagged `SUPERSEDED`.

**A. Quick one-off** — `score_dataframe()`, positional/keyword args, silent
defaults from `project_config.py`:

```python
import pandas as pd
from src.inference.predictor import score_dataframe

df = pd.read_csv("customers_to_score.csv")   # same columns as TRAIN_TABLE,
                                              # minus WORST_FUTURE_CAT/DPD
queue = score_dataframe(df, "model_bundle.pkl")
queue.to_csv("queue.csv", index=False)
```

**B. Manager/orchestration code** — `run_scoring()` + `ScoringParams`, the
recommended integration point when several data sources or call sites need
different (or per-call-overridable) business knobs. Any field left unset
(`None`) falls back to `project_config.py`, and a single `logger.warning(...)`
names exactly which fields defaulted — so a caller that forgot a parameter
sees it in the logs instead of it silently happening. `bundle_path` is the
only required field:

```python
import pandas as pd
from src.inference.scoring import run_scoring
from src.inference.scoring_params import ScoringParams

df = pd.read_csv("customers_to_score.csv")

# Minimal — everything but bundle_path defaults from project_config.py:
params = ScoringParams(bundle_path="model_bundle.pkl", output_path="queue.csv")
queue = run_scoring(df, params)
```

Every field, spelled out (pass what you need to override, omit — or leave
`None` — the rest):

```python
params = ScoringParams(
    bundle_path="model_bundle.pkl",   # REQUIRED — which trained model to score with

    # Data inputs — no config equivalent, never defaulted/warned:
    output_path="queue.csv",          # None = return the DataFrame only, don't write a CSV
    calibration_df=None,              # a matured, labelled snapshot (WORST_FUTURE_CAT
                                       # populated) to refresh calibration before scoring;
                                       # None = use the calibrator already in the bundle

    # Business knobs — override per call; omit/None = use project_config's value.
    called_log_path="calls.csv",      # NATIONAL_CODE,CALLED_AT ledger; falls back to API_CALL_LOG
    certainty_act_threshold=0.9,      # flag near-certain rows PREDICTED_SEVERE instead of
                                       # ranking them; falls back to CERTAINTY_ACT_THRESHOLD
                                       # (None = off — confirm with the business before setting)
    carve_current_cat_ge=3,           # current_cat at/above this = ALREADY_SEVERE, never ranked
    calibration_min_stratum_n=5000,   # only used when calibration_df is given
    api_data_ttl_days=30,             # enrichment freshness window for called_log_path
    pred_dedup_latest=True,           # keep only each customer's newest row when df mixes snapshots
    cost_matrix=None,                 # falls back to COST_MATRIX; drives PREDICTED_CLASS/
                                       # EXPECTED_COST only — never RISK_SCORE/the ranking
)
queue = run_scoring(df, params)
```

`df` needs one row per loan with the same columns as the source ETL table
(`NATIONAL_CODE`, `SNAPSHOT_DATE`, `DPD_DAYS`, `LOAN_CATEGORY`, ... — see
this project's `column_changes.md`). Multiple loans per customer and
multiple snapshots are handled automatically (grouped, deduped to the
newest snapshot per customer).

### ScoringParams fields

| Field | Default source | Notes |
|---|---|---|
| `bundle_path` | — (required) | No sensible project-wide default for "which model". |
| `calibration_df` | none (no config equivalent) | Optional matured, labeled DataFrame (`WORST_FUTURE_CAT` populated) to refit the calibrator before scoring. Omit to use the calibrator shipped in the bundle. |
| `output_path` | none (no config equivalent) | Omit to just get the DataFrame back without writing a CSV. |
| `called_log_path` | `API_CALL_LOG` | CSV with `NATIONAL_CODE, CALLED_AT`; recently-called customers are excluded from the queue. |
| `certainty_act_threshold` | `CERTAINTY_ACT_THRESHOLD` | Rows at/above this score are flagged `PREDICTED_SEVERE` instead of ranked. |
| `carve_current_cat_ge` | `CARVE_CURRENT_CAT_GE` | Current category at/above which customers are `ALREADY_SEVERE` (rule-flagged, never ranked). |
| `calibration_min_stratum_n` | `CALIBRATION_MIN_STRATUM_N` | Only used when `calibration_df` is given. |
| `api_data_ttl_days` | `API_DATA_TTL_DAYS` | Freshness window for the call ledger. |
| `pred_dedup_latest` | `PRED_DEDUP_LATEST` | When `df` mixes several snapshots, keep only each customer's newest row in the queue. |
| `cost_matrix` | `COST_MATRIX` | Drives the secondary `PREDICTED_CLASS`/`EXPECTED_COST` columns only — never the ranking (`RISK_SCORE` is cost-matrix-free by design). |

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
