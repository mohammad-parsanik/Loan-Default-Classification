# Deployment Guide

How to train the production model, hand it off, and generate the ranked API
queue. Everything below runs on the **training server** (Windows,
`F:\Loan Default Classification\`) where the DB and data live.

The deliverable is a ranked queue: customers not yet in the severe class,
ordered by calibrated P(entering severe) within 6 months, to be enriched by
the external API at **240 requests/hour**. Already-severe customers are
rule-flagged, never ranked.

---

## 0. One-time config check (`project_config.py`)

| Setting | Ship value | Note |
|---|---|---|
| `DEPLOY_ARM` | `"multiclass"` | Run-6 winner. `"auto"` needs a test set — not valid for `--final`. |
| `ARM_OPTUNA_TRIALS` | `0` to skip, e.g. `20` to tune | Tunes the deployed arm on val PR-AUC of P(severe). Adds time; optional. |
| `WALK_FORWARD_ENABLED` | `False` | Set `True` only for the stability check (step 2). Turn back off before `--final`. |
| `DEEPSETS_ENABLED` | `False` | Legacy neural arm; leave off. |
| `RECALIBRATE_ON_PREDICT` | `True` | At scoring time, refit the calibrator on the newest matured snapshot. |
| `API_DATA_TTL_DAYS` | `30` | Enrichment freshness window (business-confirmed). |
| `CERTAINTY_ACT_THRESHOLD` | `None` | Set (e.g. `0.9`) only once business approves acting on near-certain rows without an API call. |

---

## 1. (Optional) Evaluation run — grade the recipe

```bash
python run.py train
```

Holds out the newest mature snapshot as test, trains all `MODEL_ARMS`, and
logs the ranking headline (recall@1-day/1-week/1-month, PR-AUC) per arm and
per current-cat slice. Read `artifacts/<run>/fold_01/plots/capture_curves.png`
and `arms_metrics.json`. This does **not** produce the shipping model — it
tells you the recipe still performs. Skip if you just ran one.

## 2. (Recommended once) Walk-forward stability check

Set `WALK_FORWARD_ENABLED = True`, trim `MODEL_ARMS = ["multiclass", "binary"]`
to keep folds cheap (~40 min each), then:

```bash
python run.py train
```

`walk_forward_summary.json` reports mean ± std of ranking AP across quarterly
test snapshots — confirms performance is stable over time, not a one-snapshot
fluke. **Revert `WALK_FORWARD_ENABLED = False` and restore `MODEL_ARMS`
afterwards.**

## 3. Fit the deployment model

```bash
python run.py train --final
```

Trains `DEPLOY_ARM` on **all** mature snapshots (no test hold-out), carves a
customer-disjoint validation slice for the calibrator, and writes the
single-file bundle. Output dir is `artifacts/<timestamp>_final/fold_01/`.

## 4. Hand off the model

`model_bundle.pkl` (fitted scaler + XGBoost arm + stratified calibrator +
metadata) is the **data** — the trained parameters. It is NOT a standalone
program: unpickling it requires this project's Python class definitions
(the scaler/arm/calibrator classes are custom objects referenced by module
path) plus matching `numpy`/`scikit-learn`/`xgboost` versions. Two handoff
patterns depending on the recipient:

**A. They'll run this repo** (simplest — same team, or comfortable with the
full project): copy the whole repo plus the bundle; they run
`python run.py predict --artifact_dir model_bundle.pkl` same as you do.

**B. They're plugging scoring into another system** (a service, a different
codebase, no interest in the training pipeline): build a minimal standalone
package — no torch, no optuna, no DB driver, no training code:

```bash
python build_scoring_package.py --bundle artifacts/<ts>_final/fold_01/model_bundle.pkl \
                                --output scoring_package/
```

This copies the ~16 source files scoring actually needs (not `run.py`, not
`src/model/*` DeepSets code, not the evaluation/plotting modules) plus the
bundle, a `requirements-scoring.txt` (`numpy`, `pandas`, `scikit-learn`,
`xgboost`, `joblib` — nothing else), and a `README_SCORING.md` into one
folder. Hand that folder to the other team; they call:

```python
from src.inference.predictor import score_dataframe
queue = score_dataframe(df, "model_bundle.pkl")   # df: same columns as TRAIN_TABLE
```

No database access, no `run.py`, no knowledge of this project's training
code required — verified by an automated test that scores in a subprocess
with this repo removed from `sys.path` and `torch`/`optuna` blocked at
import time (`tests/test_pipeline_changes.py::test_build_scoring_package_runs_standalone`).

> The fold directory (not just the bundle) also holds the unbundled
> artifacts (`model_arm.pkl`, `scaler.pkl`, `calibrator.pkl`,
> `metadata.json`); `Predictor`/`ModelLoader` accept either the directory
> **or** the single bundle file.

## 5. Generate the queue (from this repo, pattern A)

```bash
python run.py predict --artifact_dir artifacts/<ts>_final/fold_01/model_bundle.pkl \
                      [--snapshot_date 20260621] \
                      [--called_log calls.csv] \
                      [--output queue.csv]
```

- `--snapshot_date` omitted → scores every currently-immature snapshot in the
  table (falls back to `PRED_SNAPSHOT_DATES`, then the latest snapshot).
- `--called_log` — CSV of past API calls (`NATIONAL_CODE, CALLED_AT`) your
  caller appends to; customers called within `API_DATA_TTL_DAYS` are flagged
  `RECENTLY_CALLED` and skipped. Omit on the first cycle.
- `--output` defaults to `<artifact_dir>/predictions/predictions_<tag>.csv`.

### Output columns

| Column | Meaning |
|---|---|
| `RISK_RANK` | 1…N queue position; **call the API in this order**. `NaN` for flagged rows. |
| `RISK_SCORE` | Calibrated, masked P(entering severe) — the sort key. |
| `RULE_FLAG` | `""` = in queue; `ALREADY_SEVERE` / `SUPERSEDED` / `RECENTLY_CALLED` / `PREDICTED_SEVERE` = handled by rule, not called. |
| `P_NO_DELAY` … `P_SEVERE_PAST_DUE` | Full calibrated class distribution. |
| `PREDICTED_CLASS`, `EXPECTED_COST` | Secondary/diagnostic (expected-cost rule). |

Consume from `RISK_RANK = 1` downward at 240/hour for as many hours as the
budget allows. After each call, append `NATIONAL_CODE, CALLED_AT` to the
ledger so the next cycle skips still-fresh enrichments.

---

## Retraining cadence

Re-run steps 3–5 when new snapshots mature (monthly). `DATA_VERSION` in
`project_config.py` invalidates the NPZ cache when the ETL changes; bump it to
force a rebuild. Re-open the arm comparison (`DEPLOY_ARM = "auto"`, full
`MODEL_ARMS`, an evaluation run) periodically to confirm `multiclass` still
wins.
