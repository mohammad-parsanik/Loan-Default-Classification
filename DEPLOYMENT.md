# Deployment Guide

How to train the production model, hand it off, and generate the ranked API
queue. Everything below runs on the **training server** (Windows,
`F:\Loan Default Classification\`) where the DB and data live.

The deliverable is a ranked queue: customers not yet in the severe class,
ordered by calibrated P(entering severe) within 6 months, to be enriched by
the external API at **240 requests/hour**. Already-severe customers are
rule-flagged, never ranked.

**Related references:** [`README.md`](README.md) for the architecture and
data flow; [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md) for every config
setting; [`MODEL_EVALUATION.md`](MODEL_EVALUATION.md) for how to tell if a
trained model is actually good before shipping it.

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

**`DEPLOY_ARM` must be an explicit arm name** (e.g. `"multiclass"`), not
`"auto"` — auto-selection needs a test set to compare arms on, which
`--final` deliberately doesn't have.

**Timing:** roughly the time of training one arm on the full dataset —
for reference, a single `multiclass` arm on ~6M training instances took
~15 minutes in Run 6 (excludes the evaluation-only stages below, which
`--final` skips). Expect longer with more mature snapshots or `ARM_OPTUNA_TRIALS > 0`.

**What gets written** (`artifacts/<ts>_final/fold_01/`):

| File | What it is |
|---|---|
| `metadata.json` | Feature list, `MAX_LOANS_PER_CUSTOMER`. |
| `scaler.pkl` | Fitted preprocessing pipeline. |
| `model_arm.pkl` | The trained arm object. |
| `calibrator.pkl` | Stratified isotonic calibrator, fit on the customer-disjoint validation slice. |
| `model_bundle.pkl` | **The one file to ship** — all four of the above packed together (§4). |

**No metrics are written.** There's no test set in a final fit, so
`arms_metrics.json`, the plots, and bootstrap CIs — all gated on having a
test set in the code — simply don't appear. This is expected, not a
failure. **Judge the recipe from your most recent evaluation run (`python
run.py train`, no `--final`) before committing to a `--final` fit** — see
`MODEL_EVALUATION.md`. `--final` re-applies an already-graded recipe to
more data; it doesn't re-grade it.

**Verify it succeeded** before moving on: confirm the log ends with
`Deployed arm '<name>' | Deployment bundle → .../model_bundle.pkl` (no
stack trace above it), and that `model_bundle.pkl` exists and is
non-trivial in size (a few MB, not a few KB — an XGBoost model with 200+
trees over 257 features isn't tiny).

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

This copies the ~18 source files scoring actually needs (not `run.py`, not
`src/model/*` DeepSets code, not the evaluation/plotting modules) plus the
bundle, a `requirements-scoring.txt` (`numpy`, `pandas`, `scikit-learn`,
`xgboost`, `joblib` — nothing else), and a `README_SCORING.md` into one
folder. Hand that folder to the other team.

**The input `df` they build:** one row per **loan** (not per customer — a
customer with 2 open loans contributes 2 rows), with the same columns as
`TRAIN_TABLE` (`column_changes.md` has the full feature dictionary), minus
`WORST_FUTURE_CAT`/`WORST_FUTURE_DPD` (this is prediction time — those
don't exist yet). At minimum: `NATIONAL_CODE`, `SNAPSHOT_DATE`,
`LOAN_CATEGORY`, `DPD_DAYS`, plus the rest of the ~64 feature columns. A
minimal illustrative example (real feature values, not just the ID
columns, obviously required in practice):

```python
import pandas as pd

df = pd.DataFrame([
    {"NATIONAL_CODE": "1234567890", "LOAN_ID": 555001, "CONTRACT_NUMBER": "C-555001",
     "SNAPSHOT_DATE": 20260621.0, "LOAN_CATEGORY": 0.0, "DPD_DAYS": 0.0, "...": "...other features..."},
    {"NATIONAL_CODE": "1234567890", "LOAN_ID": 555002, "CONTRACT_NUMBER": "C-555002",
     "SNAPSHOT_DATE": 20260621.0, "LOAN_CATEGORY": 1.0, "DPD_DAYS": 12.0, "...": "...other features..."},
    # ... one row per loan across all customers you want scored
])
```

Both loans above belong to the same customer (`NATIONAL_CODE` repeats) —
they get grouped into one portfolio, truncated to `MAX_LOANS_PER_CUSTOMER`
(currently 2, kept by current `DPD_DAYS` — never by a future label), and
produce **one row of output** for that customer. With `df` built, they
call either:

> **Which snapshot gets scored?** Neither `score_dataframe()` nor
> `run_scoring()` takes a snapshot-date argument (that's specific to the
> DB-backed `Predictor.predict()`/`run.py predict --snapshot_date` path,
> §5 below). Here, you control it entirely by which rows are in `df` —
> filter your own source data to the `SNAPSHOT_DATE` you want scored
> before calling. Mixing several snapshots in one `df` is fine; `SNAPSHOT_DATE`
> should be a **float** formatted `YYYYMMDD` (e.g. `20260621.0`, matching
> the DB/cache convention used everywhere else in this project), and
> `pred_dedup_latest` (`ScoringParams` field, default `True`) keeps only
> each customer's newest row in the ranked queue — older rows are still
> returned, flagged `SUPERSEDED`.

```python
# Quick one-off, silent config defaults:
from src.inference.predictor import score_dataframe
queue = score_dataframe(df, "model_bundle.pkl")   # df: same columns as TRAIN_TABLE
```

```python
# Manager/orchestration code with several data sources or per-call knobs —
# recommended integration point. Any field left unset (None) on ScoringParams
# falls back to project_config.py, with a warning naming which ones did.
from src.inference.scoring import run_scoring
from src.inference.scoring_params import ScoringParams

params = ScoringParams(bundle_path="model_bundle.pkl", output_path="queue.csv")
queue = run_scoring(df, params)
```

`bundle_path` is the only field you're required to pass — everything else
defaults from `project_config.py` (with a warning listing exactly which
fields defaulted, so a forgotten parameter is visible in the logs rather
than silently applied). Pass a field explicitly to override that default
for this call only, without touching `project_config.py` or affecting
other callers. Every field, spelled out:

```python
params = ScoringParams(
    bundle_path="model_bundle.pkl",   # REQUIRED — no project-wide default makes
                                       # sense for "which trained model to use"

    # Data inputs — no config equivalent; a None here is a deliberate
    # choice, not a missing parameter, so these are never defaulted/warned.
    output_path="queue.csv",          # None = return the DataFrame only, don't write a CSV
    calibration_df=None,              # a matured, labelled snapshot (WORST_FUTURE_CAT
                                       # populated) to refresh calibration before scoring;
                                       # None = use the calibrator already in the bundle

    # Business knobs — override per call; omit (None) to use project_config's value.
    called_log_path="calls.csv",      # NATIONAL_CODE,CALLED_AT ledger; falls back to API_CALL_LOG
    certainty_act_threshold=None,     # e.g. 0.9 to flag near-certain rows PREDICTED_SEVERE
                                       # instead of ranking them; falls back to
                                       # CERTAINTY_ACT_THRESHOLD (None = off, pending business sign-off)
    carve_current_cat_ge=3,           # current_cat at/above this = ALREADY_SEVERE, never ranked;
                                       # falls back to CARVE_CURRENT_CAT_GE
    calibration_min_stratum_n=5000,   # only used when calibration_df is given; falls back to
                                       # CALIBRATION_MIN_STRATUM_N
    api_data_ttl_days=30,             # enrichment freshness window for called_log_path;
                                       # falls back to API_DATA_TTL_DAYS
    pred_dedup_latest=True,           # keep only each customer's newest row when df mixes
                                       # snapshots; falls back to PRED_DEDUP_LATEST
    cost_matrix=None,                 # falls back to COST_MATRIX; drives PREDICTED_CLASS/
                                       # EXPECTED_COST only — never RISK_SCORE/the ranking
)
queue = run_scoring(df, params)
```

Passing a field's project-default value explicitly (as most of the example
above does) is equivalent to omitting it, just without the "defaulted"
warning — useful when you want the call site to document its assumptions
even though they happen to match the current config.

No database access, no `run.py`, no knowledge of this project's training
code required — verified by an automated test that scores in a subprocess
with this repo removed from `sys.path` and `torch`/`optuna` blocked at
import time (`tests/test_pipeline_changes.py::test_build_scoring_package_runs_standalone`).
See `scoring_package/README_SCORING.md` (generated) for the field-by-field
default-source table.

> The fold directory (not just the bundle) also holds the unbundled
> artifacts (`model_arm.pkl`, `scaler.pkl`, `calibrator.pkl`,
> `metadata.json`); `Predictor`/`ModelLoader` accept either the directory
> **or** the single bundle file.

### 4b. Placeholder bundle — integrate before the real one leaves the server

The trained bundle is built on the training server and getting files off
that server is slow/awkward, so the receiving team would otherwise sit
idle. `make_placeholder_bundle.py` produces a **synthetic** bundle with the
real structure, so integration can start immediately:

```bash
python make_placeholder_bundle.py --package scoring_package_placeholder
python make_placeholder_bundle.py --self-check   # build small + score, ~5s
```

It runs the actual training path (`process_raw_data` → preprocessing →
`aggregate_features` → the `DEPLOY_ARM` arm → `StratifiedCalibrator`) on
randomly generated data carrying the real column schema (the 64 feature
columns of `column_changes.md`), and writes `placeholder/model_bundle.pkl`,
a `sample_input.csv` (500 label-free loan rows to smoke-test with) and
`PLACEHOLDER.md`. With `--package` it also emits a complete scoring package
with the placeholder inside — the same folder layout the real handoff has,
so swapping is one file copy and no code change.

The scores are meaningless by construction. Guards against shipping it by
accident: `metadata["placeholder"] = True` in the bundle, and `load_bundle`
logs a `WARNING` naming it as a placeholder every time it is loaded.

Two things the placeholder cannot verify, both worth re-checking on swap:
the **feature column order** (preprocessing is positional, and the
placeholder uses the documented order, not necessarily the training
table's — compare the real bundle's `metadata["features"]`), and
**library versions** (a pickle is version-sensitive; the placeholder
records its own build versions under `metadata["built_with"]`).

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
wins — see [`MODEL_EVALUATION.md`](MODEL_EVALUATION.md) for how to read
that comparison and what "still winning" should look like.
