# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A bank early-warning system that predicts the **worst delinquency class of a customer's loan portfolio over the next 6 months** (3 classes: 0 = No Delay, 1 = Current/Minor Delay, 2 = Past Due+). Records are monthly point-in-time snapshots of an upstream ETL table (~64 numeric per-loan features, no categoricals). Prediction is at customer level (`NATIONAL_CODE`), cost-sensitive (missing a Cat-2 costs 4× a false alarm).

**Read `AGENT_HANDOFF.md` first** — it is the authoritative record of decisions, run results, and the leakage analysis. `column_changes.md` is the feature dictionary; `leakage_analysis.md` explains the temporal-split constraints.

**Outdated documents (historical context only, do not trust):** `README.md`, `Implementation_plan.md`, `OLD_Documentation.md`, `OLD_ETL Document — EDP Feature Tables.md` (still useful for the label/ETL SQL logic), `Walk-Forward Validation Implementation.md`. They reference Oracle/cx_Oracle and a Set-Transformer; the actual stack is MSSQL/pyodbc and DeepSets.

## Environment reality

- **The data and database are NOT on this machine.** Training runs on a Windows server (`F:\Loan Default Classification\`, 20-core CPU-only, MSSQL via pyodbc, table `D_ANALYTICS.DPD_SAMPLE1` aliased `EDP_Feature_Train`). On this Mac you can read/edit code but cannot execute the pipeline or load data.
- Evidence from past runs lives in `results_1/` (Run 1, biased split), `results_2/` (Run 2, val-test leakage), `results_3/` (Run 3, the only ~unbiased benchmark: baseline macro-F1 0.7046 vs DeepSets+XGB 0.6997, Cat-2 recall 0.873). `explore_output/iv_report.csv` has the IV analysis.
- After the first DB load, data is cached to `data/train_portfolios_cache.npz`; bump `DATA_VERSION` in `project_config.py` to invalidate.

## Commands (run on the training server)

```bash
python run.py train                          # full pipeline
python run.py train --resume <run_dir>       # resume; completed stages are skipped
python run.py predict --artifact_dir <dir> [--snapshot_date <YYYYMMDD> ...] [--output <csv>]
# snapshot_date defaults to PRED_SNAPSHOT_DATES, else every currently-immature snapshot
# output defaults to <artifact_dir>/predictions/predictions_<tag>.csv
python run.py explore                        # one-shot data profiling

# Standalone diagnostics (read NPZ cache, no DB needed):
python explore_iv_woe.py [--n_bins 15 --top_n 30]
python explore_umap.py
python explore_shap.py                       # needs test_embeddings.npy from server

pytest tests/                                # unit tests
pytest tests/test_dataset.py -k <name>       # single test
```

## Pipeline architecture (run.py)

Load (NPZ cache) → temporal split (or walk-forward folds) → per fold: preprocessing (fit on train only: impute → clip [1,99]pct → RobustScaler) → **Aggregated XGBoost baseline** (min/max/mean/std per feature + loan count = 257 features) → **DeepSets** (phi/pool/rho, 42K params, `CostSensitiveFocalLoss`) → **XGBoost meta-learner** on frozen 64-d embeddings → evaluation (macro F1, QWK, Brier, avg_cost, bootstrap CIs, plots).

Key behavioral toggles in `project_config.py`:
- `OPTIMIZE_ON_VALIDATION` + `VAL_SPLIT_MODE` — current default is `True` + `"customer"`: an in-time, customer-disjoint 20% holdout (stable md5 of `NATIONAL_CODE`) drives early stopping + Optuna without touching the test label window, and is then reused to fit the probability calibrator (final XGB trains on the 80% only). `VAL_SPLIT_MODE="temporal"` is the legacy leaky mode (val/test label windows overlap 5 months — Run 2). `OPTIMIZE_ON_VALIDATION=False` = no val set, `FIXED_EPOCHS` + fixed XGB params (Run 3 mode).
- `COST_MATRIX` — single source of truth for costs (DeepSets loss, expected-cost decision rule in `src/evaluation/decision.py`, `avg_cost` metric). Decisions and the top-K ranking use `argmin(probs @ COST_MATRIX)` on calibrated probs, never plain argmax.
- `WALK_FORWARD_ENABLED` — currently `False`; walk-forward finds 0 valid folds because two 6-month gaps don't fit in the 13-month data span.
- `PRED_SNAPSHOT_DATES` — snapshot(s) `predict` scores (int or list[int], overridden by `--snapshot_date`). `None` (default) auto-selects every currently-immature snapshot in `TRAIN_TABLE`; a requested date absent from the table is dropped (warn) and falls back the same way. There is no separate prediction table — `TRAIN_TABLE` holds both matured and not-yet-matured snapshots, and `Predictor` strips `WORST_FUTURE_CAT`/`WORST_FUTURE_DPD` before scoring since those columns hold a degenerate (not real) value on immature rows.

Stage checkpointing: each stage writes `<run_dir>/stages/<stage>.done` + a pickle. `--resume` skips completed stages. **Careful:** resuming after a config change silently reuses artifacts produced under the old config. Run dirs from before July 2026 (DATA_VERSION v1.0) are not resumable — instance dicts lack `current_cat`.

## Gotchas that matter

- **Active model is `src/model/deep_sets.py`.** `src/model/set_transformer.py` is dead code (never imported by run.py). Trainer class is named `TransformerTrainer` but trains DeepSets.
- `MAX_LOANS_PER_CUSTOMER` resolves at runtime to **2** (99th percentile). Most customers have 1 loan — this is why DeepSets ≈ baseline and why attention was abandoned.
- Truncation to MAX_LOANS sorts by `DPD_DAYS` desc (a prediction-time feature). It previously sorted by the label `WORST_FUTURE_DPD` — fixed July 2026; do not reintroduce label columns into `process_raw_data` sorting. The label is `max` over ALL loans while features keep only the first MAX_LOANS; each instance also carries `current_cat` (current worst category, drives stratified evaluation).
- Temporal split usability is computed against `date.today()` (`temporal_split._get_usable_snapshots`) — the same code produces different splits as calendar time passes. Snapshot dates are floats like `20241021.0`.
- Labels: features keep 5-category granularity; the target is capped at 3 classes (`min(max(cat), 2)`). For already-delinquent customers the label is largely mechanical DPD accrual — judge models on the `current_cat_0` slice (`by_current_cat` metrics), not aggregate F1.
- `explore_output/iv_report.csv` is **stale**: the old IV binning forced IV=0.0 for ~10 skewed/binary features (they are NOT constant — DB-verified). Re-run `explore_iv_woe.py` after the v1.1 cache rebuild.
- The inference deliverable is a **ranked top-K list** (external API budget limits how many customers can be enriched), not just class predictions. `Predictor` ranks by `RISK_SCORE` = expected cost of predicting class 0 (`probs @ COST_MATRIX[:,0]`) on calibrated probabilities.

## Code style

`.agent/rules/ponytail.md` applies: minimal code, stdlib/existing-deps first, no unrequested abstractions, deletion over addition. Non-trivial logic should leave one small runnable check behind.
