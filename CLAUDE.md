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
python run.py predict --artifact_dir <dir> --snapshot_date <YYYYMMDD> --output <csv>
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
- `OPTIMIZE_ON_VALIDATION` — `True`: val set + early stopping + Optuna (biased test metrics, because val and test label windows overlap by 5 months). `False`: no val set, `FIXED_EPOCHS` + fixed XGB params (unbiased metrics). This trade-off is the central methodological issue of the project.
- `WALK_FORWARD_ENABLED` — currently `False`; walk-forward finds 0 valid folds because two 6-month gaps don't fit in the 13-month data span.

Stage checkpointing: each stage writes `<run_dir>/stages/<stage>.done` + a pickle. `--resume` skips completed stages. **Careful:** resuming after a config change silently reuses artifacts produced under the old config (this affected Run 3, which reused Run 2's DeepSets checkpoint).

## Gotchas that matter

- **Active model is `src/model/deep_sets.py`.** `src/model/set_transformer.py` is dead code (never imported by run.py). Trainer class is named `TransformerTrainer` but trains DeepSets.
- `MAX_LOANS_PER_CUSTOMER` resolves at runtime to **2** (99th percentile). Most customers have 1 loan — this is why DeepSets ≈ baseline and why attention was abandoned.
- **Label-informed truncation:** `data_loader.process_raw_data` sorts loans by `WORST_FUTURE_DPD` (a label) descending before truncation to MAX_LOANS, and the label is `max` over ALL loans while features keep only the first MAX_LOANS. Also, the prediction table has no `WORST_FUTURE_*` columns, so the predict path (`load_pred_portfolios` → `sort_values`) will fail on it. Known issue — see AGENT_HANDOFF.md §What's Next.
- Temporal split usability is computed against `date.today()` (`temporal_split._get_usable_snapshots`) — the same code produces different splits as calendar time passes. Snapshot dates are floats like `20241021.0`.
- Labels: features keep 5-category granularity; the target is capped at 3 classes (`min(max(cat), 2)`).
- 10 features are constant/zero-IV in the current data (all 7 binary flags, `COUNT_90PLUS_DPD_LAST_3M`, `WORST_CLOSED_LOAN_DPD`, `AVERAGE_CLOSE_LOAN_DPD`) — suspected upstream ETL bugs, kept in the schema.
- The inference deliverable is a **ranked top-K list** (external API budget limits how many customers can be enriched), not just class predictions. `Predictor` outputs `RISK_SCORE = 2·P(cat2) + P(cat1)` sorted descending.

## Code style

`.agent/rules/ponytail.md` applies: minimal code, stdlib/existing-deps first, no unrequested abstractions, deletion over addition. Non-trivial logic should leave one small runnable check behind.
