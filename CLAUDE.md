# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A bank early-warning system that predicts the **worst delinquency class of a customer's loan portfolio over the next 6 months** (4 classes: 0 = No Delay, 1 = Current/Minor Delay, 2 = Past Due+, 3 = Severe Past Due). Records are monthly point-in-time snapshots of an upstream ETL table (~64 numeric per-loan features, no categoricals). Prediction is at customer level (`NATIONAL_CODE`).

**The deliverable is a ranked API queue** (July 2026 reframe): customers not yet severe (`current_cat < 3`), ranked by calibrated+masked P(entering class 3), consumed by an enrichment API at 240 requests/hour (`API_RATE_PER_HOUR`). Already-severe customers are rule-flagged, never ranked. **Headline metric: `ranking` block (recall/lift at K-hours, PR-AUC) from `full_evaluation`**, cost-matrix-free by design — the `COST_MATRIX` values are guessed (business gave the 4× anchor only; the class-3 row is extrapolated), so cost is a secondary diagnostic, never the target. The enrichment API returns present-time data only (cannot backtest past calls), so API value is only assessable forward.

**Key label property:** `WORST_FUTURE_CAT` includes the current month ⇒ `label >= current_cat` always (a loan can't improve past its current category). Not leakage — a definition. Exploited via `decision.mask_monotone` (zero impossible probability mass) and per-current-cat calibration (`StratifiedCalibrator`). It also makes already-delinquent slices mechanically easy — judge models on the `current_cat_0` slice and the ranking block, never aggregate F1.

**Read `AGENT_HANDOFF.md` first** — it is the authoritative record of decisions, run results, and the leakage analysis. `column_changes.md` is the feature dictionary; `leakage_analysis.md` explains the temporal-split constraints.

**Outdated documents (historical context only, do not trust):** `README.md`, `Implementation_plan.md`, `OLD_Documentation.md`, `OLD_ETL Document — EDP Feature Tables.md` (still useful for the label/ETL SQL logic), `Walk-Forward Validation Implementation.md`. They reference Oracle/cx_Oracle and a Set-Transformer; the actual stack is MSSQL/pyodbc and DeepSets.

## Environment reality

- **The data and database are NOT on this machine.** Training runs on a Windows server (`F:\Loan Default Classification\`, 20-core CPU-only, MSSQL via pyodbc, table `D_ANALYTICS.DPD_SAMPLE1` aliased `EDP_Feature_Train`). On this Mac you can read/edit code but cannot execute the pipeline or load data.
- Evidence from past runs lives in `results_1/` (Run 1, biased split), `results_2/` (Run 2, val-test leakage), `results_3/` (Run 3, the only ~unbiased benchmark: baseline macro-F1 0.7046 vs DeepSets+XGB 0.6997, Cat-2 recall 0.873). `explore_output/iv_report.csv` has the IV analysis.
- After the first DB load, data is cached to `data/train_portfolios_cache.npz`; bump `DATA_VERSION` in `project_config.py` to invalidate.

## Commands (run on the training server)

```bash
python run.py train                          # evaluation run (holds out newest mature snapshot as test)
python run.py train --final                  # deployment fit: ALL mature snapshots, no test, emits bundle
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
- `COST_MATRIX` — single source of truth for costs (DeepSets loss, expected-cost `PREDICTED_CLASS` in `src/evaluation/decision.py`, `avg_cost` metric). Class decisions use `argmin(probs @ COST_MATRIX)`; the queue RANKING does not use costs (see above).
- `CARVE_CURRENT_CAT_GE` / `API_RATE_PER_HOUR` / `RANKING_REF_WINDOWS` — define the ranked-queue population and the K values (K = rate × window hours) reported by `src/evaluation/ranking.py`.
- `CALIBRATION_MIN_STRATUM_N` — per-current-cat calibration floor; smaller strata fall back to the pooled calibrator.
- `WALK_FORWARD_ENABLED` — currently `False`. With 18 mature snapshots (2024-07…2025-12) walk-forward IS now feasible; deliberately deferred — decide after the next evaluation run whether temporal-stability evidence is worth ~1.5-2h/fold.
- `PRED_SNAPSHOT_DATES` — snapshot(s) `predict` scores (int or list[int], overridden by `--snapshot_date`). `None` (default) auto-selects every currently-immature snapshot in `TRAIN_TABLE`; a requested date absent from the table is dropped (warn) and falls back the same way. There is no separate prediction table — `TRAIN_TABLE` holds both matured and not-yet-matured snapshots, and `Predictor` strips `WORST_FUTURE_CAT`/`WORST_FUTURE_DPD` before scoring since those columns hold a degenerate (not real) value on immature rows.

Stage checkpointing: each stage writes `<run_dir>/stages/<stage>.done` + a pickle. `--resume` skips completed stages. **Careful:** resuming after a config change silently reuses artifacts produced under the old config. Run dirs from before July 2026 (DATA_VERSION v1.0) are not resumable — instance dicts lack `current_cat`.

## Gotchas that matter

- **Active model is `src/model/deep_sets.py`.** `src/model/set_transformer.py` is dead code (never imported by run.py). Trainer class is named `TransformerTrainer` but trains DeepSets.
- `MAX_LOANS_PER_CUSTOMER` resolves at runtime to **2** (99th percentile). Most customers have 1 loan — this is why DeepSets ≈ baseline and why attention was abandoned.
- Truncation to MAX_LOANS sorts by `DPD_DAYS` desc (a prediction-time feature). It previously sorted by the label `WORST_FUTURE_DPD` — fixed July 2026; do not reintroduce label columns into `process_raw_data` sorting. The label is `max` over ALL loans while features keep only the first MAX_LOANS; each instance also carries `current_cat` (current worst category, drives stratified evaluation).
- Temporal split usability is computed against `date.today()` (`temporal_split._get_usable_snapshots`) — the same code produces different splits as calendar time passes. Snapshot dates are floats like `20241021.0`.
- Labels: features keep 5-category granularity; the target is capped at `config.NUM_CLASSES` classes (`min(max(cat), NUM_CLASSES - 1)`, currently 4: raw cats 0/1/2 pass through 1:1, raw cats 3-4 collapse into class 3). For already-delinquent customers the label is largely mechanical DPD accrual — judge models on the `current_cat_0` slice (`by_current_cat` metrics), not aggregate F1.
- `run.py train` now also writes `<fold_dir>/model_bundle.pkl` — a single-file deployment artifact (fitted scaler + DeepSets state_dict/hparams + XGBoost raw bytes + calibrator) that `ModelLoader`/`Predictor` can load directly by pointing `--artifact_dir` at the `.pkl` file instead of the fold directory. It's a pure export convenience; the existing per-file directory artifacts are unchanged and still used in-pipeline.
- `explore_output/iv_report.csv` is **stale**: the old IV binning forced IV=0.0 for ~10 skewed/binary features (they are NOT constant — DB-verified). Re-run `explore_iv_woe.py` after the v1.1 cache rebuild.
- `Predictor` output: `RISK_RANK` (queue position, NaN for flagged rows), `RISK_SCORE` = calibrated+masked P(severe), `RULE_FLAG` ∈ {"", ALREADY_SEVERE, SUPERSEDED, PREDICTED_SEVERE, RECENTLY_CALLED}, `EXPECTED_COST` (secondary). When several snapshots are scored, `PRED_DEDUP_LATEST` keeps only each customer's newest row in the queue (stale scores waste API budget — the API can't be queried "as of" the past). `--called_log <csv>` (NATIONAL_CODE, CALLED_AT) flags customers called within `API_DATA_TTL_DAYS` (~30d, business-confirmed freshness window). `CERTAINTY_ACT_THRESHOLD` (default None = off, pending business decision) flags near-certain P(severe) rows to act on directly, reserving API calls for uncertain cases.
- The evaluation run also trains a **binary severe-event comparator** (`BinarySevereBaseline`, same 257 aggregated features, `binary:logistic`) — its `binary_ranking` block vs the multiclass `ranking` block answers whether the multiclass pipeline costs ranking quality, and informs the deferred per-current-cat-models idea.

## Code style

`.agent/rules/ponytail.md` applies: minimal code, stdlib/existing-deps first, no unrequested abstractions, deletion over addition. Non-trivial logic should leave one small runnable check behind.
