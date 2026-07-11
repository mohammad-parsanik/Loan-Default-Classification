# Loan Default Classification — Agent Handoff Document

> **Purpose:** This document gives a new agent the full context needed to continue work on this project without re-discovering decisions, trade-offs, or bugs that have already been resolved.

---

## 1. Problem Definition

A bank needs to predict the **worst future delinquency state** of a customer's entire loan portfolio over a **6-month forward horizon**, to prioritize collection actions.

- **Prediction level:** Customer (grouped by `NATIONAL_CODE`), not individual loan.
- **Target:** 4-class classification (bumped from 3 classes on 2026-07-08 — see §11 item 10):
  - `0` — No Delay (performing)
  - `1` — Current / Minor Delay (pre-delinquent)
  - `2` — Past Due+ (NPL, raw category 2 only)
  - `3` — Severe Past Due (raw categories 3-4 collapsed)
- **Label construction:** For each customer-snapshot, label = `min(max(WORST_FUTURE_CAT across all loans), config.NUM_CLASSES - 1)`. Features retain the full 5-category granularity internally; only the prediction target is capped (currently to 4 classes).
- **Business constraint:** This is heavily cost-sensitive. Missing a Cat-2 customer is penalized **4×** more than a false positive over-flagging; the class-3 (Severe Past Due) costs are a derived placeholder, not yet business-tuned. The single source of truth is `project_config.COST_MATRIX`.
- **Prior work:** A previous project used per-loan LightGBM classifiers and performed poorly. This project replaces that approach with a portfolio-level architecture.

---

## 2. Dataset Description

### Source & Ingestion
- Data originates from an **MSSQL** database (table `D_ANALYTICS.DPD_SAMPLE1`, aliased as `EDP_Feature_Train` in config), accessed via `pyodbc` through `src/db/mssql_connection.py`.
- **Important:** The original implementation plan and README mention Oracle/cx_Oracle — this is outdated. The actual codebase was migrated to **MSSQL (`pyodbc`)** early in development. The config file (`project_config.py`) reflects MSSQL credentials.
- After the first database load, portfolios are cached to disk as a compressed NPZ file (`data/train_portfolios_cache.npz`) with a manifest file for cache invalidation (`DATA_VERSION` in config). Subsequent runs load from this cache in ~10 seconds instead of hitting the DB.

### Snapshots & Timeline
The data contains **8 raw snapshot dates** in format `YYYYMMDD` as floats:

| Snapshot | Date | Status |
|----------|------|--------|
| S1 | `20241021` | Usable — used for training |
| S2 | `20250119` | Usable — used for training |
| S3 | `20250420` | Usable — used for training |
| S4 | `20250621` | Usable — dropped (within 6-month gap window before val/test) |
| S5 | `20250922` | Usable — dropped (within 6-month gap window before val/test) |
| S6 | `20251022` | Usable — used as Val (when enabled) |
| S7 | `20251121` | Usable — used as Test |
| S8 | `20260521` | **Dropped** — labels not yet mature (< 6 months old) |

Total usable span: ~13 months (Oct 2024 – Nov 2025). Total volume: **~5 million** customer-portfolio instances across all usable snapshots, **~64 features** per loan.

### Feature Schema
All **~64 features are strictly numeric**. No categorical features exist in this dataset. Features are grouped into:

| Group | Description | Count |
|-------|-------------|-------|
| A — Current DPD State | `DPD_DAYS`, `LOAN_CATEGORY`, `OVERDUE_RATIO`, etc. | 13 |
| B — DPD Trajectory | `DPD_DAYS_T1`–`T5`, `CATEGORY_T1`–`T3` (lag features) | 8 |
| C — Trend & Velocity | `DPD_TREND_1M/3M`, `IS_DETERIORATING`, etc. | 8 |
| D — Historical Worst | `HIST_MAX_DPD_DAYS`, `HAS_EVER_BEEN_NPL`, etc. | 7 |
| E — DPD Event Counts | `COUNT_DPD_EVENTS_LAST_3M`, `DAYS_SINCE_LAST_DPD`, etc. | 14 |
| F — Cross-Contract | `WORST_CLOSED_LOAN_DPD`, `COUNT_ACTIVE_CONTRACTS`, etc. | 10 |
| G — Contract Maturity | `CONTRACT_AGE_MONTH`, `PCT_COMPLETED`, etc. | 4 |

Full feature dictionary: see `column_changes.md`.

Meta/excluded columns (`META_COLS` in config): `LOAN_ID`, `CONTRACT_NUMBER`, `NATIONAL_CODE`, `SNAPSHOT_DATE`, `WORST_FUTURE_CAT`, `WORST_FUTURE_DPD`. These are never used as model inputs.

> **Note:** `RECORD_STATUS_CODE` was originally in META_COLS but was removed — the column does not exist in the current dataset. If it reappears, check with stakeholders whether inactive/closed records should be filtered.

### Critical Data Shape: MAX_LOANS = 2
Data profiling revealed the **99th percentile of loans per customer is 2**. The vast majority of customers have only 1 active loan. This has major architectural implications (see Section 3).

---

## 3. Architecture & Implementation

### High-Level Pipeline (run.py)

```
Stage 1: Load data (from DB or NPZ cache)
Stage 2: Temporal split (or walk-forward fold generation)
   For each fold:
     Stage 3: Compute MAX_LOANS (99th percentile on train)
     Stage 4: Preprocessing (fit on train, transform all)
     Stage 5: Aggregated XGBoost baseline
     Stage 6: Build PyTorch DataLoaders
     Stage 7: DeepSets training (with Cost-Sensitive Focal Loss)
     Stage 8: XGBoost meta-learner (Optuna HPO on embeddings)
     Stage 9: Final evaluation + plots
   Aggregate fold results (if walk-forward)
```

Each stage has crash-safe checkpointing (`StageCheckpointer` in `run.py`). On `--resume <run_dir>`, completed stages are skipped.

### Core Model: DeepSets (not Set-Transformer)

> **Critical clarification:** The README and implementation plan mention a "Set-Transformer." While `src/model/set_transformer.py` exists, **the active model used in training is `src/model/deep_sets.py`**. The `run.py` pipeline imports and trains `DeepSets`, not `SetTransformer`.

**Why DeepSets was chosen over the Set-Transformer:** With MAX_LOANS ≤ 2, self-attention on 1-2 tokens degenerates to a weighted average. DeepSets (phi + pool + rho) is provably permutation-invariant, has far fewer parameters (~42K vs ~173K), and is faster on CPU. The Set-Transformer file is retained but unused.

**DeepSets Architecture:**
```
phi (per-loan MLP):  Linear(64→128) → LN → GELU → DO → Linear(128→128) → LN → GELU → DO
pool:                concat(masked_mean, masked_max) → (B, 256)
rho (customer MLP):  Linear(256→64) → LN → GELU → DO
head (training only): Linear(64→3)
```
Parameters: **42,115**. Embedding dim fed to XGBoost: **64**.

### Loss Function: Cost-Sensitive Focal Loss (`src/model/losses.py`)
Combines Focal Loss (γ=2.0) with a cost matrix. The cost matrix penalizes:
- Missing Cat-2 → predicting 0: **cost = 4.0** (most expensive mistake)
- Missing Cat-2 → predicting 1: **cost = 2.0**
- Over-flagging Cat-0 → predicting 2: **cost = 1.0**

### XGBoost Meta-Learner (`src/model/meta_learner.py`)
After DeepSets training, the model is frozen. 64-dim embeddings are extracted for train/val/test. XGBoost trains on these embeddings with:
- `objective='multi:softprob'`, `num_class=3`
- Sample weights from inverse class frequency
- Optuna HPO (30 trials, SQLite-backed for crash resilience)
- After HPO, retrains on Train+Val with best params (using `best_ntree_limit` from early stopping)

When `OPTIMIZE_ON_VALIDATION = False`: skips Optuna entirely, trains with fixed hyperparameters (`n_estimators=100, max_depth=6, lr=0.1`).

### Aggregated XGBoost Baseline (`src/baselines/aggregated_xgboost.py`)
For each customer: computes mean, max, min, std of all ~64 features + loan count → **257 aggregated features**. XGBoost trains directly on these. Same temporal split and labels. This is the "does the Transformer add value?" benchmark.

### Preprocessing Pipeline (`src/data/preprocessing.py`)
Fully vectorized sklearn-style pipeline:
1. **DomainAwareImputer** — trajectory/binary features → 0; `days_since_*` → max_train + 1; amounts/ratios → median
2. **OutlierClipper** — continuous cols clipped at [1st, 99th] percentile
3. **PortfolioRobustScaler** — (x - median) / IQR for continuous cols; binary/small-count cols pass through

Fitted on training data only. Serialized per-fold to `scaler.pkl`.

### Trainer (`src/model/trainer.py`)
- AdamW (lr=5e-4, weight_decay=1e-4)
- CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- Early stopping on val Macro F1 (patience=10) when validation enabled
- Fixed epochs (`FIXED_EPOCHS=15`) when validation disabled
- Gradient clipping (max_norm=1.0)
- Per-epoch checkpoint saving

---

## 4. Configuration Flags (project_config.py)

These are the key toggles a new agent needs to understand:

| Flag | Current Value | Purpose |
|------|---------------|---------|
| `WALK_FORWARD_ENABLED` | `False` | `True` = rolling-window CV across all valid folds. `False` = single static temporal split. Currently `False` because walk-forward found 0 valid folds with the available data (see Section 5). |
| `OPTIMIZE_ON_VALIDATION` | `True` | `True` = use a val set for early stopping + Optuna. `False` = skip Optuna, train for FIXED_EPOCHS with fixed XGB params, no val set created (Run 3 mode). |
| `VAL_SPLIT_MODE` | `"customer"` | How the val set is built when optimizing: `"customer"` = in-time customer-disjoint holdout (leakage-free tuning, see §6). `"temporal"` = legacy second-newest-snapshot val (leaky, Run 2). |
| `CUSTOMER_VAL_FRACTION` | `0.20` | Share of customers held out when `VAL_SPLIT_MODE="customer"`. |
| `COST_MATRIX` | 4×4 | Single source of truth for costs. Shared by the DeepSets loss, the expected-cost decision rule (`src/evaluation/decision.py`), and the `avg_cost` metric. The class-3 row/col is a derived placeholder (exactly reproduces the old 3×3 sub-block) pending business tuning. |
| `BASELINE_COST_WEIGHTS` | `True` | Baseline XGBoost sample weights scaled by cost-matrix row sums (same nudge the DeepSets gets from its loss) — makes the baseline-vs-model comparison fair. |
| `RECALIBRATE_ON_PREDICT` | `True` | At predict time, refit the probability calibrator on the newest matured-label snapshot (tracks base-rate drift). |
| `FIXED_EPOCHS` | `15` | Number of DeepSets training epochs when `OPTIMIZE_ON_VALIDATION = False`. |
| `LABEL_HORIZON_MONTHS` | `6` | Forward prediction window. Also the minimum gap required between splits. |
| `DATA_VERSION` | `"v1.1"` | Bump this to force NPZ cache invalidation when ETL changes. v1.1 = truncation sort fix + `current_cats` in cache (old caches/run dirs are not resumable). |
| `MAX_LOANS_PER_CUSTOMER` | `None` | Computed at runtime (99th percentile). Currently resolves to `2`. |

---

## 5. Training Runs & Results

### Run 1 — Initial Static Split (5 snapshots, 3 Train / 1 Val / 1 Test)
- **Date:** July 1, 2026 (first run)
- **Snapshots:** Train: [Oct'24], Val: [Jun'25], Test: [Sep'25] (only 5 snapshots available)
- **Config:** `OPTIMIZE_ON_VALIDATION = True`, `WALK_FORWARD_ENABLED = False`
- **Data:** 634K train / 639K val / 634K test instances
- **Results:**

| Metric | Baseline (Agg. XGB) | DeepSets + XGBoost | Delta |
|--------|---------------------|-------------------|-------|
| Macro F1 | 0.7395 | **0.7454** | +0.006 |
| QWK | 0.7448 | 0.7477 | +0.003 |
| Cat-2 Recall | — | 0.8522 | — |
| Brier Score | 0.1169 | 0.1118 | -0.005 |

- **Notes:** DeepSets early-stopped at epoch 12 (best at epoch 2, val F1=0.7697). XGBoost Optuna ran 30 trials in ~16 min. Total pipeline time: **32 min**.
- **Key insight:** DeepSets+XGB barely beat baseline (+0.6% F1). This is expected given MAX_LOANS=2.

### Run 2 — With More Snapshots + Walk-Forward Attempt (7 usable snapshots)
- **Date:** July 1, 2026 (evening, after more snapshots became available)
- **Config:** `WALK_FORWARD_ENABLED = True`, `OPTIMIZE_ON_VALIDATION = True`
- **Data:** 7 usable snapshots, but walk-forward found **0 valid folds** (see leakage analysis below). Fell back to static split.
- **Snapshots:** Train: [Oct'24, Jan'25, Apr'25] (3 snapshots), Val: [Oct'25], Test: [Nov'25]
- **Data:** 1.92M train / 638K val / 633K test instances
- **Results:**

| Metric | Baseline (Agg. XGB) | DeepSets + XGBoost | Delta |
|--------|---------------------|-------------------|-------|
| Macro F1 | 0.7046 | **0.7090** | +0.004 |
| QWK | 0.7066 | 0.7066 | +0.000 |
| Cat-2 Recall | — | 0.7648 | — |
| Brier Score | 0.1238 | 0.1226 | -0.001 |

- **Notes:** Metrics dropped vs Run 1. This is expected — Run 1's Train/Val/Test had overlapping label windows (inflating scores). Run 2 enforced the 6-month Train→Val gap but still had **Val-Test leakage** (Val and Test only 1 month apart, 5 months of label overlap).
- Total pipeline time: **87 min** (larger train set + Optuna took 54 min).

### Run 3 — Leakage-Free Run (OPTIMIZE_ON_VALIDATION = False)
- **Date:** July 2, 2026
- **Config:** `OPTIMIZE_ON_VALIDATION = False`, `WALK_FORWARD_ENABLED = False`
- Ran in a fresh run dir (`artifacts/20260702_110830`); a mid-run crash was fixed and the run resumed, which is why the log shows `[skip]` lines. The data cache, preprocessing, and portfolio artifacts were reused; DeepSets hyperparameters came from the config file.
- **Verified clean (July 7, 2026):** the checkpoint directory contains exactly 15 `epoch_*.pt` files — the DeepSets weights were retrained fresh for `FIXED_EPOCHS=15` under the no-validation config, NOT reused from Run 2's early-stopped training. Run 3's metrics are genuinely unbiased.
- **Snapshots:** Train: [Oct'24, Jan'25, Apr'25], Val: ∅ (no val set), Test: [Nov'25]
- **Data:** 1.92M train / 0 val / 633K test instances
- **Results:**

| Metric | Baseline (Agg. XGB) | DeepSets + XGBoost | Delta |
|--------|---------------------|-------------------|-------|
| Macro F1 | 0.7046 | **0.6997** | -0.005 |
| QWK | 0.7066 | 0.7126 | +0.006 |
| Cat-2 Recall | — | **0.8730** | — |
| Brier Score | 0.1238 | 0.1245 | +0.001 |

- **Notes:** This is the **only unbiased test metric** we have. The DeepSets+XGB pipeline slightly underperforms the baseline on Macro F1 (-0.5%) but has excellent Cat-2 recall (87.3%) and better QWK. The model was trained with fixed hyperparameters (no Optuna, no early stopping), so no information from the test window leaked through optimization.
- **Key takeaway:** The baseline and model are neck-and-neck, which is expected with MAX_LOANS=2. The DeepSets model's value is in its much higher Cat-2 recall (which the cost-sensitive loss is designed to achieve).

---

## 6. The Leakage Problem (Critical Context)

This is the most important architectural decision in the project. Full analysis: `leakage_analysis.md`.

### Why Walk-Forward Found 0 Folds
Strict walk-forward requires **two** 6-month buffer gaps: Train→Val and Val→Test. That's 12+ months of "padding" + at least 3 snapshots = ~15 months minimum. We only have 13 months.

### The Val-Test Overlap Problem
In the static split fallback, Val (Oct'25) and Test (Nov'25) are only 1 month apart. Their 6-month label windows overlap by 5 months. When Optuna and early stopping optimize against the Val labels, they are indirectly optimizing against the Test window — this is **model selection leakage**. The 0.7090 Macro F1 from Run 2 is biased upward.

### Resolution 1 (Run 3): `OPTIMIZE_ON_VALIDATION = False`
When set to `False`:
- `temporal_split.py` creates no validation set; the gap is enforced directly against test
- `trainer.py` trains for `FIXED_EPOCHS` without early stopping
- `meta_learner.py` uses fixed XGB hyperparameters (no Optuna)
- Result: **unbiased test metrics**, but potentially worse model (no tuning)

### Resolution 2 (July 7, 2026 — current default): customer-disjoint in-time validation
`OPTIMIZE_ON_VALIDATION = True` + `VAL_SPLIT_MODE = "customer"`:
- Train snapshots stay ≥ 6 months before test (same as the no-val branch)
- A `CUSTOMER_VAL_FRACTION` (20%) customer-disjoint holdout is carved from the
  train snapshots (stable md5 hash of `NATIONAL_CODE` — same customer never on
  both sides)
- Early stopping + Optuna run against this holdout. Its labels come from the
  training era and never overlap the test label window → **tuning restored,
  test metrics stay unbiased**
- The final XGBoost trains on the 80% only; the 20% is reused to fit the
  probability calibrator (honest only on data the final model never saw)
- Caveat: hyperparameters are selected for cross-customer generalisation
  within the training era, not across time — a much smaller, non-leaking risk

### Trade-off Summary
| Approach | Test F1 | Bias | When to Use |
|----------|---------|------|-------------|
| `True` + `VAL_SPLIT_MODE="temporal"` | 0.7090 | **Biased** (upper bound) | Legacy Run 2 behaviour; avoid |
| `False` | 0.6997 | **Unbiased**, no tuning | Honest baseline without any val set |
| `True` + `VAL_SPLIT_MODE="customer"` | TBD (Run 4) | **Unbiased**, tuning enabled | Current default |

---

## 7. Exploration & Diagnostic Tools

Three standalone scripts exist in the project root for data quality analysis. These are documented in `EXPLORATION.md`:

| Script | Purpose |
|--------|---------|
| `explore_iv_woe.py` | Information Value & Weight of Evidence per feature, One-vs-Rest for all `config.NUM_CLASSES` classes. Reads from NPZ cache (no DB needed). |
| `explore_umap.py` | UMAP projection of raw features or model embeddings with CLI-tunable hyperparameters. |
| `explore_shap.py` | SHAP TreeExplainer on XGBoost meta-learner. Needs `.npy` embeddings copied from the training server. |

**Key finding from IV analysis:** Top features (`LOAN_CATEGORY`, `DPD_DAYS`) had very high IV values (>0.50, which normally signals leakage). After investigation, the conclusion was these are **genuinely strong signals** — they're point-in-time observations available at prediction time. Note this is partly mechanical: DPD is a counter, so for already-delinquent customers the future label is largely predetermined by continued accrual. That is why evaluation is now also reported per current-category slice (the currently-clean slice is the real early-warning task).

**Correction (July 7, 2026) — the "ten dead features" were an analysis artifact, not an ETL bug.** The old `explore_iv_woe.py` quantile binning collapsed to a single bin for any feature with ≳80% of its mass on one value, silently forcing IV = 0.0 for all 7 binary flags, `COUNT_90PLUS_DPD_LAST_3M`, and the closed-loan DPD columns. A DB check confirmed these columns ARE populated and varying. The binning is fixed (`_make_bins`: per-value bins for low-cardinality features, mode-vs-rest fallback for skewed ones); **re-run the script on the server to get true IVs** before drawing feature conclusions.

---

## 8. Environment & Infrastructure

| Item | Detail |
|------|--------|
| **Training server** | Windows, CPU-only: 20-core @ 2.5 GHz, 128 GB RAM |
| **Python environment** | `uv`-managed venv at `/Users/mohammad/.venv/` (macOS dev) or `F:\Loan Default Classification\.venv\` (Windows server) |
| **Database** | MSSQL via `pyodbc` with ODBC Driver 17 |
| **Key dependencies** | `torch>=2.0`, `xgboost>=2.0`, `optuna>=3.0`, `pyodbc>=4.0` (see `requirements.txt`) |
| **No GPU** | Pipeline is CPU-only. `torch.compile` is skipped on Windows (inductor requires MSVC). CPU threading is configured: 12 compute / 4 interop. |
| **Data cache** | `data/train_portfolios_cache.npz` (~204 MB). Invalidated by changing `DATA_VERSION` in config. |

---

## 9. Project File Map

```
Loan Default Classification/
├── project_config.py              # All hyperparameters, DB creds, feature lists, toggle flags
├── run.py                         # CLI entry point: train / predict / explore (604 lines)
├── requirements.txt               # Python dependencies
├── AGENT_HANDOFF.md               # This document
├── column_changes.md              # Full feature data dictionary
├── leakage_analysis.md            # Detailed Val-Test leakage analysis
├── EXPLORATION.md                 # Usage guide for explore_*.py scripts
│
├── src/
│   ├── db/mssql_connection.py     # MSSQL connector (pyodbc)
│   ├── data/
│   │   ├── data_loader.py         # Vectorized load + NPZ cache
│   │   ├── data_explorer.py       # One-time data profiling
│   │   ├── dataset.py             # PyTorch Dataset + padding/masking
│   │   ├── preprocessing.py       # Impute → Clip → Scale pipeline
│   │   └── temporal_split.py      # Static split + walk-forward fold generation
│   ├── model/
│   │   ├── deep_sets.py           # ★ Active model (42K params)
│   │   ├── set_transformer.py     # Retained but UNUSED in pipeline
│   │   ├── losses.py              # Cost-Sensitive Focal Loss
│   │   ├── meta_learner.py        # XGBoost on frozen embeddings + Optuna
│   │   └── trainer.py             # Training loop with early stopping
│   ├── evaluation/
│   │   ├── metrics.py             # Macro F1, QWK, Brier, Bootstrap CI
│   │   ├── visualization.py       # Confusion matrix, ROC, UMAP, training curves
│   │   └── fold_aggregator.py     # Walk-forward fold metric aggregation
│   ├── baselines/
│   │   └── aggregated_xgboost.py  # Statistical aggregation + XGBoost baseline
│   └── inference/
│       ├── predictor.py           # End-to-end scoring pipeline
│       └── model_loader.py        # Load saved artifacts
│
├── data/                          # NPZ cache + manifest
├── artifacts/                     # Per-run artifacts (models, plots, reports)
├── results_1/                     # Logs + plots from Run 1 (initial 5-snapshot run)
├── results_2/                     # Log from Run 2 (walk-forward attempt, 7 snapshots)
├── results_3/                     # Log from Run 3 (leakage-free run)
└── tests/                         # Unit tests for dataset, transformer, losses, preprocessing
```

---

## 10. Known Issues & Gotchas

1. **README.md is outdated.** It still references Oracle, `cx_Oracle`, and the Set-Transformer as the active model. The actual DB is MSSQL/pyodbc, and the active model is DeepSets.
2. **`Implementation_plan.md` is the original design doc** — many details have evolved (DeepSets replaced Transformer, Oracle replaced by MSSQL, walk-forward added then found infeasible). Treat it as historical context, not current truth.
3. **`set_transformer.py` is dead code.** It exists but is never imported by `run.py`. The pipeline uses `deep_sets.py`.
4. **Snapshot dates are stored as floats** (e.g., `20241021.0`), not integers or datetime. All temporal logic in `temporal_split.py` converts them to `datetime.date` for gap calculations.
5. **The baseline consistently matches or beats DeepSets+XGB on Macro F1.** This is expected with MAX_LOANS=2. The DeepSets model's Cat-2 recall advantage (87% vs 76-78% in Runs 2-3) was **confounded**: the DeepSets had a cost-sensitive loss while the baseline used plain argmax. Since July 7, 2026 both systems are evaluated under the same expected-cost decision rule (`cost_rule` metrics) — use those for architecture comparisons.
6. **`torch.compile` is disabled on the Windows training server** (inductor requires MSVC). The model runs in eager mode.
7. **Label-informed truncation bug (FIXED July 7, 2026):** `process_raw_data` used to sort each customer's loans by `WORST_FUTURE_DPD` (a label) before `MAX_LOANS` truncation — the label selected which loans the model saw, and the predict path crashed because `EDP_Feature_pred` has no `WORST_FUTURE_*` columns. Now sorts by `DPD_DAYS` desc in both paths; unlabelled tables get `label = -1`. Requires cache rebuild (`DATA_VERSION` v1.1).
8. **`EDP_Feature_pred` never existed as a separate table (FIXED July 7, 2026):** the live DB has one table (`TRAIN_TABLE`) holding matured snapshots plus the newest not-yet-matured one(s). `PRED_TABLE` is gone; `Predictor` now reads `TRAIN_TABLE` for prediction too. Because that table carries `WORST_FUTURE_CAT`/`WORST_FUTURE_DPD` for every row, an immature snapshot's label columns hold a **degenerate** value ("worst category observed so far", not the real future outcome) rather than being absent/NULL — `DataLoader.load_pred_portfolios` now drops those columns before `process_raw_data` so they can't masquerade as real labels.

---

## 11. Changes of July 7-8, 2026 (methodology overhaul + 4-class bump — not yet run on server)

All verified by `tests/test_pipeline_changes.py` (7 tests, synthetic data, no DB).

1. **Truncation leak + predict-path crash fixed** (`data_loader.py`) — see Gotcha 7.
2. **Expected-cost decision rule** (`src/evaluation/decision.py`): predictions and the top-K ranking use `argmin(probs @ COST_MATRIX)` instead of argmax / ad-hoc `2·P₂+P₁`. Applied identically to baseline and model via `metrics.full_evaluation`.
3. **Customer-disjoint in-time validation** (`temporal_split.split_train_by_customer`, `VAL_SPLIT_MODE="customer"`): restores early stopping + Optuna without test leakage (§6, Resolution 2).
4. **Per-class isotonic calibration** (`src/evaluation/calibration.py`): fitted on the customer holdout (both baseline and meta-learner), saved as `calibrator.pkl`, applied before decisions/ranking. `Predictor` refreshes it on the newest matured snapshot when `RECALIBRATE_ON_PREDICT=True`.
5. **Stratified evaluation**: every instance now carries `current_cat` (current worst `LOAN_CATEGORY`, capped to `config.NUM_CLASSES` classes, computed over ALL loans pre-truncation). Test metrics are reported per slice (`by_current_cat`). The `current_cat_0` slice — currently-clean customers — is the real early-warning task; aggregate F1 is inflated by the mechanical already-delinquent slices.
6. **`metadata.json` now written per fold** — `ModelLoader` always required it but nothing produced it (inference would have failed).
7. **IV binning fixed** in `explore_iv_woe.py` (see §7 correction). `explore_output/iv_report.csv` is stale until re-run.
8. Cost matrix consolidated into `project_config.COST_MATRIX` (was duplicated in `losses.py` and `metrics.py`).
9. **Prediction unified onto `TRAIN_TABLE`, multi-snapshot + auto-selection** (`project_config.PRED_SNAPSHOT_DATES`, `MSSQLConnector.get_available_snapshots`, `DataLoader.resolve_pred_snapshots`, `Predictor.predict`): `PRED_TABLE`/`EDP_Feature_pred` removed. `--snapshot_date` (CLI, now `nargs="*"`) or `PRED_SNAPSHOT_DATES` (config) pick the snapshot(s) to score; unset or not-found dates fall back to every currently-immature snapshot, then to the single latest snapshot if even that's empty. `--output` is optional, defaulting to `<artifact_dir>/predictions/predictions_<tag>.csv`. See Gotcha 8 for the accompanying degenerate-label fix.
10. **NUM_CLASSES 3 → 4, single-file deployment bundle (2026-07-08):** `config.NUM_CLASSES=4` — raw cats 0/1/2 pass through 1:1, raw cats 3-4 now collapse into a new class 3 ("Severe Past Due") instead of into class 2. `COST_MATRIX` is now 4×4 (class-3 row/col is a derived, not-yet-business-tuned placeholder — see the table above). Every hardcoded `3`/`range(3)` tied to class count was swept to `config.NUM_CLASSES` (`losses.py`, `deep_sets.py`, `meta_learner.py`, `aggregated_xgboost.py`, `metrics.py`, `visualization.py`, `explore_iv_woe.py`, `explore_shap.py`, `explore_umap.py`); `bootstrap_confidence_intervals` now tracks both `recall_class_2` and `recall_class_3`; `Predictor` emits a 4th probability column `P_SEVERE_PAST_DUE`. `DATA_VERSION` bumped to `v1.2` (labels are baked into the NPZ cache, so this forces a rebuild). Separately, `train_single_fold` now also writes `<fold_dir>/model_bundle.pkl` — everything `Predictor` needs (scaler, DeepSets state_dict + hparams, XGBoost raw model bytes, calibrator) in one file; `ModelLoader.load_pipeline()` dispatches to it when `artifact_dir` points at a file instead of a directory. Pure export convenience — the per-file directory artifacts are unchanged and still used in-pipeline. All verified locally via `tests/test_pipeline_changes.py` (20 tests, synthetic data, no DB) — **not yet run on the training server.**

## 12. Objective Reframe & Ranked API Queue (July 8, 2026 — after Run 4)

Run 4 ran on the server (results_4/): the July-7 machinery worked; aggregate F1 0.7029 vs baseline 0.6597, but the stratified slices exposed that the aggregate blends a trivial slice (current cat-2 → label 2 with prob 1.0, F1=1.000, 15% of test) with the real early-warning task (cat_0 slice: F1 ~0.58, Cat-2 recall 0.79 vs baseline 0.68). Mohammad then supplied new facts that reframed the deliverable:

- **API budget is a rate**: 240 requests/hour — so "top-K" = "hours of calling" (`API_RATE_PER_HOUR`, `RANKING_REF_WINDOWS`: 1 day ≈ 5.8K, 1 week ≈ 40K, 1 month ≈ 173K).
- **The API cannot be queried for the past** → enrichment can never be backtested; model evaluation uses only our own label; API value is assessable only forward. Score the freshest snapshot, call promptly.
- **Data expanded**: monthly snapshots 2024-07 … 2025-12 (18, all now mature) + immature 2026-05/06 for scoring. Walk-forward is now FEASIBLE (the 13-month wall is gone) but deliberately DEFERRED — Mohammad is unsure it helps; revisit after the next run. (WF grades the *recipe*, mean±std across folds; the deployed model is then retrained on all data — that's what `train --final` does.)
- **Objective sharpened**: find customers with high probability of ENTERING class 3 (severe). Already-severe customers are rule-flagged, never ranked (`CARVE_CURRENT_CAT_GE=3`).
- **Label monotonicity confirmed**: `WORST_FUTURE_CAT` includes the current month ⇒ `label >= current_cat` always. Not leakage; exploited via `decision.mask_monotone` + `StratifiedCalibrator` (per-current-cat isotonic, pooled fallback under `CALIBRATION_MIN_STRATUM_N`).
- **Metric decision** (Mohammad: cost-matrix numbers are guesses; no ground truth for "was the API call right"): headline = **ranking metrics on the observed label** — recall@K-hours / lift / PR-AUC of P(severe) over the carved population (`src/evaluation/ranking.py`, `ranking` block in `full_evaluation`). Cost metrics demoted to secondary diagnostics.

Implementation landed (2026-07-08, all local-tested — 26 tests): `ranking.py`; `mask_monotone` + `severity_scores`; `StratifiedCalibrator` everywhere a calibrator is fit; `full_evaluation` reworked (calibrate → mask → `ranking`/`argmax_cal`/`cost_rule`/`by_current_cat`); `BinarySevereBaseline` comparator (does multiclass cost ranking quality? also informs the deferred per-current-cat-models idea); `Predictor` queue output (`RISK_RANK`, `RISK_SCORE`=P(severe), `RULE_FLAG` ALREADY_SEVERE/SUPERSEDED, `PRED_DEDUP_LATEST`); `run.py train --final` (deployment fit on ALL mature snapshots, no test, bundle out); fold_aggregator/explore-script 4-class residue.

Business answers received (2026-07-08 follow-up):
- (a) **API freshness ≈ 1 month** ("the oldest data we could consider would be a month old") → `API_DATA_TTL_DAYS=30` + call-ledger support: `predict --called_log <csv>` (columns `NATIONAL_CODE, CALLED_AT`, appended by the API-calling process) flags customers with a fresh call as `RECENTLY_CALLED` (skipped by the queue).
- (b) Mohammad will ask business whether near-certain severe predictions should be acted on directly (saving API calls for genuinely uncertain cases — a call buys no information when the decision wouldn't change). Implemented behind `CERTAINTY_ACT_THRESHOLD` (default `None` = off); when set, queue rows with `RISK_SCORE >= threshold` get `RULE_FLAG=PREDICTED_SEVERE`. Note: if enabled, evaluation's ranking block does NOT yet mirror this band — revisit then. Clarified: monotone masking does not push cat-2 customers toward cat-3; it computes the honest conditional P₃/(P₂+P₃).
- (c) 2025-12 labels confirmed fresh (query executed recently, after 2026-06 closed).

Still open: real cost numbers if the cost rule is ever promoted again.

## 13. Run 5 Verdict: Architecture Switch (July 10, 2026)

Run 5 (results_5/, 18 snapshots, 12.5M instances, test=2025-12, 8.3h wall) was the first run with the ranking headline — and it settled the architecture question **against** the neural pipeline:

| Arm | pooled AP | R@1w | cat_0 AP | cat_0 R@1d |
|---|---|---|---|---|
| Binary XGB comparator (~6 min, untuned) | **0.574** | 0.500 | 0.176 | 0.258 |
| Multiclass XGB baseline | 0.567 | **0.501** | **0.179** | **0.262** |
| DeepSets+XGB (~7h incl. 5.7h Optuna) | 0.541 | 0.483 | 0.127 | 0.197 |

Per-stratum verification (`inspect_fold_metrics.py`, results_5/inspect_fold_matrix.txt): the trees win **every slice** — no hidden DeepSets advantage anywhere. Diagnosis: (a) the 64-d embedding is an information bottleneck between the 257 raw features and the meta-learner; (b) the cost-sensitive focal loss shapes embeddings for coarse boundary-drawing, not fine risk ordering (the cost-free binary arm winning is the natural ablation evidence); (c) minor: the warm-restart LR schedule reset at epoch 10 and ate the fine-tuning phase (training-curve image in results_5/). Multiclass ≈ binary (multiclass wins 2 of 3 strata) — and the business requires per-class probabilities, so full-distribution arms are the deployable ones.

Other Run-5 findings: calibration+masking gain +3.1 F1 (argmax_cal 0.699 vs raw 0.667), Brier 0.113→0.103; val→test gap shrank to ~3.7pts (more history helps, "obsolete data" concern not supported); current-cat-2 base rate into severe = 45.8% (strong evidence for the certainty band — day-one calls mostly confirm near-certainties); cat_0 (true early warning) base 1.5%, AP ~0.18, one week of budget in-stratum catches ~80% (multiclass).

**IV re-run (results_5/explore_iv.csv, 4-class, matured-only, fixed binning):** the formerly-"dead" flags are real and strong (IS_DETERIORATING 3.26, HAS_EVER_BEEN_PRENPL 2.87, HAS_EVER_BEEN_NPL 2.34) — they were always model inputs; only the old report was wrong. **Suspected ETL bug:** `IS_IN_WARNING_ZONE` and `HAS_EVER_BEEN_NPL` have byte-identical IVs across all 4 OvRs ⇒ likely identical columns; check `SELECT COUNT(*) FROM D_ANALYTICS.DPD_SAMPLE1 WHERE IS_IN_WARNING_ZONE <> HAS_EVER_BEEN_NPL` (0 ⇒ one is a copy bug and a real feature is missing). `MATURED_INST_CNT` ≈ `CONTRACT_AGE_MONTH` likewise near-identical (plausibly legitimate for monthly installments).

**Implemented (July 10, tested locally — 32 tests + synthetic end-to-end fold run):** run.py now trains `MODEL_ARMS` = multiclass / binary / ordinal (cumulative P(y>k) → full distribution) / per_cat (Mohammad's idea: one model per current_cat over its reachable classes) — all on the 257 aggregated features, identical hyperparams (`XGB_DEFAULTS`), per-arm stratified calibration, per-arm+per-slice ranking logs (closing the Run-5 logging gap), all-arm capture-curve plot, `arms_metrics.json`, deployed-arm auto-selection by pooled ranking AP (binary excluded — no class distribution), deployment artifacts `model_arm.pkl` + `calibrator.pkl`. DeepSets is legacy behind `DEEPSETS_ENABLED=False` (`_train_deepsets_legacy`). Walk-forward remains implemented and compatible with the arms flow, still disabled. **Phase C pending: Predictor/ModelLoader still expect DeepSets artifacts — `predict` is broken for arm artifacts until reworked (after Run 6).**

## 14. What's Next (Prioritized)

1. **Run 6 on the server** (`python run.py train`, cache already v1.2 → no rebuild): the four-arm shootout, ~1-2h total. Winner (best pooled AP among full-distribution arms) is auto-selected and logged; check the cat_0 slice and the capture-curve plot before accepting it.
2. **Phase C (Mac, after Run 6):** rework Predictor/ModelLoader/bundle for the winning arm (aggregate features → arm → stratified calibration → mask → queue); keep legacy loading for old bundles.
3. **Walk-forward** (now ~40 min/fold): 3 quarterly folds to confirm ranking stability across 2025 — recommended before deployment since it's cheap now.
4. **Deployment:** set `DEPLOY_ARM=<winner>` → `python run.py train --final` (trained through 2025-12) → `python run.py predict` on 2026-06 → queue CSV to the API caller at 240/h.
5. **Business items:** certainty threshold (Run-5 evidence: 88% of day-one calls confirm near-certainties), the IS_IN_WARNING_ZONE SQL check above, real cost numbers if the cost rule is ever promoted.
