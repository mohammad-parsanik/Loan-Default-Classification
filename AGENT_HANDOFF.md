# Loan Default Classification — Agent Handoff Document

> **Purpose:** This document gives a new agent the full context needed to continue work on this project without re-discovering decisions, trade-offs, or bugs that have already been resolved.

---

## 1. Problem Definition

A bank needs to predict the **worst future delinquency state** of a customer's entire loan portfolio over a **6-month forward horizon**, to prioritize collection actions.

- **Prediction level:** Customer (grouped by `NATIONAL_CODE`), not individual loan.
- **Target:** 3-class classification:
  - `0` — No Delay (performing)
  - `1` — Current / Minor Delay (pre-delinquent)
  - `2` — Past Due+ (NPL, categories 2-4 collapsed)
- **Label construction:** For each customer-snapshot, label = `min(max(WORST_FUTURE_CAT across all loans), 2)`. Features retain the full 5-category granularity internally; only the prediction target is capped to 3 classes.
- **Business constraint:** This is heavily cost-sensitive. Missing a Cat-2 customer is penalized **4×** more than a false positive over-flagging. The full cost matrix is in `src/model/losses.py`.
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
| `OPTIMIZE_ON_VALIDATION` | `False` | `True` = use val set for early stopping + Optuna. `False` = skip Optuna, train for FIXED_EPOCHS with fixed XGB params, no val set created. This was introduced to eliminate Val-Test leakage (see Section 5). |
| `FIXED_EPOCHS` | `15` | Number of DeepSets training epochs when `OPTIMIZE_ON_VALIDATION = False`. |
| `LABEL_HORIZON_MONTHS` | `6` | Forward prediction window. Also the minimum gap required between splits. |
| `DATA_VERSION` | `"v1.0"` | Bump this to force NPZ cache invalidation when ETL changes. |
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
- Ran in a fresh run dir (`artifacts/20260702_110830`); a mid-run crash was fixed and the run resumed, which is why the log shows `[skip]` lines. The data cache, preprocessing, and portfolio artifacts were reused; DeepSets hyperparameters came from the config file. **Unverified:** whether the DeepSets weights were retrained fresh for `FIXED_EPOCHS=15` in this run, or whether the loaded checkpoint originated from Run 2's early-stopped training (which was epoch-selected against the leaky Oct'25 val set — residual model-selection leakage in the encoder). To verify on the server: open `artifacts/20260702_110830/fold_01/stages/deepsets.done` — `best_val_f1: 0.0` means fresh no-validation training (leakage-free); a value ≈ 0.77 means Run 2's checkpoint was reused. A fresh run also leaves exactly `epoch_001.pt`–`epoch_015.pt` in `fold_01/checkpoints/` and a training-curves plot with no val curve.
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

### The Resolution: `OPTIMIZE_ON_VALIDATION` Toggle
When set to `False`:
- `temporal_split.py` creates no validation set; the gap is enforced directly against test
- `trainer.py` trains for `FIXED_EPOCHS` without early stopping
- `meta_learner.py` uses fixed XGB hyperparameters (no Optuna)
- Result: **unbiased test metrics**, but potentially worse model (no tuning)

### Trade-off Summary
| Approach | Test F1 | Bias | When to Use |
|----------|---------|------|-------------|
| `OPTIMIZE_ON_VALIDATION = True` | 0.7090 | **Biased** (upper bound) | When you need the best production model and accept inflated metrics |
| `OPTIMIZE_ON_VALIDATION = False` | 0.6997 | **Unbiased** | When you need honest performance estimates for stakeholder reporting |

---

## 7. Exploration & Diagnostic Tools

Three standalone scripts exist in the project root for data quality analysis. These are documented in `EXPLORATION.md`:

| Script | Purpose |
|--------|---------|
| `explore_iv_woe.py` | Information Value & Weight of Evidence per feature, One-vs-Rest for all 3 classes. Reads from NPZ cache (no DB needed). |
| `explore_umap.py` | UMAP projection of raw features or model embeddings with CLI-tunable hyperparameters. |
| `explore_shap.py` | SHAP TreeExplainer on XGBoost meta-learner. Needs `.npy` embeddings copied from the training server. |

**Key finding from IV analysis:** Top features (`LOAN_CATEGORY`, `DPD_DAYS`) had very high IV values (>0.50, which normally signals leakage). After investigation, the conclusion was these are **genuinely strong signals** — they're point-in-time observations available at prediction time. The zero-variance binary features were also retained as they may gain signal with more snapshot data.

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
5. **The baseline consistently matches or beats DeepSets+XGB on Macro F1.** This is expected with MAX_LOANS=2. The DeepSets model's advantage is in Cat-2 recall (87% vs baseline's 76-78%) due to the cost-sensitive focal loss.
6. **`torch.compile` is disabled on the Windows training server** (inductor requires MSVC). The model runs in eager mode.
7. **Label-informed truncation bug (open):** `data_loader.process_raw_data` sorts each customer's loans by `WORST_FUTURE_DPD` (a label) descending, and the dataset keeps only the first `MAX_LOANS=2` rows — so for the ~1% of customers with 3+ loans, the label selects which loans the model sees. It also breaks inference: `EDP_Feature_pred` has no `WORST_FUTURE_*` columns, so `load_pred_portfolios` → `sort_values` will fail. Fix by sorting on a prediction-time feature (e.g., `DPD_DAYS` desc) in both paths.

---

## 11. What's Next (Prioritized)

1. **Wait for More Data (highest impact):** 2-3 more monthly snapshots will enable proper walk-forward validation with 6-month gaps. This is the single most impactful improvement.
2. **Re-evaluate Architecture:** With MAX_LOANS=2, a simpler approach (direct XGBoost on raw features with cost-sensitive loss) might be competitive. Consider if the DeepSets → XGBoost two-stage pipeline is justified versus a single-stage model.
3. **Feature Re-evaluation:** Re-run `explore_iv_woe.py` after new snapshots to verify high-IV features are genuinely strong and not artifacts of limited data.
4. **Reduce Horizon (if business allows):** Shrinking from 6 to 3 months would halve the required gap sizes, enabling walk-forward immediately with existing data.
5. **Production Deployment:** The inference pipeline (`src/inference/predictor.py`) exists but has not been tested end-to-end on the prediction table (`EDP_Feature_pred`).
