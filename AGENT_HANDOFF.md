# Loan Default Classification — Agent Handoff Document

> **Purpose:** This document gives a new agent the full context needed to continue work on this project without re-discovering decisions, trade-offs, or bugs that have already been resolved.

---

## 1. Problem Definition

A bank needs to predict the **worst future delinquency state** of a customer's entire loan portfolio over a **6-month forward horizon**, to prioritize collection actions.

- **Prediction level:** **Individual loan** (`LOAN_ID` x `SNAPSHOT_DATE`) since 2026-07-26 — see §19. Was customer-level (grouped by `NATIONAL_CODE`) through Run 6; `PREDICTION_GRAIN="portfolio"` restores that.
- **Target:** 4-class classification (bumped from 3 classes on 2026-07-08 — see §11 item 10):
  - `0` — No Delay (performing)
  - `1` — Current / Minor Delay (pre-delinquent)
  - `2` — Past Due+ (NPL, raw category 2 only)
  - `3` — Severe Past Due (raw categories 3-4 collapsed)
- **Label construction:** At loan grain, label = `min(WORST_FUTURE_CAT, config.NUM_CLASSES - 1)` for that loan — the ETL already computes `WORST_FUTURE_CAT` per loan. At portfolio grain it is `min(max(WORST_FUTURE_CAT across the customer's loans), config.NUM_CLASSES - 1)`. Features retain the full 5-category granularity internally; only the prediction target is capped (currently to 4 classes).
- **Business constraint:** This is heavily cost-sensitive. Missing a Cat-2 customer is penalized **4×** more than a false positive over-flagging; the class-3 (Severe Past Due) costs are a derived placeholder, not yet business-tuned. The single source of truth is `project_config.COST_MATRIX`.
- **Prior work:** A previous project used per-loan LightGBM classifiers and performed poorly. This project replaced that with a portfolio-level architecture — and, as of 2026-07-26, has returned to a per-loan grain at the business's request. Note this is *not* a revert to the prior work: the label horizon, the ~64 engineered features, the calibration + monotone mask, and the ranked-queue objective are all different. The portfolio-level detour is what established that the per-customer aggregation was buying almost nothing (99th pct = 2 loans/customer).

---

## 2. Dataset Description

### Source & Ingestion
- Data originates from an **MSSQL** database (table `D_ANALYTICS.EDP_LOAN_FEATURES` since 2026-08-26 — see §20; previously `D_ANALYTICS.DPD_SAMPLE1`, aliased `EDP_Feature_Train`), accessed via `pyodbc` through `src/db/mssql_connection.py`.
- **Important:** The original implementation plan and README mention Oracle/cx_Oracle — this is outdated. The actual codebase was migrated to **MSSQL (`pyodbc`)** early in development. The config file (`project_config.py`) reflects MSSQL credentials.
- After the first database load, portfolios are cached to disk **one NPZ per snapshot** under `data/snapshots/<stage>_<key>/`, each with its own manifest for cache invalidation (`DATA_VERSION` in config). Subsequent runs load from this cache instead of hitting the DB. Snapshot-level granularity mirrors the ETL: a matured snapshot is never recomputed upstream so its file is permanent, while the newest 7 (immature) snapshots are rewritten on every monthly load and their caches are provisional.

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

Full feature dictionary: `etl_integration/CONSUMER_CONTRACT.md` §5 (local-only, authoritative), with `column_changes.md` as this project's older copy. The machine-readable column set, order and handling flags live in `contract/columns.json` — that is what the code actually reads (see §20 and `contract/README.md`).

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

> **Critical clarification:** The README and implementation plan mention a "Set-Transformer." No such model exists in the codebase — `src/model/set_transformer.py` was deleted July 26, 2026 as dead code (never imported by `run.py`). The legacy neural path is `src/model/deep_sets.py`.

**Why DeepSets was chosen over the Set-Transformer:** With MAX_LOANS ≤ 2, self-attention on 1-2 tokens degenerates to a weighted average. DeepSets (phi + pool + rho) is provably permutation-invariant, has far fewer parameters (~42K vs ~173K), and is faster on CPU.

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

Four standalone scripts exist in the project root for data quality analysis. These are documented in `EXPLORATION.md`:

| Script | Purpose |
|--------|---------|
| `explore_iv_woe.py` | Information Value & Weight of Evidence per feature, One-vs-Rest for all `config.NUM_CLASSES` classes. Reads from NPZ cache (no DB needed). |
| `explore_umap.py` | UMAP projection of raw features or model embeddings with CLI-tunable hyperparameters. |
| `explore_shap.py` | SHAP TreeExplainer on XGBoost meta-learner. Needs `.npy` embeddings copied from the training server. |
| `explore_clip_impact.py` | Whether `OutlierClipper`'s `[p1, p99]` clip merges away a risk-bearing tail. Reads `tail_lift` = `P(severe \| x > p99) / P(severe)`. Reads from NPZ cache (no DB needed). Added for the 700M→7B population widening (§21). |

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
| **Data cache** | `data/snapshots/<stage>_<key>/<snapshot>.npz`, one per snapshot, uncompressed. Invalidated by changing `DATA_VERSION` in config. |

---

## 9. Project File Map

```
Loan Default Classification/
├── project_config.py              # All hyperparameters, DB creds, toggle flags
├── contract/
│   ├── columns.json               # ★ The feed's column contract — feature identity
│   └── README.md                  # What it holds, what reads it, how to refresh it
├── run.py                         # CLI entry point: train / predict / explore (604 lines)
├── requirements.txt               # Python dependencies
├── AGENT_HANDOFF.md               # This document
├── column_changes.md              # Feature data dictionary (gitignored; the
│                                  #   authoritative copy is etl_integration/)
├── leakage_analysis.md            # Detailed Val-Test leakage analysis
├── EXPLORATION.md                 # Usage guide for explore_*.py scripts
│
├── src/
│   ├── db/mssql_connection.py     # MSSQL connector (pyodbc)
│   ├── data/
│   │   ├── column_contract.py     # Loads/validates contract/columns.json
│   │   ├── feed_checks.py         # Row-level feed invariants on every load
│   │   ├── data_loader.py         # Vectorized load + NPZ cache; name-based
│   │   │                          #   projection + canonical row order
│   │   ├── data_explorer.py       # One-time data profiling
│   │   ├── dataset.py             # PyTorch Dataset + padding/masking
│   │   ├── preprocessing.py       # Impute → Clip → Scale pipeline
│   │   └── temporal_split.py      # Static split + walk-forward fold generation
│   ├── model/
│   │   ├── deep_sets.py           # ★ Active model (42K params)
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
└── tests/                         # Unit + integration tests
    ├── test_pipeline_changes.py   # The main suite
    └── test_order_independence.py # Permute an input, assert the output is unchanged
```

---

## 10. Known Issues & Gotchas

1. **README.md is outdated.** It still references Oracle, `cx_Oracle`, and the Set-Transformer as the active model. The actual DB is MSSQL/pyodbc, and the active model is DeepSets.
2. **`Implementation_plan.md` is the original design doc** — many details have evolved (DeepSets replaced Transformer, Oracle replaced by MSSQL, walk-forward added then found infeasible). Treat it as historical context, not current truth.
3. **`set_transformer.py` was deleted** (July 26, 2026) — it was never imported by `run.py`. The neural path is `deep_sets.py`; the deployed model is `src/baselines/aggregated_xgboost.py`.
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

## 14. Performance Fix: Checkpoint Pickling + Redundant Aggregation (July 10, 2026, mid-Run-6)

The first live run under the arms code (started attempting Run 6) showed multi-minute unexplained gaps between a stage's "completed in Xs" log line and its "[checkpoint] complete" line — e.g. 31 min after Stage 1 (load_data, "completed in 36.9s") and 20 min after Stage 2 (split, "completed in 20.5s"). Root cause: `ckpt.save(...)` (a `joblib.dump` of the FULL per-instance dict list — now 15-20M+ instances) runs **outside** the `timed()` block that produces those log lines, so the pickling cost was invisible in the log yet dominated wall time. Confirmed from Run 5's own log too (11 min gap after preprocessing, previously unflagged) — the pattern scales with instance count, which is exactly what's been growing.

Second, compounding issue found by inspection: the arms refactor called the (already-slow, ~6.5 min/call at 6M rows per Run 5's log) per-instance aggregation loop independently in every arm — roughly 12-13 total calls across 4 arms (train/val/test each) plus a redundant deployed-arm re-call — instead of once.

**Fix (implemented, tested — 33 local tests incl. a resume-semantics check that arms are still skipped on re-run):**
- `load_data`, `split`, `preprocessing` are no longer checkpointed via joblib — they're cheap to recompute (NPZ cache + simple ops: ~30s/~20s/~4min) and `--resume` just always redoes them, going straight to whichever arm stage wasn't finished. Per-arm and DeepSets checkpointing (the genuinely expensive stages) is unchanged.
- `aggregated_xgboost.aggregate_features(instances) -> (X, y)` replaces the old per-arm `_aggregate` method: vectorized (flat loan matrix + `np.reduceat`), called ONCE per split in `run.py`, shared by every arm. Verified bit-identical to the old per-instance computation (`test_aggregate_features_matches_naive_reference`). Local Mac benchmark showed only a modest ~1.4-1.7x per-call speedup (this Mac's numpy per-call overhead is apparently much lower than whatever the Windows server exhibits, going by Run 5's 6.5min/6M-rows figure) — the **guaranteed** win is the call-count reduction (12-13 → 3), independent of that uncertain per-call factor.
- Arm interface changed accordingly: `train(X_train, y_train, cat_train, X_val, y_val, cat_val)` / `predict_proba(X, current_cat=None)` (array-based, not instance-list-based). `severity_scores()` removed — redundant now that `predict_proba(X)[:, -1]` is P(severe) for every arm uniformly (binary arm's 2 columns included).
- Deployed-arm CI/plots section now reuses the arm's already-computed test probs from the arms loop instead of calling `predict_proba` a second time.

**Net effect:** a run that was on track for 2+ hours (checkpoint tax already paid + ~4 arms × redundant aggregation + real fit time) should now complete in roughly the real compute time alone — order ~1h for the four-arm shootout at current data scale.

## 15. Run 6 Verdict + Ship-Readiness (July 11, 2026)

Run 6 (results_6/, 54 min vs Run-5's 8.3h) ran the four-arm shootout on the same split (test 2025-12). All four arms within ~1pt pooled AP — the decomposition barely matters once it's boosted trees + stratified calibration + masking:

| Arm | pooled AP | cat_0 AP | cat_1 AP | cat_2 AP |
|---|---|---|---|---|
| **multiclass** (deployed) | **0.5761** | **0.1737** | 0.3281 | 0.7442 |
| per_cat | 0.5759 | 0.1604 | 0.3295 | 0.7441 |
| binary (diagnostic) | 0.5700 | 0.1704 | 0.3165 | 0.7373 |
| ordinal | 0.5660 | 0.1705 | 0.3188 | 0.7379 |

Multiclass wins pooled AND the cat_0 early-warning slice — auto-selected, now **locked** (`DEPLOY_ARM="multiclass"`). Notable: per_cat (Mohammad's per-category idea) ties overall but is *worst* on cat_0 (0.160) — the specialist can't borrow deterioration signal from other strata, while the pooled model + masking + per-stratum calibration + `LOAN_CATEGORY` feature already capture the conditional structure. Dropping cost weights (BASELINE_COST_WEIGHTS=False) lifted multiclass +1pt AP vs Run 5 (0.5665→0.5761). `IS_IN_WARNING_ZONE == HAS_EVER_BEEN_NPL` confirmed identical *by definition* (not an ETL bug) — a harmless redundant column.

**Implemented July 11 (36 tests + synthetic end-to-end fold & deploy round-trip):**
- **Phase C DONE** — `predict` now works with arm artifacts. `ModelLoader.load_pipeline()` → `(scorer, calibrator, features)`; `Scorer` abstraction (`ArmScorer`: scale → `aggregate_features` → `arm.predict_proba`, no torch; `DeepSetsScorer` legacy). `Predictor` uses `scorer.raw_probs`. Directory loader prefers `model_arm.pkl`; single-file `model_bundle.pkl` (kind="arm") is the shippable artifact; legacy bundle renamed `deepsets_bundle.pkl`.
- **`DEPLOY_ARM="multiclass"` locked**; `ARM_OPTUNA_TRIALS` (default 0) tunes the deployed arm on val PR-AUC of P(severe) via `tune_arm_params` (lazy optuna) — objective is the deliverable, not macro F1.
- **Fold aggregator reworked** for the ranking headline (mean±std of pooled AP + recall@windows across folds); WF best-fold now by AP. Walk-forward stays implemented, still disabled.
- **`DEPLOYMENT.md`** — the train → (walk-forward check) → `--final` → move `model_bundle.pkl` → `predict` runbook with output-column reference and the call-ledger format.

## 16. Walk-Forward Stability Confirmed (July 12, 2026)

Mohammad ran the 15-fold walk-forward check (`WALK_FORWARD_ENABLED=True`, `MODEL_ARMS=["multiclass","binary"]`, test snapshots Aug–Dec 2025). The console log was lost when the session closed, but `run.py` saves `<run_dir>/stages/fold_results.pkl` after every completed fold — `analyze_walk_forward.py` (new script) reconstructed the conclusion from it.

**Finding:** pooled AP swung from 0.11 to 0.66 across folds — looked alarming until broken down by `train_snaps`. Every unstable fold shares the same cause: **trained on only 1 (sometimes 2) snapshots** — `train=[20240721]` alone is unstable on every test month regardless of which one (AP 0.11–0.37). This is an artifact of walk-forward enumerating every valid `(train, val, test)` combination, including thin early ones with almost no history — a regime production never occupies (it always trains on all mature snapshots).

**Restricting to the 6 folds with ≥3 training snapshots — the only ones resembling real deployment volume — results are tight and consistent: AP 0.55–0.58 across four different test months (Oct–Dec 2025), and `multiclass` wins all 6 of those folds** (the naive all-fold average had "binary winning" only because it averages in the thin-data folds, where multiclass — being more data-hungry — degrades harder than binary on scraps of data; e.g. worst fold: multiclass 0.11 vs binary 0.36). Mean AP in the mature-fold regime (~0.565) closely tracks Run 6's full-scale static-split result (0.576) — **this is the confirmation that Run 6 wasn't a lucky single snapshot; the multiclass verdict is temporally stable.**

**A real gap surfaced along the way, not yet fixed:** walk-forward's `build_fold_instances` still assigns validation as one whole calendar snapshot — it was never updated to the customer-disjoint carve-out (`VAL_SPLIT_MODE="customer"`) the single-split/`--final` path has used since July 7–8. This likely amplifies the thin-fold instability (calibration/early-stopping driven by an entire separate month rather than a customer-disjoint slice of the same training data) but wasn't the root cause — data volume was. Low priority to fix given walk-forward is off by default and thin folds aren't the production regime anyway; note it if walk-forward is ever leaned on more heavily.

`analyze_walk_forward.py --min_train_snaps N` (default 3) is now a permanent tool for this: prints both the naive all-fold verdict and one restricted to folds with realistic training volume, so this mistake can't repeat silently on a future run.

## 17. Cost-Matrix / Calibration Clarification + Manager-Code Scoring Module (July 2026)

Mohammad asked two clarifying questions before greenlighting a tech-team request; both surfaced small real gaps.

**Q1 (cost matrix, training vs. inference):** confirmed aligned. Training-time cost weighting is OFF (`BASELINE_COST_WEIGHTS=False`; the one model with a cost-sensitive *loss*, DeepSets, is disabled entirely) because it was found to hurt ranking (§13/§15). Inference-time cost matrix drives only `PREDICTED_CLASS`/`EXPECTED_COST` (secondary columns) — never `RISK_SCORE`. **Gap found:** `decision.py`'s `COST_MATRIX` was a module-level constant frozen at import time — impossible to override per call. Fixed: `expected_costs`/`cost_decisions`/`risk_scores` now accept an optional `cost_matrix=` param.

**Q2 (calibration sample grouping):** confirmed `StratifiedCalibrator` groups the fitting sample by `current_cat` ALONE, never combined with predicted class — within each stratum it fits one isotonic curve per *output* class (that's the "predicted category" axis Mohammad was recalling, but it's a per-class curve, not a second grouping dimension). This was true in code but under-documented — CLAUDE.md now states it explicitly.

**Tech team request:** a module (or set of modules) importable into their manager/orchestration code — pass a DataFrame + parameters, get a ranked-queue DataFrame/CSV back, independent of data source, with base-config fallback + a warning when parameters aren't passed explicitly. Training pipeline stays as-is (run ~2x/year).

Implemented:
- `src/inference/scoring_params.py` — `ScoringParams` dataclass: `bundle_path` (required), `calibration_df`/`output_path` (no config equivalent, never warned), and 7 business knobs (`called_log_path`, `certainty_act_threshold`, `carve_current_cat_ge`, `calibration_min_stratum_n`, `api_data_ttl_days`, `pred_dedup_latest`, `cost_matrix`) that fall back to `project_config` via `.resolve()`, logging one consolidated `logger.warning` naming exactly which fields defaulted.
- `src/inference/scoring.py` — `run_scoring(df, params)`: the manager-code entry point. Resolves params, loads the bundle, optional DB-free calibration refresh from a supplied `calibration_df`, scores via the same shared `score_instances()` transform as `Predictor`/`score_dataframe()`, optional CSV write.
- `apply_queue_flags`/`score_instances` (`predictor.py`) gained keyword-only explicit-override params threaded from `ScoringParams`, while keeping their old config-getattr defaults for the pre-existing `Predictor.predict()`/`score_dataframe()` call sites (fully backward compatible, no new warnings on old paths).
- `build_scoring_package.py` updated (both new files added to the manifest; README documents both entry points — `score_dataframe()` for quick one-offs, `run_scoring`/`ScoringParams` for manager code).

All verified locally (new tests + full suite); see `tests/test_pipeline_changes.py`.

## 18. What's Next (Prioritized)

1. **Optuna** (optional): set `ARM_OPTUNA_TRIALS=20` for a tuned `--final` fit — headroom likely small given how tightly the arms cluster.
2. **Ship:** `python run.py train --final` → move `artifacts/<ts>_final/fold_01/model_bundle.pkl` to production → `python run.py predict` on 2026-06 → queue CSV to the API caller at 240/h (`RISK_RANK` order). For the tech team's manager code: `build_scoring_package.py` → `run_scoring(df, ScoringParams(...))`.
3. **Business items (non-blocking):** `CERTAINTY_ACT_THRESHOLD` (Run-5 evidence: 88% of day-one calls confirm near-certainties); real cost numbers only if the cost rule is ever promoted; survivorship in the ETL sample remains the main open data question.
4. **Nice-to-have:** update walk-forward's `build_fold_instances` to use customer-disjoint validation, matching the single-split path — only worth it if walk-forward becomes a routine check rather than an occasional one.

---

## 19. Prediction Grain: Customer → Loan (July 26, 2026 — not yet run on server)

**Decision:** business wants risk and a prediction for **each individual loan**, so `PREDICTION_GRAIN = "loan"` is now the default. One scored row = one (`LOAN_ID`, `SNAPSHOT_DATE`).

**This is a deletion, not an addition.** The ETL already emits a per-loan label — `OLD_ETL Document` line 465 is `MAX(D.LABEL_DPD) AS WORST_FUTURE_DPD ... GROUP BY D.LOAN_ID`, and the final select groups by `LC.LOAN_ID, LC.SNAPSHOT_DATE`. The customer rollup (`label = max` over the customer's loans) was applied in Python, in `process_raw_data`. Loan grain removes that groupby.

**Expected accuracy gain: small, and that was accepted going in.** Single-loan customers produce bit-identical rows under either grain (min == max == mean == the feature, std == 0, count == 1), and the 99th percentile of loans per customer is 2 — so the two models can only differ on the multi-loan minority.

**The real win is the carve-out, not accuracy.** `current_cat` used to be the portfolio MAX, so `CARVE_CURRENT_CAT_GE` removed a customer holding *one* severe loan from the queue entirely — including their still-healthy loans, which are exactly what an early-warning system should keep watching (`mask_monotone` compounded it, forcing P(severe)→1 for the healthy loan). Per loan, only the severe loan is flagged.

**What changed**
- `PREDICTION_GRAIN` in `project_config.py`; `DATA_VERSION` → `v1.3`; the grain is part of the NPZ cache key, so the two grains cache separately.
- Instances gain `loan_id` and `portfolio_n_loans` (the customer's true loan count — output context, **not** a model feature).
- `build_features()` dispatches on grain. Loan grain skips aggregation: X is ~4× smaller and the XGBoost fit correspondingly cheaper.
- The grain is written to `metadata.json` and the bundle, and **inference follows that, not the local config** — a model always sees features built the way it was fit. Pre-grain bundles read as `"portfolio"`.
- Output adds `LOAN_ID` and `CUSTOMER_MAX_RISK_SCORE` (worst loan per customer; context only, the queue still ranks per row).
- `DEEPSETS_ENABLED` now raises unless the grain is `"portfolio"`.

**Deliberately deferred:** portfolio-context features on each loan row (portfolio worst category, loan count, portfolio max DPD). A cat-0 loan belonging to someone with a cat-3 loan is genuinely higher-risk, and the aggregated features used to carry that signal — but the decision was to ship the plain per-loan model first and add these only if the grain change is accepted.

**Also fixed in the same pass (independent bug):** the train/serve truncation skew — see §10 and the `truncate_loans` note in `CLAUDE.md`.

**Reading the next run's numbers:** compare `ranking_single_loan`, NOT `ranking`, against `results_3`…`results_6`. Loan grain changes the population (healthy siblings now enter the queue) and the severe base rate with it, and PR-AUC moves with the base rate regardless of model quality. `ranking.base_rate` sits next to `pr_auc` so the shift is visible. `recall@K` is unchanged and stays keyed to loan-slots: the API is customer-keyed, so two loans of one customer cost two slots but one call, making the metric slightly conservative — the safe direction. `score_instances` logs the duplicate-customer count in the callable window.

---

## 20. ETL Feed Alignment + Order Independence (August 26, 2026 — not yet run on server)

The upstream ETL was rebuilt. Two work orders landed together, because the column contract they share makes them one change: `etl_integration/SESSION_HANDOFF.md` (align with the new feed) and `etl_integration/ORDER_INDEPENDENCE_ML_PLAN.md` (stop depending on row/column order). Both are local-only documents; this section is the tracked record.

**Still blocked, unchanged by any of this:** do **not** retrain, and do **not** compare any metric against `results_1`…`results_6`. The upstream validation gate has not passed and the 12-snapshot rebuild has not happened. Several features changed *meaning*, not just value, so a model trained on the new table is not comparable to one trained on the old — and the difference is not noise. Everything below is a correctness and reproducibility change to the code that *will* do that training; the point of landing it now is that the rebuild becomes the first result this project can reproduce.

### The table

`TRAIN_TABLE` is now `D_ANALYTICS.EDP_LOAN_FEATURES` (71 columns), read from the contract rather than written as a literal. Three names had been circulating for what should be one table; that was settled upstream on 2026-08-26, and the 70-column table behind the other two is deprecated and left in place. `src/db/create_dpd_sample.sql` and its runner were **deleted** rather than updated — the authoritative DDL lives beside the SQL that has to satisfy it, in the ETL repo, and two DDLs for one table is how the dictionary drifted in the first place.

### The column contract

`contract/columns.json` (tracked) pins the column set, ordinal order, and per-column handling flags; `src/data/column_contract.py` loads and validates it at import. `project_config.META_COLS` / `BINARY_FEATURES` / `NO_CLIP` / `NO_SCALE` are now **derived** from it. It carries no column semantics — this repo is public and the prose dictionary stays in the gitignored `etl_integration/`. `column_contract` cross-checks the tracked file against the vendored copy when that folder is present, so a forgotten refresh warns instead of drifting silently. See `contract/README.md` for the refresh procedure.

Deriving `META_COLS` is what fixes the sharpest edge in the new feed. Column 71 is `LABEL_HORIZON_DATE`, and `FEATURE_COLS` is built as *every column minus `META_COLS`* from a `SELECT *` — so on the first read against the new table it would have become feature #65: a monotonically increasing date that proxies snapshot recency and correlates with label maturity. Nothing raises. The model just gets quietly better on the training set.

### Label maturity is now read, not computed

`filter_mature_snapshots` compares `LABEL_HORIZON_DATE <= today` (both Gregorian `YYYYMMDD` ints) against a snapshot→horizon map that `DataLoader` populates from the feed and carries in the NPZ manifest. The old `months_apart(snapshot, today) >= 6` re-derived, in Gregorian months, a fact the row now carries — derived upstream on a different calendar, so the two can disagree at month boundaries. `LABEL_HORIZON_MONTHS` is still used for the train/val/test gap logic, and as a warned fallback when no horizon is known.

### The sentinel columns

The four `DAYS_SINCE_LAST_*` features now carry `99999` = "never reached this band", against a scale where `0` = "in that band right now" — the two ends of one axis. `OutlierClipper` was clipping them to `[p1, p99]`, which rewrites *never delinquent* as *cleared a long time ago*. What makes this worth writing down: the bug is a **no-op in one of the four columns and destructive in its sibling**. Where "never" is the overwhelming majority, `p99` *is* the sentinel and the clip does nothing; where it is a minority, `p99` sits far below and the clip destroys the distinction. Spot-checking one column proves nothing about the others. They are now exempt from both clipping and scaling via the contract, and the sentinel must never be imputed to a median.

### Order independence

Six findings, from an audit of both repos. The first could be silently wrong in production; the rest made results depend on arrival order.

| # | What was wrong | Fix |
|---|---|---|
| M1 | Feature identity was positional end to end and never name-checked. The training order *was* saved to `metadata.json` and `load_pipeline()` *did* return it — and all three call sites threw it away (`scorer, calibrator, _ = ...`). A reordered source applies the wrong median, clip bounds and scaler to each column, at identical width, with no error. | `DataLoader.project_features(df, order)` reindexes by name; every call site binds the model's saved list; `preprocessing.assert_pipeline_features` verifies it against what the transformers were fitted on; `ModelLoader` refuses an artifact that cannot state its own feature order. |
| M2 | `_cache_key` hashed `DATA_VERSION`, table, `META_COLS`, database and grain — not the column list or its order. A reorder survived the cache. | The key now covers `CONTRACT_VERSION` and the feature list **in order**. |
| M3 | `subsample=0.8, colsample_bytree=0.8` sample by index. A fixed `random_state` fixes the draw, not what gets drawn. | No hyperparameter change — the sampling was never the problem, the undefined order underneath it was. Fixed by M1 + M7. |
| M4 | The queue sorted on `["_flagged", "RISK_SCORE"]`. Multi-key `sort_values` is stable, so ties kept their arrival order — and calibrated probabilities tie heavily, isotonic regression being a step function. Who got called today was decided by page layout. | Sort on `_flagged, RISK_SCORE, NATIONAL_CODE, LOAN_ID`. |
| M5 | `ranking.py`'s `argsort(kind="stable")` inherited the same dependence, so `recall@K` and `lift@K` did too. | `ranking_metrics(..., tie_break=)` / `capture_curve(..., tie_break=)`, fed `LOAN_ID` by `run.py`. |
| M6 | Truncation kept the first N rows after a sort ending on `DPD_DAYS desc` — 0 for most loans, so the ties were enormous. | `LOAN_ID` appended to the canonical sort makes the key unique; no tie survives it. |

Already correct and left alone: `aggregate_features`' min/max/mean/std reduction is permutation-invariant, and `_customer_bucket` uses md5 rather than `hash()` precisely so the split survives `PYTHONHASHSEED` — that is the in-repo precedent M1 now follows one layer up.

Reported `recall@K` / `lift@K` may move very slightly against earlier runs. They were never stable to begin with; this is the first version of them that can be reproduced.

### Also landed

- `src/data/feed_checks.py` — the feed's row-level invariants (`WORST_FUTURE_CAT >= LOAN_CATEGORY`, the `DAYS_SINCE_LAST_*` ladder, category ranges, grain uniqueness, `NPL ⇒ PRE-NPL`, horizon constant within a snapshot, sentinels still present), checked on every DB load. Logged, not raised: a handful of bad rows in a 577k-row snapshot should not abort a multi-hour run.
- `MSSQLConnector.get_etl_runs()` reads the upstream job ledger before loading, so a snapshot missing from the table is distinguishable from one whose run never happened. Advisory — a run that can read the table is not blocked on bookkeeping.
- `MSSQLConnector.get_label_horizons()` — one cheap `DISTINCT` supplying the maturity map for every snapshot.
- `DATA_VERSION` → `v1.4`. Every earlier cache is stale.
- `VAL_SPLIT_MODE = "customer"` is unchanged and now carries a comment saying why it is a **correctness requirement**: ten of the 64 features are customer-level and identical across every loan a borrower holds, so a random split puts those ten values on both sides of the fold boundary.
- `tests/test_order_independence.py` — 15 tests, each verified to fail when its corresponding change is reverted.

### Reading the first run on the new feed

Judge it on its own terms. `results_1`…`results_6` are not a baseline for it. Within the run, the usual rule still holds: the `current_cat_0` slice and the `ranking` block, never aggregate F1.

---

## 21. Population Widened: contract amount ≤ 700M → ≤ 7B (September 2, 2026 — not yet run on server)

The ETL's `q1` scope filter changed. A loan enters a snapshot if its contract amount is **≤ 7,000,000,000** rials, where the ceiling used to be 700,000,000. Everything else about the filter is unchanged: first installment on or before T−6, last installment on or after T+6, monthly repayment schedule, individual customer.

Upstream records this as a **correction of a miscommunicated figure, not a policy change** (`etl_integration/CONSUMER_CONTRACT.md` §8: "the population is retail either way"). That reframes every earlier run rather than excusing it: `results_1`…`results_6` were trained on an unintended *subset* of the population the project was always meant to serve, not on a deliberate low-value segment that has now been extended. Nothing technical below changes because of it — the model still has to be refitted on a population it has never seen — but it is the reason not to treat the old population as a "core" segment worth preserving a model for.

**Nothing in this repo encodes the cap** — it was never a constant here, only a fact about which rows arrive. The 71-column contract, `CONTRACT_VERSION`, every column meaning and every handling flag are untouched, so the pipeline runs against the new table with no code change. That is exactly the risk: this is the first change to the feed that **no schema check, contract check or feed invariant can see**. The table name, the column set, the snapshot dates and the column semantics are all identical; only the row population differs.

### What changed here

- **`DATA_VERSION` → `v1.5`.** The one mandatory edit. `_cache_key` hashes `DATA_VERSION`, table, `META_COLS`, database, grain, `CONTRACT_VERSION` and the ordered feature list — none of which move when the population does. Without the bump, the cached snapshots stay valid and silently serve the ≤700M population forever.
- `explore_clip_impact.py` — new diagnostic, see below and `EXPLORATION.md` Script 4.

### Backfilled, and therefore consistent

The tables and **all** snapshots are being rebuilt under the new filter. This is the good case, and it is worth naming why: had the widening applied forward-only, the population shift would sit exactly on the temporal split — training on ≤700M loans, deploying against a queue containing 7B loans the model had never seen, with test-fold metrics unable to show it because the test fold would be old-population too. Backfilled, train and score populations match and the only consequence is that every historical number is superseded. Re-reading the same `SNAPSHOT_DATE` now returns different rows; that is expected, not a bug (the ETL publishes per-snapshot with `DELETE WHERE SNAPSHOT_DATE = ?` then insert).

### Comparability: decided

**`results_1`…`results_6` are not a baseline. Do not compare against them, in any block, including `ranking_single_loan`.** That slice controls for prediction grain, not for population — it has no power here. The severe base rate moves with the new segment, and PR-AUC moves with the base rate regardless of model quality, so a lower AP on the new feed is not evidence of a worse model and a higher one is not evidence of a better one. `ranking.base_rate` is reported next to `pr_auc` so the shift is at least visible. The first run on the rebuilt table is the new baseline; judge it on its own terms, on the `current_cat_0` slice and the `ranking` block, never aggregate F1.

This compounds with §20 — the feed alignment already made old results non-comparable because several features changed meaning. Two independent reasons now, same conclusion.

### The clipping question (open, has a tool)

`OutlierClipper` clips every non-binary, non-`NO_CLIP` feature to `[p1, p99]` fit on train. **Clipping is the one preprocessing step XGBoost cannot shrug off.** Imputation and `PortfolioRobustScaler` are monotone and trees split on order, so they are invisible to the model; clipping merges every value above p99 into a single number and destroys the ordering up there.

Three features are amount-scaled and inherit the 10× wider range directly: `PAYED_OVERDUE_AMNT` (8), `UPCOMING_AMNT` (14), `REMAINING_AMNT` (66). The concern is structural, not hypothetical: the newly admitted large loans are a *minority* sitting at the *top* of the distribution, which is precisely the region clipping flattens. If severe events concentrate there, the clip is deleting the signal the widening was meant to add. Note the shape of this is the same trap as the `DAYS_SINCE_LAST_*` sentinels in §20 — harmless in the columns where the extreme value is the majority, destructive where it is a minority — which is why it gets measured rather than assumed.

**Not pre-emptively changed.** Setting `clip: false` on those three in `contract/columns.json` is a one-line fix and it is the wrong move to make blind: clipping exists because unbounded tails hurt, and the tail could as easily be noise as risk. Measure first with `explore_clip_impact.py`, then A/B on validation lift@K before committing. Whichever way it goes, changing a `clip` flag bumps `contract_version` and invalidates the cache on its own.

### Secondary consequences, no action

- **`COST_MATRIX` is flat per loan** and now spans a 10× exposure range, so it is considerably more wrong than it was. Harmless: it feeds `PREDICTED_CLASS` / `EXPECTED_COST` only, never `RISK_SCORE`, never training. It was already advisory (business gave only the 4× anchor); it is now more so.
- **`MAX_LOANS_PER_CUSTOMER`** resolves from the 99th percentile at runtime and is inert at loan grain. Ignore.
- **`assert_feed_invariants`** was measured against the old segment. Nullability and range assumptions are worth re-reading in the log on the first load rather than trusting.
- **Row volume** grows. If it grows a lot, the arm-training stages are where it will show.
- **`etl_integration/CONSUMER_CONTRACT.md` was refreshed from upstream on 2026-09-02** and is current: §7 carries the new ceiling, §8 carries a row-population entry for it. Diffed against the previous copy, that ceiling is the *only* substantive change — no column was added, removed or redefined, `contract_version` is still 1, and the vendored `etl_integration/columns.json` still matches the tracked `contract/columns.json` projection exactly, so no contract refresh is needed. (The refreshed copy arrived without the provenance header the convention calls for; it was restored by hand, with the ETL commit noted as unrecorded.)

### The business question this raises

The queue ranks by P(severe) alone and `recall@K` counts loans, so a 7B loan and a 70M loan occupy identical slots at identical value. If the objective is money at risk, that is now a real distortion rather than a rounding error — either rank on `P3 × REMAINING_AMNT` or report exposure-weighted recall beside the count-based one. Deliberately **not** implemented: it changes what the deliverable optimises, which is a business decision, not a maintenance one. Raised with the business, pending.

---

## 22. Load Stage Rebuilt for the ≤7B Row Volume (September 5, 2026 — not yet run on server)

### What happened

The first run against the rebuilt (≤7B) tables died in the load stage:

```
INFO - Loading training data from D_ANALYTICS.EDP_LOAN_FEATURES
MemoryError: unable to allocate 326 MiB for an array with shape (42713503,) and data type uint64
```

The number is the row count: **42,713,503**, up from the ~6M every prior run
was built around. §21 predicted volume growth would show in the arm-training
stages; it showed a stage earlier.

`load_training_data` issued a single `SELECT * FROM D_ANALYTICS.EDP_LOAN_FEATURES`
— every snapshot, mature and immature, all 71 columns — through
`pd.read_sql`, which on a DBAPI connection does one `cursor.fetchall()` into a
list of pyodbc `Row` objects before building the frame. The float64 frame alone
is ~24 GB; the row-object intermediate is several times that. The 326 MiB
allocation that actually failed was just the next thing asked for.

### What changed

`DataLoader.load_train_portfolios` no longer reads the table in one shot:

- **One `SELECT` per snapshot**, vectorised and released before the next. Peak
  frame is ~1/20th of the table. Equivalent by construction — instances group
  by `(customer, snapshot)`, so no group spans a snapshot boundary.
- **Mature snapshots only.** The split discarded immature ones anyway; loading
  them was ~25% of the rows for nothing.
- **One NPZ per snapshot** under `data/snapshots/<stage>_<key>/`, replacing the
  monolithic `train_portfolios_cache.npz`. This mirrors the ETL: each monthly
  load recomputes the newest 7 snapshots (T new, T-7 just matured, five
  rewritten in between), so a **matured** snapshot never changes again and its
  file is permanent, while an **immature** one is provisional. Next month's run
  reads one new snapshot from the DB and the rest from disk instead of
  rebuilding ~40M rows.
- **Immature caches expire with the ETL cycle.** `load_pred_portfolios` reuses a
  cached immature snapshot only while the ETL run that built it is still the
  newest `SUCCESS` in `etl_job_control`; if that ledger is unreadable the
  snapshot is re-read every time. Serving last month's rows under this month's
  snapshot date is the failure this prevents, and nothing in the data would
  reveal it.
- **`np.savez`, not `np.savez_compressed`.** Single-threaded zlib over a
  multi-GB `features_flat` cost tens of minutes per rebuild to save a few GB of
  disk. `np.load` reads either.
- **Diagnostics** (`explore_iv_woe`, `explore_umap`, `explore_clip_impact`) now
  read through `data_loader.load_cached_arrays()`, which concatenates the
  per-snapshot NPZs and rebases `offsets`. They take `--cache_dir` / `--cache`
  pointing at a snapshot directory.

### Instance order changed, deliberately

Per-snapshot loading produces `(SNAPSHOT_DATE, NATIONAL_CODE, DPD_DAYS desc,
LOAN_ID)` — snapshot-major, where the whole-table load was customer-major. A
global re-sort to restore the old order was implemented, then removed: the
order-independence requirement (§20) is that the order be a function of the
data rather than of arrival order, not that it be customer-major specifically,
and snapshot-major is what per-snapshot caching yields for free. Consequence:
**XGBoost's index-based `subsample`/`colsample_bytree` draw differently than
they would have before**, so a pre- and post-September-2026 run are not
bit-comparable even on identical data. Accepted — `results_1`…`results_6` are
already dead as a baseline (§21), so nothing was being compared against.

### Residual risk — the next wall

The read is fixed; the **instance representation is not**. At loan grain,
~32M mature rows become ~32M instance dicts: roughly 19 GB of dict and array
object overhead, plus ~8 GB of `X_all` blocks held alive by the per-instance
`features` views, plus another ~8 GB while `_save_cache` fills `features_flat`.
On the 128 GB server that should fit alongside preprocessing and four arms, but
not comfortably. The array form already exists — it is exactly what the NPZ
cache stores and what `load_cached_arrays` returns — so the fix, if the run
falls over again, is to stop exploding it back into dicts in `_load_cache` and
teach `build_features` / the arms to consume the flat arrays directly. Not done
here: it touches the arms, the splits and the scorers.

`process_raw_data` also still materialises a float64 intermediate
(`df[feature_cols].apply(pd.to_numeric)`) and two full boolean frames for the
unparseable-value warning — ~27 GB of transient peak at whole-table scale, but
only ~1.4 GB now that it sees one snapshot at a time. Left alone.

### Regression tests

`tests/test_order_independence.py` — streaming load equals the whole-table load
(both grains) and is invariant to input row order; immature snapshots are never
queried; a cached mature snapshot is served from disk and a newly matured one
costs exactly one DB read; an immature prediction cache is rebuilt when the ETL
run tag moves and reused when it does not; `load_cached_arrays` rebases
`offsets` correctly across snapshot files (portfolio grain, variable group
sizes — the case that can silently be wrong).
