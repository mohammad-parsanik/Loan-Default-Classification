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
Snapshot dates are `YYYYMMDD` floats. The figures below are **what Run 7 actually
loaded on 2026-09-05** (§23); they move every month as snapshots mature, so read
them from the run log rather than from this table.

| | |
|---|---|
| Rows in `D_ANALYTICS.EDP_LOAN_FEATURES` | ~42.7 M (all snapshots, mature + immature) |
| Mature snapshots loaded | **23**, `20240419` … `20260219` |
| Mature loan instances | **32,538,715** |
| Columns | 71 contract columns = 64 features + 7 meta |
| Train / Val / Test instances | 16,872,770 / 4,204,534 / 1,655,036 |

Run 7's split shape: test is the newest mature snapshot (`20260219`); the six
snapshots from `20250822` to `20260120` are dropped to enforce the 6-month label
gap; the remaining 16 (`20240419`…`20250722`) are train, from which the
customer-disjoint 20% validation carve-out is taken. `VAL_SPLIT_MODE="customer"`
is why the log's "Val snapshots" line is empty — that is correct, not a bug.

Both ends of the range have moved since earlier runs: the ≤7B rebuild (§21)
backfilled history to 2024-04, and calendar time keeps maturing snapshots at the
front. The pre-rebuild 8-snapshot / ~5 M-instance shape this section used to
document is gone, and is not a baseline for anything (§20, §21).

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
Data profiling revealed the **99th percentile of loans per customer is 2**. The vast majority of customers have only 1 active loan. This has major architectural implications (see Section 3) — it is why DeepSets ≈ baseline and why attention was abandoned.

At `PREDICTION_GRAIN="loan"` (the default since §19) the runtime value is mechanically **1** — one loan per instance, nothing to truncate — as Run 7's log shows. `MAX_LOANS_PER_CUSTOMER` is inert there; it only gates the legacy DeepSets padding width. The "2" above is the portfolio-grain figure and the reason the portfolio detour bought so little.

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
| `explore_clip_impact.py` | Whether `OutlierClipper`'s `[p1, p99]` clip merges away a risk-bearing tail. Reads `tail_lift` = `P(severe \| x > p99) / P(severe)`. Reads from NPZ cache (no DB needed). Added for the 700M→7B population widening (§21); first run in Run 7, which closed that question (§23). **Caveat:** it measures over all mature rows, not the ranked `current_cat < 3` population, so DPD-family features score at the ceiling for mechanical reasons — see §23. |

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
├── run.py                         # CLI entry point: train / predict / explore
├── build_scoring_package.py       # Emit the ~21-file standalone scoring folder
├── analyze_walk_forward.py        # Recover WF verdicts from fold_results.pkl
├── explore_iv_woe.py              # ┐
├── explore_umap.py                # │ standalone diagnostics, read the NPZ
├── explore_shap.py                # │ cache, no DB needed (see EXPLORATION.md)
├── explore_clip_impact.py         # ┘
├── requirements.txt               # Python dependencies
├── AGENT_HANDOFF.md               # This document
├── CONFIG_REFERENCE.md            # Every project_config.py setting
├── MODEL_EVALUATION.md            # How to judge a trained model's numbers
├── DEPLOYMENT.md                  # train --final → bundle → predict runbook
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
│   ├── model/                     # LEGACY — dead unless DEEPSETS_ENABLED=True
│   │   ├── deep_sets.py           #   the neural arm (42K params), off by default
│   │   ├── losses.py              #   Cost-Sensitive Focal Loss
│   │   ├── meta_learner.py        #   XGBoost on frozen embeddings + Optuna
│   │   └── trainer.py             #   `TransformerTrainer` — name only, no transformer
│   ├── evaluation/
│   │   ├── metrics.py             # full_evaluation: ranking + classification blocks
│   │   ├── ranking.py             # ★ The deliverable's metrics: recall/lift@K, PR-AUC
│   │   ├── calibration.py         # StratifiedCalibrator (per-current-cat isotonic)
│   │   ├── decision.py            # mask_monotone, expected-cost decisions
│   │   ├── visualization.py       # Confusion matrix, ROC, capture curves
│   │   └── fold_aggregator.py     # Walk-forward fold metric aggregation
│   ├── baselines/
│   │   └── aggregated_xgboost.py  # ★ THE DEPLOYED MODEL — the four arms + ARM_BUILDERS
│   └── inference/
│       ├── predictor.py           # score_instances: calibrate → mask → flag → rank
│       ├── model_loader.py        # Load bundle or fold dir → (scorer, calibrator, features)
│       ├── scoring.py             # run_scoring(df, params) — manager-code entry point
│       └── scoring_params.py      # ScoringParams: explicit knobs + warned config fallback
│
├── data/snapshots/<stage>_<key>/  # NPZ cache, one file + manifest per snapshot
├── artifacts/                     # Per-run artifacts (models, plots, reports)
├── results_1/ … results_6/        # Runs 1–6 — DEAD as a baseline (§20, §21)
├── results_7/                     # ★ Run 7 (§23): the current baseline. Partial
│                                  #   console log + clip_impact.csv only
└── tests/                         # Unit + integration tests
    ├── test_pipeline_changes.py   # The main suite
    └── test_order_independence.py # Permute an input, assert the output is unchanged
```

---

## 10. Known Issues & Gotchas

1. **The active model is XGBoost, not DeepSets.** `src/baselines/aggregated_xgboost.py`, arm `"multiclass"`. DeepSets lost the Run-5/Run-6 shootouts and is off by default (`DEEPSETS_ENABLED=False`, §13/§15); it also now *raises* unless `PREDICTION_GRAIN="portfolio"`. Any document in this repo calling DeepSets or the Set-Transformer "active" predates July 2026 — that includes older passages of this file's §3, which is kept for the architecture history rather than as a statement of what runs. `README.md` was corrected in August 2026 (it once claimed Oracle/`cx_Oracle`; the DB is MSSQL/pyodbc).
2. **`Implementation_plan.md` is the original design doc** — many details have evolved (DeepSets replaced Transformer, Oracle replaced by MSSQL, walk-forward added then found infeasible). Treat it as historical context, not current truth.
3. **`set_transformer.py` was deleted** (July 26, 2026) — it was never imported by `run.py`. The neural path is `deep_sets.py`; the deployed model is `src/baselines/aggregated_xgboost.py`.
4. **Snapshot dates are stored as floats** (e.g., `20241021.0`), not integers or datetime. All temporal logic in `temporal_split.py` converts them to `datetime.date` for gap calculations.
5. **The baseline consistently matches or beats DeepSets+XGB.** This is expected with MAX_LOANS=2. The DeepSets model's early Cat-2 recall advantage (87% vs 76-78% in Runs 2-3) was **confounded**: DeepSets had a cost-sensitive loss while the baseline used plain argmax. Since July 7, 2026 both are evaluated under the same expected-cost decision rule (`cost_rule` metrics). Run 5 then settled it on the metric that actually matters — ranking, on every slice — which is why the architecture question is closed (§13).
6. **`torch.compile` is disabled on the Windows training server** (inductor requires MSVC). The model runs in eager mode.
7. **Label-informed truncation bug (FIXED July 7, 2026):** `process_raw_data` used to sort each customer's loans by `WORST_FUTURE_DPD` (a label) before `MAX_LOANS` truncation — the label selected which loans the model saw, and the predict path crashed because `EDP_Feature_pred` has no `WORST_FUTURE_*` columns. Now sorts by `DPD_DAYS` desc in both paths; unlabelled tables get `label = -1`. Requires cache rebuild (`DATA_VERSION` v1.1).
8. **`EDP_Feature_pred` never existed as a separate table (FIXED July 7, 2026):** the live DB has one table (`TRAIN_TABLE`) holding matured snapshots plus the newest not-yet-matured one(s). `PRED_TABLE` is gone; `Predictor` now reads `TRAIN_TABLE` for prediction too. Because that table carries `WORST_FUTURE_CAT`/`WORST_FUTURE_DPD` for every row, an immature snapshot's label columns hold a **degenerate** value ("worst category observed so far", not the real future outcome) rather than being absent/NULL — `DataLoader.load_pred_portfolios` now drops those columns before `process_raw_data` so they can't masquerade as real labels.

---

## 11. Changes of July 7-8, 2026 (methodology overhaul + 4-class bump — items 1-9 first ran in Run 4, §12; item 10 in Run 5, §13)

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
10. **NUM_CLASSES 3 → 4, single-file deployment bundle (2026-07-08):** `config.NUM_CLASSES=4` — raw cats 0/1/2 pass through 1:1, raw cats 3-4 now collapse into a new class 3 ("Severe Past Due") instead of into class 2. `COST_MATRIX` is now 4×4 (class-3 row/col is a derived, not-yet-business-tuned placeholder — see the table above). Every hardcoded `3`/`range(3)` tied to class count was swept to `config.NUM_CLASSES` (`losses.py`, `deep_sets.py`, `meta_learner.py`, `aggregated_xgboost.py`, `metrics.py`, `visualization.py`, `explore_iv_woe.py`, `explore_shap.py`, `explore_umap.py`); `bootstrap_confidence_intervals` now tracks both `recall_class_2` and `recall_class_3`; `Predictor` emits a 4th probability column `P_SEVERE_PAST_DUE`. `DATA_VERSION` bumped to `v1.2` (labels are baked into the NPZ cache, so this forces a rebuild). Separately, `train_single_fold` now also writes `<fold_dir>/model_bundle.pkl` — everything `Predictor` needs (scaler, DeepSets state_dict + hparams, XGBoost raw model bytes, calibrator) in one file; `ModelLoader.load_pipeline()` dispatches to it when `artifact_dir` points at a file instead of a directory. Pure export convenience — the per-file directory artifacts are unchanged and still used in-pipeline. All verified locally via `tests/test_pipeline_changes.py` (20 tests, synthetic data, no DB); first run on the server in Run 5 (§13).

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

*Rewritten 2026-09-05 after Run 7 (§23), reprioritised the same day after §24.*

0. **Establish how much of the base-rate drift is retroactive source deletion (§24).** *(Confirmed with the source owner: hard delete, inconsistently applied, nothing available upstream to fix it with today.)* The severe rate runs 8.37% → 9.96% → 12.95% across the train / dropped / test windows, the ETL owner has confirmed hard deletion of closed and post-NPL installments, and one as-of rebuild of every snapshot makes old snapshots systematically depleted of exactly the class-3 rows. This gates what the training labels *mean*, so it comes before any modelling decision below — including item 2. Cheapest first move: `python explore_snapshot_drift.py` (cache only, no DB). Then a cohort-persistence count for `d(age)`. **Also: copy the snapshot cache aside before the next rebuild** — the two-vintage comparison, the conclusive test, was lost this cycle because the cache path is reused.
1. **Pull Run 7's artifacts off the server.** `F:\EDP\Loan Default Classification\artifacts\20260905_133423\` — `metrics.json`, `arms_metrics.json`, `model_bundle.pkl`, `plots/`. `results_7/` currently holds only a partial console log (capture starts after the load stage) and `clip_impact.csv`; several questions below need numbers that log does not print, including the whole `ranking_single_loan` block and every bootstrap CI.
2. **Settle `DEPLOY_ARM`.** Run 7 has `per_cat` beating the hard-locked `multiclass` on pooled AP (0.6721 vs 0.6555) *and* on the cat_0 early-warning slice (0.1780 vs 0.1460) — the exact slice the Run-6 lock was justified by (§23). One snapshot is not enough to reverse a verdict that §16 confirmed across 15 folds. Re-check with `WALK_FORWARD_ENABLED=True`, `MODEL_ARMS=["multiclass","per_cat"]`, then `analyze_walk_forward.py --min_train_snaps 3`. With 23 mature snapshots and a warm cache this is now an overnight job, not the 1.5–2 h/fold §16-era estimate.
3. **Ship:** `python run.py train --final` → move `artifacts/<ts>_final/fold_01/model_bundle.pkl` to production → `python run.py predict` → queue CSV to the API caller at 240/h (`RISK_RANK` order). For the tech team's manager code: `build_scoring_package.py` → `run_scoring(df, ScoringParams(...))`.
4. ~~**Contract edits, batched.**~~ **DONE** (contract v2, §24): `clip: false` on the nine bounded integer counts, including the `COUNT_90PLUS_DPD_LAST_3M` constant. **The next run rebuilds the cache** — copy the existing snapshot dir aside first (item 0), it is the two-vintage deletion test. The ETL team still needs to mirror the flags upstream.
5. ~~**Make `explore_clip_impact.py` carve-aware.**~~ **DONE** (§24): `--ranked_only`, a `head_lift` column for the p1 side, and a hard error on any `p1 >= p99`. Re-run it with `--ranked_only` after the next cache rebuild and re-read the DPD family and `CATEGORY_TREND_1M`/`_3M` — those numbers were never trustworthy without the carve.
6. **Refit calibration on the newest mature snapshot** rather than on val (§24). Cheapest real mitigation for the drift, strictly an improvement, and it does not touch ranking. Not a substitute for item 0 — it corrects the output layer while the training labels stay depleted.
7. ~~**Test whether an amount interaction exists before banding.**~~ **DONE** (§24): the deployed arm now reports `ranking_by_exposure` — pooled AP and lift per `REMAINING_AMNT` decile — and logs the AP spread. Read it off the next run: flat ⇒ banding buys nothing and a base-rate shift is one tree split; a dip in one decile ⇒ a real interaction, and the first thing to rule out is that post-NPL deletion varies with loan size.
8. **Re-read `assert_feed_invariants`' load-stage output** against the ≤7B population — §21 flagged it and Run 7's captured log starts one stage too late to show it. Costs one `grep` of the next run's log.
9. **Optuna** (optional): `ARM_OPTUNA_TRIALS=20` on the `--final` fit. All four Run-7 arms sit within ~1.7 pt pooled AP on `XGB_DEFAULTS`, so headroom is probably small — but it has never been measured on this population.
10. **Business items (non-blocking):** exposure-weighted ranking (§21 — a 10× exposure range makes "every queue slot is worth the same" a real distortion now; raised with business, pending); `CERTAINTY_ACT_THRESHOLD` (Run-5 evidence: 88% of day-one calls confirm near-certainties); real cost numbers only if the cost rule is ever promoted. *(Survivorship in the ETL sample is no longer an open question filed here — it is item 0 and §24.)*
11. **Nice-to-have, promoted by item 2:** update walk-forward's `build_fold_instances` to use customer-disjoint validation, matching the single-split path (§16). §16 rated this low priority *because walk-forward was off by default* — if walk-forward is now going to decide the arm question, it should run the same validation scheme as the path it is deciding for.

---

## 19. Prediction Grain: Customer → Loan (July 26, 2026 — first ran on the server in Run 7, §23)

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

## 20. ETL Feed Alignment + Order Independence (August 26, 2026 — first ran on the server in Run 7, §23)

The upstream ETL was rebuilt. Two work orders landed together, because the column contract they share makes them one change: `etl_integration/SESSION_HANDOFF.md` (align with the new feed) and `etl_integration/ORDER_INDEPENDENCE_ML_PLAN.md` (stop depending on row/column order). Both are local-only documents; this section is the tracked record.

**Status (2026-09-05):** the block described below has lifted — the rebuild landed and Run 7 (§23) trained against it successfully. The *comparability* half of it has not and never will: **do not compare any metric against `results_1`…`results_6`.** Several features changed *meaning*, not just value, so a model trained on the new table is not comparable to one trained on the old, and the difference is not noise. §21 then added a second, independent reason (the population widened). Run 7 is the new baseline.

*Original wording, kept because it states the reasoning:* "do not retrain, and do not compare any metric against `results_1`…`results_6`. The upstream validation gate has not passed and the 12-snapshot rebuild has not happened. […] Everything below is a correctness and reproducibility change to the code that *will* do that training; the point of landing it now is that the rebuild becomes the first result this project can reproduce." — that is exactly what happened.

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

## 21. Population Widened: contract amount ≤ 700M → ≤ 7B (September 2, 2026 — first ran on the server in Run 7, §23)

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

### The clipping question — ANSWERED in Run 7 (§23): no change, leave `clip: true`

> **Outcome first:** measured on the Run-7 cache, all three amount features have `tail_lift` **below 1.0** (`REMAINING_AMNT` 0.588, `UPCOMING_AMNT` 0.470, `PAYED_OVERDUE_AMNT` 0.264) — the tail the widening admitted is *less* likely to go severe than the population average. Large loans here are safer, not riskier. `clip: true` stays on all three. The reasoning below is retained because it is the template for the next time a feature's distribution moves; the numbers and the one remaining candidate are in §23.

`OutlierClipper` clips every non-binary, non-`NO_CLIP` feature to `[p1, p99]` fit on train. **Clipping is the one preprocessing step XGBoost cannot shrug off.** Imputation and `PortfolioRobustScaler` are monotone and trees split on order, so they are invisible to the model; clipping merges every value above p99 into a single number and destroys the ordering up there.

Three features are amount-scaled and inherit the 10× wider range directly: `PAYED_OVERDUE_AMNT` (8), `UPCOMING_AMNT` (14), `REMAINING_AMNT` (66). The concern is structural, not hypothetical: the newly admitted large loans are a *minority* sitting at the *top* of the distribution, which is precisely the region clipping flattens. If severe events concentrate there, the clip is deleting the signal the widening was meant to add. Note the shape of this is the same trap as the `DAYS_SINCE_LAST_*` sentinels in §20 — harmless in the columns where the extreme value is the majority, destructive where it is a minority — which is why it gets measured rather than assumed.

**Not pre-emptively changed.** Setting `clip: false` on those three in `contract/columns.json` is a one-line fix and it is the wrong move to make blind: clipping exists because unbounded tails hurt, and the tail could as easily be noise as risk. Measure first with `explore_clip_impact.py`, then A/B on validation lift@K before committing. Whichever way it goes, changing a `clip` flag bumps `contract_version` and invalidates the cache on its own.

**That is what happened, and "the tail could as easily be noise as risk" is the half that came true.** Had the flags been flipped pre-emptively the model would have gained three unbounded features for a tail that is *below* base rate. See §23 for the full table, for why the 22 features that tripped the tool's 1.5 warning threshold are mostly an artifact of measuring over the un-carved population, and for the one column (`CATEGORY_TREND_1M/3M`) still worth an A/B.

### Secondary consequences, no action

- **`COST_MATRIX` is flat per loan** and now spans a 10× exposure range, so it is considerably more wrong than it was. Harmless: it feeds `PREDICTED_CLASS` / `EXPECTED_COST` only, never `RISK_SCORE`, never training. It was already advisory (business gave only the 4× anchor); it is now more so.
- **`MAX_LOANS_PER_CUSTOMER`** resolves from the 99th percentile at runtime and is inert at loan grain. Ignore.
- **`assert_feed_invariants`** was measured against the old segment. Nullability and range assumptions are worth re-reading in the log on the first load rather than trusting.
- **Row volume** grows. If it grows a lot, the arm-training stages are where it will show.
- **`etl_integration/CONSUMER_CONTRACT.md` was refreshed from upstream on 2026-09-02** and is current: §7 carries the new ceiling, §8 carries a row-population entry for it. Diffed against the previous copy, that ceiling is the *only* substantive change — no column was added, removed or redefined, `contract_version` is still 1, and the vendored `etl_integration/columns.json` still matches the tracked `contract/columns.json` projection exactly, so no contract refresh is needed. (The refreshed copy arrived without the provenance header the convention calls for; it was restored by hand, with the ETL commit noted as unrecorded.)

### The business question this raises

The queue ranks by P(severe) alone and `recall@K` counts loans, so a 7B loan and a 70M loan occupy identical slots at identical value. If the objective is money at risk, that is now a real distortion rather than a rounding error — either rank on `P3 × REMAINING_AMNT` or report exposure-weighted recall beside the count-based one. Deliberately **not** implemented: it changes what the deliverable optimises, which is a business decision, not a maintenance one. Raised with the business, pending.

---

## 22. Load Stage Rebuilt for the ≤7B Row Volume (September 5, 2026 — held up in Run 7, §23)

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

### Residual risk — the next wall (did NOT arrive in Run 7)

> **Outcome:** Run 7 (§23) carried ~32.5 M loan instances through load, preprocessing
> and four arms on the 128 GB server in 74 minutes without falling over. The
> array-native refactor below stays unnecessary — keep it as the known fix if a
> future population change pushes it over, not as pending work.

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

---

## 23. Run 7 — First Result on the Rebuilt ≤7B Feed (September 5, 2026)

Everything landed in §19–§22 has now actually run. Run 7 is the first end-to-end
training pass against the rebuilt `D_ANALYTICS.EDP_LOAN_FEATURES` at loan grain,
and it is **the project's new and only baseline** — `results_1`…`results_6` are
dead for the two independent reasons in §20 (features changed meaning) and §21
(the population widened).

**What is local vs. what is on the server.** `results_7/` holds the console log
*from the end of the load stage onward* and `clip_impact.csv`. The run's own
artifacts — `metrics.json`, `arms_metrics.json`, `model_bundle.pkl`, the plots,
the `ranking_single_loan` block and every bootstrap CI — are on the server under
`artifacts/20260905_133423/` and have not been pulled. Everything below is read
off the log and the CSV; anything not stated here needs those files.

### It ran, and the §22 memory wall did not arrive

| Stage | Wall time |
|---|---|
| Load — 23 mature snapshots, DB read + NPZ write | 2,271 s |
| Temporal split | 38 s |
| Preprocessing — fit on train, transform 22.7 M instances | 783 s |
| Build features | 34 s |
| Arms: multiclass / binary / ordinal / per_cat | 366 / 78 / 269 / 290 s |
| Deployed-arm CI + plots | 316 s |
| **Total** | **4,445 s (74 min)** |

32,538,715 loan instances × 64 features, on the 128 GB / 20-core server. §22's
residual risk — ~32 M instance dicts at ~19 GB plus the array blocks alive
underneath them — fit. The array-native refactor is not pending work; it is the
known fix if a future population change pushes it over.

Note the shape of the bill has changed: **the load stage is now 51% of wall
time, and it is the part that mostly disappears next month.** 22 of the 23
snapshot NPZs are permanent (a matured snapshot is never recomputed upstream),
so the next monthly run reads one snapshot from the DB and the rest from disk.
That is exactly what the §22 rebuild was for.

### Results

Ranking block. Test = `20260219`; 1,583,717 ranked rows (`current_cat < 3`, the
rest rule-flagged); severe base rate **0.0903**. K = `API_RATE_PER_HOUR` × window
⇒ 5,760 / 40,320 / 172,800.

| Arm | pooled AP | cat_0 AP | cat_1 AP | cat_2 AP | R@1d | R@1w | R@1m |
|---|---|---|---|---|---|---|---|
| **multiclass** (deployed) | 0.6555 | 0.1460 | 0.4660 | 0.8196 | 0.037 | 0.228 | 0.678 |
| per_cat | **0.6721** | **0.1780** | 0.4732 | 0.8212 | 0.037 | 0.234 | 0.693 |
| binary (diagnostic, never deployed) | 0.6693 | 0.1538 | 0.4705 | 0.8209 | 0.037 | 0.235 | 0.688 |
| ordinal | 0.6563 | 0.1498 | 0.4724 | 0.8211 | 0.037 | 0.231 | 0.677 |

Classification block, secondary: multiclass F1 0.6362 / QWK 0.7154 / cost 0.4203;
ordinal F1 0.6366 / QWK 0.7215 / cost 0.3577; per_cat F1 0.6370 / QWK 0.7136 /
cost 0.4416. Read these as diagnostics only — the cost matrix is guessed, and the
aggregate F1 is inflated by the `current_cat_3` slice, where F1 is 1.0000 by the
`label >= current_cat` identity rather than by anything the model did.

Judged on its own terms (the only way it can be judged): the queue finds **67.8%
of future-severe loans in the first month of calling at 240/h**, 6.2× better than
random, and on the cat_0 early-warning slice — healthy loans today, severe within
six months — **59.8% land inside one week of calling, a 15.0× lift**.

> **Read this block against §24.** The test snapshot's severe base rate (0.0903
> ranked) is roughly twice the training window's (0.0463), because the upstream
> installment table is hard-deleted for closed and post-NPL loans and older
> snapshots were rebuilt after more of their rows had gone. The ranking numbers
> above survive it — isotonic calibration is monotone, so the ordering is
> untouched — but the calibrated probabilities are biased low, which is why the
> classification block's cat-0 Cat-3 recall looks as weak as it does.

### The one decision Run 7 forces: `DEPLOY_ARM`

`DEPLOY_ARM` is hard-locked to `"multiclass"` on Run-6 evidence, and Run 7
contradicts that on both metrics the lock was justified by. **`per_cat` wins
pooled AP (0.6721 vs 0.6555) and wins the cat_0 slice by 22% relative (0.1780 vs
0.1460)** — the exact slice §15 recorded per_cat as *worst* on. Had `DEPLOY_ARM`
been `"auto"`, this run would have deployed per_cat. It did not, because the
config says `"multiclass"`; the log's "Deployed arm (configured)" line is the
tell.

Reasons not to flip the lock on this alone:

- One test snapshot, one split, and **no tuning** — `ARM_OPTUNA_TRIALS=0`, so all
  four arms ran on `XGB_DEFAULTS`. The spread across arms (~1.7 pt pooled AP) is
  within the range a tuning pass could reshuffle.
- The cat_0 gap is concentrated at the sharpest end of the queue (`R@1_day` 0.127
  vs 0.073, lift 22.3× vs 12.9×) and has closed by `1_month` (0.998 vs 0.991).
  That is the least stable part of a single-snapshot estimate: ~9,600 severe
  cat-0 rows competing for the first 5,760 slots.
- §15's per_cat verdict was itself confirmed across 15 walk-forward folds (§16).
  Reversing it deserves the same standard, not a lower one.

**Recommended:** `WALK_FORWARD_ENABLED=True`, `MODEL_ARMS=["multiclass","per_cat"]`,
then `analyze_walk_forward.py --min_train_snaps 3`. Walk-forward is well-supplied
now — §16's alarming fold spread was a 1–2-training-snapshot artifact, and there
are 23 mature snapshots. With the load stage cached, a fold costs roughly
preprocessing + 2 arms + eval ≈ 30 min, so this is an overnight job. If per_cat
holds across folds, flip the lock; if it does not, record that its cat_0 win was
snapshot noise so the question does not get reopened every run.

### Clipping: §21's question is answered, and not the way it was expected

`explore_clip_impact.py` on the Run-7 cache → `results_7/clip_impact.csv`. 53
rows (64 features − 4 `NO_CLIP` − 7 binary), of which the clip actually moves a
value on 43 and is a no-op on 10 bounded categoricals;
`tail_lift = P(severe | x > p99) / base_rate`, base rate 9.08% over all mature
loan-rows.

**The three amount-scaled features that motivated the tool come back below 1.0.**
The tail the ≤7B widening admitted is *less* likely to go severe than the
population average:

| Feature | p1 | p99 | max | tail_lift |
|---|---|---|---|---|
| `REMAINING_AMNT` | 3.42e7 | 4.04e9 | 2.90e10 | **0.588** |
| `UPCOMING_AMNT` | 2.96e7 | 3.99e9 | 2.85e10 | **0.470** |
| `PAYED_OVERDUE_AMNT` | 1.96e7 | 1.99e9 | 7.86e9 | **0.264** |

**Decision: leave `clip: true` on all three.** Big loans in this population are
safer, not riskier; exempting them would hand the model three unbounded features
to buy nothing. §21's concern was the right one to have and is now closed. (The
p99s incidentally confirm the population really did widen — they sit ~6× above
anywhere a ≤700M filter could have put them.)

**31 features trip the tool's 1.5 warning threshold (22 of them exceed 5×), and
the warning is largely an artifact.** The top of the list is the DPD family — `DPD_DAYS` 11.01,
`UNPAYED_INST_CNT` 11.01, `DPD_DAYS_T1`…`T5` 10.88–10.98, `MAX_DPD_LAST_6M`
10.83, `HIST_MAX_DPD_DAYS` 10.17. **11.01 is the arithmetic ceiling** (1 /
0.0908): *every* row above p99 is severe. That is the `label >= current_cat`
identity, not discovered signal — a loan 700 days past due is already class 3,
and `CARVE_CURRENT_CAT_GE=3` removes those rows from the queue before the
model's ordering up there could matter.

**Tool limitation, fix before trusting the number again:** `load_loan_rows`
computes `tail_lift` over *all* mature rows, not over the ranked
(`current_cat < 3`) population, so it cannot distinguish "this tail predicts
severity" from "this tail *is* severity". A `--ranked_only` flag mirroring
`ranking_metrics`' carve would make it answer the question the deliverable
actually asks. Until then, read the DPD-family rows as uninformative — the
amount-feature rows are unaffected, since those tails are not mechanically
severe.

**One row is not obviously mechanical:** `CATEGORY_TREND_1M` (p1 −1, p99 1, max
2, `tail_lift` 11.01 on 0.11% of rows) and `CATEGORY_TREND_3M` (10.53 on 0.95%).
The clip merges "jumped 2 categories" into "jumped 1" — a *deterioration* signal
rather than a current-state one, which is precisely what the cat_0 slice is short
of. It may still be mechanical (a +2 jump probably lands you at cat ≥ 2 already);
the carve-aware rerun settles it. If it survives, it is the cheapest A/B on the
list: a one-line `clip: false` on a 3-valued column with no unbounded-tail risk
to speak of. It bumps `contract_version` and invalidates the whole cache, so
batch it with something else.

**`COUNT_90PLUS_DPD_LAST_3M` is destroyed outright — `p1 == p99 == 0`.**
`OutlierClipper` clips it to `[0, 0]`, so every row becomes 0 and the column
enters all four arms as a **constant**. Its `tail_lift` (8.19 on the 0.36% of
rows above 0) is carve-contaminated like the rest of the DPD family and is
beside the point: a feature reduced to a constant is a defect whatever its
signal was. This one does not need the carve-aware rerun to justify a fix — it
is a bounded 0–3 count with no tail to protect against. **`clip: false`.**

That generalises to **12 columns with `max <= 14`** where clipping is merging
values it was never meant to touch: `COUNT_90PLUS/60PLUS/30PLUS_DPD_LAST_3M`,
`PRE_UPTO30/60/120/150_DPD_LOANS`, `COUNT_ACTIVE_CONTRACTS`,
`COUNT_DELINQUENT_CONTRACTS`, `CATEGORY_TREND_1M`/`_3M`, `PCT_COMPLETED`.
Clipping exists to bound unbounded tails; a 0–3 count has none, so `clip: false`
on bounded integer counts is defensible on principle without waiting on the
lift numbers.

**The low side is unmeasured and is doing damage in the direction that matters.**
`tail_lift` only looks above p99, but `pct_lo` shows the p1 clip collapsing the
*clean* end — `PAYED_OVERDUE_INST_CNT` has `p1 = 3`, so 0.91% of rows are pushed
**up** to 3, merging "never paid an overdue installment" with "paid three";
`HIST_MAX_DPD_DAYS` has `p1 = 1`, pushing 0.83% of rows at 0 ("never any DPD")
up to 1. Both collapse exactly the `current_cat_0` population the early-warning
task lives in. A `head_lift` mirroring `tail_lift` on `x < p1` is four lines and
belongs in the same edit as `--ranked_only`.

### Loose ends this run left

- **`assert_feed_invariants`' first report against the ≤7B population was not
  seen** — the captured log starts at "Stage 1 completed". §21 flagged re-reading
  it; still outstanding, and free on the next run.
- **`ranking_single_loan` is computed but never logged** (`metrics.py:107` writes
  it to `metrics.json`; `run.py` prints only the `ranking` block). It is moot as a
  cross-run comparison now (§21 killed that), but it is still the cleanest
  within-run read of what the grain change did. Consider logging it, or stop
  documenting it as the comparison to make.
- **Two sklearn warnings repeat on every arm** — `confusion_matrix` and
  `cohen_kappa_score` seeing a single label in `y_true`/`y_pred`. Benign given the
  `current_cat_3` slice is a single class by definition, but they are noise in
  every log and are fixed by passing `labels=range(NUM_CLASSES)`.

---

## 24. Label Base-Rate Drift and Retroactive Source Deletion (September 5, 2026)

The largest finding in Run 7 is not in the metrics block. **The severe base rate
rises monotonically with snapshot recency, and the training window sits at the
bottom of that trend.**

### The measurement

Three disjoint windows, reconstructed from the Run-7 log (the ordinal arm's
`P(y > 2)` positive count gives the train label distribution; the per-cat arm's
strata give train `current_cat`; the classification block gives the test slice
counts) and cross-checked against `explore_clip_impact.py`'s independently
computed pooled figure:

| Window | Snapshots | Rows | Severe (class-3) rate |
|---|---|---|---|
| 2024-04 … 2025-07 — train + val | 16 | 21,077,304 | **8.37%** |
| 2025-08 … 2026-01 — dropped for the 6-month gap | 6 | 9,806,375 | **9.96%** |
| 2026-02-19 — test | 1 | 1,655,036 | **12.95%** |
| pooled, all mature | 23 | 32,538,715 | 9.08% |

The reconstruction closes to 9.0805%, exactly the figure `explore_clip_impact.py`
prints from the cache, so the decomposition is sound. On the ranked
(`current_cat < 3`) population the gap is starker still: **train 4.63% vs test
9.03%, a factor of 1.95.**

This is not composition. The test snapshot's `current_cat` mix is *healthier*
than train's (61.2% vs 56.6% cat-0). Had train's strata carried test's
conditional rates it would hold 1,484,256 severe rows; it holds 749,762 — train
rates average **50.5% of test rates**, and the shortfall survives any
reallocation across strata (give train's cat-2 the full test rate of 0.5939 and
cat-0+cat-1 are still left at 0.97% against test's 4.99%).

### The mechanism: hard deletion upstream, one as-of rebuild

Confirmed with the source-data owner: **the installment table is hard-deleted for
loans the bank is done with — fully paid, or post-NPL.** No soft delete, no
status flag, nothing recoverable, and not reliably applied either (loans have
been found retaining full installment history after payoff, alongside loans with
none). Every snapshot in `EDP_LOAN_FEATURES` was rebuilt from a single as-of view
during the ≤7B backfill (§21), so **an old snapshot was reconstructed after more
of its loans had been deleted than a recent one.**

The bias is one-directional, and the scope filter is why. `q1` requires **last
installment ≥ T+6**:

- **Payoff-driven removal is structurally suppressed.** To be in snapshot T a
  loan needed ≥ 6 months of remaining schedule, so it cannot have reached
  scheduled maturity by T. Only genuine prepayment escapes that, and it is the
  weaker channel.
- **NPL-driven removal is not.** A loan can go NPL at any point with years of
  schedule left. Deleting it removes a row that *would* have satisfied the
  filter — and whose `WORST_FUTURE_CAT` would have been 3.

So the deletion **selectively removes positive labels**, and removes more of them
the older the snapshot. That is exactly the observed shape.

The magnitude is consistent too. If a fraction `d` of rows is deleted and
essentially all carried label 3, then `observed = (s − d)/(1 − d)`. Taking the
test window as the least-biased estimate of `s`:

| Window | Age at rebuild | Observed | Implied `d` |
|---|---|---|---|
| 2024-04 … 2025-07 | 17–29 months | 8.37% | **5.0%** |
| 2025-08 … 2026-01 | 8–13 months | 9.96% | **3.3%** |

Deletion roughly linear in snapshot age, at ~5% attrition over two years of NPL
workout. Ordinary numbers. Genuine portfolio deterioration is not excluded and
the two are not mutually exclusive — but deletion alone reconciles the windows,
and it is the hypothesis with a confirmed mechanism behind it.

### Why this is worse than a drifting base rate

**It hits features, not just labels.** Ten of the 64 features are customer-level,
computed over the customer's *other* loans — `AVG_DPD_OTHER_LOANS`,
`COUNT_ACTIVE_CONTRACTS`, `COUNT_DELINQUENT_CONTRACTS`, `MAX_DPD_ANY_PAST_LOAN`,
`PRE_UPTO*_DPD_LOANS`, `CNT_RECOVERED_BEFORE`, and the two `*_CLOSE*_LOAN_DPD`
columns that are *about closed loans specifically*. If siblings were deleted these
are computed over a survivor subset and biased low, worst in the oldest
snapshots. At serving time you score the newest immature snapshot, where nothing
has been deleted yet — so those features arrive **higher than anything the model
was trained on**. Train/serve skew in the cross-loan risk signal, silent, and in
the same direction as the label bias.

**The 6-month gap guarantees the training data is contaminated.** Horizon 6
months plus gap 6 months means the freshest usable training snapshot is ~7 months
before test — precisely the region where deletion has had time to bite. *There is
no configuration of this pipeline that trains on clean data.* That tension may be
the defining constraint of the project and belongs in front of the business, not
buried in a config note.

**It compounds the `LABEL_HORIZON_DATE` hazard.** That column is kept out of
`FEATURE_COLS` because it proxies snapshot recency. Snapshot recency now also
predicts label prevalence for a second, independent reason. Any feature that
leaks recency leaks more than it used to.

### What this changes

- **Calibration.** `StratifiedCalibrator` is fit on val — customer-disjoint, but
  drawn from the *train* snapshots, i.e. the lowest window — and applied to a fold
  running at ~2× those rates. Per-stratum isotonic maps onto the training
  window's base rates precisely, so calibrated `P(severe)` is biased low by
  roughly the window ratio. **Ranking is unaffected** (isotonic is monotone, so
  `RISK_SCORE` order, PR-AUC, recall@K and lift all stand). `PREDICTED_CLASS`,
  `EXPECTED_COST` and any future `CERTAINTY_ACT_THRESHOLD` are not — this is the
  mechanical explanation for multiclass's cat-0 Cat-3 recall of 0.1127, and it
  makes the `Cost:` column uninterpretable as a cross-arm comparison.
  Refitting calibration on the newest mature snapshot is necessary but **not
  sufficient**: it corrects the output layer while the training labels stay
  depleted.
- **`results_7` remains the baseline** — every alternative is worse, and the
  ranking block is the metric that survives.
- **Nothing here changes the `DEPLOY_ARM` question** (§23), which is a
  within-fold comparison under identical contamination.
- **The amount-feature clip verdict (§23) stands but its measurement is now
  provisional.** `tail_lift < 1` on the three amount columns comes from the same
  contaminated population; if large NPL loans are written off faster because they
  are material, large loans would *look* safer than they are. The recommendation
  does not move — `clip: true` is the safe default either way — but treat "big
  loans are safer" as unestablished.
- **Amount banding (2B split, or any other) is premature.** The case for a
  segmented model is that the feature→target *relationship* differs, not that the
  base rate does; a base-rate shift is one tree split, which XGBoost finds on its
  own at a learned threshold. `per_cat` is the precedent and it works because
  `current_cat` changes the label *mechanics* (the monotone floor). Amount does
  not. Test the interaction before building anything: add an amount-decile
  stratification to the ranking block, mirroring `by_current_cat` —
  `ranking_metrics` already takes `strata` and recurses. Flat lift@K across
  deciles ⇒ banding buys nothing. Note also that if the real concern is "big
  loans matter more", exposure-weighted ranking (§21, already with the business)
  is the cheaper and more direct lever.

### How to test it — status

1. **Ask the ETL owner.** ✅ Done — hard delete, confirmed, not consistently
   applied, nothing available upstream to fix it with as of now.
2. **Same snapshot, two vintages.** ❌ Dead. `explore_clip_impact.py --baseline`
   returns "no cached snapshot in …" — no pre-v1.5 cache survived the rebuild.
   The script's own docstring warns the cache path is reused; that warning was
   the one to act on and it is now too late for this cycle. **Copy the cache
   aside before the next rebuild** and this becomes the conclusive test.
3. **Feature-recency test.** ▶ Tool written: `explore_snapshot_drift.py` (cache
   only, no DB). Reports per-snapshot row count, severe rate, mean portfolio
   size and the mean of 11 customer-history "canary" columns, then Spearman rho
   of each series against snapshot order. The discriminator: **deletion moves the
   severe rate AND drags the customer-history means along with it; genuine
   deterioration moves the severe rate alone.** Portfolio growth confounds the
   feature means (row volume is already up ~25% across the range), so this is
   strong evidence, not proof.
4. **Cohort persistence.** Loans present in snapshot T whose schedule also spans
   T−1, checked for presence in T−1. Measures the deletion rate directly and
   gives `d` per snapshot age empirically. It diagnoses rather than fixes — but
   `d(age)` is the input any reweighting scheme would need, so it is the
   prerequisite for the only mitigation that addresses the training labels
   themselves rather than the output layer.

### Landed 2026-09-05 (contract v2 — the cache is invalidated)

Implemented against §23/§24; none of it has run on the server yet.

- **`contract_version` 1 → 2**, `clip: false` on the **nine bounded integer
  counts** (`COUNT_90PLUS/60PLUS/30PLUS_DPD_LAST_3M`,
  `PRE_UPTO30/60/120/150_DPD_LOANS`, `COUNT_ACTIVE_CONTRACTS`,
  `COUNT_DELINQUENT_CONTRACTS`). `COUNT_90PLUS_DPD_LAST_3M` was the proof —
  `p1 == p99 == 0` made it a constant in every arm — and the rest follow on
  principle: clipping bounds unbounded tails, and a 0–3 count has none.
  **`clip` only, not `scale`**: scaling is monotone and XGBoost splits on
  order, so exempting them from it would claim something untrue.
- **`NO_CLIP` now has two populations**, so `column_contract`'s self-check moved
  from `NO_CLIP == NO_SCALE` to `NO_SCALE <= NO_CLIP` with
  `set(SENTINELS) == NO_SCALE`. Sentinel-bearing columns are exactly the
  unscaled ones; anything else in `NO_CLIP` is a bounded count.
- **`_LOCAL_OVERRIDES` in `column_contract.py`** records the nine as a deliberate
  divergence from the vendored ETL copy. `_check_vendored_copy` compares with
  the overrides applied, so it still catches *unexpected* drift, and it now logs
  when upstream adopts one so the entry gets deleted rather than accumulating.
  **The ETL team should mirror this** — until they do, the tracked contract is
  ahead of the vendored copy by exactly these nine flags.
- **`TRAIN_WINDOW_SNAPSHOTS`** (`project_config`, default `None`): positive N
  keeps the newest N eligible training snapshots, negative N the oldest N. This
  is the §24 recency A/B — train both, score on the same test fold, and see
  whether depletion costs ranking or only calibration. Applied in both the
  single-split and walk-forward paths, and in walk-forward it narrows *before*
  the `MIN_TRAIN_SNAPSHOTS` check so the minimum means what it says.
- **`ranking_by_exposure`** — the ranking block cut by `EXPOSURE_FEATURE`
  (`REMAINING_AMNT`) decile, via `metrics.exposure_decile_ranking`, logged for
  the deployed arm with the AP spread across deciles. Deciles are cut on the
  ranked population only and are rank-based, so a skewed amount distribution
  still yields balanced bins. **Read `pr_auc` and `lift`, not `recall`** — K is
  the whole API budget spent inside one decile. This is the banding test: flat
  ⇒ an amount-banded model buys nothing.
- **`compute_metrics` passes `labels=range(NUM_CLASSES)`** to `f1_score` and
  `cohen_kappa_score`, and returns `nan` for QWK directly when nothing varies.
  Fixes the two sklearn warnings per arm, and makes the `current_cat_3` slice
  report macro-F1 **0.25** instead of a free **1.0000** — that slice is one
  class by the label identity, and scoring it over the single observed class
  read as a perfect model on 4% of the test set. **Pooled numbers are
  unchanged** (all four classes are present, so `labels=` is a no-op there).
- **`explore_clip_impact.py`**: `--ranked_only` (carve to `current_cat <
  CARVE_CURRENT_CAT_GE`, which is what makes `tail_lift` answer the
  deliverable's question), a `head_lift` column for the p1 side, an error on
  any `p1 >= p99` column, and a warning when run without `--ranked_only`.
- **`MAX_LOANS (99th pct on train)`** now says `[inert at loan grain]`, because
  `n_loans` is 1 by construction there and the line otherwise reads as a claim
  about how many loans customers hold.

Regression tests in `tests/test_pipeline_changes.py`: the window knob's two arms
are disjoint and over-asking is a no-op; exposure deciles partition the ranked
population exactly and stay balanced; the degenerate slice scores against all
`NUM_CLASSES`.

### Mitigations, none free

- **Restrict or downweight old snapshots.** Cleanest in principle, and directly
  opposed by the 6-month gap: the snapshots nearest the test window are exactly
  the ones the gap forbids. Cutting the 16 train snapshots to the newest few
  trades bias for variance, and §16 already established that folds trained on
  1–2 snapshots are unstable.
- **Reweight by estimated `d(age)`.** Needs test 4 first, and rests on deletion
  being a smooth function of age — which the "some paid-off loans keep full
  history, some do not" report argues against. Partial, inconsistent deletion is
  harder to correct than systematic deletion, because absence is unobservable
  and cannot be distinguished from a loan that was never there.
- **Refit calibration on the newest mature snapshot.** Cheapest, safest, and
  strictly an improvement — but it corrects the output layer only.
- **Fix it upstream.** Not available now. Worth standing as a request: a
  soft-delete flag or a tombstone row would make the whole problem measurable and
  most of it correctable.
