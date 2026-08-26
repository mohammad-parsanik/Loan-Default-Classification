# Loan Default Classification — Ranked Early-Warning Queue

A bank early-warning system that ranks **loans** by their probability of
entering **severe delinquency** within the next 6 months, so a limited
external enrichment API (240 requests/hour) can be spent where a call
actually changes the decision. Already-severe loans are excluded by rule —
they're a known risk, not a ranking question.

> **Grain:** one scored row is one loan (`LOAN_ID` x `SNAPSHOT_DATE`) as of
> 2026-07-26. Through Run 6 it was one row per customer, with the label
> collapsed to the worst of their loans; `PREDICTION_GRAIN = "portfolio"`
> in `project_config.py` restores that. The enrichment API itself is still
> keyed by `NATIONAL_CODE`, so two loans of one customer cost two queue
> slots but a single call. See AGENT_HANDOFF.md §19.

> **New to this project? Read in this order:**
> 1. This file — architecture and data flow at a glance.
> 2. **[AGENT_HANDOFF.md](AGENT_HANDOFF.md)** — the full decision history: what was
>    tried, what worked, what didn't, and why. This is the authoritative record.
> 3. **[CLAUDE.md](CLAUDE.md)** — terse day-to-day reference (commands, config
>    flags, gotchas) for whoever is actively coding.
> 4. **[DEPLOYMENT.md](DEPLOYMENT.md)** — how to train the production model,
>    package it, and generate predictions from it.
> 5. **[MODEL_EVALUATION.md](MODEL_EVALUATION.md)** — how to tell if a trained
>    model is actually good (what to read, what "normal" numbers look like).
> 6. **[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)** — every setting in
>    `project_config.py` explained.

---

## The problem, precisely

Each month, an upstream ETL table (`D_ANALYTICS.EDP_LOAN_FEATURES`) gives one
row per loan: 64 numeric features (current days-past-due, trend, history,
cross-contract signals) plus a label, `WORST_FUTURE_CAT`, capped to 4 classes.
The column set, its order, and each column's handling flags are pinned in
[`contract/columns.json`](contract/columns.json) — see
[contract/README.md](contract/README.md).

| Class | Meaning |
|---|---|
| 0 | No Delay |
| 1 | Current / Minor Delay |
| 2 | Past Due+ |
| 3 | **Severe Past Due** — the event we're forecasting |

**Key label property:** the label window includes the current month, so
`label >= current_category` always — a loan can't improve past where it
already is. This isn't leakage, it's a definition, and the pipeline
exploits it directly (see "Monotonicity" below).

**The deliverable is a ranked queue**, not a classifier: customers with
`current_category < 3`, sorted by calibrated `P(entering class 3)`. The
business calls the enrichment API down that list until the hourly budget
runs out. See [DEPLOYMENT.md](DEPLOYMENT.md) for the exact output format.

---

## Architecture at a glance

```
MSSQL (D_ANALYTICS.EDP_LOAN_FEATURES)
  │  one row per loan per monthly snapshot; 71 columns, pinned by
  │  contract/columns.json
  ▼
Projection + grouping (src/data/data_loader.py)
  │  project columns BY NAME to the contract's 64 features (or, when scoring,
  │  to the trained model's own list); sort rows into one canonical order
  │  group by (customer, snapshot); truncate to MAX_LOANS≈2 loans/customer
  │  (kept by CURRENT delinquency, never by the future label — that would leak)
  ▼
Temporal split (src/data/temporal_split.py)
  │  test = newest snapshot whose LABEL_HORIZON_DATE has passed
  │  train = snapshots ≥6mo before test; val = customer-disjoint 20% of train
  │  (val is in-time, so early stopping/tuning never touch the test window)
  ▼
Preprocessing (src/data/preprocessing.py)
  │  domain-aware impute → clip outliers → RobustScaler (fit on train only)
  │  sentinel-bearing columns are exempt from clip and scale (see below)
  ▼
Feature build (src/baselines/aggregated_xgboost.py::build_features)
  │  loan grain: the ~64 raw per-loan features, used as-is
  │  portfolio grain: min/max/mean/std per feature + loan count → 257
  ▼
Model arms (src/baselines/aggregated_xgboost.py — see project_config.MODEL_ARMS)
  │  XGBoost variants trained on the SAME features; compared by ranking
  │  quality; DEPLOYED: "multiclass" (locked after the Run-6 shootout)
  ▼
Per-current-category calibration (src/evaluation/calibration.py)
  │  P(severe) means something different for a clean customer vs. one
  │  already at cat-2 — one isotonic calibrator per stratum
  ▼
Monotonicity masking (src/evaluation/decision.py::mask_monotone)
  │  zero out P(class < current_category) — logically impossible — renormalize
  ▼
Ranked queue (src/inference/predictor.py)
     sort by P(severe); flag ALREADY_SEVERE / SUPERSEDED / RECENTLY_CALLED /
     PREDICTED_SEVERE; rank the rest → predictions_<snapshot>.csv
```

### The model, precisely

**Features:** at loan grain, the **64 raw per-loan features** unchanged —
one row per loan, nothing to aggregate. (Aggregating here would give
`min == max == mean == the feature`, `std == 0`, `count == 1`: 4x the
columns for zero extra information.) At portfolio grain, those 64 become 4
summary statistics each across a customer's loans plus a loan count —
**257 features per customer** (`4 × 64 + 1`). This is what every model
candidate below actually trains on.

**Model candidates ("arms", `src/baselines/aggregated_xgboost.py`)** — all
XGBoost, all on the same features, all with identical hyperparameters
(`n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
colsample_bytree=0.8`) so comparisons isolate the objective, not tuning
luck:

| Arm | What it fits | Deployable? |
|---|---|---|
| **`multiclass`** (deployed) | One `multi:softprob` model over all 4 classes directly. | Yes — full class distribution. |
| `ordinal` | `NUM_CLASSES-1` cumulative binaries `P(label > k)`; the full distribution is recovered by differencing consecutive cumulative probabilities. | Yes, but didn't win. |
| `per_cat` | One model per `current_cat` stratum, trained only over that stratum's *reachable* classes (a cat-1 customer can only end at 1/2/3). Natively monotone by construction. | Yes, but didn't win — see `AGENT_HANDOFF.md` §15 for why the per-stratum specialist underperforms the pooled model on the hardest slice. |
| `binary` | Direct `P(label == severe)`. | **No** — no per-class distribution, so it can't produce the required `P_NO_DELAY`/`P_CURRENT`/`P_PAST_DUE` columns. Kept only as a ranking-ceiling diagnostic. |

An earlier, much more complex candidate — **DeepSets** (a permutation-invariant
neural set encoder feeding an XGBoost meta-learner on 64-dim embeddings) —
lost a controlled shootout to plain `multiclass` on every ranking slice.
Its code still exists (`src/model/`, `DEEPSETS_ENABLED=False`) but isn't
maintained as a live candidate.

**Calibration (`src/evaluation/calibration.py`):** raw XGBoost probabilities
aren't honest frequencies once class weighting or an ordered-target
decomposition is involved. A separate isotonic regression curve is fit
per class, per `current_cat` stratum (falling back to a pooled curve when
a stratum has too few validation samples — `CALIBRATION_MIN_STRATUM_N`),
on a validation slice the model never trained on. This matters because
P(severe) means something very different for a currently-clean customer
(rare event) than for one already at cat-2 (common event) — pooling them
would systematically miscalibrate one relative to the other, which
directly corrupts the ranking since the queue mixes strata.

**Monotonicity masking (`src/evaluation/decision.py::mask_monotone`):**
since `label >= current_cat` always, any predicted probability mass on a
class below the customer's current category is provably impossible. That
mass is zeroed and the remaining probabilities renormalized — this
sharpens the calibrated distribution for free, using a constraint that
costs nothing to enforce.

**Sentinel columns.** Four features encode "never reached this delinquency
band" as `99999`, at the far end of the same axis on which `0` means "in that
band right now". Running them through a percentile clipper rewrites the best
state as one of the worst, and it does so *invisibly* in the columns where the
sentinel is the majority value (there `p99` is the sentinel, so clipping is a
no-op) while being destructive in the ones where it is a minority. They carry
`clip: false, scale: false` in the contract and both transformers skip them;
XGBoost splits on raw values, so nothing is lost. Never impute the sentinel to
a median.

**Ranking score:** `RISK_SCORE` = the calibrated, masked `P(class == 3)`
for that customer. That's the entire sort key — no cost matrix, no
learned ranking objective, just the class-3 probability from whichever
arm is deployed. `PREDICTED_CLASS`/`EXPECTED_COST` are a separate,
cost-matrix-driven diagnostic and can legitimately disagree with
`argmax(P_i)` — see "`RISK_SCORE` vs. `EXPECTED_COST`" in
[DEPLOYMENT.md](DEPLOYMENT.md) for the worked example.

## Major decisions (why it looks like this)

Full rationale and the evidence behind each is in
[AGENT_HANDOFF.md](AGENT_HANDOFF.md); short version:

- **XGBoost on aggregated features, not a neural set model.** An earlier
  DeepSets (portfolio-set) architecture was tried and lost a controlled
  four-arm shootout (Run 6) on every ranking slice, including the hardest
  one (currently-clean customers). Its code path still exists
  (`DEEPSETS_ENABLED=False`) but is not maintained as the primary path.
- **Ranking metrics, not classification accuracy, are the headline.**
  Aggregate macro-F1 is dominated by the mechanically easy slice
  (already-delinquent customers, whose future label is near-certain).
  The metric that matters is recall/lift of the ranked queue at the
  actual API budget (`src/evaluation/ranking.py`).
- **The business cost matrix is a secondary diagnostic, not the
  objective.** Its numbers are estimates, not measured costs; the ranking
  headline needs no cost assumption at all.
- **Calibration and monotonicity masking are load-bearing, not
  cosmetic.** They measurably improved ranking quality in evaluation and
  are required for probabilities to mean what the downstream rule system
  assumes they mean.
- **Walk-forward validation exists but is off by default** — the team
  deliberately deferred routine time-stability checks; flip
  `WALK_FORWARD_ENABLED` when you want one.
- **Nothing depends on the order rows or columns arrive in.** Feature
  identity is by name against `contract/columns.json`; rows are put into one
  canonical order (`customer, snapshot, DPD desc, LOAN_ID`) before anything
  reads them; the queue and the ranking metrics break ties explicitly. This
  was not cosmetic — XGBoost's `subsample`/`colsample_bytree` draw by index,
  calibrated probabilities tie in large blocks, and every preprocessing
  statistic is keyed by column position, so all three quietly followed
  whatever order the database returned. See `tests/test_order_independence.py`.
- **Validation is customer-disjoint, as a correctness requirement.** Ten of
  the 64 features describe the borrower rather than the loan and are stamped
  identically onto every row of every loan they hold, so a random split would
  put the same ten values on both sides of the fold boundary.
  `VAL_SPLIT_MODE = "customer"` is not a preference.

---

## Environment & installation

Training and prediction run on a Windows server with MSSQL access via
`pyodbc` (**not** Oracle — `oracledb`/`cx_Oracle` references in
`OLD_*` docs are historical). This repo's `.venv` is managed with `uv`:

```bash
uv pip install -r requirements.txt
```

`pyodbc`, `torch`, `optuna`, and `umap-learn` are only needed for the
DB-backed / legacy-DeepSets / diagnostic paths — a pure scoring deployment
needs none of them (see [DEPLOYMENT.md](DEPLOYMENT.md) §4).

## Project structure

```
project_config.py       All hyperparameters, DB creds, toggle flags
contract/columns.json   The upstream feed's column contract (see contract/README.md)
run.py                  CLI: explore / train [--final] [--resume] / predict
build_scoring_package.py  Package a trained model for handoff to another system

src/
  db/                   MSSQL connector (pyodbc)
  data/
    column_contract.py    Loads/validates contract/columns.json — feature identity
    feed_checks.py         Row-level invariant checks on a freshly-loaded frame
    data_loader.py        DB/cache loading, name-based projection, canonical row
                           order, raw→portfolio grouping (leak-safe truncation)
    temporal_split.py      Static split, label maturity, customer-disjoint validation
    preprocessing.py       Impute → clip → scale pipeline (+ feature-list checks)
    dataset.py, data_explorer.py   Legacy-DeepSets dataloaders / one-shot profiling
  baselines/
    aggregated_xgboost.py  THE model: feature aggregation + all XGBoost arms
  model/                 Legacy DeepSets architecture (dead by default, see above)
  evaluation/
    decision.py            Expected-cost rule, monotonicity masking, severity score
    calibration.py          Per-class / per-stratum isotonic calibration
    ranking.py              Recall@K, lift, PR-AUC — the headline metrics
    metrics.py, fold_aggregator.py, visualization.py
  inference/
    model_loader.py         Scorer abstraction (ArmScorer / legacy DeepSetsScorer)
    predictor.py             Queue construction: score_instances(), score_dataframe(), Predictor
    scoring.py               run_scoring() — manager-code integration entry point
    scoring_params.py        ScoringParams — explicit, per-call-overridable business knobs

explore_iv_woe.py, explore_umap.py, explore_shap.py   Standalone diagnostics (see EXPLORATION.md)
tests/                  Unit + integration tests (pytest; also runnable without it, see below)
```

See [DEPLOYMENT.md](DEPLOYMENT.md) §4 for when to use `score_dataframe()`
vs. `run_scoring()`/`ScoringParams` vs. the `run.py predict` CLI.

## Commands

```bash
python run.py explore                         # one-shot data profiling
python run.py train                            # evaluation run (holds out newest mature snapshot)
python run.py train --final                    # deployment fit: all mature snapshots, no test
python run.py train --resume <run_dir>         # resume a crashed run
python run.py predict --artifact_dir <dir_or_bundle.pkl> [--snapshot_date ...] [--called_log calls.csv]

# Diagnostics (read the NPZ cache; no DB needed):
python explore_iv_woe.py
python explore_umap.py --mode raw
python explore_shap.py --bundle <model_bundle.pkl> --data <snapshot.csv>

# Package a trained model for a system that shouldn't depend on this whole repo:
python build_scoring_package.py --bundle <model_bundle.pkl> --output scoring_package/

pytest tests/
```

See [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for every config flag that
changes this behavior (`MODEL_ARMS`, `DEPLOY_ARM`, `CARVE_CURRENT_CAT_GE`,
`API_RATE_PER_HOUR`, `CERTAINTY_ACT_THRESHOLD`, ...), and
[MODEL_EVALUATION.md](MODEL_EVALUATION.md) for how to judge a training
run's output once you have one.
