import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
# Re-exported deliberately: the rest of the project reads these off
# project_config, as it always has. Only their SOURCE moved.
from src.data.column_contract import (            # noqa: E402, F401
    BINARY_FEATURES,
    CONTRACT_VERSION,
    FEATURE_ORDER,
    META_COLS,
    NO_CLIP,
    NO_SCALE,
    TABLE as CONTRACT_TABLE,
)

# ── Database ─────────────────────────────────────────────
MSSQL_SERVER = os.getenv("MSSQL_SERVER", "localhost")
MSSQL_DATABASE = os.getenv("MSSQL_DATABASE", "EDP")
MSSQL_USER = os.getenv("MSSQL_USER", "sa")
MSSQL_PASSWORD = os.getenv("MSSQL_PASSWORD", "password")

# ── Tables & Columns ────────────────────────────────────
# Single live table: contains matured snapshots (for training) plus the
# newest, not-yet-matured snapshot(s) (for prediction). No separate pred
# table — predict reads TRAIN_TABLE too, see PRED_SNAPSHOT_DATES below.
# The 71-column feed (contract/columns.json). The 70-column table that
# "EDP_Feature_Train" / "D_ANALYTICS.DPD_SAMPLE1" referred to is deprecated
# and left in place; nothing here reads it any more.
TRAIN_TABLE = CONTRACT_TABLE

ID_COL       = "LOAN_ID"
CONTRACT_COL = "CONTRACT_NUMBER"
CUSTOMER_COL = "NATIONAL_CODE"
SNAPSHOT_COL = "SNAPSHOT_DATE"
TARGET_COL   = "WORST_FUTURE_CAT"
# T+6 as a Gregorian YYYYMMDD int: the date this row's label becomes
# trustworthy. Carried per row by the feed, so maturity is a comparison
# rather than a re-derivation of the horizon from the wall clock in a
# different calendar. See src/data/temporal_split.filter_mature_snapshots.
HORIZON_COL  = "LABEL_HORIZON_DATE"

# META_COLS / BINARY_FEATURES / FEATURE_ORDER are NOT maintained here — they
# are derived from contract/columns.json (imported above), so they cannot
# drift from the feed. The five *_COL names above stay literal: each names a
# particular column for a particular job, which is a different thing from
# "the set of non-feature columns".
#
# Notably this is how LABEL_HORIZON_DATE (column 71) becomes a meta column by
# construction. Without it, FEATURE_COLS = "every column minus META_COLS"
# would silently adopt it as feature #65 — a monotonically increasing date
# that proxies snapshot recency and correlates with label maturity. Nothing
# would raise; the model would just get quietly better on the training set.

NUM_CLASSES = 4  # {0: No Delay, 1: Current, 2: Past Due+, 3: Severe Past Due}

# ── Prediction grain ─────────────────────────────────────────────────────
# What one scored row IS.
#   "loan"      — one row per (LOAN_ID, SNAPSHOT_DATE); the ETL's native
#                 grain. WORST_FUTURE_CAT is already a per-loan label
#                 (MAX(LABEL_DPD) GROUP BY LOAN_ID upstream), so this is
#                 the label as the source computes it. Business asked for
#                 per-loan risk; also lets a healthy loan stay in the queue
#                 while a sibling loan of the same customer is carved out
#                 as ALREADY_SEVERE.
#   "portfolio" — one row per (NATIONAL_CODE, SNAPSHOT_DATE); loans are
#                 collapsed with label = max(loan labels) and features
#                 aggregated to min/max/mean/std + count. The pre-July-2026
#                 behaviour that results_1..results_6 were measured under.
# Kept switchable so the loan-grain run can be compared against the
# existing benchmark; delete the portfolio path once loan grain is accepted.
PREDICTION_GRAIN = "loan"

# ── Cost matrix (single source of truth) ─────────────────
# COST_MATRIX[true][predicted]. Used by:
#   - losses.CostSensitiveFocalLoss (training-time nudge, DeepSets)
#   - evaluation/decision.py (expected-cost decision rule at prediction time)
#   - metrics.avg_cost
# Business anchor: missing a Cat-2 costs 4x a false alarm.
# PLACEHOLDER for the new class 3 row/col: derived (not business-tuned) by
# extrapolating the same per-step-cost formula that produces the original
# 3x3 block exactly (false-alarm cost = 0.5/step regardless of true class;
# miss cost from true class t = (1.0 + 0.5*t)/step). Retune once real costs
# for "Severe Past Due" misses/false-alarms are known.
COST_MATRIX = [
    [0.0, 0.5, 1.0, 1.5],  # True 0
    [1.5, 0.0, 0.5, 1.0],  # True 1
    [4.0, 2.0, 0.0, 0.5],  # True 2
    [7.5, 5.0, 2.5, 0.0],  # True 3
]

# ── Feature typing ───────────────────────────────────────
# No categorical features. BINARY_FEATURES (0/1 flags), NO_CLIP and NO_SCALE
# all come from the contract:
#   - binary flags are exempt from clipping and scaling, as before;
#   - NO_CLIP/NO_SCALE additionally exempt the four sentinel-bearing
#     DAYS_SINCE_LAST_* columns. Their 99999 is a CODE ("never reached this
#     band"), at the opposite end of the axis from 0 ("in the band right
#     now"). Percentile-clipping rewrites "never delinquent" as "cleared a
#     long time ago" — and it does so invisibly in one of the four columns
#     while being a no-op in another, so spot-checking one proves nothing.
#     XGBoost splits on raw values and needs no scaling here.
#     Do NOT impute the sentinel to the median.

# ── Label window ────────────────────────────────────────
# WORST_FUTURE_CAT uses a 6-month forward horizon.
# For leakage-free splits, the last train snapshot must be
# >= LABEL_HORIZON_MONTHS before the first test snapshot.
LABEL_HORIZON_MONTHS = 6

# ── Walk-Forward Validation & Optimization ───────────────
# Set WALK_FORWARD_ENABLED = True to run rolling-window cross-validation
# across all valid (train, val, test) fold combinations.
# Set to False to use a single static temporal split.
WALK_FORWARD_ENABLED  = False

# Minimum number of training snapshots required for a fold to be valid.
# Folds with fewer training snapshots are skipped.
MIN_TRAIN_SNAPSHOTS   = 1

# Keep only part of the eligible training window. None = every snapshot that
# clears the LABEL_HORIZON_MONTHS gap (the default, and what every run so far
# used). A positive N keeps the NEWEST N; a negative N keeps the OLDEST N.
#
# This exists for one experiment: the upstream installment table is hard-deleted
# for closed and post-NPL loans, so an old snapshot was rebuilt after more of
# its future-severe rows had vanished — the measured severe rate runs 8.37% ->
# 9.96% -> 12.95% across the train / gap-dropped / test windows
# (AGENT_HANDOFF.md §24). Training on the newest N vs the oldest N and scoring
# both on the same test fold says whether that depletion costs RANKING quality,
# or only calibration. Uniform undersampling of positives shifts the intercept
# and leaves the ordering alone; feature-dependent deletion does not.
#
# Leave at None for production runs — it throws away training data by design.
TRAIN_WINDOW_SNAPSHOTS = None

# Set OPTIMIZE_ON_VALIDATION = True to use a validation set for early
# stopping and Optuna hyperparameter tuning. Set to False to train on
# more data and test directly without any validation set.
OPTIMIZE_ON_VALIDATION = True

# How the validation set is built when OPTIMIZE_ON_VALIDATION = True:
#   "customer" — in-time, customer-disjoint holdout carved from the train
#                snapshots. Val labels never overlap the test label window,
#                so tuning/early-stopping cause no test leakage.
#   "temporal" — legacy behaviour: val = second-newest usable snapshot.
#                LEAKY when val and test label windows overlap (Run 2).
# Keep this at "customer". It is a CORRECTNESS requirement, not a
# preference: the ten cross-loan (Group F) features are customer-level and
# are stamped identically onto every row of every loan a borrower holds. A
# random split puts two rows of the same customer on both sides of the fold
# boundary carrying ten identical values, and leaks them across it.
VAL_SPLIT_MODE        = "customer"
CUSTOMER_VAL_FRACTION = 0.20   # share of customers held out for validation

# Train the multiclass arm with cost-informed sample weights (inverse class
# frequency x cost-matrix row sum). Default OFF since the July-8 reframe:
# the headline is cost-free ranking, the cost values are guesses, and less
# training-time distortion leaves the calibrator less to undo. (Run 5's
# binary comparator — no cost machinery at all — beat the cost-loss-trained
# DeepSets pipeline on every ranking slice.)
BASELINE_COST_WEIGHTS = False

# ── Model arms (July 10 architecture switch) ─────────────
# Run-5 verdict: XGBoost on the 257 aggregated features beats DeepSets+XGB
# on every ranking slice. Evaluation runs train and compare these arms:
#   "multiclass" — one multi:softprob model; full class distribution
#   "ordinal"    — cumulative binaries P(y>k); full distribution via
#                  differences; exploits the ordered target
#   "per_cat"    — one model per current_cat stratum over its REACHABLE
#                  classes {c..3}; natively monotone full distribution
#   "binary"     — direct P(severe); ranking-ceiling diagnostic ONLY (no
#                  per-class probabilities, so never deployed)
MODEL_ARMS = ["multiclass", "binary", "ordinal", "per_cat"]

# Deployed arm: "auto" = best pooled ranking PR-AUC among full-distribution
# arms (business needs the per-class probabilities; sorting stays on P3).
# Locked to "multiclass" after the Run-6 shootout (all four arms within
# ~1pt AP; multiclass best pooled AND best on the cat_0 early-warning
# slice). Re-run with "auto" to re-open the comparison in a few months.
DEPLOY_ARM = "multiclass"

# Optuna trials for tuning the DEPLOYED arm's XGBoost hyperparameters on
# the customer-disjoint val set, maximising PR-AUC of P(severe) — the
# actual deliverable objective (Run 5 wasted 5.7h tuning macro F1, the
# wrong number). 0 = disabled (arms use XGB_DEFAULTS). Only the arm(s)
# that could deploy are tuned; diagnostic arms always use defaults.
ARM_OPTUNA_TRIALS = 0

# Legacy neural arm (DeepSets encoder + XGB meta-learner). Lost the Run-5
# shootout on every slice; kept behind this flag for reproducibility.
# If ever re-enabled: fix the LR schedule first (CosineAnnealingWarmRestarts
# restarted at epoch 10 and ate the fine-tuning phase — use plain cosine).
DEEPSETS_ENABLED = False

# At predict time, refresh the probability calibrator on the most recent
# snapshot whose labels have matured (requires train-table/cache access).
RECALIBRATE_ON_PREDICT = True

# Snapshot(s) to score at predict time (YYYYMMDD int or list[int]).
# None (default) = auto-select every currently-immature snapshot in
# TRAIN_TABLE (the standard early-warning use case). A --snapshot_date CLI
# value always overrides this. Any requested date not found in the table is
# dropped (with a warning) and falls back the same way as an unset value.
PRED_SNAPSHOT_DATES = None

# When multiple snapshots are scored in one predict call, keep only each
# customer's NEWEST row in the ranked queue. The enrichment API returns
# present-time information only (cannot query the past), so acting on a
# stale score wastes budget; the older rows are still written to the CSV
# flagged as superseded.
PRED_DEDUP_LATEST = True

# ── Deliverable: ranked API queue ────────────────────────
# The business objective is to find customers NOT yet in the severe class
# (NUM_CLASSES-1) with the highest probability of entering it within the
# label horizon. Customers already severe are rule-flagged, never ranked.
CARVE_CURRENT_CAT_GE = NUM_CLASSES - 1   # current_cat >= this ⇒ rule, not model queue

# Enrichment API budget is a RATE, so "top-K" is really "hours of calling".
# Ranking metrics are reported at K = API_RATE_PER_HOUR * window_hours.
API_RATE_PER_HOUR   = 240
RANKING_REF_WINDOWS = {          # label -> hours of continuous calling
    "1_day":   24,
    "1_week":  168,
    "1_month": 720,
}

# The ranking block is also cut by decile of this feature, to answer whether
# the queue ranks equally well at every loan size. That is the precondition for
# an amount-banded model: a band is worth building when the feature->target
# RELATIONSHIP differs, not when the base rate does (one tree split handles a
# base-rate shift, at a threshold the data picks). Flat lift@K across deciles
# means banding buys nothing. It also watches the §24 deletion bias — if
# post-NPL deletion upstream varies with loan size, the model learns a
# size-dependent distortion as if it were signal. Set to None to skip.
EXPOSURE_FEATURE = "REMAINING_AMNT"

# Per-current-cat calibration: P(severe) is a rare event for cat-0 but a
# common one for cat-2 — pooled isotonic miscalibrates the strata against
# each other, and the ranked queue mixes strata. Strata with fewer val
# samples than this floor fall back to the pooled calibrator.
CALIBRATION_MIN_STRATUM_N = 5000

# Enrichment data freshness: an API result stays usable for ~this many
# days (business: "the oldest data we could consider would be a month
# old"). Customers whose last call is fresher than the TTL are flagged
# RECENTLY_CALLED and skipped by the queue — re-calling them buys nothing.
# Feed predict the call ledger (CSV with NATIONAL_CODE, CALLED_AT columns,
# appended by the API-calling process) via --called_log or API_CALL_LOG.
API_DATA_TTL_DAYS = 30
API_CALL_LOG      = None   # optional default path to the ledger CSV

# Optional certainty band (PENDING business decision): when set (e.g.
# 0.90), queue rows with RISK_SCORE >= threshold are flagged
# PREDICTED_SEVERE — treated as certain enough to act on directly, saving
# API calls for cases where enrichment could still change the decision.
# None = disabled.
CERTAINTY_ACT_THRESHOLD = None


# ── Cache ────────────────────────────────────────────────
# Bump this string when raw data changes (re-ETL, schema updates, etc.)
# to force cache invalidation without touching any code logic.
# v1.1: truncation sort key WORST_FUTURE_DPD -> DPD_DAYS; cache adds current_cats.
# v1.2: NUM_CLASSES 3 -> 4 (raw cats 3-4 now collapse into a new class 3
# instead of into class 2) -- labels baked into the cache change, must rebuild.
# v1.3: instances carry loan_id + portfolio_n_loans, and PREDICTION_GRAIN
# changes what an instance IS -- new NPZ arrays, must rebuild. (The grain
# itself is also part of the cache key, so the two grains cache separately.)
# v1.4: new ETL feed (contract/columns.json, table above). LABEL_HORIZON_DATE
# added to META_COLS; DAYS_SINCE_LAST_* now carry a 99999 sentinel and are
# exempt from clip/scale; HAS_EVER_BEEN_NPL/PRENPL thresholds corrected; the
# cross-loan block, CONSECUTIVE_MONTHS_WITH_DPD, MONTHS_IN_CURRENT_CATEGORY,
# COUNT_CATEGORY_CHANGES and the COUNT_* thresholds all changed MEANING, not
# just value. Feature identity is also name-based from here on and the cache
# key now covers the column list. Every earlier cache is stale — must rebuild.
# See etl_integration/CONSUMER_CONTRACT.md section 8 (local-only).
# v1.5: POPULATION change, not a schema change (Sept 2, 2026). The ETL's q1
# scope filter widened from contract amount <= 700,000,000 to <= 7,000,000,000,
# so the table now covers loans an order of magnitude larger. Same 71 columns,
# same meanings, same contract version -- nothing in this repo encodes the cap,
# and no schema or contract check can see it, so THIS STRING is the only thing
# that stops a stale cache from silently serving the old population. Tables and
# all snapshots are being rebuilt under the new filter, so the change is
# backfilled: train and score populations stay consistent. Expect the severe
# base rate, the monetary features' p99 (PAYED_OVERDUE_AMNT / UPCOMING_AMNT /
# REMAINING_AMNT) and therefore PR-AUC to move. Not comparable to results_1-6.
DATA_VERSION = "v1.5"

# ── Model — DeepSets (CPU-optimized) ────────────────────
MAX_LOANS_PER_CUSTOMER = None  # Computed from data (99th percentile)
DEEPSETS_HIDDEN_DIM  = 128    # phi hidden layer width
DEEPSETS_EMBED_DIM   = 64     # rho output embedding dimension (fed to XGBoost)
DROPOUT              = 0.15
BATCH_SIZE           = 512
LEARNING_RATE        = 5e-4
WEIGHT_DECAY         = 1e-4
EPOCHS               = 80
FIXED_EPOCHS         = 15     # Used when OPTIMIZE_ON_VALIDATION = False
PATIENCE             = 10
NUM_WORKERS          = 2      # Small payload (MAX_LOANS≤7); 0 is also fine
RANDOM_SEED          = 42

# ── CPU threading ────────────────────────────────────────
# Reserve 12 cores for PyTorch compute; 8 for DataLoader + XGBoost
TORCH_NUM_THREADS        = 12
TORCH_NUM_INTEROP        = 4

# ── Paths ────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATA_DIR     = BASE_DIR / "data"

# Create directories if they don't exist
ARTIFACT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
