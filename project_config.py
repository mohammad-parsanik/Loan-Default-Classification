import os
from pathlib import Path

# ── Database ─────────────────────────────────────────────
MSSQL_SERVER = os.getenv("MSSQL_SERVER", "localhost")
MSSQL_DATABASE = os.getenv("MSSQL_DATABASE", "EDP")
MSSQL_USER = os.getenv("MSSQL_USER", "sa")
MSSQL_PASSWORD = os.getenv("MSSQL_PASSWORD", "password")

# ── Tables & Columns ────────────────────────────────────
# Single live table: contains matured snapshots (for training) plus the
# newest, not-yet-matured snapshot(s) (for prediction). No separate pred
# table — predict reads TRAIN_TABLE too, see PRED_SNAPSHOT_DATES below.
TRAIN_TABLE = "EDP_Feature_Train"

ID_COL       = "LOAN_ID"
CONTRACT_COL = "CONTRACT_NUMBER"
CUSTOMER_COL = "NATIONAL_CODE"
SNAPSHOT_COL = "SNAPSHOT_DATE"
TARGET_COL   = "WORST_FUTURE_CAT"

# RECORD_STATUS_CODE removed — column does not exist in the current dataset
META_COLS = [ID_COL, CONTRACT_COL, CUSTOMER_COL,
             SNAPSHOT_COL, TARGET_COL, "WORST_FUTURE_DPD"]

NUM_CLASSES = 4  # {0: No Delay, 1: Current, 2: Past Due+, 3: Severe Past Due}

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
# No categorical features — Collateral_type removed from dataset
BINARY_FEATURES = [
    "IS_IN_WARNING_ZONE", "IS_DETERIORATING", "IS_IMPROVING",
    "IS_ACCELERATING", "HAS_EVER_BEEN_NPL", "HAS_EVER_BEEN_PRENPL",
    "HAS_RECOVERED_BEFORE",
]
# FEATURE_COLS built dynamically: all columns minus META_COLS

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
VAL_SPLIT_MODE        = "customer"
CUSTOMER_VAL_FRACTION = 0.20   # share of customers held out for validation

# Train the aggregated-XGBoost baseline with cost-informed sample weights
# (inverse class frequency x cost-matrix row sum) instead of frequency only.
BASELINE_COST_WEIGHTS = True

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

# Per-current-cat calibration: P(severe) is a rare event for cat-0 but a
# common one for cat-2 — pooled isotonic miscalibrates the strata against
# each other, and the ranked queue mixes strata. Strata with fewer val
# samples than this floor fall back to the pooled calibrator.
CALIBRATION_MIN_STRATUM_N = 5000


# ── Cache ────────────────────────────────────────────────
# Bump this string when raw data changes (re-ETL, schema updates, etc.)
# to force cache invalidation without touching any code logic.
# v1.1: truncation sort key WORST_FUTURE_DPD -> DPD_DAYS; cache adds current_cats.
# v1.2: NUM_CLASSES 3 -> 4 (raw cats 3-4 now collapse into a new class 3
# instead of into class 2) -- labels baked into the cache change, must rebuild.
DATA_VERSION = "v1.2"

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
