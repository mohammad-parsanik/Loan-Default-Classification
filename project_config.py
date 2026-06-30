import os
from pathlib import Path

# ── Database ─────────────────────────────────────────────
MSSQL_SERVER = os.getenv("MSSQL_SERVER", "localhost")
MSSQL_DATABASE = os.getenv("MSSQL_DATABASE", "EDP")
MSSQL_USER = os.getenv("MSSQL_USER", "sa")
MSSQL_PASSWORD = os.getenv("MSSQL_PASSWORD", "password")

# ── Tables & Columns ────────────────────────────────────
TRAIN_TABLE = "EDP_Feature_Train"
PRED_TABLE  = "EDP_Feature_pred"

ID_COL       = "LOAN_ID"
CONTRACT_COL = "CONTRACT_NUMBER"
CUSTOMER_COL = "NATIONAL_CODE"
SNAPSHOT_COL = "SNAPSHOT_DATE"
TARGET_COL   = "WORST_FUTURE_CAT"

# RECORD_STATUS_CODE removed — column does not exist in the current dataset
META_COLS = [ID_COL, CONTRACT_COL, CUSTOMER_COL,
             SNAPSHOT_COL, TARGET_COL, "WORST_FUTURE_DPD"]

NUM_CLASSES = 3  # {0: No Delay, 1: Current, 2: Past Due+}

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

# ── Cache ────────────────────────────────────────────────
# Bump this string when raw data changes (re-ETL, schema updates, etc.)
# to force cache invalidation without touching any code logic.
DATA_VERSION = "v1.0"

# ── Model — DeepSets (CPU-optimized) ────────────────────
MAX_LOANS_PER_CUSTOMER = None  # Computed from data (99th percentile)
DEEPSETS_HIDDEN_DIM  = 128    # phi hidden layer width
DEEPSETS_EMBED_DIM   = 64     # rho output embedding dimension (fed to XGBoost)
DROPOUT              = 0.15
BATCH_SIZE           = 512
LEARNING_RATE        = 5e-4
WEIGHT_DECAY         = 1e-4
EPOCHS               = 80
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
