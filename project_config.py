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

META_COLS = [ID_COL, CONTRACT_COL, CUSTOMER_COL,
             SNAPSHOT_COL, TARGET_COL, "WORST_FUTURE_DPD",
             "RECORD_STATUS_CODE"]

NUM_CLASSES = 3  # {0: No Delay, 1: Current, 2: Past Due+}

# ── Feature typing ───────────────────────────────────────
# No categorical features — Collateral_type removed from dataset
BINARY_FEATURES = [
    "IS_IN_WARNING_ZONE", "IS_DETERIORATING", "IS_IMPROVING",
    "IS_ACCELERATING", "HAS_EVER_BEEN_NPL", "HAS_EVER_BEEN_PRENPL",
    "HAS_RECOVERED_BEFORE",
]
# FEATURE_COLS built dynamically: all columns minus META_COLS

# ── Model (CPU-optimized) ───────────────────────────────
MAX_LOANS_PER_CUSTOMER = None  # Computed from data
D_MODEL        = 64
N_HEADS        = 4
N_LAYERS       = 2
D_FEEDFORWARD  = 256
DROPOUT        = 0.15
BATCH_SIZE     = 512
LEARNING_RATE  = 5e-4
WEIGHT_DECAY   = 1e-4
EPOCHS         = 80
PATIENCE       = 10
NUM_WORKERS    = 8
RANDOM_SEED    = 42

# ── Paths ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATA_DIR     = BASE_DIR / "data"

# Create directories if they don't exist
ARTIFACT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
