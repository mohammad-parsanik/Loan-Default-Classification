import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import project_config as config
from src.db.mssql_connection import MSSQLConnector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def explore_data(sample_size=1000):
    """Utility function to check data distributions and basic stats."""
    logger.info("Initializing data exploration...")
    
    try:
        # Load a sample from database
        conn = MSSQLConnector()
        # Instead of loading everything, we can get counts and sample data
        # Check snapshot dates
        snapshot_query = f"SELECT {config.SNAPSHOT_COL}, COUNT(*) as CNT FROM {config.TRAIN_TABLE} GROUP BY {config.SNAPSHOT_COL} ORDER BY {config.SNAPSHOT_COL}"
        snapshot_df = conn.read_sql(snapshot_query)
        logger.info(f"Snapshots available:\n{snapshot_df}")
        
        if snapshot_df.empty:
            logger.warning("No data found in the training table.")
            return
            
        snapshots = snapshot_df[config.SNAPSHOT_COL].tolist()
        
        # Load a sample for feature analysis (just the latest snapshot)
        latest_snap = snapshots[-1]
        logger.info(f"Loading data for the latest snapshot ({latest_snap}) for feature profiling...")
        
        df = conn.load_training_data(snapshot_dates=[latest_snap])
        conn.close()
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.info("Generating dummy data for exploration testing since DB is unavailable.")
        # Fallback for testing when DB is not available
        snapshots = [14040131, 14040231, 14040331, 14040431, 14040531]
        
        # Generate dummy dataset based on the feature schema
        n_samples = 10000
        n_customers = 2000
        
        data = {
            config.ID_COL: np.arange(n_samples),
            "CONTRACT_NUMBER": [f"C{i}" for i in range(n_samples)],
            config.CUSTOMER_COL: np.random.randint(100000, 100000 + n_customers, n_samples),
            config.SNAPSHOT_COL: [snapshots[-1]] * n_samples,
            config.TARGET_COL: np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.8, 0.1, 0.05, 0.03, 0.02]),
            "WORST_FUTURE_DPD": np.random.randint(0, 300, n_samples),
            "RECORD_STATUS_CODE": ["ACTIVE"] * n_samples
        }
        
        # Add random features
        for f in config.BINARY_FEATURES:
            data[f] = np.random.choice([0, 1], n_samples)
            
        # Add some continuous features
        continuous = ["DPD_DAYS", "LOAN_CATEGORY", "DAYS_TO_NEXT_THRESHOLD", "PAYED_OVERDUE_INST_CNT", 
                     "UNPAYED_INST_CNT", "PAYED_OVERDUE_AMNT", "OVERDUE_RATIO", "ONTIME_RATIO",
                     "CNT_INSTALLMENT_WARNING_ZONE", "MATURED_INST_CNT", "UPCOMING_INST_CNT", 
                     "UPCOMING_AMNT", "DPD_DAYS_T1", "DPD_DAYS_T2", "DPD_DAYS_T3", "DPD_DAYS_T4", 
                     "DPD_DAYS_T5", "CATEGORY_T1", "CATEGORY_T2", "CATEGORY_T3"]
                     
        for f in continuous:
            if "RATIO" in f:
                data[f] = np.random.rand(n_samples)
            elif "AMNT" in f:
                data[f] = np.random.randint(0, 1000000, n_samples)
            elif "CATEGORY" in f:
                data[f] = np.random.randint(0, 6, n_samples)
            else:
                data[f] = np.random.randint(0, 100, n_samples)
                
        df = pd.DataFrame(data)

    logger.info(f"Loaded {len(df)} rows.")

    # 1. Compute MAX_LOANS_PER_CUSTOMER
    loans_per_cust = df.groupby(config.CUSTOMER_COL).size()
    max_loans_99 = int(np.percentile(loans_per_cust, 99))
    max_loans_max = int(loans_per_cust.max())
    
    logger.info(f"Loans per customer - Max: {max_loans_max}, 99th Percentile: {max_loans_99}")
    
    # 2. Class Distribution
    raw_class_dist = df[config.TARGET_COL].value_counts().to_dict()
    
    # Capped target
    capped_target = df[config.TARGET_COL].clip(upper=2)
    capped_class_dist = capped_target.value_counts().to_dict()
    
    logger.info(f"Raw class distribution: {raw_class_dist}")
    logger.info(f"Capped class distribution: {capped_class_dist}")
    
    # 3. Missingness
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_features = missing_pct[missing_pct > 0].to_dict()
    logger.info(f"Features with missing values: {missing_features}")
    
    # Identify feature columns
    feature_cols = [c for c in df.columns if c not in config.META_COLS]
    logger.info(f"Identified {len(feature_cols)} feature columns")
    
    # Compile report
    report = {
        "snapshots": snapshots,
        "max_loans_per_customer_99th": max_loans_99,
        "max_loans_per_customer_max": max_loans_max,
        "class_distribution_raw": {str(k): int(v) for k, v in raw_class_dist.items()},
        "class_distribution_capped": {str(k): int(v) for k, v in capped_class_dist.items()},
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "missing_features": missing_features
    }
    
    report_path = config.ARTIFACT_DIR / "data_exploration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Exploration report saved to {report_path}")
    
    # Plots
    try:
        plt.figure(figsize=(10, 5))
        sns.histplot(loans_per_cust, bins=range(1, max_loans_max + 2), discrete=True)
        plt.axvline(max_loans_99, color='r', linestyle='--', label=f'99th percentile ({max_loans_99})')
        plt.title('Distribution of Loans per Customer')
        plt.xlabel('Number of Loans')
        plt.ylabel('Number of Customers')
        plt.legend()
        plt.savefig(config.ARTIFACT_DIR / "loans_per_customer.png")
        plt.close()
        
        plt.figure(figsize=(10, 5))
        capped_target.value_counts().sort_index().plot(kind='bar')
        plt.title('Capped Target Class Distribution')
        plt.xlabel('Class (0=No Delay, 1=Current, 2=Past Due+)')
        plt.ylabel('Count')
        plt.savefig(config.ARTIFACT_DIR / "class_distribution.png")
        plt.close()
        logger.info("Plots saved to artifacts directory.")
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")

if __name__ == "__main__":
    explore_data()
