import pandas as pd
import numpy as np
import logging
import sys
import joblib
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import project_config as config
from src.db.mssql_connection import MSSQLConnector

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, mssql_connector=None):
        self.conn = mssql_connector
        
    def _compute_capped_label(self, group: pd.DataFrame) -> int:
        """
        Input features retain all 5 categories (0-4).
        Target is capped: Cat 2,3,4 -> 2.
        Customer label = min(max across loans, 2).
        """
        max_cat = group[config.TARGET_COL].max()
        # Cap at NUM_CLASSES - 1 (which is 2)
        return int(min(max_cat, config.NUM_CLASSES - 1))
        
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Extract all columns that are features (not meta columns)."""
        return [c for c in df.columns if c not in config.META_COLS]

    def process_raw_data(self, df: pd.DataFrame, max_loans: int = None) -> list:
        """
        Group flat table by (NATIONAL_CODE, SNAPSHOT_DATE) to create
        customer portfolio instances.
        """
        logger.info(f"Processing {len(df)} rows into customer portfolios...")
        
        # Determine features
        features = self.get_feature_columns(df)
        
        # Sort so we keep the worst DPD loans if we need to truncate
        df = df.sort_values(by=['WORST_FUTURE_DPD'], ascending=False)
        
        grouped = df.groupby([config.CUSTOMER_COL, config.SNAPSHOT_COL])
        
        instances = []
        for (national_code, snapshot_date), group in tqdm(grouped, desc="Customer Portfolios", total=len(grouped)):
            # Calculate target
            label = self._compute_capped_label(group)
            
            # Truncate to max_loans keeping the worst (due to sorting)
            if max_loans and len(group) > max_loans:
                group = group.head(max_loans)
                
            # Extract feature matrix as float32
            feature_matrix = group[features].values.astype(np.float32)
            
            instances.append({
                'national_code': national_code,
                'snapshot_date': snapshot_date,
                'n_loans': len(group),
                'features': feature_matrix,
                'label': label
            })
            
        logger.info(f"Created {len(instances)} customer portfolio instances.")
        return instances, features

    def load_train_portfolios(self, snapshot_dates: list = None, max_loans: int = None, use_cache: bool = True) -> tuple:
        """
        End-to-end load of training data from Oracle to portfolio instances.
        Returns (instances, feature_names).
        """
        cache_path = config.DATA_DIR / "train_portfolios_cache.joblib"
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached training portfolios from {cache_path}")
            return joblib.load(cache_path)
            
        if self.conn is None:
            self.conn = MSSQLConnector()
            close_conn = True
        else:
            close_conn = False
            
        try:
            df = self.conn.load_training_data(snapshot_dates=snapshot_dates)
            instances, features = self.process_raw_data(df, max_loans)
            if use_cache:
                logger.info(f"Saving training portfolios to cache {cache_path}")
                joblib.dump((instances, features), cache_path)
            return instances, features
        finally:
            if close_conn:
                self.conn.close()
                
    def load_pred_portfolios(self, snapshot_date: int, max_loans: int = None, use_cache: bool = True) -> tuple:
        """
        End-to-end load of prediction data from Oracle to portfolio instances.
        """
        cache_path = config.DATA_DIR / f"pred_portfolios_cache_{snapshot_date}.joblib"
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached prediction portfolios from {cache_path}")
            return joblib.load(cache_path)
            
        if self.conn is None:
            self.conn = MSSQLConnector()
            close_conn = True
        else:
            close_conn = False
            
        try:
            df = self.conn.load_prediction_data(snapshot_date=snapshot_date)
            instances, features = self.process_raw_data(df, max_loans)
            if use_cache:
                logger.info(f"Saving prediction portfolios to cache {cache_path}")
                joblib.dump((instances, features), cache_path)
            return instances, features
        finally:
            if close_conn:
                self.conn.close()
