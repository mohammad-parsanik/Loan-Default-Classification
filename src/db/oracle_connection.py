import cx_Oracle
import pandas as pd
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import project_config as config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OracleConnector:
    def __init__(self, host=None, port=None, service=None, user=None, password=None):
        self.host = host or config.ORACLE_HOST
        self.port = port or config.ORACLE_PORT
        self.service = service or config.ORACLE_SERVICE
        self.user = user or config.ORACLE_USER
        self.password = password or config.ORACLE_PASSWORD
        
        try:
            # Requires Oracle Instant Client (e.g., OraClient12Home1) to be installed and in PATH/LD_LIBRARY_PATH
            self.dsn = cx_Oracle.makedsn(self.host, self.port, service_name=self.service)
            self.pool = cx_Oracle.SessionPool(
                user=self.user, 
                password=self.password, 
                dsn=self.dsn,
                min=2, max=8, increment=1, 
                encoding="UTF-8"
            )
            logger.info(f"Successfully connected to Oracle DB {self.host}:{self.port}/{self.service}")
        except cx_Oracle.Error as e:
            logger.error(f"Failed to connect to Oracle: {e}")
            raise
    
    def read_sql(self, query: str, params: dict = None) -> pd.DataFrame:
        params = params or {}
        try:
            with self.pool.acquire() as conn:
                return pd.read_sql(query, conn, params=params)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def load_training_data(self, table: str = config.TRAIN_TABLE, snapshot_dates: list = None) -> pd.DataFrame:
        logger.info(f"Loading training data from {table}")
        query = f"SELECT * FROM {table}"
        params = {}
        
        if snapshot_dates:
            placeholders = ",".join(f":s{i}" for i in range(len(snapshot_dates)))
            query += f" WHERE SNAPSHOT_DATE IN ({placeholders})"
            params = {f"s{i}": int(d) for i, d in enumerate(snapshot_dates)}
            
        return self.read_sql(query, params)
    
    def load_prediction_data(self, table: str = config.PRED_TABLE, snapshot_date: int = None) -> pd.DataFrame:
        logger.info(f"Loading prediction data from {table} for snapshot {snapshot_date}")
        query = f"SELECT * FROM {table}"
        params = {}
        
        if snapshot_date:
            query += " WHERE SNAPSHOT_DATE = :snap"
            params = {"snap": int(snapshot_date)}
            
        return self.read_sql(query, params)
    
    def close(self):
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            logger.info("Oracle connection pool closed")

if __name__ == "__main__":
    # Test connection
    try:
        conn = OracleConnector()
        print("Oracle connector initialized successfully.")
        conn.close()
    except Exception as e:
        print(f"Connection test failed: {e}")
