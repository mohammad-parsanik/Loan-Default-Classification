# pyrefly: ignore [missing-import]
import pyodbc
import pandas as pd
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import project_config as config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MSSQLConnector:
    def __init__(self, server=None, database=None, user=None, password=None):
        self.server = server or config.MSSQL_SERVER
        self.database = database or config.MSSQL_DATABASE
        self.user = user or config.MSSQL_USER
        self.password = password or config.MSSQL_PASSWORD
        
        # Change this driver string if you have a different version installed 
        # (e.g. 'ODBC Driver 18 for SQL Server' or 'SQL Server Native Client 11.0')
        self.driver = "{ODBC Driver 17 for SQL Server}"
        
        try:
            self.conn_str = (
                f"DRIVER={self.driver};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.user};"
                f"PWD={self.password};"
                "TrustServerCertificate=yes;" # Needed if using Driver 18 without SSL certs
            )
            self.conn = pyodbc.connect(self.conn_str)
            logger.info(f"Successfully connected to MSSQL DB {self.server}/{self.database}")
        except pyodbc.Error as e:
            logger.error(f"Failed to connect to MSSQL: {e}")
            raise
    
    def read_sql(self, query: str, params: tuple = None) -> pd.DataFrame:
        try:
            if params:
                return pd.read_sql(query, self.conn, params=params)
            else:
                return pd.read_sql(query, self.conn)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def load_training_data(self, table: str = config.TRAIN_TABLE, snapshot_dates: list = None) -> pd.DataFrame:
        logger.info(f"Loading training data from {table}")
        query = f"SELECT * FROM {table}"
        params = None
        
        if snapshot_dates:
            # Create parameterized query with '?' for pyodbc
            placeholders = ",".join("?" for _ in snapshot_dates)
            query += f" WHERE SNAPSHOT_DATE IN ({placeholders})"
            params = tuple(int(d) for d in snapshot_dates)
            
        return self.read_sql(query, params)
    
    def load_prediction_data(self, table: str = config.PRED_TABLE, snapshot_date: int = None) -> pd.DataFrame:
        logger.info(f"Loading prediction data from {table} for snapshot {snapshot_date}")
        query = f"SELECT * FROM {table}"
        params = None
        
        if snapshot_date:
            query += " WHERE SNAPSHOT_DATE = ?"
            params = (int(snapshot_date),)
            
        return self.read_sql(query, params)
    
    def close(self):
        if hasattr(self, 'conn') and self.conn is not None:
            self.conn.close()
            logger.info("MSSQL connection closed")

if __name__ == "__main__":
    # Test connection
    try:
        conn = MSSQLConnector()
        print("MSSQL connector initialized successfully.")
        
        # Test query to check if we can read from the DB
        df = conn.read_sql("SELECT @@VERSION AS Version")
        print("\nSQL Server Version:")
        print(df.iloc[0]['Version'])
        
        conn.close()
    except Exception as e:
        print(f"Connection test failed: {e}")
