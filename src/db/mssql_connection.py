# pyrefly: ignore [missing-import]
import pyodbc
import pandas as pd
import logging
import sys
from pathlib import Path
from typing import Optional

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
    
    def load_prediction_data(self, table: str = config.TRAIN_TABLE, snapshot_date: int = None) -> pd.DataFrame:
        logger.info(f"Loading prediction data from {table} for snapshot {snapshot_date}")
        query = f"SELECT * FROM {table}"
        params = None

        if snapshot_date:
            query += " WHERE SNAPSHOT_DATE = ?"
            params = (int(snapshot_date),)

        return self.read_sql(query, params)

    # The upstream ETL's own bookkeeping. `job_name` identifies the monthly
    # load; rows written before 2026-08 hold a Jalali "YYYY-MM" in run_month
    # and are inert here — the two formats cannot collide with a Gregorian
    # YYYYMMDD, so they simply never match a snapshot we ask about.
    ETL_JOB_CONTROL_TABLE = "etl_job_control"
    ETL_JOB_NAME          = "ORACLE_X_MONTHLY_LOAD"

    def get_etl_runs(self, limit: int = 24) -> pd.DataFrame:
        """
        Recent upstream load attempts, newest first, as
        (snapshot_date, status, last_step, error_message).

        Only status = 'SUCCESS' means the snapshot is complete. Publishing is
        one transaction upstream, so a half-published snapshot is not
        observable — this is about ABSENCE: distinguishing "that month was
        never run / failed" from "that month has no data".

        Raises if the ledger is not readable (absent table, no grant); callers
        treat that as "completeness unverified", not as a failure.
        """
        query = (
            f"SELECT TOP {int(limit)} run_month AS snapshot_date, status, "
            f"last_step, error_message "
            f"FROM {self.ETL_JOB_CONTROL_TABLE} WHERE job_name = ? "
            f"ORDER BY run_month DESC"
        )
        return self.read_sql(query, (self.ETL_JOB_NAME,))

    def get_available_snapshots(self, table: str = None) -> list:
        table = table or config.TRAIN_TABLE
        query = f"SELECT DISTINCT {config.SNAPSHOT_COL} FROM {table} ORDER BY {config.SNAPSHOT_COL}"
        df = self.read_sql(query)
        return df[config.SNAPSHOT_COL].tolist()

    def get_label_horizons(self, table: str = None) -> dict:
        """
        snapshot -> LABEL_HORIZON_DATE for every snapshot in the table.

        A cheap DISTINCT (the horizon is constant within a snapshot). For
        display and the gap logic only — maturity is get_label_ledger().
        """
        table = table or config.TRAIN_TABLE
        query = (f"SELECT DISTINCT {config.SNAPSHOT_COL}, {config.HORIZON_COL} "
                 f"FROM {table} ORDER BY {config.SNAPSHOT_COL}")
        df = self.read_sql(query)
        return {int(s): int(h)
                for s, h in df.dropna().itertuples(index=False)}

    def get_label_ledger(self) -> dict:
        """
        snapshot -> {"status", "loans_unlabelled", "loans_missing_months"} for
        every snapshot whose label is final. The ETL writes one row per
        snapshot at finalisation (STATUS 'FINAL') or at go-live for history
        (STATUS 'LEGACY', counts NULL), and never rewrites it — so this table,
        not LABEL_HORIZON_DATE, is what makes a snapshot mature. Raises if the
        table is unreadable; there is no fallback by design.
        """
        df = self.read_sql(
            "SELECT SNAPSHOT_DATE, STATUS, LOANS_UNLABELLED, LOANS_MISSING_MONTHS "
            f"FROM {config.LABEL_LEDGER_TABLE}"
        )

        def _count(v):
            return None if pd.isna(v) else int(v)

        return {
            int(r.SNAPSHOT_DATE): {
                "status": str(r.STATUS).strip().upper(),
                "loans_unlabelled": _count(r.LOANS_UNLABELLED),
                "loans_missing_months": _count(r.LOANS_MISSING_MONTHS),
            }
            for r in df.itertuples(index=False)
        }

    def get_snapshot_build_time(self, snapshot_date: int) -> Optional[str]:
        """
        end_time of the SUCCESSful ETL run that published `snapshot_date`, as a
        string; None if there is no such row. Snapshots are built once and
        never rebuilt (a rebuild of a finalised one needs its ledger row
        deleted by hand), so an unchanged end_time means unchanged rows.
        run_month is varchar (it also holds legacy Jalali 'YYYY-MM' values), so
        the snapshot is passed as a string.
        """
        df = self.read_sql(
            f"SELECT TOP 1 end_time FROM {self.ETL_JOB_CONTROL_TABLE} "
            "WHERE job_name = ? AND run_month = ? AND status = 'SUCCESS' "
            "ORDER BY end_time DESC",
            (self.ETL_JOB_NAME, str(int(snapshot_date))),
        )
        if df.empty or pd.isna(df.iloc[0, 0]):
            return None
        return str(df.iloc[0, 0])

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
