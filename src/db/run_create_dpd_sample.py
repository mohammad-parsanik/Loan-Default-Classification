import sys
from pathlib import Path
import pyodbc

# Add the directory containing the connection script to path
sys.path.append(str(Path(__file__).resolve().parent))
from mssql_connection import MSSQLConnector

def run_sql_script(filepath):
    try:
        # Initialize connection using your existing setup
        db = MSSQLConnector()
        cursor = db.conn.cursor()
        
        with open(filepath, 'r') as f:
            sql_script = f.read()
        
        # 'GO' is a batch separator used in SSMS, but not valid in pyodbc/ADO.NET.
        # We need to split the script by 'GO' and execute each batch individually.
        # Split by GO (case-insensitive) on its own line
        import re
        statements = re.split(r'(?i)^\s*GO\s*$', sql_script, flags=re.MULTILINE)
        
        for i, statement in enumerate(statements):
            statement = statement.strip()
            if statement:
                print(f"Executing Batch {i+1}...")
                cursor.execute(statement)
                
        # Commit the transaction
        db.conn.commit()
        print("Table successfully created!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'db' in locals():
            db.close()

if __name__ == "__main__":
    script_path = Path(__file__).resolve().parent / "create_dpd_sample.sql"
    print(f"Running script: {script_path}")
    run_sql_script(script_path)
