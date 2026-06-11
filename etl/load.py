import os
import sqlite3
import pandas as pd
from etl.transform import transform_stock_data

def load_stock_data(df, db_name="stocks.db", table_name="stocks"):
    """
    Loads transformed stock dataframe into an SQLite database.
    """
    db_dir = "database"
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, db_name)
    
    print(f"Loading data into SQLite database at: {db_path}...")
    
    # Establish connection
    conn = sqlite3.connect(db_path)
    
    try:
        # Load df into sqlite table
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Successfully loaded {len(df)} rows into table '{table_name}'.")
        
        # Verify load by running a simple query
        count_df = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table_name}", conn)
        print(f"Verified rows in DB: {count_df['count'].values[0]}")
        
    except Exception as e:
        print(f"Error loading data to SQLite: {e}")
        raise e
    finally:
        conn.close()
        
    return db_path

def run_etl():
    """
    Runs the full ETL pipeline: Transform -> Load.
    Assumes raw data is already extracted.
    """
    df = transform_stock_data()
    db_path = load_stock_data(df)
    return db_path

if __name__ == "__main__":
    run_etl()
