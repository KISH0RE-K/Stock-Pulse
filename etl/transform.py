import os
import pandas as pd
import numpy as np

def transform_stock_data(ticker="AAPL"):
    """
    Cleans stock data, creates technical indicators, and saves transformed data.
    """
    raw_path = os.path.join("data", "raw", f"{ticker.lower()}_raw.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}. Run extract.py first.")
        
    print(f"Transforming raw stock data from: {raw_path}...")
    df = pd.read_csv(raw_path)
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        
    # Standardize column names
    # yfinance output columns: Date, Open, High, Low, Close, Adj Close, Volume
    # Ensure columns exist, stripping potential extra spaces
    df.columns = [col.strip() for col in df.columns]
    
    # Make sure Date is parsed correctly
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by Date ascending
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate daily returns
    # Daily Return = (Close_t - Close_t-1) / Close_t-1
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Calculate price swing
    df['Price_Swing'] = df['High'] - df['Low']
    
    # Calculate simple moving averages (SMA)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Calculate daily volume percentage change
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Calculate lag features for returns (lags 1 to 5)
    for lag in range(1, 6):
        df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)
        
    # Convert Date back to string format YYYY-MM-DD for clean storage in SQLite
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # Print transformation summary
    print(f"Transformation complete. New columns added.")
    print(f"Columns: {list(df.columns)}")
    print(f"Total rows: {len(df)}")
    
    # Ensure transformed directory exists (even though we'll load to database, we can also return df)
    return df

if __name__ == "__main__":
    transform_stock_data()
