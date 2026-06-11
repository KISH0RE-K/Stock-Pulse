import os
import yfinance as yf
import pandas as pd

def extract_stock_data(ticker="AAPL", start_date="2025-01-01", end_date="2025-12-31"):
    """
    Downloads historical stock data using yfinance and saves it to CSV.
    """
    print(f"Extracting stock data for {ticker} from {start_date} to {end_date}...")
    
    # Download stock data
    # Note: to get the full 2025-12-31 data, yfinance end_date should be 2026-01-01
    # because the end date is exclusive.
    df = yf.download(ticker, start=start_date, end="2026-01-01")
    
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker} in the specified date range.")
    
    # If columns are MultiIndex (common in recent yfinance), drop the ticker level (level 1)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Reset index to make 'Date' a column
    df = df.reset_index()
    
    # Ensure raw data directory exists
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(raw_dir, f"{ticker.lower()}_raw.csv")
    df.to_csv(output_path, index=False)
    print(f"Raw data successfully saved to: {output_path}")
    print(f"Total rows extracted: {len(df)}")
    return output_path

if __name__ == "__main__":
    # Change working directory to the file's parent's parent directory if run from etl/
    # but we assume execution from stock-etl-project/ directory
    # Set standard directory structure relative to current working directory
    extract_stock_data()
