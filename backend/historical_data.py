import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# Base directory for storing historical data CSVs
HISTORICAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend', 'public', 'data')

def get_crypto_historical_data(ticker_symbol: str):
    file_path = os.path.join(HISTORICAL_DATA_DIR, f'{ticker_symbol.lower()}_historical_data.csv')
    
    df = None
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        last_date = df.index.max()
        
        # Fetch new data from the day after the last recorded date
        start_date = last_date + pd.Timedelta(days=1)
        today = datetime.now().date()

        if start_date.date() <= today:
            print(f"Fetching new data for {ticker_symbol} from {start_date.strftime('%Y-%m-%d')} to today...")
            new_data = yf.Ticker(ticker_symbol).history(start=start_date, end=today)
            if not new_data.empty:
                new_data.index = pd.to_datetime(new_data.index.date)
                df = pd.concat([df, new_data])
                df = df[~df.index.duplicated(keep='last')] # Remove duplicates if any
                df.to_csv(file_path)
                print(f"New data for {ticker_symbol} appended and saved.")
            else:
                print(f"No new data available to fetch for {ticker_symbol}.")
        else:
            print(f"Data for {ticker_symbol} is already up to date.")
    else:
        print(f"CSV not found for {ticker_symbol}. Fetching full history...")
        df = yf.Ticker(ticker_symbol).history(period="max")
        df.index = pd.to_datetime(df.index.date)
        df.to_csv(file_path)
        print(f"Full history for {ticker_symbol} fetched and saved.")

    df.columns = [c.lower() for c in df.columns]
    if "dividends" in df.columns: del df["dividends"]
    if "stock splits" in df.columns: del df["stock splits"]

    # Return only relevant columns for charting
    return df[["close"]].reset_index().rename(columns={'index': 'date'}).to_dict(orient='records')
