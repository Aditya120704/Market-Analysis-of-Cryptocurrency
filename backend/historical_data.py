import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import numpy as np
import sys
from utils.data_processing import clean_historical_dataframe
from utils.technical_indicators import calculate_all_indicators # NEW IMPORT

# Base directory for storing historical data CSVs
HISTORICAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'historical_crypto_data')

from utils.data_processing import clean_historical_dataframe

def get_crypto_historical_data(ticker_symbol: str, period: str = "max"):
    file_name = f'{ticker_symbol.lower()}_historical_data.csv'
    file_path = os.path.join(HISTORICAL_DATA_DIR, file_name)
    
    df = None
    today = datetime.now().date()

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            # Read the CSV
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            if df.empty:
                df = yf.Ticker(ticker_symbol).history(period="max")
                df = clean_historical_dataframe(df)
                df.to_csv(file_path)
            else:
                # If data exists, proceed with incremental update logic
                last_date = df.index.max()
                
                start_date = last_date + pd.Timedelta(days=1)
                
                if start_date.date() <= today:
                    new_data = yf.Ticker(ticker_symbol).history(start=start_date, end=today)
                    if not new_data.empty:
                        new_data = clean_historical_dataframe(new_data)
                        df = pd.concat([df, new_data])
                        df = df[~df.index.duplicated(keep='last')] # Remove duplicates if any
                        df.to_csv(file_path)
                    else:
                        return df
                else:
                    pass
        except pd.errors.EmptyDataError:
            df = yf.Ticker(ticker_symbol).history(period="max")
            df = clean_historical_dataframe(df)
            df.to_csv(file_path)
        except Exception as e:
            print(f"ERROR: {ticker_symbol} - Error reading CSV or processing data: {e}. Attempting to fetch from yfinance.", file=sys.stderr)
            df = yf.Ticker(ticker_symbol).history(period="max")
            df = clean_historical_dataframe(df)
            df.to_csv(file_path)
    else:
        df = yf.Ticker(ticker_symbol).history(period="max")
        df = clean_historical_dataframe(df)
        df.to_csv(file_path)

    if not df.empty:
        df = calculate_all_indicators(df) # Call the new function here
    
    return df

def calculate_volatility(ticker: str):
    try:
        # Fetch historical data
        df = get_crypto_historical_data(ticker, period="max")
        if isinstance(df, dict) and "error" in df:
            return df

        # Validate data
        if 'close' not in df.columns:
            return {"error": "Close prices not available for volatility calculation."}

        # Compute daily returns and 30-day annualized volatility
        daily_returns = df['close'].pct_change(fill_method=None).dropna()
        if len(daily_returns) < 30:
            return {"error": "Insufficient data for volatility", "details": "Need at least 30 daily return observations."}

        window_returns = daily_returns.tail(30)
        daily_volatility = float(window_returns.std())
        # Crypto trades 365 days/year; annualize accordingly
        annualized_volatility = daily_volatility * float(np.sqrt(365))

        return {"ticker": ticker, "volatility": annualized_volatility * 100.0}

    except Exception as e:
        return {"error": "Failed to calculate volatility", "details": str(e)}