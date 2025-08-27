import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys
from historical_data import get_crypto_historical_data
from sentiment_analysis import get_historical_fear_and_greed_index, get_google_trends

SP500_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sp500.csv')
BTC_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend', 'public', 'data', 'btc.csv')

from pandas.errors import ParserError

def custom_date_parser(date_string):
    try:
        return pd.to_datetime(date_string, format='%Y-%m-%d %H:%M:%S%z', utc=True)
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(date_string, utc=True)
        except (ValueError, TypeError):
            return pd.NaT

from utils.data_processing import clean_historical_dataframe

def get_sp500_data():
    sp500_df = pd.DataFrame() # Initialize sp500_df as an empty DataFrame
    today = datetime.now().date()

    if os.path.exists(SP500_CSV_PATH) and os.path.getsize(SP500_CSV_PATH) > 0:
        try:
            sp500_df = pd.read_csv(SP500_CSV_PATH, index_col=0, parse_dates=True)
            if sp500_df.index.tz is None:
                sp500_df = sp500_df.tz_localize('UTC')
            else:
                sp500_df = sp500_df.tz_convert('UTC')
            sp500_df = clean_historical_dataframe(sp500_df)
            last_date = sp500_df.index.max().date()
            
            if last_date < today:
                
                new_data = yf.Ticker("^GSPC").history(start=last_date + pd.Timedelta(days=1), end=today)
                if not new_data.empty:
                    if new_data.index.tz is None:
                        new_data = new_data.tz_localize('UTC')
                    else:
                        new_data = new_data.tz_convert('UTC')
                    new_data = clean_historical_dataframe(new_data)
                    sp500_df = pd.concat([sp500_df, new_data])
                    sp500_df = sp500_df[~sp500_df.index.duplicated(keep='last')]
                    sp500_df.to_csv(SP500_CSV_PATH)
                else:
                    print("No new S&P 500 data available.")
            else:
                print("S&P 500 data is already up to date.")
        except Exception as e:
            print(f"Error reading S&P 500 CSV or fetching new data: {e}. Re-fetching all data.", file=sys.stderr)
            # Ensure sp500_df is a DataFrame before passing to clean_historical_dataframe
            sp500_df = yf.Ticker("^GSPC").history(period="max")
            if sp500_df.index.tz is None:
                sp500_df = sp500_df.tz_localize('UTC')
            else:
                sp500_df = sp500_df.tz_convert('UTC')
            sp500_df = clean_historical_dataframe(sp500_df)
            sp500_df.to_csv(SP500_CSV_PATH)
    else:
        print("S&P 500 CSV not found or empty. Fetching all historical data.")
        sp500_df = yf.Ticker("^GSPC").history(period="max")
        if sp500_df.index.tz is None:
            sp500_df = sp500_df.tz_localize('UTC')
        else:
            sp500_df = sp500_df.tz_convert('UTC')
        sp500_df = clean_historical_dataframe(sp500_df)
        sp500_df.to_csv(SP500_CSV_PATH)
    
    return sp500_df

def get_sp500_crypto_correlation():
    try:
        # Load and update S&P 500 data
        sp500_df = get_sp500_data()
        if sp500_df.empty:
            return {"error": "Failed to load S&P 500 data."}
        
        # The data is already cleaned in get_sp500_data, just need to format columns
        sp500_df.columns = [c.lower() for c in sp500_df.columns]

        # Load Bitcoin data
        btc_df = get_crypto_historical_data("BTC-USD", period="max")
        if isinstance(btc_df, dict) and "error" in btc_df:
            return btc_df
        btc_df.columns = [c.lower() for c in btc_df.columns]
        

        # Determine the common date range
        start_date = max(sp500_df.index.min(), btc_df.index.min())
        end_date = sp500_df.index.max() # Use S&P 500 end date
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Resample both to daily and reindex to the full date range
        sp500_df_daily = sp500_df['close'].resample('D').last().reindex(full_date_range).ffill()
        btc_df_daily = btc_df['close'].resample('D').last().reindex(full_date_range).ffill()

        # Create DataFrames with only the 'close' column for merging
        btc_close_df = btc_df_daily.to_frame(name='close_btc')
        sp500_close_df = sp500_df_daily.to_frame(name='close_sp500')

        # Merge dataframes on date
        merged_df = pd.merge(btc_close_df, sp500_close_df, left_index=True, right_index=True)
        

        if merged_df.empty or len(merged_df) < 2:
            return {"error": "Insufficient data for correlation analysis.", "details": "Merged dataframe is too small."}

        # Calculate daily returns
        merged_df['returns_btc'] = merged_df['close_btc'].pct_change(fill_method=None)
        merged_df['returns_sp500'] = merged_df['close_sp500'].pct_change(fill_method=None)
        merged_df = merged_df.dropna()
        

        if merged_df.empty:
            return {"error": "Insufficient data after calculating returns.", "details": "No common return data."}

        # Calculate correlation over different periods
        correlation_30d = merged_df['returns_btc'].tail(30).corr(merged_df['returns_sp500'].tail(30))
        correlation_90d = merged_df['returns_btc'].tail(90).corr(merged_df['returns_sp500'].tail(90))

        # Calculate recent performance changes
        latest_btc_close = merged_df['close_btc'].iloc[-1]
        latest_sp500_close = merged_df['close_sp500'].iloc[-1]

        btc_change_1d = merged_df['close_btc'].pct_change().iloc[-1] * 100
        sp500_change_1d = merged_df['close_sp500'].pct_change().iloc[-1] * 100

        btc_change_7d = (merged_df['close_btc'].iloc[-1] / merged_df['close_btc'].iloc[-7] - 1) * 100 if len(merged_df) >= 7 else None
        sp500_change_7d = (merged_df['close_sp500'].iloc[-1] / merged_df['close_sp500'].iloc[-7] - 1) * 100 if len(merged_df) >= 7 else None

        btc_change_30d = (merged_df['close_btc'].iloc[-1] / merged_df['close_btc'].iloc[-30] - 1) * 100 if len(merged_df) >= 30 else None
        sp500_change_30d = (merged_df['close_sp500'].iloc[-1] / merged_df['close_sp500'].iloc[-30] - 1) * 100 if len(merged_df) >= 30 else None

        return {
            "correlation_30d": correlation_30d if not pd.isna(correlation_30d) else None,
            "correlation_90d": correlation_90d if not pd.isna(correlation_90d) else None,
            "btc_change_1d": btc_change_1d if not pd.isna(btc_change_1d) else None,
            "sp500_change_1d": sp500_change_1d if not pd.isna(sp500_change_1d) else None,
            "btc_change_7d": btc_change_7d if btc_change_7d is not None and not pd.isna(btc_change_7d) else None,
            "sp500_change_7d": sp500_change_7d if sp500_change_7d is not None and not pd.isna(sp500_change_7d) else None,
            "btc_change_30d": btc_change_30d if btc_change_30d is not None and not pd.isna(btc_change_30d) else None,
            "sp500_change_30d": sp500_change_30d if sp500_change_30d is not None and not pd.isna(sp500_change_30d) else None,
            "latest_btc_close": latest_btc_close,
            "latest_sp500_close": latest_sp500_close,
            "recent_data": merged_df[['close_btc', 'close_sp500']].tail(90).reset_index().rename(columns={'index': 'date'}).assign(date=lambda df: df.date.dt.strftime('%Y-%m-%d')).to_dict(orient='records'),
            "latest_common_date": end_date.strftime('%Y-%m-%d')
        }
    except Exception as e:
        print(f"Error in get_sp500_crypto_correlation: {e}", file=sys.stderr)
        return {"error": "Failed to perform S&P 500 crypto correlation analysis", "details": str(e)}








def get_correlation_matrix(tickers: list[str]):
    try:
        all_returns = pd.DataFrame()
        for ticker in tickers:
            if ticker == "^GSPC":
                try:
                    df = get_sp500_data()
                except Exception as e:
                    print(f"ERROR: Failed to load S&P 500 data: {e}", file=sys.stderr)
                    continue
            else:
                try:
                    df = get_crypto_historical_data(ticker, period="max")
                except Exception as e:
                    print(f"ERROR: Failed to load crypto data for {ticker}: {e}", file=sys.stderr)
                    continue

            if isinstance(df, dict) and "error" in df:
                continue
            
            if 'close' not in df.columns:
                continue

            returns = df['close'].pct_change(fill_method=None)
            returns.name = ticker
            if all_returns.empty:
                all_returns = returns.to_frame()
            else:
                all_returns = pd.merge(all_returns, returns, left_index=True, right_index=True, how='inner')
            
        
        all_returns = all_returns.dropna()
        

        if all_returns.empty or len(all_returns) < 2:
            return {"error": "Insufficient common data for correlation matrix.", "details": "Not enough overlapping data points."}

        correlation_matrix = all_returns.corr()
        # Replace NaN values with None for JSON serialization
        correlation_matrix = correlation_matrix.replace({np.nan: None})
        return correlation_matrix.to_dict(orient='index')

    except Exception as e:
        print(f"Error in get_correlation_matrix: {e}", file=sys.stderr)
        return {"error": "Failed to compute correlation matrix", "details": str(e)}

def get_sentiment_correlation_analysis():
    """
    Calculates the rolling correlation between BTC price returns and various sentiment indicators.
    """
    try:
        # 1. Fetch all necessary data for the last year
        btc_df = get_crypto_historical_data("BTC-USD", period="1y")
        if btc_df.empty:
            return {"error": "Could not fetch Bitcoin historical data."}
        if btc_df.index.tz is None:
            btc_df = btc_df.tz_localize('UTC')
        else:
            btc_df = btc_df.tz_convert('UTC')

        fng_data = get_historical_fear_and_greed_index(days=365)
        if not fng_data.empty:
            if fng_data.index.tz is None:
                fng_data = fng_data.tz_localize('UTC')
            else:
                fng_data = fng_data.tz_convert('UTC')

        trends_df = get_google_trends(keywords=['bitcoin'], timeframe='today 12-m')
        if not trends_df.empty:
            if trends_df.index.tz is None:
                trends_df = trends_df.tz_localize('UTC')
            else:
                trends_df = trends_df.tz_convert('UTC')

        # 2. Prepare and merge data into a single DataFrame
        df = btc_df[['close']].copy()
        df['btc_returns'] = df['close'].pct_change()

        if not fng_data.empty:
            df = df.join(fng_data[['value']].rename(columns={'value': 'fear_and_greed'}))
            df['fear_and_greed'] = df['fear_and_greed'].ffill().bfill()

        if not trends_df.empty:
            df = df.join(trends_df[['bitcoin']].rename(columns={'bitcoin': 'google_trends'}))
            df['google_trends'] = df['google_trends'].ffill().bfill()
        
        df.dropna(inplace=True)

        if len(df) < 30:
            return {"error": "Not enough combined data for a meaningful correlation analysis."}

        # 3. Calculate 30-day rolling correlations safely
        fng_corr, trends_corr = None, None
        try:
            fng_corr = df['btc_returns'].rolling(window=30).corr(df['fear_and_greed']).iloc[-1]
        except Exception as e:
            pass

        try:
            trends_corr = df['btc_returns'].rolling(window=30).corr(df['google_trends']).iloc[-1]
        except Exception as e:
            pass

        return {
            "fear_and_greed_correlation": fng_corr if pd.notna(fng_corr) else None,
            "google_trends_correlation": trends_corr if pd.notna(trends_corr) else None,
            "last_updated": datetime.now().isoformat()
        }

    except Exception as e:
        return {"error": "Failed to compute sentiment correlation analysis", "details": str(e)}

