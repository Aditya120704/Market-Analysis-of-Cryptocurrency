import pandas as pd
import numpy as np
import sys

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a set of common technical indicators and adds them to the DataFrame.
    df must contain 'open', 'high', 'low', 'close', 'volume' columns.
    """
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        print("Warning: Missing OHLCV columns for full indicator calculation.", file=sys.stderr)
        # Attempt to proceed with available data, but some indicators might fail
    
    # Moving Averages
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['ema_7'] = df['close'].ewm(span=7, adjust=False).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    df['ema_30'] = df['close'].ewm(span=30, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['middle_band'] = df['close'].rolling(window=20).mean()
    df['std_dev'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)

    # Drop intermediate columns
    df = df.drop(columns=['middle_band', 'std_dev'], errors='ignore')

    return df