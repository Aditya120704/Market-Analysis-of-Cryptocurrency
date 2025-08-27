import pandas as pd
import numpy as np
import sys

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a set of common technical indicators and adds them to the DataFrame.
    """
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        print("Warning: Missing OHLCV columns for full indicator calculation.", file=sys.stderr)
    
    # Moving Averages
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['ema_7'] = df['close'].ewm(span=7, adjust=False).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    df['ema_30'] = df['close'].ewm(span=30, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    alpha = 1.0 / 14
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False, min_periods=12).mean()
    exp2 = df['close'].ewm(span=26, adjust=False, min_periods=26).mean()
    df['macd'] = exp1 - exp2
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False, min_periods=9).mean()

    # Bollinger Bands
    df['middle_band'] = df['close'].rolling(window=20).mean()
    df['std_dev'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)

    # Drop intermediate columns
    df = df.drop(columns=['middle_band', 'std_dev'], errors='ignore')
    
    # Clean up data
    indicator_columns = ['sma_7', 'ema_7', 'sma_30', 'ema_30', 'rsi', 'macd', 'signal_line', 'upper_band', 'lower_band']
    
    for col in indicator_columns:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


    

