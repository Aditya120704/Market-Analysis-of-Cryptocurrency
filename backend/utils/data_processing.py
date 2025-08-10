import pandas as pd

def clean_historical_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes historical data DataFrames.
    Ensures date columns are in a consistent format and handles common issues.
    """
    if df.empty:
        return df

    # Reset index to make date a regular column if it's not already
    if isinstance(df.index, pd.DatetimeIndex) or df.index.name is not None:
        df = df.reset_index()

    # Identify the date column (assuming it's the first column or named 'Date'/'date')
    date_col = None
    if 'date' in df.columns:
        date_col = 'date'
    elif 'Date' in df.columns:
        date_col = 'Date'
    elif df.columns[0] == 'index': # After reset_index, original index might be named 'index'
        date_col = 'index'
    else:
        # Fallback: assume the first column is the date column
        date_col = df.columns[0]

    # Convert date column to datetime and handle errors
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')

    # Drop rows where the date is NaT after conversion
    df = df.dropna(subset=[date_col])

    # Set the cleaned date column as the index
    df = df.set_index(date_col)

    # Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Remove specific columns that are often not needed for analysis
    columns_to_drop = ['dividends', 'stock splits']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    return df
