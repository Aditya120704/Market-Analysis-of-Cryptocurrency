import pandas as pd
import os

# Create the output directory if it doesn't exist
if not os.path.exists('frontend/public/data'):
    os.makedirs('frontend/public/data')

# Load the Excel file
xls = pd.ExcelFile('data/crypto_data_collection.xlsx')

# Process the first sheet for market analysis
df_market = pd.read_excel(xls, sheet_name=0)
df_market.to_json('frontend/public/data/market_data.json', orient='records')

# Process the remaining sheets for historical data
for i in range(1, 6):
    df_historical = pd.read_excel(xls, sheet_name=i)
    # Get the sheet name and use it for the filename
    sheet_name = xls.sheet_names[i].lower().replace(' ', '_')
    df_historical.to_json(f'frontend/public/data/{sheet_name}_historical.json', orient='records')

print("Excel file successfully converted to JSON.")
