import requests
import json
import os

def get_live_market_data(vs_currency='usd', per_page=100):
    url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency={vs_currency}&order=market_cap_desc&per_page={per_page}&page=1&sparkline=false"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        
        # Process data to match CoinTable's expected format
        processed_data = []
        for coin in data:
            processed_data.append({
                'name': coin.get('name'),
                'symbol': coin.get('symbol').upper(),
                'price_usd': coin.get('current_price'),
                'market_cap_usd': coin.get('market_cap'),
                'volume_usd': coin.get('total_volume'),
                'volatility_24h': coin.get('price_change_percentage_24h'),
            })
        return processed_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching live market data from CoinGecko: {e}", file=os.sys.stderr)
        return {"error": "Failed to fetch live market data", "details": str(e)}
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=os.sys.stderr)
        return {"error": "An unexpected error occurred", "details": str(e)}
