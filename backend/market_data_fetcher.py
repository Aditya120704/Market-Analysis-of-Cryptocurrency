import requests
import json
import os
import time

# Cache for API responses
_cache = {}
_cache_expiry_time = 300 # Cache for 5 minutes (300 seconds)

def cached_api_call(func):
    def wrapper(*args, **kwargs):
        cache_key = f"{func.__name__}-{json.dumps(args)}-{json.dumps(kwargs)} "
        if cache_key in _cache and (time.time() - _cache[cache_key]['timestamp'] < _cache_expiry_time):
            return _cache[cache_key]['data']
        
        # Add a small delay before making the API call to respect rate limits
        time.sleep(1) 
        
        result = func(*args, **kwargs)
        _cache[cache_key] = {'data': result, 'timestamp': time.time()}
        return result
    return wrapper

@cached_api_call
def get_live_market_data(vs_currency='usd', per_page=50):
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
                'price_change_percentage_24h': coin.get('price_change_percentage_24h'),
                'image': coin.get('image')
            })
        return processed_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching live market data from CoinGecko: {e}", file=os.sys.stderr)
        return {"error": "Failed to fetch live market data", "details": str(e)}
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=os.sys.stderr)
        return {"error": "An unexpected error occurred", "details": str(e)}

@cached_api_call
def get_trending_coins():
    trending_url = "https://api.coingecko.com/api/v3/search/trending"
    try:
        # First, get the list of trending coin IDs
        response = requests.get(trending_url)
        response.raise_for_status()
        trending_data = response.json()
        
        trending_coin_ids = [item['item']['id'] for item in trending_data.get('coins', [])[:10]] # Fetch more to ensure enough for sorting
        
        if not trending_coin_ids:
            return []

        # Second, get the full market data for those specific coins
        ids_param = ",".join(trending_coin_ids)
        market_url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={ids_param}"
        
        market_response = requests.get(market_url)
        market_response.raise_for_status()
        market_data = market_response.json()

        # Process data and sort by absolute price change percentage
        processed_data = []
        for coin in market_data:
            if coin.get('price_change_percentage_24h') is not None:
                processed_data.append({
                    'name': coin.get('name'),
                    'symbol': coin.get('symbol').upper(),
                    'price_usd': coin.get('current_price'),
                    'price_change_percentage_24h': coin.get('price_change_percentage_24h'),
                    'image': coin.get('image')
                })
        
        # Sort by absolute value of price_change_percentage_24h to find most movement
        processed_data.sort(key=lambda x: abs(x['price_change_percentage_24h']), reverse=True)

        return processed_data[:3] # Return top 3 by movement

    except requests.exceptions.RequestException as e:
        print(f"Error fetching trending coins from CoinGecko: {e}", file=os.sys.stderr)
        return {"error": "Failed to fetch trending coins", "details": str(e)}
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=os.sys.stderr)
        return {"error": "An unexpected error occurred", "details": str(e)}

@cached_api_call
def get_global_market_data():
    url = "https://api.coingecko.com/api/v3/global"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data and 'data' in data:
            return {
                'total_market_cap_usd': data['data']['total_market_cap']['usd'],
                'total_volume_24h_usd': data['data']['total_volume']['usd'],
                'bitcoin_dominance': data['data']['market_cap_percentage']['btc']
            }
        else:
            return {"error": "No global market data found"}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching global market data from CoinGecko: {e}", file=os.sys.stderr)
        return {"error": "Failed to fetch global market data", "details": str(e)}
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=os.sys.stderr)
        return {"error": "An unexpected error occurred", "details": str(e)}

@cached_api_call
def get_top_gainers_losers(vs_currency='usd', limit=5):
    url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency={vs_currency}&order=market_cap_desc&per_page=250&page=1&sparkline=false"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        all_coins = response.json()

        # Filter out coins with no price change data
        filtered_coins = [coin for coin in all_coins if coin.get('price_change_percentage_24h') is not None]

        # Sort by price change percentage
        sorted_coins = sorted(filtered_coins, key=lambda x: x['price_change_percentage_24h'], reverse=True)

        processed_gainers = []
        processed_losers = []

        # Get top gainers
        for coin in sorted_coins:
            if coin['price_change_percentage_24h'] > 0:
                processed_gainers.append({
                    'name': coin.get('name'),
                    'symbol': coin.get('symbol').upper(),
                    'price_usd': coin.get('current_price'),
                    'price_change_percentage_24h': coin.get('price_change_percentage_24h'),
                    'image': coin.get('image')
                })
            if len(processed_gainers) == limit:
                break
        
        # Get top losers
        for coin in reversed(sorted_coins):
            if coin['price_change_percentage_24h'] < 0:
                processed_losers.append({
                    'name': coin.get('name'),
                    'symbol': coin.get('symbol').upper(),
                    'price_usd': coin.get('current_price'),
                    'price_change_percentage_24h': coin.get('price_change_percentage_24h'),
                    'image': coin.get('image')
                })
            if len(processed_losers) == limit:
                break

        return {
            "gainers": processed_gainers,
            "losers": processed_losers
        }

    except requests.exceptions.RequestException as e:
        print(f"Error fetching top gainers/losers from CoinGecko: {e}", file=os.sys.stderr)
        return {"error": "Failed to fetch top gainers/losers", "details": str(e)}
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=os.sys.stderr)
        return {"error": "An unexpected error occurred", "details": str(e)}
