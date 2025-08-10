from flask import Flask, jsonify, request
import pandas as pd
from flask_cors import CORS
import json
import os
import time
import sys
from prediction_model import get_prediction_direction, get_price_prediction_regression
from datetime import datetime
from sentiment_analysis import get_news_sentiment, get_overall_market_sentiment, get_currency_sentiment, get_overall_currency_sentiment, get_fear_and_greed_index
from macro_data import get_sp500_data, get_sp500_crypto_correlation, get_correlation_matrix
from historical_data import get_crypto_historical_data, calculate_volatility
from utils.technical_indicators import calculate_all_indicators
from utils.data_processing import clean_historical_dataframe
from market_data_fetcher import get_live_market_data, get_trending_coins, get_global_market_data, get_top_gainers_losers

app = Flask(__name__)
CORS(app)

# Define the absolute path to the data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'public', 'data')

@app.route('/api/market/overview')
def get_market_data():
    live_data = get_live_market_data()
    if "error" in live_data:
        return jsonify(live_data), 500
    return jsonify(live_data)

@app.route('/api/market/trending')
def get_trending_coins_route():
    trending_data = get_trending_coins()
    if "error" in trending_data:
        return jsonify(trending_data), 500
    return jsonify(trending_data)

@app.route('/api/historical-data/<crypto_name>')
@app.route('/api/historical-data/<crypto_name>/<period>')
def get_historical_data(crypto_name, period="max"):
    ticker_map = {
        "bitcoin": "BTC-USD",
        "ethereum": "ETH-USD",
        "ripple": "XRP-USD",
        "tether": "USDT-USD",
        "binancecoin": "BNB-USD",
        "solana": "SOL-USD",
        "cardano": "ADA-USD"
    }
    
    ticker_symbol = ticker_map.get(crypto_name.lower())
    
    if not ticker_symbol:
        return jsonify({"error": "Cryptocurrency not found"}), 404
        
    historical_data = get_crypto_historical_data(ticker_symbol, period=period)
    if isinstance(historical_data, dict) and "error" in historical_data:
        return jsonify(historical_data), 500

    # Calculate technical indicators
    # historical_data = calculate_all_indicators(historical_data)
    # Ensure date index is serialized as a 'date' column (ISO string)
    df = historical_data.reset_index()
    # Handle different possible index names
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'date'}, inplace=True)
    elif 'index' in df.columns:
        df.rename(columns={'index': 'date'}, inplace=True)
    # Format to ISO date if dtype is datetime-like
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        except Exception:
            pass
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/predict/price')
def predict_price_route():
    try:
        # Combine regression metrics (MAE) and direction metrics (accuracy, probability)
        regression = get_price_prediction_regression()
        if isinstance(regression, dict) and "error" in regression:
            return jsonify(regression), 500

        direction = get_prediction_direction()
        if isinstance(direction, dict) and "error" in direction:
            # Still return regression if direction fails, with nulls for missing fields
            return jsonify({
                "predicted_price": regression.get("predicted_price"),
                "mae": regression.get("mae"),
                "accuracy": None,
                "prediction_probability": None
            })

        return jsonify({
            "predicted_price": regression.get("predicted_price"),
            "mae": regression.get("mae"),
            "accuracy": direction.get("accuracy"),
            "prediction_probability": direction.get("prediction_probability"),
            "last_data_date": regression.get("last_data_date"),
            "prediction_date": regression.get("prediction_date"),
            "latest_data": direction.get("latest_data")
        })
    except Exception as e:
        return jsonify({"error": "Failed to get price prediction", "details": str(e)}), 500


@app.route('/api/predict/train', methods=['POST'])
def train_models_route():
    try:
        # Force retrain both models and persist metrics
        regression = get_price_prediction_regression(force_retrain=True)
        if isinstance(regression, dict) and "error" in regression:
            return jsonify(regression), 500

        direction = get_prediction_direction(force_retrain=True)
        if isinstance(direction, dict) and "error" in direction:
            # Return at least regression metrics
            return jsonify({
                "predicted_price": regression.get("predicted_price"),
                "mae": regression.get("mae"),
                "accuracy": None,
                "prediction_probability": None,
                "trained": True
            })

        return jsonify({
            "predicted_price": regression.get("predicted_price"),
            "mae": regression.get("mae"),
            "accuracy": direction.get("accuracy"),
            "prediction_probability": direction.get("prediction_probability"),
            "trained": True
        })
    except Exception as e:
        return jsonify({"error": "Failed to train models", "details": str(e)}), 500

@app.route('/api/sentiment/news')
def get_news_sentiment_route():
    sentiment_data = get_news_sentiment()
    if "error" in sentiment_data:
        return jsonify(sentiment_data), 500
    return jsonify(sentiment_data)

@app.route('/api/sentiment/overall')
def get_overall_sentiment_route():
    overall_sentiment_data = get_overall_market_sentiment()
    if "error" in overall_sentiment_data:
        return jsonify(overall_sentiment_data), 500
    return jsonify(overall_sentiment_data)

@app.route('/api/sentiment/currency/<currency_name>')
def get_currency_sentiment_route(currency_name):
    normalized_map = {
        'bitcoin': 'Bitcoin BTC',
        'ethereum': 'Ethereum ETH',
        'ripple': 'Ripple XRP',
        'tether': 'Tether USDT',
        'binancecoin': 'Binance Coin BNB',
        'bnb': 'Binance Coin BNB',
    }
    query_name = normalized_map.get(currency_name.lower(), currency_name)
    currency_sentiment_data = get_currency_sentiment(query_name)
    if "error" in currency_sentiment_data:
        return jsonify(currency_sentiment_data), 500
    return jsonify(currency_sentiment_data)

@app.route('/api/sentiment/currency/overall/<currency_name>')
def get_overall_currency_sentiment_route(currency_name):
    normalized_map = {
        'bitcoin': 'Bitcoin BTC',
        'ethereum': 'Ethereum ETH',
        'ripple': 'Ripple XRP',
        'tether': 'Tether USDT',
        'binancecoin': 'Binance Coin BNB',
        'bnb': 'Binance Coin BNB',
    }
    query_name = normalized_map.get(currency_name.lower(), currency_name)
    overall_currency_sentiment_data = get_overall_currency_sentiment(query_name)
    if "error" in overall_currency_sentiment_data:
        return jsonify(overall_currency_sentiment_data), 500
    return jsonify(overall_currency_sentiment_data)

@app.route('/api/sentiment/fear_and_greed')
def get_fear_and_greed_route():
    fear_and_greed_data = get_fear_and_greed_index()
    if "error" in fear_and_greed_data:
        return jsonify(fear_and_greed_data), 500
    return jsonify(fear_and_greed_data)

@app.route('/api/macro/sp500')
def get_sp500_route():
    sp500_data = get_sp500_data()
    if "error" in sp500_data:
        return jsonify(sp500_data), 500
    return jsonify(sp500_data.to_dict(orient='records'))

@app.route('/api/analysis/sp500_correlation')
def get_sp500_crypto_analysis_route():
    analysis_data = get_sp500_crypto_correlation()
    # Always return 200 with a safe payload; macro layer already provides neutral fields when data is insufficient
    return jsonify(analysis_data)

@app.route('/api/market/global-data')
def get_global_market_data_route():
    global_data = get_global_market_data()
    if "error" in global_data:
        return jsonify(global_data), 500
    return jsonify(global_data)

@app.route('/api/market/top-movers')
def get_top_movers_route():
    top_movers_data = get_top_gainers_losers()
    if "error" in top_movers_data:
        return jsonify(top_movers_data), 500
    return jsonify(top_movers_data)

@app.route('/api/analysis/correlation')
def get_correlation_route():
    tickers_str = request.args.get('tickers')
    if not tickers_str:
        return jsonify({"error": "Missing 'tickers' parameter"}), 400
    tickers = [t.strip() for t in tickers_str.split(',')]
    correlation_data = get_correlation_matrix(tickers)
    if "error" in correlation_data:
        return jsonify(correlation_data), 500
    return jsonify(correlation_data)

@app.route('/api/analysis/volatility/<ticker>')
def get_volatility_route(ticker):
    volatility_data = calculate_volatility(ticker)
    if "error" in volatility_data:
        return jsonify(volatility_data), 500
    return jsonify(volatility_data)

if __name__ == '__main__':
    app.run(debug=True)
