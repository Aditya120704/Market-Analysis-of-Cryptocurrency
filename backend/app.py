from flask import Flask, jsonify
from flask_cors import CORS
import json
import os
from prediction_model import get_prediction_direction, get_price_prediction_regression
from sentiment_analysis import get_news_sentiment, get_overall_market_sentiment, get_currency_sentiment, get_overall_currency_sentiment
from macro_data import get_sp500_data, get_sp500_crypto_correlation
from historical_data import get_crypto_historical_data
from market_data_fetcher import get_live_market_data

app = Flask(__name__)
CORS(app)

# Define the absolute path to the data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'public', 'data')

@app.route('/api/market_data')
def get_market_data():
    # This endpoint will now serve live data
    live_data = get_live_market_data()
    if "error" in live_data:
        return jsonify(live_data), 500
    return jsonify(live_data)

@app.route('/api/historical-data/<crypto_name>')
def get_historical_data(crypto_name):
    # Map user-friendly names to Yahoo Finance ticker symbols
    ticker_map = {
        "bitcoin": "BTC-USD",
        "ethereum": "ETH-USD",
        "ripple": "XRP-USD",
        "tether": "USDT-USD",
        "binancecoin": "BNB-USD"
    }
    
    ticker_symbol = ticker_map.get(crypto_name.lower())
    
    if not ticker_symbol:
        return jsonify({"error": "Cryptocurrency not found"}), 404
        
    historical_data = get_crypto_historical_data(ticker_symbol)
    if "error" in historical_data:
        return jsonify(historical_data), 500
    return jsonify(historical_data)

@app.route('/api/price-prediction')
def price_prediction_direction_route():
    prediction_data = get_prediction_direction()
    return jsonify(prediction_data)

@app.route('/api/price-prediction/value')
def price_prediction_value_route():
    prediction_data = get_price_prediction_regression()
    return jsonify(prediction_data)

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
    currency_sentiment_data = get_currency_sentiment(currency_name)
    if "error" in currency_sentiment_data:
        return jsonify(currency_sentiment_data), 500
    return jsonify(currency_sentiment_data)

@app.route('/api/sentiment/currency/overall/<currency_name>')
def get_overall_currency_sentiment_route(currency_name):
    overall_currency_sentiment_data = get_overall_currency_sentiment(currency_name)
    if "error" in overall_currency_sentiment_data:
        return jsonify(overall_currency_sentiment_data), 500
    return jsonify(overall_currency_sentiment_data)

@app.route('/api/macro/sp500')
def get_sp500_route():
    sp500_data = get_sp500_data()
    if "error" in sp500_data:
        return jsonify(sp500_data), 500
    return jsonify(sp500_data)

@app.route('/api/macro/sp500_crypto_analysis')
def get_sp500_crypto_analysis_route():
    analysis_data = get_sp500_crypto_correlation()
    if "error" in analysis_data:
        return jsonify(analysis_data), 500
    return jsonify(analysis_data)

if __name__ == '__main__':
    app.run(debug=True)