import yfinance as yf
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sentiment_analysis import get_overall_market_sentiment
import numpy as np

def _prepare_data(btc_data):
    # --- Integrate NewsAPI Sentiment ---
    overall_sentiment_data = get_overall_market_sentiment()
    if "error" in overall_sentiment_data:
        print(f"Error fetching overall sentiment: {overall_sentiment_data['details']}")
        current_sentiment = "Neutral"
    else:
        current_sentiment = overall_sentiment_data['overall_sentiment']

    sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    btc_data['overall_sentiment_score'] = sentiment_map.get(current_sentiment, 0)

    # --- Technical Indicators ---
    btc_data['SMA_7'] = btc_data['close'].rolling(window=7).mean()
    btc_data['EMA_7'] = btc_data['close'].ewm(span=7, adjust=False).mean()
    btc_data['SMA_30'] = btc_data['close'].rolling(window=30).mean()
    btc_data['EMA_30'] = btc_data['close'].ewm(span=30, adjust=False).mean()

    delta = btc_data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    btc_data['RSI'] = 100 - (100 / (1 + rs))

    exp1 = btc_data['close'].ewm(span=12, adjust=False).mean()
    exp2 = btc_data['close'].ewm(span=26, adjust=False).mean()
    btc_data['MACD'] = exp1 - exp2
    btc_data['Signal_Line'] = btc_data['MACD'].ewm(span=9, adjust=False).mean()

    btc_data['MA_20'] = btc_data['close'].rolling(window=20).mean()
    btc_data['StdDev'] = btc_data['close'].rolling(window=20).std()
    btc_data['Upper_Band'] = btc_data['MA_20'] + (btc_data['StdDev'] * 2)
    btc_data['Lower_Band'] = btc_data['MA_20'] - (btc_data['StdDev'] * 2)

    btc_data = btc_data.dropna()
    return btc_data

def _create_sequences(data, target_column, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        y.append(target_column.iloc[i + look_back])
    return np.array(X), np.array(y)

def get_prediction_direction():
    # --- Data Loading ---
    btc_ticker = yf.Ticker("BTC-USD")
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    btc_csv_path = os.path.join(backend_dir, '..', 'frontend', 'public', 'data', 'btc.csv')

    if os.path.exists(btc_csv_path):
        btc = pd.read_csv(btc_csv_path, index_col=0)
    else:
        btc = btc_ticker.history(period="max")
        btc.to_csv(btc_csv_path)

    btc.index = pd.to_datetime(pd.to_datetime(btc.index).date)
    del btc["Dividends"]
    del btc["Stock Splits"]
    btc.columns = [c.lower() for c in btc.columns]

    btc = _prepare_data(btc.copy())

    # --- Feature Engineering for Classification ---
    btc["tomorrow"] = btc["close"].shift(-1)
    btc["target"] = (btc["tomorrow"] > btc["close"]).astype(int)
    btc = btc.dropna() # Drop NA after creating target

    features = ["close", "overall_sentiment_score", 
                'SMA_7', 'EMA_7', 'SMA_30', 'EMA_30', 
                'RSI', 'MACD', 'Signal_Line', 
                'Upper_Band', 'Lower_Band']

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(btc[features])

    look_back = 90
    X, y = _create_sequences(scaled_data, btc["target"], look_back)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    last_data_points = btc[features].iloc[-look_back:].values
    last_data_points_scaled = scaler.transform(last_data_points)
    X_predict = np.array([last_data_points_scaled])

    next_day_prediction_proba = model.predict(X_predict)[0][0]
    next_day_prediction = 1 if next_day_prediction_proba > 0.5 else 0

    latest_data = btc.iloc[-1].to_dict()
    
    return {
        "prediction": int(next_day_prediction),
        "accuracy": float(accuracy),
        "latest_data": {
            "close": latest_data.get('close'),
            "overall_sentiment_score": latest_data.get('overall_sentiment_score'),
            "SMA_7": latest_data.get('SMA_7'),
            "EMA_7": latest_data.get('EMA_7'),
            "SMA_30": latest_data.get('SMA_30'),
            "EMA_30": latest_data.get('EMA_30'),
            "RSI": latest_data.get('RSI'),
            "MACD": latest_data.get('MACD'),
            "Signal_Line": latest_data.get('Signal_Line'),
            "Upper_Band": latest_data.get('Upper_Band'),
            "Lower_Band": latest_data.get('Lower_Band'),
        }
    }

def get_price_prediction_regression():
    # --- Data Loading ---
    btc_ticker = yf.Ticker("BTC-USD")
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    btc_csv_path = os.path.join(backend_dir, '..', 'frontend', 'public', 'data', 'btc.csv')

    if os.path.exists(btc_csv_path):
        btc = pd.read_csv(btc_csv_path, index_col=0)
    else:
        btc = btc_ticker.history(period="max")
        btc.to_csv(btc_csv_path)

    btc.index = pd.to_datetime(pd.to_datetime(btc.index).date)
    del btc["Dividends"]
    del btc["Stock Splits"]
    btc.columns = [c.lower() for c in btc.columns]

    btc = _prepare_data(btc.copy())

    # --- Feature Engineering for Regression ---
    btc["tomorrow_close"] = btc["close"].shift(-1)
    btc = btc.dropna() # Drop NA after creating target

    features = ["close", "overall_sentiment_score", 
                'SMA_7', 'EMA_7', 'SMA_30', 'EMA_30', 
                'RSI', 'MACD', 'Signal_Line', 
                'Upper_Band', 'Lower_Band']

    # Scale features and target for regression
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_X.fit_transform(btc[features])

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_y.fit_transform(btc[["tomorrow_close"]])

    look_back = 90
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back), :])
        y.append(scaled_target[i + look_back, 0])
    X, y = np.array(X), np.array(y)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    # Build LSTM model for Regression
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=1, activation='linear')) # Linear activation for regression

    model.compile(optimizer='adam', loss='mse') # Mean Squared Error loss for regression
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)

    # Evaluate the model
    y_pred_scaled = model.predict(X_test)
    y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled)
    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)

    # Prepare data for next day prediction
    last_data_points = btc[features].iloc[-look_back:].values
    last_data_points_scaled = scaler_X.transform(last_data_points)
    X_predict = np.array([last_data_points_scaled])

    next_day_prediction_scaled = model.predict(X_predict)[0][0]
    next_day_prediction_unscaled = scaler_y.inverse_transform([[next_day_prediction_scaled]])[0][0]

    return {
        "predicted_price": float(next_day_prediction_unscaled),
        "mae": float(mae),
    }