import yfinance as yf
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sentiment_analysis import get_sentiment_score, get_historical_fear_and_greed_index, get_google_trends
import numpy as np
import joblib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import sys

# Define paths for saved models and scalers (UNCHANGED)
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

PRICE_MODEL_PATH = os.path.join(MODELS_DIR, 'price_prediction_model.json')
PRICE_SCALER_X_PATH = os.path.join(MODELS_DIR, 'price_scaler_X.pkl')
PRICE_MAE_PATH = os.path.join(MODELS_DIR, 'price_mae.pkl')

DIRECTION_MODEL_PATH = os.path.join(MODELS_DIR, 'direction_prediction_model.keras')
DIRECTION_SCALER_PATH = os.path.join(MODELS_DIR, 'direction_scaler.pkl')
DIRECTION_ACCURACY_PATH = os.path.join(MODELS_DIR, 'direction_accuracy.pkl')

# NEW: Ensemble model paths (only additions)
ENSEMBLE_WEIGHT_PATH = os.path.join(MODELS_DIR, 'ensemble_weights.pkl')

def _prepare_data(btc_data):
    # --- EXISTING CODE (UNCHANGED) ---
    btc_data['return'] = btc_data['close'].pct_change()
    btc_data['log_return'] = np.log(btc_data['close']).diff()
    btc_data['vol_14'] = btc_data['return'].rolling(14, center=False).std()

    # --- Technical Indicators (UNCHANGED) ---
    btc_data['SMA_7'] = btc_data['close'].rolling(window=7, center=False).mean()
    btc_data['EMA_7'] = btc_data['close'].ewm(span=7, adjust=False).mean()
    btc_data['SMA_30'] = btc_data['close'].rolling(window=30, center=False).mean()
    btc_data['EMA_30'] = btc_data['close'].ewm(span=30, adjust=False).mean()

    delta = btc_data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, center=False).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, center=False).mean()
    rs = gain / loss
    btc_data['RSI'] = 100 - (100 / (1 + rs))

    exp1 = btc_data['close'].ewm(span=12, adjust=False).mean()
    exp2 = btc_data['close'].ewm(span=26, adjust=False).mean()
    btc_data['MACD'] = exp1 - exp2
    btc_data['Signal_Line'] = btc_data['MACD'].ewm(span=9, adjust=False).mean()

    btc_data['MA_20'] = btc_data['close'].rolling(window=20, center=False).mean()
    btc_data['StdDev'] = btc_data['close'].rolling(window=20, center=False).std()
    btc_data['Upper_Band'] = btc_data['MA_20'] + (btc_data['StdDev'] * 2)
    btc_data['Lower_Band'] = btc_data['MA_20'] - (btc_data['StdDev'] * 2)
    
    # --- NEW ENHANCEMENTS (ADDITIONS ONLY) ---
    # Multi-timeframe RSI for better market condition detection
    for period in [9, 21, 30]:
        delta_temp = btc_data['close'].diff()
        gain_temp = (delta_temp.where(delta_temp > 0, 0)).rolling(window=period, center=False).mean()
        loss_temp = (-delta_temp.where(delta_temp < 0, 0)).rolling(window=period, center=False).mean()
        rs_temp = gain_temp / loss_temp
        btc_data[f'RSI_{period}'] = 100 - (100 / (1 + rs_temp))
    
    # Price position within recent range (market structure)
    btc_data['price_position'] = (btc_data['close'] - btc_data['close'].rolling(20, center=False).min()) / (
        btc_data['close'].rolling(20, center=False).max() - btc_data['close'].rolling(20, center=False).min() + 1e-8
    )
    
    # Volatility regime
    btc_data['vol_regime'] = btc_data['return'].rolling(30, center=False).std().rolling(10, center=False).rank(pct=True)
    
    # Volume analysis
    btc_data['volume_sma'] = btc_data['volume'].rolling(20, center=False).mean()
    btc_data['volume_ratio'] = btc_data['volume'] / (btc_data['volume_sma'] + 1e-8)

    btc_data = btc_data.dropna()
    return btc_data

def _create_sequences(data, target_column, look_back):
    # UNCHANGED
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        y.append(target_column.iloc[i + look_back])
    return np.array(X), np.array(y)

def _create_enhanced_lstm_model(input_shape):
    """Enhanced LSTM with attention-like mechanism via multiple layers"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(100, return_sequences=True),  # Keep sequences for better pattern learning
        Dropout(0.3),
        LSTM(100, return_sequences=False),
        Dropout(0.3),
        Dense(50, activation='relu'),  # Additional dense layer
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

def _ensemble_prediction(lstm_pred, xgb_pred, lstm_confidence, xgb_confidence):
    """Weight predictions based on model confidence/accuracy"""
    total_confidence = lstm_confidence + xgb_confidence
    lstm_weight = lstm_confidence / total_confidence
    xgb_weight = xgb_confidence / total_confidence
    
    ensemble_pred = (lstm_weight * lstm_pred) + (xgb_weight * xgb_pred)
    ensemble_confidence = (lstm_confidence + xgb_confidence) / 2
    
    return ensemble_pred, ensemble_confidence

def get_prediction_direction(force_retrain: bool = False):
    # --- Data Loading (UNCHANGED) ---
    btc_ticker = yf.Ticker("BTC-USD")
    
    try:
        latest_btc_data = btc_ticker.history(period="120d")
        if latest_btc_data.empty:
            raise ValueError("No data fetched from yfinance")
            
        latest_btc_data.index = pd.to_datetime(latest_btc_data.index).tz_localize(None).tz_localize('UTC')
        del latest_btc_data["Dividends"]
        del latest_btc_data["Stock Splits"]
        latest_btc_data.columns = [c.lower() for c in latest_btc_data.columns]
        
        btc = _prepare_data(latest_btc_data.copy())
        
        sentiment_score = get_sentiment_score()
        btc['sentiment_score'] = sentiment_score

        # ENHANCED: Extended features list with new indicators
        features = ["close", "volume", 'sentiment_score',
                    'SMA_7', 'EMA_7', 'SMA_30', 'EMA_30', 
                    'RSI', 'MACD', 'Signal_Line', 
                    'Upper_Band', 'Lower_Band', 'return', 'log_return', 'vol_14',
                    'RSI_9', 'RSI_21', 'RSI_30', 'price_position', 'vol_regime', 'volume_ratio']

    except Exception as e:
        return {"error": "Failed to fetch or prepare recent Bitcoin data for direction prediction", "details": str(e)}

    look_back = 90

    if os.path.exists(DIRECTION_MODEL_PATH) and not force_retrain:
        model = load_model(DIRECTION_MODEL_PATH)
        scaler = joblib.load(DIRECTION_SCALER_PATH)
        accuracy = joblib.load(DIRECTION_ACCURACY_PATH)
        
        # Load ensemble weights if available
        if os.path.exists(ENSEMBLE_WEIGHT_PATH):
            ensemble_weights = joblib.load(ENSEMBLE_WEIGHT_PATH)
        else:
            ensemble_weights = {'lstm_confidence': 0.8, 'xgb_confidence': 0.7}

    else:
        try:
            full_btc_history = btc_ticker.history(period="5y")
            full_btc_history.index = pd.to_datetime(full_btc_history.index).tz_localize(None).tz_localize('UTC')
            del full_btc_history["Dividends"]
            del full_btc_history["Stock Splits"]
            full_btc_history.columns = [c.lower() for c in full_btc_history.columns]
            
            train_btc = _prepare_data(full_btc_history.copy())
            
            sentiment_score = get_sentiment_score()
            train_btc['sentiment_score'] = sentiment_score

            train_btc["tomorrow"] = train_btc["close"].shift(-1)
            train_btc["target"] = (train_btc["tomorrow"] > train_btc["close"]).astype(int)
            print("Target variable class distribution:")
            print(train_btc["target"].value_counts())
            train_btc = train_btc.dropna()

            features_for_training = ["close", "volume", 'sentiment_score',
                                    'SMA_7', 'EMA_7', 'SMA_30', 'EMA_30', 
                                    'RSI', 'MACD', 'Signal_Line', 
                                    'Upper_Band', 'Lower_Band', 'return', 'log_return', 'vol_14',
                                    'RSI_9', 'RSI_21', 'RSI_30', 'price_position', 'vol_regime', 'volume_ratio']

            train_size = int(len(train_btc) * 0.8)
            train_df = train_btc[:train_size]
            test_df = train_btc[train_size:]

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_train_data = scaler.fit_transform(train_df[features_for_training])
            scaled_test_data = scaler.transform(test_df[features_for_training])

            X_train, y_train = _create_sequences(scaled_train_data, train_df["target"], look_back)
            X_test, y_test = _create_sequences(scaled_test_data, test_df["target"], look_back)

            # ENHANCED: Use improved LSTM architecture
            model = _create_enhanced_lstm_model((X_train.shape[1], X_train.shape[2]))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Enhanced callbacks with learning rate reduction
            from tensorflow.keras.callbacks import ReduceLROnPlateau
            es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            mc = ModelCheckpoint(DIRECTION_MODEL_PATH.replace('.keras', '_best.keras'), save_best_only=True, monitor='val_loss')
            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.0001)

            model.fit(X_train, y_train, epochs=100, batch_size=64, 
                     validation_data=(X_test, y_test), 
                     callbacks=[es, mc, lr_reducer], verbose=1)

            _, accuracy = model.evaluate(X_test, y_test, verbose=0)
            
            # ENHANCED: Train XGBoost for ensemble (optional enhancement)
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            xgb_model.fit(X_train_flat, y_train)
            xgb_accuracy = xgb_model.score(X_test_flat, y_test)
            
            # Store ensemble weights
            ensemble_weights = {
                'lstm_confidence': float(accuracy),
                'xgb_confidence': float(xgb_accuracy)
            }

            # Save everything
            model.save(DIRECTION_MODEL_PATH)
            joblib.dump(scaler, DIRECTION_SCALER_PATH)
            joblib.dump(accuracy, DIRECTION_ACCURACY_PATH)
            joblib.dump(ensemble_weights, ENSEMBLE_WEIGHT_PATH)
            print("Enhanced direction prediction model, scaler, and accuracy trained and saved.")

        except Exception as e:
            return {"error": "Failed to train and save the direction prediction model", "details": str(e)}

    # Prepare data for next day prediction
    last_data_points = btc[features].iloc[-look_back:].values
    last_data_points_scaled = scaler.transform(last_data_points)
    X_predict = np.array([last_data_points_scaled])

    next_day_prediction_proba = model.predict(X_predict)[0][0]
    next_day_prediction = 1 if next_day_prediction_proba > 0.5 else 0

    # ENHANCED: Add confidence scoring based on prediction probability
    confidence_score = max(next_day_prediction_proba, 1 - next_day_prediction_proba)

    latest_data = btc.iloc[-1].to_dict()
    
    return {
        "prediction": int(next_day_prediction),
        "accuracy": float(accuracy),
        "prediction_probability": float(next_day_prediction_proba),
        "confidence_score": float(confidence_score),  # NEW
        "latest_data": {
            "close": latest_data.get('close'),
            "overall_sentiment_score": latest_data.get('sentiment_score'),
            "SMA_7": latest_data.get('SMA_7'),
            "EMA_7": latest_data.get('EMA_7'),
            "SMA_30": latest_data.get('SMA_30'),
            "EMA_30": latest_data.get('EMA_30'),
            "RSI": latest_data.get('RSI'),
            "MACD": latest_data.get('MACD'),
            "Signal_Line": latest_data.get('Signal_Line'),
            "Upper_Band": latest_data.get('Upper_Band'),
            "Lower_Band": latest_data.get('Lower_Band'),
            "price_position": latest_data.get('price_position'),  # NEW
            "vol_regime": latest_data.get('vol_regime'),  # NEW
        }
    }

# Keep existing _merge_fear_and_greed and _merge_google_trends functions UNCHANGED

def get_price_prediction_regression(force_retrain: bool = False):
    look_back = 30
    base_features = ['close', 'volume', 'sentiment_score', 'SMA_7', 'SMA_30', 'RSI', 'MACD', 'Upper_Band', 'Lower_Band']
    
    # ENHANCED: Add new features to base features
    enhanced_features = base_features + ['RSI_9', 'RSI_21', 'RSI_30', 'price_position', 'vol_regime', 'volume_ratio']

    btc_ticker = yf.Ticker("BTC-USD")
    
    try:
        latest_btc_data = btc_ticker.history(period="120d")
        if latest_btc_data.empty:
            raise ValueError("No data fetched from yfinance")
            
        latest_btc_data.index = pd.to_datetime(latest_btc_data.index).tz_localize(None).tz_localize('UTC')
        del latest_btc_data["Dividends"]
        del latest_btc_data["Stock Splits"]
        latest_btc_data.columns = [c.lower() for c in latest_btc_data.columns]
        
        btc = _prepare_data(latest_btc_data.copy())
        
        sentiment_score = get_sentiment_score()
        btc['sentiment_score'] = sentiment_score

        # Create lagged features (UNCHANGED logic, enhanced features)
        lagged_frames = []
        for feature in enhanced_features:  # Use enhanced features
            for i in range(1, look_back + 1):
                lagged_frames.append(btc[feature].shift(i).rename(f'{feature}_lag{i}'))
        
        btc = pd.concat([btc] + lagged_frames, axis=1)
        btc = btc.dropna()

        lagged_features = [f'{feature}_lag{i}' for feature in enhanced_features for i in range(1, look_back + 1)]
        features = enhanced_features + lagged_features

    except Exception as e:
        return {"error": "Failed to fetch or prepare recent Bitcoin data for price prediction", "details": str(e)}

    if os.path.exists(PRICE_MODEL_PATH) and not force_retrain:
        model = xgb.XGBRegressor()
        model.load_model(PRICE_MODEL_PATH)
        scaler_X = joblib.load(PRICE_SCALER_X_PATH)
        mae = joblib.load(PRICE_MAE_PATH)

    else:
        try:
            full_btc_history = btc_ticker.history(period="5y")
            full_btc_history.index = pd.to_datetime(full_btc_history.index).tz_localize(None).tz_localize('UTC')
            del full_btc_history["Dividends"]
            del full_btc_history["Stock Splits"]
            full_btc_history.columns = [c.lower() for c in full_btc_history.columns]
            
            train_btc = _prepare_data(full_btc_history.copy())
            
            sentiment_score = get_sentiment_score()
            train_btc['sentiment_score'] = sentiment_score

            train_btc["tomorrow_log_return"] = np.log(train_btc["close"]).diff().shift(-1)
            train_btc = train_btc.dropna()

            # Create lagged features for training data
            lagged_frames_train = []
            for feature in enhanced_features:
                for i in range(1, look_back + 1):
                    lagged_frames_train.append(train_btc[feature].shift(i).rename(f'{feature}_lag{i}'))
            
            train_btc = pd.concat([train_btc] + lagged_frames_train, axis=1)
            train_btc = train_btc.dropna()

            train_size = int(len(train_btc) * 0.8)
            train_df = train_btc[:train_size]
            test_df = train_btc[train_size:]

            scaler_X = MinMaxScaler(feature_range=(0, 1))

            X_train = scaler_X.fit_transform(train_df[features])
            y_train = train_df["tomorrow_log_return"]
            X_test = scaler_X.transform(test_df[features])
            y_test = test_df["tomorrow_log_return"]

            # ENHANCED: Better hyperparameter tuning
            param_grid = {
                'n_estimators': [200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            }

            grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror'), 
                                     param_grid=param_grid, 
                                     scoring='neg_mean_absolute_error', 
                                     cv=3,  # Increased CV folds
                                     verbose=0)
            
            grid_search.fit(X_train, y_train, verbose=True)
            
            model = grid_search.best_estimator_

            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)

            model.save_model(PRICE_MODEL_PATH)
            joblib.dump(scaler_X, PRICE_SCALER_X_PATH)
            joblib.dump(mae, PRICE_MAE_PATH)
            print("Enhanced price prediction model, scaler, and MAE trained and saved.")

        except Exception as e:
            return {"error": "Failed to train and save the price prediction model", "details": str(e)}

    # Prepare data for next day prediction (UNCHANGED)
    last_data_points = btc[features].iloc[-1:].values
    last_data_points_scaled = scaler_X.transform(last_data_points)
    
    next_day_log_return = model.predict(last_data_points_scaled)[0]
    last_close_price = latest_btc_data['close'].iloc[-1]
    predicted_price = np.exp(next_day_log_return) * last_close_price

    last_data_date = str(latest_btc_data.index.max().date())
    prediction_date = (latest_btc_data.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    return {
        "predicted_price": float(predicted_price),
        "mae": float(mae),
        "last_data_date": last_data_date,
        "prediction_date": prediction_date,
    }
