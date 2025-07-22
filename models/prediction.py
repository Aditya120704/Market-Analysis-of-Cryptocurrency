import yfinance as yf
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import xgboost as xgb

btc_ticker = yf.Ticker("BTC-USD")

if os.path.exists("bitcoin_price_predictor/btc.csv"):
    btc = pd.read_csv("bitcoin_price_predictor/btc.csv", index_col=0)
else:
    btc = btc_ticker.history(period="max")
    btc.to_csv("bitcoin_price_predictor/btc.csv")

btc.index = pd.to_datetime(pd.to_datetime(btc.index).date)

del btc["Dividends"]
del btc["Stock Splits"]

btc.columns = [c.lower() for c in btc.columns]

wiki = pd.read_csv("bitcoin_price_predictor/wikipedia_edits.csv", index_col=0, parse_dates=True)
wiki.index = wiki.index.normalize()

btc = btc.merge(wiki, left_index=True, right_index=True)

btc["tomorrow"] = btc["close"].shift(-1)

btc["target"] = (btc["tomorrow"] > btc["close"]).astype(int)

model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)

train = btc.iloc[:-200]
test = btc.iloc[-200:]

predictors = ["close", "volume", "open", "high", "low", "edit_count", "sentiment", "neg_sentiment"]
model.fit(train[predictors], train["target"])

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

print(precision_score(test["target"], preds))

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="predictions")
    combined = pd.concat([test["target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=1095, step=150):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

model = xgb.XGBClassifier(random_state=1, learning_rate=.1, n_estimators=200)
predictions = backtest(btc, model, predictors)

print("Backtest precision score:", precision_score(predictions["target"], predictions["predictions"]))

def compute_rolling(btc):
    horizons = [2,7,60,365]
    new_predictors = ["close", "sentiment", "neg_sentiment"]

    for horizon in horizons:
        rolling_averages = btc.rolling(horizon, min_periods=1).mean()

        ratio_column = f"close_ratio_{horizon}"
        btc[ratio_column] = btc["close"] / rolling_averages["close"]
        
        edit_column = f"edit_{horizon}"
        btc[edit_column] = rolling_averages["edit_count"]

        rolling = btc.rolling(horizon, closed='left', min_periods=1).mean()
        trend_column = f"trend_{horizon}"
        btc[trend_column] = rolling["target"]

        new_predictors+= [ratio_column, trend_column, edit_column]
    return btc, new_predictors

btc, new_predictors = compute_rolling(btc.copy())

predictions = backtest(btc, model, new_predictors)

# --- Final Prediction ---

# Train the model on the entire dataset
model.fit(btc[new_predictors], btc["target"])

# Predict for the next day
next_day_prediction = model.predict(btc[new_predictors].iloc[-1:])

print("----------------------------------------")
print("Bitcoin Price Prediction")
print("----------------------------------------")

# Print the last known precision score from backtesting
print(f"Backtested Precision Score: {precision_score(predictions['target'], predictions['predictions']):.4f}")

# Print the prediction
if next_day_prediction[0] == 1:
    print("Predicted outcome for tomorrow: Price will go UP")
else:
    print("Predicted outcome for tomorrow: Price will go DOWN")
print("----------------------------------------")
