from prediction_model import get_prediction_direction, get_price_prediction_regression

print("Forcing retraining of direction prediction model...")
get_prediction_direction(force_retrain=True)

print("\nForcing retraining of price prediction model...")
get_price_prediction_regression(force_retrain=True)
