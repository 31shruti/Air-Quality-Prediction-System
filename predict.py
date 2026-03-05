# # ----- AQI Model Testing Script (All Models Comparison) -----

# import numpy as np
# import joblib
# from tensorflow.keras.models import load_model  # type: ignore

# print("Loading models and scaler...")

# # Load trained models
# lstm_model = load_model("aqi_lstm_model.h5", compile=False)
# lr_model = joblib.load("linear_model.pkl")
# rf_model = joblib.load("random_forest_model.pkl")
# scaler = joblib.load("scaler.save")

# print("All models loaded successfully.")

# # ----- Sample Input (Last 24 hours data simulation) -----
# # Format:
# # [aqi_index, temp_c, humidity, windspeed_kph, pm2_5, pm10, no2, co, o3, so2, pressure_mb]

# sample_input = np.array([
#     [200, 8.0, 95, 5.0, 150, 160, 40, 0.8, 30, 5, 995]
# ] * 24)

# # ----- Scaling Input -----
# scaled_input = scaler.transform(sample_input)

# # LSTM input shape: (samples, time_steps, features)
# lstm_input = scaled_input.reshape(1, 24, 11)

# # Flatten full 24-hour sequence for LR & RF
# flat_input = scaled_input.reshape(1, -1)

# print("Making predictions...")

# # -------- LSTM Prediction --------
# lstm_pred_scaled = lstm_model.predict(lstm_input)
# dummy_lstm = np.zeros((1, 11))
# dummy_lstm[:, 0] = lstm_pred_scaled[:, 0]
# lstm_pred = scaler.inverse_transform(dummy_lstm)[:, 0][0]

# # -------- Linear Regression Prediction --------
# lr_pred_scaled = lr_model.predict(flat_input)
# dummy_lr = np.zeros((1, 11))
# dummy_lr[:, 0] = lr_pred_scaled
# lr_pred = scaler.inverse_transform(dummy_lr)[:, 0][0]

# # -------- Random Forest Prediction --------
# rf_pred_scaled = rf_model.predict(flat_input)
# dummy_rf = np.zeros((1, 11))
# dummy_rf[:, 0] = rf_pred_scaled
# rf_pred = scaler.inverse_transform(dummy_rf)[:, 0][0]


# # ----- AQI Category Interpretation -----
# def aqi_category(aqi):
#     if aqi <= 50:
#         return "Good"
#     elif aqi <= 100:
#         return "Moderate"
#     elif aqi <= 150:
#         return "Unhealthy for Sensitive Groups"
#     elif aqi <= 200:
#         return "Unhealthy"
#     elif aqi <= 300:
#         return "Very Unhealthy"
#     else:
#         return "Hazardous"


# print("\n========= Prediction Comparison =========")
# print(f"Linear Regression AQI : {round(lr_pred,2)}")
# print(f"Random Forest AQI     : {round(rf_pred,2)}")
# print(f"LSTM AQI              : {round(lstm_pred,2)}")
# print("------------------------------------------")
# print(f"LSTM Category         : {aqi_category(lstm_pred)}")
# print("==========================================")