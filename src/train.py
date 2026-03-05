# #-----AQI LSTM Model Training Script-----
# print("Step 1: Script is running")
# import pandas as pd
# import joblib
# from sklearn.metrics import mean_squared_error
# import numpy as np

# print("Step 2: Fetching historical API data...")

# import requests

# lat = 33.28
# lon = 75.34

# url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide"

# response = requests.get(url)
# data = response.json()

# hourly = data["hourly"]

# df = pd.DataFrame({
#     "pm2_5": hourly["pm2_5"],
#     "pm10": hourly["pm10"],
#     "no2": hourly["nitrogen_dioxide"],
#     "co": hourly["carbon_monoxide"],
#     "o3": hourly["ozone"],
#     "so2": hourly["sulphur_dioxide"]
# })

# # Adding environmental variables
# df["temp_c"] = 25
# df["humidity"] = 50
# df["windspeed_kph"] = 5
# df["pressure_mb"] = 1013

# # Creating AQI index (target variable)
# df["aqi_index"] = df["pm2_5"] * 1.5 + df["pm10"] * 0.5

# print("API dataset created successfully.")
# print(df.head())

# # Selecting only required features for prediction
# FEATURES = [
#     'aqi_index',
#     'temp_c',
#     'humidity',
#     'windspeed_kph',
#     'pm2_5',
#     'pm10',
#     'no2',
#     'co',
#     'o3',
#     'so2',
#     'pressure_mb'
# ]

# # Keeping only selected columns
# df = df[FEATURES]

# print("Selected features:")
# print(df.head())

# print("Shape of dataset:")
# print(df.shape)

# #-----Data Scaling-----

# from sklearn.preprocessing import MinMaxScaler
# import numpy as np

# print("Step 3: Scaling data...")

# # Scaling data between 0 and 1 for better LSTM performance
# scaler = MinMaxScaler()

# scaled_data = scaler.fit_transform(df)

# print("Data scaled successfully.")
# print("First 3 scaled rows:")
# print(scaled_data[:3])

# #-----Creating Time Series Sequences-----

# print("Step 4: Creating sequences...")

# sequence_length = 24  # using past 24 hours
# X = []
# y = []

# # Creating sliding window sequences
# for i in range(sequence_length, len(scaled_data)):
#     X.append(scaled_data[i-sequence_length:i])
#     y.append(scaled_data[i][0])  # predicting AQI (first column)

# X = np.array(X)
# y = np.array(y)

# print("Sequences created successfully.")
# print("Shape of X:", X.shape)
# print("Shape of y:", y.shape)

# #-----Train-Test Split-----
# print("Step 5: Splitting data...")

# split = int(0.8 * len(X))

# X_train = X[:split]
# X_test = X[split:]
# y_train = y[:split]
# y_test = y[split:]

# print("Training samples:", X_train.shape)
# print("Testing samples:", X_test.shape)


# #-----Linear Regression Baseline Model-----

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# print("Step 6A: Training Linear Regression...")


# # Reshaped 3D sequence data to 2D for ML model
# X_train_flat = X_train.reshape(X_train.shape[0], -1)
# X_test_flat = X_test.reshape(X_test.shape[0], -1)

# # If multi-step output, take first value
# if len(y_train.shape) > 1:
#     y_train_flat = y_train[:, 0]
#     y_test_flat = y_test[:, 0]
# else:
#     y_train_flat = y_train
#     y_test_flat = y_test

# # Training Linear Regression
# lr = LinearRegression()
# lr.fit(X_train_flat, y_train_flat)

# # Predicting
# lr_pred = lr.predict(X_test_flat)

# # Converting predictions back to real AQI
# dummy_lr = np.zeros((len(lr_pred), scaled_data.shape[1]))
# dummy_lr[:, 0] = lr_pred

# lr_pred_real = scaler.inverse_transform(dummy_lr)[:, 0]

# # Converting actual values back
# dummy_actual_lr = np.zeros((len(y_test_flat), scaled_data.shape[1]))
# dummy_actual_lr[:, 0] = y_test_flat

# actual_real_lr = scaler.inverse_transform(dummy_actual_lr)[:, 0]

# # Evaluating
# mae_lr = mean_absolute_error(actual_real_lr, lr_pred_real)
# rmse_lr = np.sqrt(mean_squared_error(actual_real_lr, lr_pred_real))
# r2_lr = r2_score(actual_real_lr, lr_pred_real)

# print("\nLinear Regression Results:")
# print("MAE:", mae_lr)
# print("RMSE:", rmse_lr)
# print("R2 Score:", r2_lr)

# joblib.dump(lr, "linear_model.pkl")
# print("Linear Regression model saved.")

# #-----Building LSTM Model-----
# print("Step 6: Building LSTM model...")

# #-----Random Forest Baseline Model-----


# from sklearn.ensemble import RandomForestRegressor

# print("Step 6B: Training Random Forest...")

# rf = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)

# rf.fit(X_train_flat, y_train_flat)

# rf_pred = rf.predict(X_test_flat)

# # Convert predictions back to real AQI
# dummy_rf = np.zeros((len(rf_pred), scaled_data.shape[1]))
# dummy_rf[:, 0] = rf_pred

# rf_pred_real = scaler.inverse_transform(dummy_rf)[:, 0]

# # Convert actual values back
# dummy_actual_rf = np.zeros((len(y_test_flat), scaled_data.shape[1]))
# dummy_actual_rf[:, 0] = y_test_flat

# actual_real_rf = scaler.inverse_transform(dummy_actual_rf)[:, 0]

# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# mae_rf = mean_absolute_error(actual_real_rf, rf_pred_real)
# rmse_rf = np.sqrt(mean_squared_error(actual_real_rf, rf_pred_real))
# r2_rf = r2_score(actual_real_rf, rf_pred_real)

# print("\nRandom Forest Results:")
# print("MAE:", mae_rf)
# print("RMSE:", rmse_rf)
# print("R2 Score:", r2_rf)

# joblib.dump(rf, "random_forest_model.pkl")
# print("Random Forest model saved.")

# from tensorflow.keras.models import Sequential # type: ignore
# from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

# model = Sequential()

# model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dropout(0.2))

# model.add(LSTM(50))
# model.add(Dropout(0.2))

# model.add(Dense(1))

# model.compile(optimizer='adam', loss='mse')

# print("Model built successfully.")
# model.summary()

# #-----Training Model-----
# print("Step 7: Training model...")

# history = model.fit(
#     X_train,
#     y_train,
#     epochs=10,
#     batch_size=32,
#     validation_data=(X_test, y_test)
# )

# print("Model training completed.")

# #-----Making Predictions-----
# print("Step 8: Making predictions...")

# predictions = model.predict(X_test)

# print("Predictions shape:", predictions.shape)

# #------Converting Back to Real AQI Values-----
# print("Step 9: Converting predictions back to real AQI values...")

# # Creating dummy array with same number of features
# dummy = np.zeros((len(predictions), scaled_data.shape[1]))
# dummy[:, 0] = predictions[:, 0]

# predicted_aqi = scaler.inverse_transform(dummy)[:, 0]

# # Same process for actual values
# dummy_actual = np.zeros((len(y_test), scaled_data.shape[1]))
# dummy_actual[:, 0] = y_test

# actual_aqi = scaler.inverse_transform(dummy_actual)[:, 0]

# print("First 5 Predicted AQI:", predicted_aqi[:5])
# print("First 5 Actual AQI:", actual_aqi[:5])

# #-----Model Evaluation-----
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import numpy as np

# mae = mean_absolute_error(actual_aqi, predicted_aqi)
# rmse = np.sqrt(mean_squared_error(actual_aqi, predicted_aqi))

# print("MAE:", mae)
# print("RMSE:", rmse)
# from sklearn.metrics import r2_score

# r2_lstm = r2_score(predicted_aqi, actual_aqi)

# print("R2 Score:", r2_lstm)
# # Visualization: Actual vs Predicted (LSTM)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 6))
# plt.plot(actual_aqi[:200], label="Actual AQI")
# plt.plot(predicted_aqi[:200], label="Predicted AQI")
# plt.title("Actual vs Predicted AQI (LSTM Model)")
# plt.xlabel("Time Steps")
# plt.ylabel("AQI")
# plt.legend()
# plt.tight_layout()

# plt.savefig("results_lstm.png")
# plt.show()

# print("Prediction graph saved as results_lstm.png")

# #-----Saving Model and Scaler-----
# print("Step 10: Saving model...")

# model.save("aqi_lstm_model.h5")

# print("Model saved successfully.")

# import joblib

# # Saving scaler for future predictions in Streamlit app
# joblib.dump(scaler, "scaler.save")

# print("Scaler saved successfully.")

# # Final Model Comparison Table

# import pandas as pd

# results = pd.DataFrame({
#     "Model": ["Linear Regression", "Random Forest", "LSTM"],
#     "MAE": [mae_lr, mae_rf, mae],
#     "RMSE": [rmse_lr, rmse_rf, rmse],
#     "R2 Score": [r2_lr, r2_rf, r2_lstm]
# })

# print("\n================ Final Model Comparison ================")
# print(results)
# print("========================================================")

# # ----- Saving Model Metrics for Streamlit -----

# metrics = {
#     "Linear Regression": rmse_lr,
#     "Random Forest": rmse_rf,
#     "LSTM": rmse
# }

# joblib.dump(metrics, "model_metrics.pkl")

# print("Model metrics saved successfully.")



# -------- Train AQI Models Using OpenWeather API Data --------

import requests
import pandas as pd
import numpy as np
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

print("Collecting pollution data from API...")

API_KEY = "YOUR_OPENWEATHER_API_KEY"

# Cities used for training
cities = [
    (28.6139, 77.2090),  # Delhi
    (19.0760, 72.8777),  # Mumbai
    (12.9716, 77.5946),  # Bangalore
    (13.0827, 80.2707),  # Chennai
    (22.5726, 88.3639),  # Kolkata
    (26.9124, 75.7873),  # Jaipur
    (33.28, 75.34)       # Kashmir
]

rows = []

for lat, lon in cities:

    end = int(time.time())
    start = end - (5 * 24 * 60 * 60)  # last 5 days

    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={API_KEY}"

    response = requests.get(url).json()

    if "list" not in response:
        continue

    for item in response["list"]:

        row = {
            "aqi_index": item["main"]["aqi"],
            "pm2_5": item["components"]["pm2_5"],
            "pm10": item["components"]["pm10"],
            "no2": item["components"]["no2"],
            "co": item["components"]["co"],
            "o3": item["components"]["o3"],
            "so2": item["components"]["so2"]
        }

        rows.append(row)

df = pd.DataFrame(rows)

print("Collected rows:", len(df))

# Convert AQI category to numeric scale
df["aqi_index"] = df["aqi_index"].map({
    1: 25,
    2: 75,
    3: 125,
    4: 175,
    5: 300
})

print("Preparing dataset...")

features = ["pm2_5","pm10","no2","co","o3","so2"]

X = df[features]
y = df["aqi_index"]

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Linear Regression...")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("Training Random Forest...")

rf_model = RandomForestRegressor(n_estimators=200)
rf_model.fit(X_train, y_train)

print("Preparing LSTM data...")

X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

print("Training LSTM model...")

lstm_model = Sequential()

lstm_model.add(LSTM(64, input_shape=(1,6)))
lstm_model.add(Dense(32, activation="relu"))
lstm_model.add(Dense(1))

lstm_model.compile(
    optimizer="adam",
    loss="mse"
)

lstm_model.fit(
    X_lstm,
    y,
    epochs=20,
    batch_size=16
)

print("Saving models...")

joblib.dump(lr_model, "linear_model.pkl")
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.save")

lstm_model.save("aqi_lstm_model.h5")

print("Models retrained successfully.")