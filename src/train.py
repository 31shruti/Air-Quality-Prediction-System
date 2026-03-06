# ----- AQI LSTM Model Training Script -----

print("Step 1: Script is running")

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
import matplotlib.pyplot as plt

os.makedirs("models", exist_ok=True)

# ── AQI Calculation (single source of truth) ──────────────────────────────
def calculate_aqi(pm25):
    if pm25 <= 30:  return (50 / 30) * pm25
    if pm25 <= 60:  return ((100 - 51)  / (60 - 31))  * (pm25 - 31)  + 51
    if pm25 <= 90:  return ((200 - 101) / (90 - 61))  * (pm25 - 61)  + 101
    if pm25 <= 120: return ((300 - 201) / (120 - 91)) * (pm25 - 91)  + 201
    if pm25 <= 250: return ((400 - 301) / (250 - 121))* (pm25 - 121) + 301
    return 500

# ── Load & Clean Dataset ──────────────────────────────────────────────────
print("Step 2: Loading dataset")

df = pd.read_csv("training_data.csv")
df = df.dropna()
df = df[df["pm2_5"] < 200]
df = df[df["pm10"]  < 300]

# Calculate AQI here — single consistent definition for the whole pipeline
# Realistic pm2_5 caps per city type
CITY_PM25_CAPS = {
    # Hill stations
    "Shimla": 35, "Manali": 30, "Leh": 25,
    "Gangtok": 30, "Aizawl": 30,
    # International
    "London": 30, "Tokyo": 35,
    "New York": 25, "Paris": 30, "Sydney": 20,
    # Moderate Indian cities
    "Bangalore": 60, "Chennai": 65, "Kochi": 50,
    "Pune": 70, "Hyderabad": 70,
}

df["pm2_5"] = df.apply(
    lambda row: min(row["pm2_5"], CITY_PM25_CAPS.get(row["city"], 150)),
    axis=1
)
df["aqi_index"] = df["pm2_5"].apply(calculate_aqi)

print(df.groupby("city")["aqi_index"].mean().sort_values())

print("Dataset loaded. Shape:", df.shape)
print(df.head())

# ── Feature Selection ─────────────────────────────────────────────────────
FEATURES = [
    "pm2_5",
    "pm10",
    "no2",
    "so2",
    "o3",
    "co",
    "temp_c",
    "humidity",
    "windspeed_kph",
    "pressure_mb",
]
TARGET = "aqi_index"

# aqi_index is always the LAST column — every inverse_transform uses index -1
df = df[FEATURES + [TARGET]]
N_COLS = len(FEATURES) + 1   # 11  (indices 0–9 = features, 10 = aqi_index)
AQI_COL = N_COLS - 1         # 10

print("Selected features:", FEATURES)
print("Shape:", df.shape)

# ── Scale ─────────────────────────────────────────────────────────────────
print("Step 3: Scaling data...")

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

print("Scaled. First 3 rows:")
print(scaled_data[:3])

# ── Sliding-window Sequences ──────────────────────────────────────────────
print("Step 4: Creating sequences...")

SEQ_LEN = 24
X, y = [], []

for i in range(SEQ_LEN, len(scaled_data)):
    X.append(scaled_data[i - SEQ_LEN : i])          # shape (24, 11)
    y.append(scaled_data[i][AQI_COL])               # scalar — aqi_index column

X = np.array(X)   # (samples, 24, 11)
y = np.array(y)   # (samples,)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ── Train / Test Split ────────────────────────────────────────────────────
print("Step 5: Splitting data...")

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Train samples:", X_train.shape)
print("Test  samples:", X_test.shape)

# ── Shared inverse-transform helper ───────────────────────────────────────
def inverse_aqi(scaled_values):
    """
    Inverse-transform a 1-D array of scaled aqi_index values back to real AQI.
    Places values in the correct AQI_COL position.
    """
    dummy = np.zeros((len(scaled_values), N_COLS))
    dummy[:, AQI_COL] = scaled_values
    return scaler.inverse_transform(dummy)[:, AQI_COL]

# Actual AQI for the test set (computed once, reused by all models)
actual_aqi = inverse_aqi(y_test)

# ── Flatten 3-D sequences → 2-D for sklearn models ───────────────────────
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat  = X_test.reshape (X_test.shape[0],  -1)

# ── Linear Regression ─────────────────────────────────────────────────────
print("\nStep 6A: Training Linear Regression...")

lr = LinearRegression()
lr.fit(X_train_flat, y_train)
lr_pred_scaled = lr.predict(X_test_flat)

lr_pred_real = inverse_aqi(lr_pred_scaled)

mae_lr  = mean_absolute_error(actual_aqi, lr_pred_real)
rmse_lr = np.sqrt(mean_squared_error(actual_aqi, lr_pred_real))
r2_lr   = r2_score(actual_aqi, lr_pred_real)

print(f"Linear Regression — MAE: {mae_lr:.2f}  RMSE: {rmse_lr:.2f}  R²: {r2_lr:.4f}")

joblib.dump(lr, "models/linear_model.pkl")
print("Linear Regression model saved.")

# ── Random Forest ─────────────────────────────────────────────────────────
print("\nStep 6B: Training Random Forest...")

rf = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
rf.fit(X_train_flat, y_train)
rf_pred_scaled = rf.predict(X_test_flat)

rf_pred_real = inverse_aqi(rf_pred_scaled)

mae_rf  = mean_absolute_error(actual_aqi, rf_pred_real)
rmse_rf = np.sqrt(mean_squared_error(actual_aqi, rf_pred_real))
r2_rf   = r2_score(actual_aqi, rf_pred_real)

print(f"Random Forest       — MAE: {mae_rf:.2f}  RMSE: {rmse_rf:.2f}  R²: {r2_rf:.4f}")

joblib.dump(rf, "models/random_forest_model.pkl")
print("Random Forest model saved.")

# ── LSTM ──────────────────────────────────────────────────────────────────
print("\nStep 6C: Building LSTM model...")

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.summary()

print("Step 7: Training LSTM...")

history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_test, y_test),
)

# ── LSTM Predictions ──────────────────────────────────────────────────────
print("Step 8: Making predictions...")

predictions = model.predict(X_test)           # shape (samples, 1)

lstm_pred_real = inverse_aqi(predictions[:, 0])

mae_lstm  = mean_absolute_error(actual_aqi, lstm_pred_real)
rmse_lstm = np.sqrt(mean_squared_error(actual_aqi, lstm_pred_real))
r2_lstm   = r2_score(actual_aqi, lstm_pred_real)

print(f"LSTM                — MAE: {mae_lstm:.2f}  RMSE: {rmse_lstm:.2f}  R²: {r2_lstm:.4f}")

print("First 5 Predicted AQI:", lstm_pred_real[:5])
print("First 5 Actual    AQI:", actual_aqi[:5])

# ── Visualisation ─────────────────────────────────────────────────────────
plt.figure(figsize=(12, 6))
plt.plot(actual_aqi[:200],     label="Actual AQI")
plt.plot(lstm_pred_real[:200], label="Predicted AQI")
plt.title("Actual vs Predicted AQI (LSTM)")
plt.xlabel("Time Steps")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.savefig("results_lstm.png")
plt.close()
print("Graph saved as results_lstm.png")

# ── Save Models & Scaler ──────────────────────────────────────────────────
print("Step 10: Saving model and scaler...")

model.save("models/aqi_lstm_model.h5")
joblib.dump(scaler, "models/scaler.pkl")
print("LSTM model and scaler saved.")

# ── Final Comparison Table ────────────────────────────────────────────────
results = pd.DataFrame({
    "Model":   ["Linear Regression", "Random Forest", "LSTM"],
    "MAE":     [mae_lr,   mae_rf,   mae_lstm],
    "RMSE":    [rmse_lr,  rmse_rf,  rmse_lstm],
    "R2 Score":[r2_lr,    r2_rf,    r2_lstm],
})

print("\n================ Final Model Comparison ================")
print(results.to_string(index=False))
print("=========================================================")

# Save metrics for Streamlit
metrics = {
    "Linear Regression": {"MAE": mae_lr,   "RMSE": rmse_lr,   "R2": r2_lr},
    "Random Forest":     {"MAE": mae_rf,   "RMSE": rmse_rf,   "R2": r2_rf},
    "LSTM":              {"MAE": mae_lstm, "RMSE": rmse_lstm, "R2": r2_lstm},
}
joblib.dump(metrics, "models/model_metrics.pkl")
print("Model metrics saved.")

# Also save N_COLS so app.py never has to hardcode it
joblib.dump({"N_COLS": N_COLS, "AQI_COL": AQI_COL}, "models/scaler_meta.pkl")
print("Scaler meta saved.")

print("\nAQI Training Pipeline Completed Successfully 🚀")