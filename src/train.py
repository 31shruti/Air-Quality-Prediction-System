# -----Section 1 Imported important libraries----
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
import matplotlib.pyplot as plt

os.makedirs("models", exist_ok=True)

#----Section 2 AQI Formula using India CPCB breakpoints----
def calculate_aqi(pm25):
    if pm25 <= 30:  return (50 / 30) * pm25
    if pm25 <= 60:  return ((100 - 51)  / (60 - 31))  * (pm25 - 31)  + 51
    if pm25 <= 90:  return ((200 - 101) / (90 - 61))  * (pm25 - 61)  + 101
    if pm25 <= 120: return ((300 - 201) / (120 - 91)) * (pm25 - 91)  + 201
    if pm25 <= 250: return ((400 - 301) / (250 - 121))* (pm25 - 121) + 301
    return 500

#----Section 3 Load dataset and remove bad values----
print("Step 2: Loading dataset")

df = pd.read_csv("training_data.csv")
df = df.dropna()
df = df[df["pm2_5"] < 200]
df = df[df["pm10"]  < 300]
df = df.sort_values("timestamp").reset_index(drop=True)

#----Section 4 Apply tier-based PM2.5 caps----
# collect_training.py already applies caps during collection
# but we re-apply here as a safety net in case old training_data.csv is used
# tier column may or may not exist depending on when data was collected

TIER_PM25_CAPS = {
    1: 150,   # Very polluted — Delhi, UP, Punjab, Haryana, Bihar
    2: 100,   # Moderately polluted — Rajasthan, Gujarat, Maharashtra etc.
    3:  70,   # Less polluted — South India, coastal, Goa
    4:  40,   # Clean — hills, high altitude, northeast
}

# city-level overrides for specific known cases
CITY_PM25_CAPS = {
    # Tier 4 hill stations
    "Shimla": 35, "Manali": 30, "Leh": 25, "Gangtok": 30, "Aizawl": 30,
    "Solan": 35,  "Mandi": 40,  "Dharamshala": 35,
    # J&K
    "Srinagar": 45, "Anantnag": 40, "Baramulla": 40, "Jammu": 55,
    # Uttarakhand
    "Haridwar": 75, "Roorkee": 75, "Dehradun": 70,
    # South India
    "Bangalore": 60, "Chennai": 65, "Kochi": 50, "Pune": 70, "Hyderabad": 70,
}

if "tier" in df.columns:
    # use tier from CSV if available (new collect_training.py)
    df["pm2_5"] = df.apply(
        lambda row: min(row["pm2_5"], TIER_PM25_CAPS.get(int(row["tier"]), 150)), axis=1
    )
    print("Applied tier-based PM2.5 caps from CSV tier column.")
else:
    # fallback: use city-level caps only (old collect_training.py)
    print("No tier column found — using city-level caps only.")

if "city" in df.columns:
    df["pm2_5"] = df.apply(
        lambda row: min(row["pm2_5"], CITY_PM25_CAPS.get(row["city"], row["pm2_5"])), axis=1
    )

df["aqi_index"] = df["pm2_5"].apply(calculate_aqi)

# save copy with city/state/tier before dropping those columns
df_original = df.copy()

print("Dataset loaded. Shape:", df.shape)
if "state" in df.columns:
    print("Cities per state:")
    print(df.groupby(["state", "tier"])["city"].nunique().to_string())
print(df.head())

#----Section 5 Select features and target----
FEATURES = [
    "pm2_5", "pm10", "no2", "so2", "o3", "co",
    "temp_c", "humidity", "windspeed_kph", "pressure_mb",
]
TARGET = "aqi_index"

# aqi_index kept as last column (index 10) — required for inverse_aqi()
df = df[FEATURES + [TARGET]]
N_COLS = len(FEATURES) + 1   # 11
AQI_COL = N_COLS - 1         # 10

print("Selected features:", FEATURES)
print("Shape:", df.shape)

#----Section 6 Scale all values between 0 and 1----
print("Step 3: Scaling data...")

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

print("Scaled. First 3 rows:")
print(scaled_data[:3])

#----Section 7 Create 24-hour sliding window sequences per city----
# sequences created per city separately to avoid cross-city contamination
# e.g. Delhi hour 720 -> Shimla hour 1 would be a nonsense sequence
print("Step 4: Creating sequences...")

SEQ_LEN = 24
X, y = [], []

for city_name in df_original["city"].unique():
    city_mask   = df_original["city"].values == city_name
    city_scaled = scaled_data[city_mask]

    if len(city_scaled) < SEQ_LEN + 1:
        print(f"  Skipping {city_name} — not enough rows ({len(city_scaled)})")
        continue

    for i in range(SEQ_LEN, len(city_scaled)):
        X.append(city_scaled[i - SEQ_LEN : i])
        y.append(city_scaled[i][AQI_COL])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

#----Section 8 Split into train and test sets----
print("Step 5: Splitting data...")

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Train samples:", X_train.shape)
print("Test  samples:", X_test.shape)

#----Section 9 Helper to convert scaled AQI back to real values----
def inverse_aqi(scaled_values):
    dummy = np.zeros((len(scaled_values), N_COLS))
    dummy[:, AQI_COL] = scaled_values
    return scaler.inverse_transform(dummy)[:, AQI_COL]

actual_aqi = inverse_aqi(y_test)

#----Section 10 Persistence baseline----
persistence_pred   = actual_aqi[:-1]
persistence_actual = actual_aqi[1:]
mae_p  = mean_absolute_error(persistence_actual, persistence_pred)
rmse_p = np.sqrt(mean_squared_error(persistence_actual, persistence_pred))
r2_p   = r2_score(persistence_actual, persistence_pred)
print(f"Persistence Baseline — MAE: {mae_p:.2f}  RMSE: {rmse_p:.2f}  R²: {r2_p:.4f}")

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0],  -1)

#----Section 11 Linear Regression----
print("\nStep 6A: Training Linear Regression...")

lr = LinearRegression()
lr.fit(X_train_flat, y_train)
lr_pred_real = inverse_aqi(lr.predict(X_test_flat))

mae_lr  = mean_absolute_error(actual_aqi, lr_pred_real)
rmse_lr = np.sqrt(mean_squared_error(actual_aqi, lr_pred_real))
r2_lr   = r2_score(actual_aqi, lr_pred_real)

print(f"Linear Regression — MAE: {mae_lr:.2f}  RMSE: {rmse_lr:.2f}  R²: {r2_lr:.4f}")
joblib.dump(lr, "models/linear_model.pkl")
print("Linear Regression model saved.")

#----Section 12 Random Forest----
# n_estimators reduced from 30 to 10 and max_depth=10 added
# this keeps file size under 10MB (was 100MB+ with n_estimators=30)
# accuracy loss is minimal — R2 stays above 0.97
print("\nStep 6B: Training Random Forest...")

rf = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_flat, y_train)
rf_pred_real = inverse_aqi(rf.predict(X_test_flat))

mae_rf  = mean_absolute_error(actual_aqi, rf_pred_real)
rmse_rf = np.sqrt(mean_squared_error(actual_aqi, rf_pred_real))
r2_rf   = r2_score(actual_aqi, rf_pred_real)

print(f"Random Forest — MAE: {mae_rf:.2f}  RMSE: {rmse_rf:.2f}  R²: {r2_rf:.4f}")
joblib.dump(rf, "models/random_forest_model.pkl")
print("Random Forest model saved.")

#----Section 13 LSTM model----
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

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
)

#----Section 14 LSTM predictions----
print("Step 8: Making predictions...")

predictions    = model.predict(X_test)
lstm_pred_real = inverse_aqi(predictions[:, 0])

mae_lstm  = mean_absolute_error(actual_aqi, lstm_pred_real)
rmse_lstm = np.sqrt(mean_squared_error(actual_aqi, lstm_pred_real))
r2_lstm   = r2_score(actual_aqi, lstm_pred_real)

print(f"LSTM — MAE: {mae_lstm:.2f}  RMSE: {rmse_lstm:.2f}  R²: {r2_lstm:.4f}")
print("First 5 Predicted AQI:", lstm_pred_real[:5])
print("First 5 Actual    AQI:", actual_aqi[:5])

#----Section 15 Plot actual vs predicted----
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

#----Section 16 Save models and scaler----
print("Step 10: Saving model and scaler...")

model.save("models/aqi_lstm_model.h5")
joblib.dump(scaler, "models/scaler.pkl")
print("LSTM model and scaler saved.")

#----Section 17 Final comparison table----
results = pd.DataFrame({
    "Model":    ["Persistence Baseline", "Linear Regression", "Random Forest", "LSTM"],
    "MAE":      [mae_p,    mae_lr,   mae_rf,   mae_lstm],
    "RMSE":     [rmse_p,   rmse_lr,  rmse_rf,  rmse_lstm],
    "R2 Score": [r2_p,     r2_lr,    r2_rf,    r2_lstm],
})

print("\n================ Final Model Comparison ================")
print(results.to_string(index=False))
print("=========================================================")

#----Section 18 Save metrics and scaler meta for app.py----
metrics = {
    "Persistence":       {"MAE": mae_p,    "RMSE": rmse_p,   "R2": r2_p},
    "Linear Regression": {"MAE": mae_lr,   "RMSE": rmse_lr,  "R2": r2_lr},
    "Random Forest":     {"MAE": mae_rf,   "RMSE": rmse_rf,  "R2": r2_rf},
    "LSTM":              {"MAE": mae_lstm, "RMSE": rmse_lstm, "R2": r2_lstm},
}
joblib.dump(metrics, "models/model_metrics.pkl")

# save state/tier city mapping so app.py can do nearest-neighbour lookups
city_state_map = {}
if "state" in df_original.columns and "tier" in df_original.columns:
    for _, row in df_original[["city","state","tier"]].drop_duplicates().iterrows():
        city_state_map[row["city"]] = {"state": row["state"], "tier": int(row["tier"])}
joblib.dump(city_state_map, "models/city_state_map.pkl")
print("City-state map saved.")

joblib.dump({"N_COLS": N_COLS, "AQI_COL": AQI_COL}, "models/scaler_meta.pkl")
print("Scaler meta saved.")
print("Model metrics saved.")

print("\nAQI Training Pipeline Completed Successfully")