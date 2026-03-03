#-----AQI LSTM Model Training Script-----
print("Step 1: Script is running")
import pandas as pd

print("Step 2: Loading dataset...")

# Reading air quality dataset
df = pd.read_csv("data/air_quality.csv")

# Selecting only required features for prediction
FEATURES = [
    'aqi_index',
    'temp_c',
    'humidity',
    'windspeed_kph',
    'pm2_5',
    'pm10',
    'pressure_mb'
]

# Keeping only selected columns
df = df[FEATURES]

print("Selected features:")
print(df.head())

print("Shape of dataset:")
print(df.shape)

#-----Data Scaling-----

from sklearn.preprocessing import MinMaxScaler
import numpy as np

print("Step 3: Scaling data...")

# Scaling data between 0 and 1 for better LSTM performance
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

print("Data scaled successfully.")
print("First 3 scaled rows:")
print(scaled_data[:3])

#-----Creating Time Series Sequences-----

print("Step 4: Creating sequences...")

sequence_length = 24  # using past 24 hours
X = []
y = []

# Creating sliding window sequences
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i][0])  # predicting AQI (first column)

X = np.array(X)
y = np.array(y)

print("Sequences created successfully.")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

#-----Train-Test Split-----
print("Step 5: Splitting data...")

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

#-----Building LSTM Model-----
print("Step 6: Building LSTM model...")

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(50))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

print("Model built successfully.")
model.summary()

#-----Training Model-----
print("Step 7: Training model...")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

print("Model training completed.")

#-----Making Predictions-----
print("Step 8: Making predictions...")

predictions = model.predict(X_test)

print("Predictions shape:", predictions.shape)

#------Converting Back to Real AQI Values-----
print("Step 9: Converting predictions back to real AQI values...")

# Creating dummy array with same number of features
dummy = np.zeros((len(predictions), scaled_data.shape[1]))
dummy[:, 0] = predictions[:, 0]

predicted_aqi = scaler.inverse_transform(dummy)[:, 0]

# Same process for actual values
dummy_actual = np.zeros((len(y_test), scaled_data.shape[1]))
dummy_actual[:, 0] = y_test

actual_aqi = scaler.inverse_transform(dummy_actual)[:, 0]

print("First 5 Predicted AQI:", predicted_aqi[:5])
print("First 5 Actual AQI:", actual_aqi[:5])

#-----Model Evaluation-----
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(actual_aqi, predicted_aqi)
rmse = np.sqrt(mean_squared_error(actual_aqi, predicted_aqi))

print("MAE:", mae)
print("RMSE:", rmse)

#-----Saving Model and Scaler-----
print("Step 10: Saving model...")

model.save("aqi_lstm_model.h5")

print("Model saved successfully.")

import joblib

# Saving scaler for future predictions in Streamlit app
joblib.dump(scaler, "scaler.save")

print("Scaler saved successfully.")