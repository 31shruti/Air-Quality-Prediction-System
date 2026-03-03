import numpy as np
import joblib
from tensorflow.keras.models import load_model # type: ignore

print ("Loading model and scaler...")
model = load_model("aqi_lstm_model.h5", compile=False)
scaler = joblib.load("scaler.save")

print("Model loaded successfully.")

# Example input (last 24 hours data simulation)
# You must provide 24 rows with 7 features
# Format: [aqi_index, temp_c, humidity, windspeed_kph, pm2_5, pm10, pressure_mb]

sample_input = np.array([
    [200, 8.0, 95, 5.0, 150, 160, 995]
] * 24)

# Scale input
scaled_input = scaler.transform(sample_input)

# Reshape for LSTM (1 sample, 24 time steps, 7 features)
scaled_input = np.reshape(scaled_input, (1, 24, 7))

print("Making prediction...")

prediction = model.predict(scaled_input)

# Convert prediction back
dummy = np.zeros((1, 7))
dummy[:, 0] = prediction[:, 0]

predicted_aqi = scaler.inverse_transform(dummy)[:, 0]

print("Predicted AQI:", predicted_aqi[0])