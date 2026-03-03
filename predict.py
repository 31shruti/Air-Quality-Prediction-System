#-----AQI Model Testing Script-----
import numpy as np
import joblib
from tensorflow.keras.models import load_model # type: ignore

print ("Loading model and scaler...")

# Loading trained LSTM model
model = load_model("aqi_lstm_model.h5", compile=False)

# Loading saved MinMax scaler
scaler = joblib.load("scaler.save")

print("Model loaded successfully.")

#-----Creating Sample Input for Testing-----

# Example input (last 24 hours data simulation)
# You must provide 24 rows with 7 features
# Format: [aqi_index, temp_c, humidity, windspeed_kph, pm2_5, pm10, pressure_mb]

sample_input = np.array([
    [200, 8.0, 95, 5.0, 150, 160, 995]
] * 24)  # Repeating same values for 24 hours

#-----Scaling Input Data-----

# Scaling input using previously fitted scaler
scaled_input = scaler.transform(sample_input)

# Reshaping data to match LSTM expected input shape
# Shape required: (samples, time_steps, features)
scaled_input = np.reshape(scaled_input, (1, 24, 7))

print("Making prediction...")


# Predicting next hour AQI (scaled value)
prediction = model.predict(scaled_input)

#-----Converting Prediction Back to Real Value-----

# Creating dummy array for inverse scaling
dummy = np.zeros((1, 7))
dummy[:, 0] = prediction[:, 0]

# Getting actual AQI value
predicted_aqi = scaler.inverse_transform(dummy)[:, 0]
print("Predicted AQI:", predicted_aqi[0])