#----Section 1 Imported Important Libraries----
import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv # type: ignore


#----Section 2 Loaded API key from .env file----
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY:
    raise ValueError("OPENWEATHER_API_KEY not found. Add it to your .env file.")

#----Section 3 Defined Cities with longitude & latitude Coordinates----
cities = [

    # North India
    ("Delhi",       28.6139,  77.2090),
    ("Chandigarh",  30.7333,  76.7794),
    ("Jaipur",      26.9124,  75.7873),
    ("Lucknow",     26.8467,  80.9462),
    ("Kanpur",      26.4499,  80.3319),
    ("Agra",        27.1767,  78.0081),
    ("Varanasi",    25.3176,  82.9739),
    ("Amritsar",    31.6340,  74.8723),

    # International
    ("London",      51.5074,  -0.1278),
    ("New York",    40.7128, -74.0060),
    ("Paris",       48.8566,   2.3522),
    ("Tokyo",       35.6762, 139.6503),
    ("Sydney",     -33.8688, 151.2093),

    # Jammu & Kashmir
    ("Srinagar",    34.0837,  74.7973),
    ("Jammu",       32.7266,  74.8570),
    ("Anantnag",    33.7311,  75.1487),
    ("Baramulla",   34.1980,  74.3636),

    # West India
    ("Mumbai",      19.0760,  72.8777),
    ("Pune",        18.5204,  73.8567),
    ("Ahmedabad",   23.0225,  72.5714),
    ("Surat",       21.1702,  72.8311),
    ("Rajkot",      22.3039,  70.8022),

    # South India
    ("Bangalore",   12.9716,  77.5946),
    ("Chennai",     13.0827,  80.2707),
    ("Hyderabad",   17.3850,  78.4867),
    ("Kochi",        9.9312,  76.2673),
    ("Coimbatore",  11.0168,  76.9558),

    # East India
    ("Kolkata",     22.5726,  88.3639),
    ("Bhubaneswar", 20.2961,  85.8245),
    ("Patna",       25.5941,  85.1376),
    ("Ranchi",      23.3441,  85.3096),

    # Central India
    ("Bhopal",      23.2599,  77.4126),
    ("Indore",      22.7196,  75.8577),
    ("Nagpur",      21.1458,  79.0882),
    ("Raipur",      21.2514,  81.6296),

    # Hill / low pollution cities
    ("Shimla",      31.1048,  77.1734),
    ("Manali",      32.2396,  77.1887),
    ("Leh",         34.1526,  77.5770),
    ("Gangtok",     27.3389,  88.6065),
    ("Aizawl",      23.7271,  92.7176),
]

#----Section 4 Initialized Empty list to store all rows----
rows = []

#----Section 5 Looped through each city and fetched data----
for city, lat, lon in cities:

    print(f"Collecting data for: {city}")

    end   = int(time.time())
    start = end - (30 * 24 * 60 * 60)   # last 30 days

    # ----Section 5.1 fetched pollution history from Api----
    pollution_url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={start}&end={end}&appid={API_KEY}"
    )

    try:
        pollution_resp = requests.get(pollution_url, timeout=10)
        pollution_resp.raise_for_status()
        pollution_data = pollution_resp.json()

        if "list" not in pollution_data:
            print(f" No pollution data for {city}, skipping.")
            continue

    except Exception as e:
        print(f" Pollution fetch failed for {city}: {e}")
        continue

    # ----Section 5.2 fetched historical weather data per hour----
    # fetched weather snapshot per hour from the historical weather API so that temp/humidity/wind actually vary across the dataset.
    weather_rows = {}

    for hour_offset in range(0, 30 * 24, 6):      # every 6 hours over 30 days
        ts = start + hour_offset * 3600
        hist_url = (
            f"https://api.openweathermap.org/data/2.5/onecall/timemachine"
            f"?lat={lat}&lon={lon}&dt={ts}&appid={API_KEY}&units=metric"
        )
        try:
            w = requests.get(hist_url, timeout=10).json()
            if "hourly" in w:
                for h in w["hourly"]:
                    weather_rows[h["dt"]] = {
                        "temp_c":        h.get("temp",     20),
                        "humidity":      h.get("humidity", 50),
                        "windspeed_kph": h.get("wind_speed", 5) * 3.6,
                        "pressure_mb":   h.get("pressure", 1013),
                    }
        except Exception:
            pass    # fell through to snapshot fallback below

    #----Section 5.3 Fallback to current weather if historical fetch failed----
    fallback_weather = {"temp_c": 25, "humidity": 50, "windspeed_kph": 10, "pressure_mb": 1013}
    if not weather_rows:
        try:
            w_url = (
                f"https://api.openweathermap.org/data/2.5/weather"
                f"?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
            )
            w = requests.get(w_url, timeout=10).json()
            fallback_weather = {
                "temp_c":        w["main"]["temp"],
                "humidity":      w["main"]["humidity"],
                "windspeed_kph": w["wind"]["speed"] * 3.6,
                "pressure_mb":   w["main"]["pressure"],
            }
        except Exception as e:
            print(f" Weather fallback failed for {city}: {e}")

    # ----Section 5.4 Matched weather to each pollution timestamp and build rows----
    for item in pollution_data["list"]:
        ts = item["dt"]

        # Picked nearest weather snapshot (within ±3 h), else fallback is used
        if weather_rows:
            nearest_ts = min(weather_rows.keys(), key=lambda t: abs(t - ts))
            wx = weather_rows[nearest_ts] if abs(nearest_ts - ts) <= 10800 else fallback_weather
        else:
            wx = fallback_weather

        pm25 = item["components"]["pm2_5"]

        rows.append({
            "city":           city,
            "timestamp":      ts,
            "pm2_5":          pm25,
            "pm10":           item["components"]["pm10"],
            "no2":            item["components"]["no2"],
            "so2":            item["components"]["so2"],
            "o3":             item["components"]["o3"],
            "co":             item["components"]["co"],
            "temp_c":         wx["temp_c"],
            "humidity":       wx["humidity"],
            "windspeed_kph":  wx["windspeed_kph"],
            "pressure_mb":    wx["pressure_mb"],
            # aqi_index is NOT saved here, train.py calculated it consistently
        })

    print(f" {len(pollution_data['list'])} rows added for {city}")

#----Section 6 Saved all collected data to CSV----
df = pd.DataFrame(rows)
df.to_csv("training_data.csv", index=False)

print("\nDataset created successfully.")
print("Total rows:", len(df))
print(df.head())