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

#----Section 3 Defined Cities grouped by State with coordinates----
# Tier 1 = Very polluted (Indo-Gangetic Plain)
# Tier 2 = Moderately polluted
# Tier 3 = Less polluted (South India / coastal)
# Tier 4 = Clean (hills, northeast, high altitude)

STATE_CITIES = {

    "Delhi": {
        "tier": 1,
        "cities": [
            ("Delhi",           28.6139,  77.2090),
        ]
    },

    "Uttar Pradesh": {
        "tier": 1,
        "cities": [
            ("Lucknow",         26.8467,  80.9462),
            ("Kanpur",          26.4499,  80.3319),
            ("Agra",            27.1767,  78.0081),
            ("Varanasi",        25.3176,  82.9739),
            ("Meerut",          28.9845,  77.7064),
            ("Ghaziabad",       28.6692,  77.4538),
            ("Allahabad",       25.4358,  81.8463),
            ("Bareilly",        28.3670,  79.4304),
            ("Moradabad",       28.8386,  78.7733),
            ("Gorakhpur",       26.7606,  83.3732),
            ("Mathura",         27.4924,  77.6737),
        ]
    },

    "Punjab": {
        "tier": 1,
        "cities": [
            ("Amritsar",        31.6340,  74.8723),
            ("Ludhiana",        30.9010,  75.8573),
            ("Jalandhar",       31.3260,  75.5762),
            ("Patiala",         30.3398,  76.3869),
        ]
    },

    "Haryana": {
        "tier": 1,
        "cities": [
            ("Gurugram",        28.4595,  77.0266),
            ("Faridabad",       28.4089,  77.3178),
            ("Ambala",          30.3782,  76.7767),
            ("Hisar",           29.1492,  75.7217),
        ]
    },

    "Bihar": {
        "tier": 1,
        "cities": [
            ("Patna",           25.5941,  85.1376),
            ("Gaya",            24.7955,  85.0002),
            ("Muzaffarpur",     26.1209,  85.3647),
            ("Bhagalpur",       25.2425,  86.9842),
        ]
    },

    "Chandigarh": {
        "tier": 2,
        "cities": [
            ("Chandigarh",      30.7333,  76.7794),
        ]
    },

    "Rajasthan": {
        "tier": 2,
        "cities": [
            ("Jaipur",          26.9124,  75.7873),
            ("Jodhpur",         26.2389,  73.0243),
            ("Udaipur",         24.5854,  73.7125),
            ("Kota",            25.2138,  75.8648),
            ("Ajmer",           26.4499,  74.6399),
            ("Bikaner",         28.0229,  73.3119),
        ]
    },

    "Madhya Pradesh": {
        "tier": 2,
        "cities": [
            ("Bhopal",          23.2599,  77.4126),
            ("Indore",          22.7196,  75.8577),
            ("Gwalior",         26.2183,  78.1828),
            ("Jabalpur",        23.1815,  79.9864),
            ("Ujjain",          23.1765,  75.7885),
        ]
    },

    "Chhattisgarh": {
        "tier": 2,
        "cities": [
            ("Raipur",          21.2514,  81.6296),
        ]
    },

    "Gujarat": {
        "tier": 2,
        "cities": [
            ("Ahmedabad",       23.0225,  72.5714),
            ("Surat",           21.1702,  72.8311),
            ("Rajkot",          22.3039,  70.8022),
            ("Vadodara",        22.3072,  73.1812),
            ("Bhavnagar",       21.7645,  72.1519),
            ("Gandhinagar",     23.2156,  72.6369),
            ("Jamnagar",        22.4707,  70.0577),
        ]
    },

    "Maharashtra": {
        "tier": 2,
        "cities": [
            ("Mumbai",          19.0760,  72.8777),
            ("Pune",            18.5204,  73.8567),
            ("Nashik",          19.9975,  73.7898),
            ("Aurangabad",      19.8762,  75.3433),
            ("Nagpur",          21.1458,  79.0882),
            ("Solapur",         17.6599,  75.9064),
            ("Kolhapur",        16.7050,  74.2433),
            ("Thane",           19.2183,  72.9781),
        ]
    },

    "West Bengal": {
        "tier": 2,
        "cities": [
            ("Kolkata",         22.5726,  88.3639),
            ("Howrah",          22.5958,  88.2636),
            ("Siliguri",        26.7271,  88.3953),
            ("Durgapur",        23.5204,  87.3119),
            ("Asansol",         23.6739,  86.9524),
        ]
    },

    "Jharkhand": {
        "tier": 2,
        "cities": [
            ("Ranchi",          23.3441,  85.3096),
            ("Jamshedpur",      22.8046,  86.2029),
            ("Dhanbad",         23.7957,  86.4304),
            ("Bokaro",          23.6693,  86.1511),
        ]
    },

    "Odisha": {
        "tier": 2,
        "cities": [
            ("Bhubaneswar",     20.2961,  85.8245),
            ("Cuttack",         20.4625,  85.8830),
            ("Rourkela",        22.2604,  84.8536),
            ("Sambalpur",       21.4669,  83.9756),
        ]
    },

    "Telangana": {
        "tier": 2,
        "cities": [
            ("Hyderabad",       17.3850,  78.4867),
            ("Warangal",        17.9784,  79.5941),
            ("Nizamabad",       18.6725,  78.0941),
        ]
    },

    "Andhra Pradesh": {
        "tier": 2,
        "cities": [
            ("Visakhapatnam",   17.6868,  83.2185),
            ("Vijayawada",      16.5062,  80.6480),
            ("Tirupati",        13.6288,  79.4192),
            ("Guntur",          16.3067,  80.4365),
            ("Nellore",         14.4426,  79.9865),
        ]
    },

    "Karnataka": {
        "tier": 3,
        "cities": [
            ("Bangalore",       12.9716,  77.5946),
            ("Mysore",          12.2958,  76.6394),
            ("Mangalore",       12.9141,  74.8560),
            ("Hubli",           15.3647,  75.1240),
            ("Belgaum",         15.8497,  74.4977),
        ]
    },

    "Tamil Nadu": {
        "tier": 3,
        "cities": [
            ("Chennai",         13.0827,  80.2707),
            ("Coimbatore",      11.0168,  76.9558),
            ("Madurai",          9.9252,  78.1198),
            ("Tiruchirappalli", 10.7905,  78.7047),
            ("Salem",           11.6643,  78.1460),
            ("Tirunelveli",      8.7139,  77.7567),
        ]
    },

    "Kerala": {
        "tier": 3,
        "cities": [
            ("Kochi",            9.9312,  76.2673),
            ("Thiruvananthapuram", 8.5241, 76.9366),
            ("Kozhikode",       11.2588,  75.7804),
            ("Thrissur",        10.5276,  76.2144),
            ("Kollam",           8.8932,  76.6141),
        ]
    },

    "Goa": {
        "tier": 3,
        "cities": [
            ("Panaji",          15.4909,  73.8278),
            ("Margao",          15.2993,  73.9862),
        ]
    },

    "Uttarakhand": {
        "tier": 3,
        "cities": [
            ("Dehradun",        30.3165,  78.0322),
            ("Haridwar",        29.9457,  78.1642),
            ("Roorkee",         29.8543,  77.8880),
        ]
    },

    "Himachal Pradesh": {
        "tier": 4,
        "cities": [
            ("Shimla",          31.1048,  77.1734),
            ("Manali",          32.2396,  77.1887),
            ("Dharamshala",     32.2190,  76.3234),
            ("Solan",           30.9045,  77.0967),
            ("Mandi",           31.7080,  76.9318),
        ]
    },

    "Jammu and Kashmir": {
        "tier": 4,
        "cities": [
            ("Srinagar",        34.0837,  74.7973),
            ("Jammu",           32.7266,  74.8570),
            ("Anantnag",        33.7311,  75.1487),
            ("Baramulla",       34.1980,  74.3636),
        ]
    },

    "Ladakh": {
        "tier": 4,
        "cities": [
            ("Leh",             34.1526,  77.5770),
        ]
    },

    "Northeast": {
        "tier": 4,
        "cities": [
            ("Guwahati",        26.1445,  91.7362),
            ("Shillong",        25.5788,  91.8933),
            ("Imphal",          24.8170,  93.9368),
            ("Agartala",        23.8315,  91.2868),
            ("Dibrugarh",       27.4728,  94.9120),
            ("Silchar",         24.8333,  92.7789),
            ("Gangtok",         27.3389,  88.6065),
            ("Aizawl",          23.7271,  92.7176),
        ]
    },

}

#----Section 3b PM2.5 caps per tier----
TIER_PM25_CAPS = {
    1: 150,   # Very polluted
    2: 100,   # Moderately polluted
    3:  70,   # Less polluted
    4:  40,   # Clean / hill stations
}

#----Section 4 Flatten city list----
cities_flat = []
for state, info in STATE_CITIES.items():
    tier    = info["tier"]
    pm25cap = TIER_PM25_CAPS[tier]
    for city, lat, lon in info["cities"]:
        cities_flat.append((city, lat, lon, state, tier, pm25cap))

print(f"Total cities to collect: {len(cities_flat)}")

#----Section 5 Initialize rows list----
rows = []

#----Section 6 Loop through each city and fetch data----
for city, lat, lon, state, tier, pm25_cap in cities_flat:

    print(f"[{state} | Tier {tier}] Collecting: {city}")

    end   = int(time.time())
    start = end - (30 * 24 * 60 * 60)

    #----Section 6.1 Fetch pollution history----
    pollution_url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={start}&end={end}&appid={API_KEY}"
    )

    try:
        pollution_resp = requests.get(pollution_url, timeout=60)
        pollution_resp.raise_for_status()
        pollution_data = pollution_resp.json()

        if "list" not in pollution_data:
            print(f"  No pollution data for {city}, skipping.")
            continue

    except Exception as e:
        print(f"  Pollution fetch failed for {city}: {e}")
        continue

    #----Section 6.2 Fetch weather using free forecast endpoint----
    weather_rows = {}
    try:
        w_url = (
            f"https://api.openweathermap.org/data/2.5/forecast"
            f"?lat={lat}&lon={lon}&appid={API_KEY}&units=metric&cnt=40"
        )
        w = requests.get(w_url, timeout=30).json()
        if "list" in w:
            for h in w["list"]:
                weather_rows[h["dt"]] = {
                    "temp_c":        h["main"]["temp"],
                    "humidity":      h["main"]["humidity"],
                    "windspeed_kph": h["wind"]["speed"] * 3.6,
                    "pressure_mb":   h["main"]["pressure"],
                }
    except Exception:
        pass

    #----Section 6.3 Fallback to current weather snapshot----
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
            print(f"  Weather fallback failed for {city}: {e}")

    #----Section 6.4 Match weather to pollution and build rows----
    for item in pollution_data["list"]:
        ts   = item["dt"]
        pm25 = min(item["components"]["pm2_5"], pm25_cap)   # apply tier cap here

        if weather_rows:
            nearest_ts = min(weather_rows.keys(), key=lambda t: abs(t - ts))
            wx = weather_rows[nearest_ts] if abs(nearest_ts - ts) <= 10800 else fallback_weather
        else:
            wx = fallback_weather

        rows.append({
            "city":          city,
            "state":         state,
            "tier":          tier,
            "timestamp":     ts,
            "pm2_5":         pm25,
            "pm10":          item["components"]["pm10"],
            "no2":           item["components"]["no2"],
            "so2":           item["components"]["so2"],
            "o3":            item["components"]["o3"],
            "co":            item["components"]["co"],
            "temp_c":        wx["temp_c"],
            "humidity":      wx["humidity"],
            "windspeed_kph": wx["windspeed_kph"],
            "pressure_mb":   wx["pressure_mb"],
        })

    print(f"  {len(pollution_data['list'])} rows added for {city}")
    time.sleep(1)

#----Section 7 Save to CSV----
df = pd.DataFrame(rows)
df.to_csv("training_data.csv", index=False)

print("\nDataset created successfully.")
print(f"Total rows   : {len(df)}")
print(f"Cities saved : {df['city'].nunique()}")
print(f"States saved : {df['state'].nunique()}")
print("\nRows per state:")
print(df.groupby(["state","tier"])["city"].nunique().sort_values(ascending=False).to_string())
print("\nSample:")
print(df.head())