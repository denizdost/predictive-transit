import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = BASE_DIR

print("Loading data...")
df = pd.read_csv(os.path.join(DATA_DIR, "stop_arrivals.csv"))
print(f"  Rows: {len(df)}")

CAT_COLS = ["weather_condition", "traffic_level", "stop_type", "time_bucket", "line_id"]
encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

DELAY_FEATURES = [
    "stop_sequence", "hour_of_day", "day_of_week", "is_weekend",
    "cumulative_delay_min", "speed_factor", "minutes_to_next_bus",
    "weather_condition_enc", "traffic_level_enc",
    "stop_type_enc", "time_bucket_enc", "line_id_enc",
]

CROWD_FEATURES = [
    "stop_sequence", "hour_of_day", "day_of_week", "is_weekend",
    "minutes_to_next_bus", "dwell_time_min",
    "weather_condition_enc", "traffic_level_enc",
    "stop_type_enc", "time_bucket_enc", "line_id_enc",
]

print("\nTraining Model 1: Arrival Delay...")
X1 = df[DELAY_FEATURES].fillna(0)
y1 = df["delay_min"]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
model_delay = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
model_delay.fit(X1_train, y1_train)
mae = mean_absolute_error(y1_test, model_delay.predict(X1_test))
print(f"  MAE: {mae:.2f} minutes")

print("\nTraining Model 2: Crowd Estimation...")
X2 = df[CROWD_FEATURES].fillna(0)
y2 = df["passengers_waiting"]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
model_crowd = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
model_crowd.fit(X2_train, y2_train)
rmse = np.sqrt(mean_squared_error(y2_test, model_crowd.predict(X2_test)))
print(f"  RMSE: {rmse:.2f} people")

print("\nSaving models...")
for name, obj in [("model_delay.pkl", model_delay), ("model_crowd.pkl", model_crowd),
                   ("encoders.pkl", encoders), ("features.pkl", {"delay": DELAY_FEATURES, "crowd": CROWD_FEATURES})]:
    with open(os.path.join(MODEL_DIR, name), "wb") as f:
        pickle.dump(obj, f)

print(f"\nDone! MAE: {mae:.2f} min | RMSE: {rmse:.2f} people")
