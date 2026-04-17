import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from skl2onnx import to_onnx
from onnxconverter_common.float16 import convert_float_to_float16

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = BASE_DIR

print("Loading data...")
df = pd.read_csv(os.path.join(DATA_DIR, "stop_arrivals.csv"))
df = df.sort_values(by=['trip_id', 'stop_sequence'])

# --- 1. SPATIOTEMPORAL FEATURE ENGINEERING ---
# The "Ripple Effect": Was the bus delayed at the stop immediately before this one?
df['upstream_delay_min'] = df.groupby('trip_id')['cumulative_delay_min'].shift(1).fillna(0)

# Rolling Delay: Average delay of the last 2 stops
df['rolling_delay_last_2_stops'] = df.groupby('trip_id')['cumulative_delay_min'].transform(
    lambda x: x.shift(1).rolling(window=2, min_periods=1).mean().fillna(0)
)

# Weather x Traffic Interaction (Combining them to simplify the model)
traffic_map = {'low': 1, 'moderate': 2, 'high': 3, 'congested': 4}
df['traffic_num'] = df['traffic_level'].map(traffic_map).fillna(1)
df['is_raining_or_snowing'] = df['weather_condition'].apply(lambda x: 1 if x in ['rain', 'snow'] else 0)
df['weather_traffic_impact'] = df['is_raining_or_snowing'] * df['traffic_num']

# --- 2. PIPELINE SETUP ---
CAT_FEATURES = ["stop_type", "time_bucket", "line_id"]
NUM_FEATURES = [
    "stop_sequence", "hour_of_day", "day_of_week", "is_weekend", 
    "cumulative_delay_min", "speed_factor", "minutes_to_next_bus",
    "upstream_delay_min", "rolling_delay_last_2_stops", "weather_traffic_impact" 
]

preprocessor = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='median'), NUM_FEATURES),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), CAT_FEATURES)
])

# Model A: 50th Percentile (Expected Delay)
pipeline_50 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(loss='absolute_error', random_state=42))
])

# Model B: 90th Percentile (Worst-Case Risk)
pipeline_90 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(loss='quantile', quantile=0.90, random_state=42))
])

X = df[NUM_FEATURES + CAT_FEATURES]
y = df["delay_min"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Expected Arrival (50th Percentile)...")
pipeline_50.fit(X_train, y_train)

print("Training Worst-Case Scenario (90th Percentile)...")
pipeline_90.fit(X_train, y_train)

# --- 3. EXPORTING ---
print("\nExporting Models...")
# Save the Sklearn model for SHAP Explainability
joblib.dump(pipeline_50, os.path.join(MODEL_DIR, "model_delay_sklearn.pkl"))

# Convert to ONNX Float16 for raw speed
onx_50 = convert_float_to_float16(to_onnx(pipeline_50, X_train[:1]))
onx_90 = convert_float_to_float16(to_onnx(pipeline_90, X_train[:1]))

with open(os.path.join(MODEL_DIR, "model_delay_50_quant.onnx"), "wb") as f: f.write(onx_50.SerializeToString())
with open(os.path.join(MODEL_DIR, "model_delay_90_quant.onnx"), "wb") as f: f.write(onx_90.SerializeToString())

print("MLOps Pipeline execution complete! Models saved.")
