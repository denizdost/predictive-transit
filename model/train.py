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

print("Loading data from CSV...")
df = pd.read_csv(os.path.join(DATA_DIR, "stop_arrivals.csv"))
df = df.sort_values(by=['trip_id', 'stop_sequence'])

# --- 1. SPATIOTEMPORAL FEATURE ENGINEERING ---
# (Purely data-driven pandas shifts, no hardcoded mappings)
df['upstream_delay_min'] = df.groupby('trip_id')['cumulative_delay_min'].shift(1).fillna(0)
df['rolling_delay_last_2_stops'] = df.groupby('trip_id')['cumulative_delay_min'].transform(
    lambda x: x.shift(1).rolling(window=2, min_periods=1).mean().fillna(0)
)

# --- 2. PIPELINE SETUP ---
# Notice: weather_condition and traffic_level are now dynamically handled by the pipeline
CAT_FEATURES = ["stop_type", "time_bucket", "line_id", "weather_condition", "traffic_level"]
NUM_FEATURES = [
    "stop_sequence", "hour_of_day", "day_of_week", "is_weekend", 
    "cumulative_delay_min", "speed_factor", "minutes_to_next_bus",
    "upstream_delay_min", "rolling_delay_last_2_stops"
]

preprocessor = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='median'), NUM_FEATURES),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), CAT_FEATURES)
])

# Model A: Expected Delay
pipeline_50 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(loss='absolute_error', random_state=42))
])

# Model B: Worst-Case Risk
pipeline_90 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(loss='quantile', quantile=0.90, random_state=42))
])

X = df[NUM_FEATURES + CAT_FEATURES]
y = df["delay_min"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Models strictly from CSV data...")
pipeline_50.fit(X_train, y_train)
pipeline_90.fit(X_train, y_train)

# --- 3. EXPORTING ---
joblib.dump(pipeline_50, os.path.join(MODEL_DIR, "model_delay_sklearn.pkl"))

onx_50 = convert_float_to_float16(to_onnx(pipeline_50, X_train[:1]))
onx_90 = convert_float_to_float16(to_onnx(pipeline_90, X_train[:1]))

with open(os.path.join(MODEL_DIR, "model_delay_50_quant.onnx"), "wb") as f: f.write(onx_50.SerializeToString())
with open(os.path.join(MODEL_DIR, "model_delay_90_quant.onnx"), "wb") as f: f.write(onx_90.SerializeToString())

print("MLOps Pipeline execution complete! Models saved.")
