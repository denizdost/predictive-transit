import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ONNX Imports
from skl2onnx import to_onnx
from onnxconverter_common.float16 import convert_float_to_float16

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = BASE_DIR

print("Loading data...")
df = pd.read_csv(os.path.join(DATA_DIR, "stop_arrivals.csv"))
print(f"  Rows: {len(df)}")

CAT_FEATURES = ["weather_condition", "traffic_level", "stop_type", "time_bucket", "line_id"]
NUM_FEATURES_DELAY = ["stop_sequence", "hour_of_day", "day_of_week", "is_weekend", "cumulative_delay_min", "speed_factor", "minutes_to_next_bus"]
NUM_FEATURES_CROWD = ["stop_sequence", "hour_of_day", "day_of_week", "is_weekend", "minutes_to_next_bus", "dwell_time_min"]

preprocessor_delay = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="median"), NUM_FEATURES_DELAY),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES)
])

preprocessor_crowd = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="median"), NUM_FEATURES_CROWD),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES)
])

pipeline_delay = Pipeline([
    ("preprocessor", preprocessor_delay),
    ("regressor", HistGradientBoostingRegressor(max_iter=200, max_depth=5, learning_rate=0.05, random_state=42))
])

pipeline_crowd = Pipeline([
    ("preprocessor", preprocessor_crowd),
    ("regressor", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1))
])

print("\nTraining Model 1: Arrival Delay...")
X1 = df[NUM_FEATURES_DELAY + CAT_FEATURES]
y1 = df["delay_min"]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
pipeline_delay.fit(X1_train, y1_train)

print("\nTraining Model 2: Crowd Estimation...")
X2 = df[NUM_FEATURES_CROWD + CAT_FEATURES]
y2 = df["passengers_waiting"]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
pipeline_crowd.fit(X2_train, y2_train)

# --- ONNX CONVERSION & QUANTIZATION ---
print("\nConverting and Quantizing to ONNX (Float16)...")

# Convert Scikit-Learn Pipelines to ONNX graph
# We pass a tiny slice of the training data so ONNX can infer the exact input data types
onx_delay = to_onnx(pipeline_delay, X1_train[:1])
onx_crowd = to_onnx(pipeline_crowd, X2_train[:1])

# Quantize weights from Float32 to Float16 to halve the model size and boost CPU throughput
onx_delay_quant = convert_float_to_float16(onx_delay)
onx_crowd_quant = convert_float_to_float16(onx_crowd)

# Save the compiled ONNX models
with open(os.path.join(MODEL_DIR, "model_delay_quant.onnx"), "wb") as f:
    f.write(onx_delay_quant.SerializeToString())

with open(os.path.join(MODEL_DIR, "model_crowd_quant.onnx"), "wb") as f:
    f.write(onx_crowd_quant.SerializeToString())

print("✅ ONNX Models successfully compiled and quantized!")
