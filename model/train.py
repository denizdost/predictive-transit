"""
train.py  –  Terminal Sivas ML pipeline
Trains two HistGradientBoosting regressors (expected delay + 90th-percentile
worst-case) and saves them as joblib pickle files.

Why not ONNX?
  skl2onnx >= 1.17 + scikit-learn >= 1.6 dropped support for converting
  HistGradientBoostingRegressor pipelines that contain a ColumnTransformer
  with mixed float/string columns.  The pure-sklearn pickle route is equally
  fast for a web API (predictions are sub-millisecond) and requires no
  additional conversion tooling.

Outputs (written to model/):
  model_delay_p50.pkl   – expected delay (MAE / L1 loss)
  model_delay_p90.pkl   – worst-case delay (90th-percentile quantile loss)
  features.json         – dynamically saves feature schema for the backend
"""

import logging
import os
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = BASE_DIR

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
log.info("Loading stop_arrivals.csv ...")
df = pd.read_csv(os.path.join(DATA_DIR, "stop_arrivals.csv"))
df = df.sort_values(by=["trip_id", "stop_sequence"]).reset_index(drop=True)
log.info("  %d rows loaded", len(df))

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
log.info("Engineering lag/rolling features ...")
df["upstream_delay_min"] = (
    df.groupby("trip_id")["cumulative_delay_min"].shift(1).fillna(0)
)
df["rolling_delay_last_2_stops"] = (
    df.groupby("trip_id")["cumulative_delay_min"]
    .transform(lambda x: x.shift(1).rolling(window=2, min_periods=1).mean().fillna(0))
)

# ---------------------------------------------------------------------------
# 3. Feature lists  (dynamically generated and exported)
# ---------------------------------------------------------------------------
NUM_FEATURES = [
    "stop_sequence",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "cumulative_delay_min",
    "speed_factor",
    "minutes_to_next_bus",
    "dwell_time_min",
    "upstream_delay_min",
    "rolling_delay_last_2_stops",
]
CAT_FEATURES = [
    "stop_type",
    "time_bucket",
    "line_id",
    "weather_condition",
    "traffic_level",
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

# Write the exact feature schema to disk so the backend can read it safely
features_path = os.path.join(MODEL_DIR, "features.json")
with open(features_path, "w") as f:
    json.dump({
        "NUM_FEATURES": NUM_FEATURES,
        "CAT_FEATURES": CAT_FEATURES,
        "ALL_FEATURES": ALL_FEATURES
    }, f)
log.info("  Feature schema saved to %s", features_path)

# ---------------------------------------------------------------------------
# 4. Preprocessor
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            CAT_FEATURES,
        ),
    ]
)

pipeline_p50 = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=300, random_state=42
    )),
])

pipeline_p90 = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", HistGradientBoostingRegressor(
        loss="quantile", quantile=0.90, max_iter=300, random_state=42
    )),
])

# ---------------------------------------------------------------------------
# 5. Train / evaluate
# ---------------------------------------------------------------------------
X = df[ALL_FEATURES]
y = df["delay_min"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

log.info("Training expected-delay model (p50 / MAE loss) ...")
pipeline_p50.fit(X_train, y_train)
log.info("  Test MAE: %.3f min", mean_absolute_error(y_test, pipeline_p50.predict(X_test)))

log.info("Training worst-case model (p90 / quantile loss) ...")
pipeline_p90.fit(X_train, y_train)
log.info("  Test MAE: %.3f min", mean_absolute_error(y_test, pipeline_p90.predict(X_test)))

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
p50_path = os.path.join(MODEL_DIR, "model_delay_p50.pkl")
p90_path = os.path.join(MODEL_DIR, "model_delay_p90.pkl")
joblib.dump(pipeline_p50, p50_path)
joblib.dump(pipeline_p90, p90_path)

log.info("Models saved:")
log.info("  p50 -> %s", p50_path)
log.info("  p90 -> %s", p90_path)
log.info("Training complete.")
