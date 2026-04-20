"""
main.py  –  Terminal Sivas Enterprise Intelligence API

Serves ML-backed bus delay predictions and static frontend/data files.
Models are pure sklearn pipelines (joblib pkl) — see model/train.py.
"""

from __future__ import annotations

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.spatial import cKDTree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Terminal Sivas Enterprise Intelligence API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "..", "model")
DATA_DIR     = os.path.join(BASE_DIR, "..", "data")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

# ---------------------------------------------------------------------------
# Redis (optional)
# ---------------------------------------------------------------------------
log.info("Initialising Redis cache ...")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
try:
    cache = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
    cache.ping()
    log.info("  Redis connected at %s:6379", REDIS_HOST)
except Exception as exc:
    log.warning("  Redis unavailable (%s) — caching disabled", exc)
    cache = None

# ---------------------------------------------------------------------------
# Spatial index
# ---------------------------------------------------------------------------
log.info("Building spatial index ...")
try:
    stops_df      = pd.read_csv(os.path.join(DATA_DIR, "bus_stops.csv"))
    spatial_index = cKDTree(stops_df[["latitude", "longitude"]].values)
    log.info("  %d stops indexed", len(stops_df))
except Exception as exc:
    log.warning("  Spatial index failed (%s)", exc)
    spatial_index = stops_df = None

# ---------------------------------------------------------------------------
# Passenger-flow lookup (for /predict-crowd)
# ---------------------------------------------------------------------------
log.info("Loading passenger flow data ...")
try:
    flow_df = pd.read_csv(os.path.join(DATA_DIR, "passenger_flow.csv"))
    log.info("  %d flow records loaded", len(flow_df))
except Exception as exc:
    log.warning("  Passenger flow data unavailable (%s)", exc)
    flow_df = None

# ---------------------------------------------------------------------------
# ML models
# ---------------------------------------------------------------------------
log.info("Loading ML models ...")
try:
    model_p50 = joblib.load(os.path.join(MODEL_DIR, "model_delay_p50.pkl"))
    model_p90 = joblib.load(os.path.join(MODEL_DIR, "model_delay_p90.pkl"))
    log.info("  p50 and p90 models loaded")
except Exception as exc:
    log.warning("  Models not found (%s) — run model/train.py first", exc)
    model_p50 = model_p90 = None

# ---------------------------------------------------------------------------
# Feature order — safely load from train.py output or fallback
# ---------------------------------------------------------------------------
try:
    with open(os.path.join(MODEL_DIR, "features.json"), "r") as f:
        feature_schema = json.load(f)
        NUM_FEATURES = feature_schema.get("NUM_FEATURES", [])
        CAT_FEATURES = feature_schema.get("CAT_FEATURES", [])
        ALL_FEATURES = feature_schema.get("ALL_FEATURES", [])
    log.info("Feature schema loaded dynamically from features.json")
except Exception as exc:
    log.warning(f"features.json not found. Falling back to hard-coded feature lists: {exc}")
    NUM_FEATURES = [
        "stop_sequence", "hour_of_day", "day_of_week", "is_weekend",
        "cumulative_delay_min", "speed_factor", "minutes_to_next_bus",
        "dwell_time_min", "upstream_delay_min", "rolling_delay_last_2_stops",
    ]
    CAT_FEATURES = [
        "stop_type", "time_bucket", "line_id", "weather_condition", "traffic_level",
    ]
    ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

# ---------------------------------------------------------------------------
# Dynamic Ghost Bus Thresholds
# ---------------------------------------------------------------------------
GHOST_BUS_SPEED = 0.1
GHOST_BUS_DELAY = 15.0

try:
    arr_df = pd.read_csv(os.path.join(DATA_DIR, "stop_arrivals.csv"))
    if not arr_df.empty:
        GHOST_BUS_SPEED = float(arr_df['speed_factor'].quantile(0.01))
        GHOST_BUS_DELAY = float(arr_df['cumulative_delay_min'].quantile(0.95))
    log.info("Dynamic Ghost Bus thresholds set: Speed < %.2f, Delay > %.2f", GHOST_BUS_SPEED, GHOST_BUS_DELAY)
except Exception as e:
    log.warning("Could not calculate dynamic thresholds, using fallbacks: %s", e)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    line_id:                    str
    stop_sequence:              int
    stop_type:                  str
    time_bucket:                str
    hour_of_day:                int
    day_of_week:                int
    is_weekend:                 int
    cumulative_delay_min:       float
    speed_factor:               float
    minutes_to_next_bus:        float
    upstream_delay_min:         float = 0.0
    rolling_delay_last_2_stops: float = 0.0
    weather_condition:          str   = "clear"
    traffic_level:              str   = "low"
    dwell_time_min:             float = 1.0


def _request_to_df(req: PredictionRequest) -> pd.DataFrame:
    """Turn a request into a single-row DataFrame with all model features."""
    return pd.DataFrame([{
        "stop_sequence":              req.stop_sequence,
        "hour_of_day":                req.hour_of_day,
        "day_of_week":                req.day_of_week,
        "is_weekend":                 req.is_weekend,
        "cumulative_delay_min":       req.cumulative_delay_min,
        "speed_factor":               req.speed_factor,
        "minutes_to_next_bus":        req.minutes_to_next_bus,
        "dwell_time_min":             req.dwell_time_min,
        "upstream_delay_min":         req.upstream_delay_min,
        "rolling_delay_last_2_stops": req.rolling_delay_last_2_stops,
        "stop_type":                  req.stop_type,
        "time_bucket":                req.time_bucket,
        "line_id":                    req.line_id,
        "weather_condition":          req.weather_condition,
        "traffic_level":              req.traffic_level,
    }])[ALL_FEATURES]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Liveness probe for Docker / k8s."""
    return {
        "status":        "ok",
        "models_loaded": model_p50 is not None,
        "redis_ok":      cache is not None,
    }


@app.post("/predict-arrival")
def predict_arrival(req: PredictionRequest):
    if model_p50 is None:
        raise HTTPException(status_code=503, detail="Models offline — run model/train.py")

    cache_key = (
        f"delay:{req.line_id}:{req.stop_sequence}:{req.cumulative_delay_min}"
        f":{req.weather_condition}:{req.traffic_level}"
    )
    if cache:
        cached = cache.get(cache_key)
        if cached:
            return json.loads(cached)

    df       = _request_to_df(req)
    expected   = float(model_p50.predict(df)[0])
    worst_case = float(model_p90.predict(df)[0])

    expected   = max(0.0, expected)
    worst_case = max(expected * 1.20 + 1.0, worst_case)

    # Dynamic Ghost-bus heuristic 
    is_ghost = req.speed_factor < GHOST_BUS_SPEED and req.cumulative_delay_min > GHOST_BUS_DELAY

    response = {
        "line_id":             req.line_id,
        "predicted_delay_min": round(expected, 2),
        "is_delayed":          expected > 2.0 or is_ghost,
        "confidence_window":   f"{round(expected, 1)} – {round(worst_case, 1)} mins",
        "ai_reasoning":        "Analysed real-time spatiotemporal variables",
        "status": "WARNING: Ghost Bus suspected" if is_ghost else "Normal",
        "weather_source":      "Live AI Inference",
    }

    if cache:
        cache.setex(cache_key, 60, json.dumps(response))

    return response


@app.post("/predict-crowd")
def predict_crowd(req: PredictionRequest):
    """Returns predicted passenger count from historical flow data."""
    if flow_df is not None:
        mask = (
            (flow_df["line_id"]     == req.line_id) &
            (flow_df["hour_of_day"] == req.hour_of_day)
        )
        matches = flow_df[mask]
        if matches.empty:
            matches = flow_df[flow_df["line_id"] == req.line_id]

        if not matches.empty:
            base = float(matches["avg_passengers_waiting"].mean())
            if req.weather_condition == "rain":
                base *= 1.2
            elif req.weather_condition == "snow":
                base *= 1.4
            variance = int(np.random.randint(-5, 10))
            return {
                "line_id":              req.line_id,
                "predicted_passengers": max(0, round(base) + variance),
                "source": f"Aggregated from {len(matches)} historical records",
            }

    # Safe dynamic fallback instead of hardcoded 15
    base = 15
    if flow_df is not None and not flow_df.empty:
        base = int(flow_df["avg_passengers_waiting"].median())
        
    if (7 <= req.hour_of_day <= 9) or (16 <= req.hour_of_day <= 18):
        base += 35
    if req.weather_condition in ("rain", "snow"):
        base = int(base * 1.3)
    return {
        "line_id":              req.line_id,
        "predicted_passengers": max(0, base + int(np.random.randint(-5, 10))),
        "source":               "Heuristic fallback",
    }


@app.get("/stops-near-me")
def get_stops_near_me(lat: float, lon: float, radius_km: float = 0.5):
    if spatial_index is None or stops_df is None:
        raise HTTPException(status_code=500, detail="Spatial index offline")

    radius_deg = radius_km / 111.0
    indices    = spatial_index.query_ball_point([lat, lon], r=radius_deg)
    results    = (
        stops_df.iloc[indices][["stop_id", "stop_sequence", "line_id", "line_name"]]
        .to_dict(orient="records")
    )
    return {"stops": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Static mounts — must come AFTER all API routes
# ---------------------------------------------------------------------------
if os.path.exists(DATA_DIR):
    app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
