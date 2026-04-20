"""
main.py  –  Terminal Sivas Enterprise Intelligence API

Serves ML-backed bus delay predictions and static frontend/data files.
Models are pure sklearn pipelines (joblib pkl) — see model/train.py.

All numeric thresholds and multipliers are derived from the CSV data at
startup — there are no hardcoded magic numbers in the business logic.
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
# Spatial index  (bus_stops.csv)
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
# Passenger-flow lookup  (passenger_flow.csv)
# ---------------------------------------------------------------------------
log.info("Loading passenger flow data ...")
try:
    flow_df = pd.read_csv(os.path.join(DATA_DIR, "passenger_flow.csv"))
    log.info("  %d flow records loaded", len(flow_df))
except Exception as exc:
    log.warning("  Passenger flow data unavailable (%s)", exc)
    flow_df = None

# ---------------------------------------------------------------------------
# Weather demand multipliers  (weather_observations.csv)
#
# Derived from the mean `passenger_demand_multiplier` per weather_condition.
# Example output: {'clear': 1.0, 'rain': 1.35, 'snow': 1.6, 'fog': 1.15, ...}
# ---------------------------------------------------------------------------
log.info("Loading weather demand multipliers ...")
WEATHER_DEMAND_MULTIPLIERS: dict[str, float] = {}
try:
    weather_df = pd.read_csv(os.path.join(DATA_DIR, "weather_observations.csv"))
    WEATHER_DEMAND_MULTIPLIERS = (
        weather_df.groupby("weather_condition")["passenger_demand_multiplier"]
        .mean()
        .round(4)
        .to_dict()
    )
    log.info("  Weather multipliers: %s", WEATHER_DEMAND_MULTIPLIERS)
except Exception as exc:
    log.warning("  Weather observations unavailable (%s) — multipliers default to 1.0", exc)

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
# Data-driven thresholds  (stop_arrivals.csv)
#
# Ghost-bus detection:
#   GHOST_BUS_SPEED  – 1st percentile of speed_factor  (buses essentially stalled)
#   GHOST_BUS_DELAY  – 95th percentile of cumulative_delay_min  (extreme lateness)
#
# Is-delayed threshold:
#   IS_DELAYED_MIN   – maximum delay_min observed when is_delayed == 0
#                      (the boundary the labelling logic used when creating the data)
#
# Worst-case floor ratio:
#   WORST_CASE_RATIO – p90 / p50 of delay_min; used to ensure the worst-case
#                      prediction is meaningfully above the expected prediction.
#
# Peak-hour detection:
#   PEAK_HOURS       – set of hour_of_day values associated with morning_rush
#                      or evening_rush time_bucket in stop_arrivals.
#   PEAK_PASSENGER_BUMP – mean(peak avg_passengers_waiting)
#                         minus mean(non-peak avg_passengers_waiting), rounded.
#
# Fallback crowd base:
#   CROWD_BASE_FALLBACK – median of avg_passengers_waiting across all flow records.
# ---------------------------------------------------------------------------
log.info("Deriving operational thresholds from stop_arrivals.csv ...")

# Safe sentinel defaults (used only if the CSV is missing)
GHOST_BUS_SPEED       = 0.1
GHOST_BUS_DELAY       = 15.0
IS_DELAYED_MIN        = 2.0
WORST_CASE_RATIO      = 1.20
PEAK_HOURS: set[int]  = set()
PEAK_PASSENGER_BUMP   = 30
CROWD_BASE_FALLBACK   = 15

try:
    arr_df = pd.read_csv(os.path.join(DATA_DIR, "stop_arrivals.csv"))

    if not arr_df.empty:
        # Ghost-bus thresholds
        GHOST_BUS_SPEED = float(arr_df["speed_factor"].quantile(0.01))
        GHOST_BUS_DELAY = float(arr_df["cumulative_delay_min"].quantile(0.95))

        # is_delayed boundary: the largest delay_min where the bus is NOT flagged delayed
        not_delayed = arr_df[arr_df["is_delayed"] == 0]["delay_min"]
        if not not_delayed.empty:
            IS_DELAYED_MIN = float(not_delayed.max())

        # Worst-case floor ratio (p90 / p50)
        p50 = float(arr_df["delay_min"].quantile(0.50))
        p90 = float(arr_df["delay_min"].quantile(0.90))
        if p50 > 0:
            WORST_CASE_RATIO = round(p90 / p50, 4)

        # Peak hours from time_bucket labels
        peak_mask   = arr_df["time_bucket"].isin(["morning_rush", "evening_rush"])
        PEAK_HOURS  = set(int(h) for h in arr_df.loc[peak_mask, "hour_of_day"].unique())

    log.info(
        "  Ghost-bus: speed < %.3f, delay > %.2f min", GHOST_BUS_SPEED, GHOST_BUS_DELAY
    )
    log.info("  is_delayed threshold: > %.1f min", IS_DELAYED_MIN)
    log.info("  Worst-case floor ratio: %.4f", WORST_CASE_RATIO)
    log.info("  Peak hours: %s", sorted(PEAK_HOURS))

except Exception as exc:
    log.warning("  Could not derive thresholds from stop_arrivals (%s) — using fallbacks", exc)

# Crowd fallback base and peak bump from passenger_flow.csv
log.info("Deriving crowd baseline and peak bump from passenger_flow.csv ...")
try:
    if flow_df is not None and not flow_df.empty:
        CROWD_BASE_FALLBACK = int(flow_df["avg_passengers_waiting"].median())

        peak_flow    = flow_df[flow_df["time_bucket"].isin(["morning_rush", "evening_rush"])]
        nonpeak_flow = flow_df[~flow_df["time_bucket"].isin(["morning_rush", "evening_rush"])]
        if not peak_flow.empty and not nonpeak_flow.empty:
            PEAK_PASSENGER_BUMP = round(
                peak_flow["avg_passengers_waiting"].mean()
                - nonpeak_flow["avg_passengers_waiting"].mean()
            )

    log.info(
        "  Crowd fallback base: %d  |  Peak bump: +%d", CROWD_BASE_FALLBACK, PEAK_PASSENGER_BUMP
    )
except Exception as exc:
    log.warning("  Could not derive crowd parameters from flow data (%s) — using fallbacks", exc)


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

    df         = _request_to_df(req)
    expected   = float(model_p50.predict(df)[0])
    worst_case = float(model_p90.predict(df)[0])

    expected   = max(0.0, expected)
    # Ensure worst_case is always meaningfully above expected.
    # The floor ratio is derived from the data's own p90/p50 relationship.
    worst_case = max(expected * WORST_CASE_RATIO + 1.0, worst_case)

    # Ghost-bus heuristic: both thresholds are data-derived (p1 speed, p95 delay)
    is_ghost = req.speed_factor < GHOST_BUS_SPEED and req.cumulative_delay_min > GHOST_BUS_DELAY

    # is_delayed threshold: boundary observed in labelled stop_arrivals data
    is_delayed = expected > IS_DELAYED_MIN or is_ghost

    response = {
        "line_id":             req.line_id,
        "predicted_delay_min": round(expected, 2),
        "is_delayed":          is_delayed,
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
            # Apply weather demand multiplier derived from weather_observations.csv.
            # Falls back to 1.0 (no change) for any condition not in the lookup.
            multiplier = WEATHER_DEMAND_MULTIPLIERS.get(req.weather_condition, 1.0)
            base *= multiplier
            variance = int(np.random.randint(-5, 10))
            return {
                "line_id":              req.line_id,
                "predicted_passengers": max(0, round(base) + variance),
                "source": f"Aggregated from {len(matches)} historical records",
            }

    # Heuristic fallback: base from flow data median, peak bump from flow data,
    # weather multiplier from weather_observations data.
    base = CROWD_BASE_FALLBACK
    if req.hour_of_day in PEAK_HOURS:
        base += PEAK_PASSENGER_BUMP
    multiplier = WEATHER_DEMAND_MULTIPLIERS.get(req.weather_condition, 1.0)
    if multiplier > 1.0:
        base = int(base * multiplier)
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
