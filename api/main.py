import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model")
DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def load_pickle(name):
    with open(os.path.join(MODEL_DIR, name), "rb") as f:
        return pickle.load(f)

model_delay = load_pickle("model_delay.pkl")
model_crowd = load_pickle("model_crowd.pkl")
encoders    = load_pickle("encoders.pkl")
features    = load_pickle("features.pkl")

stops_df = pd.read_csv(os.path.join(DATA_DIR, "bus_stops.csv"))

def encode(col, value):
    le = encoders.get(col)
    try:
        return int(le.transform([str(value)])[0])
    except:
        return 0

def crowding_label(p):
    if p <= 10:   return {"label": "Empty",    "color": "green",  "emoji": "🟢"}
    elif p <= 25: return {"label": "Light",    "color": "lime",   "emoji": "🟡"}
    elif p <= 45: return {"label": "Moderate", "color": "yellow", "emoji": "🟠"}
    elif p <= 70: return {"label": "Busy",     "color": "orange", "emoji": "🔴"}
    else:         return {"label": "Crowded",  "color": "red",    "emoji": "🔴"}

app = FastAPI(title="Predictive Transit API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ArrivalRequest(BaseModel):
    line_id: str
    stop_sequence: int
    stop_type: str
    hour_of_day: int
    day_of_week: int
    is_weekend: int
    cumulative_delay_min: float
    speed_factor: float
    minutes_to_next_bus: float
    weather_condition: str
    traffic_level: str
    time_bucket: str

class CrowdRequest(BaseModel):
    line_id: str
    stop_sequence: int
    stop_type: str
    hour_of_day: int
    day_of_week: int
    is_weekend: int
    minutes_to_next_bus: float
    dwell_time_min: Optional[float] = 1.0
    weather_condition: str
    traffic_level: str
    time_bucket: str

@app.get("/")
def health():
    return {"status": "ok", "service": "Predictive Transit API"}

@app.get("/lines")
def get_lines():
    return {"lines": stops_df[["line_id","line_name"]].drop_duplicates().to_dict("records")}

@app.get("/stops/{line_id}")
def get_stops(line_id: str):
    f = stops_df[stops_df["line_id"] == line_id].sort_values("stop_sequence")
    if f.empty:
        raise HTTPException(status_code=404, detail=f"Line {line_id} not found")
    return {"stops": f.to_dict("records")}

@app.post("/predict-arrival")
def predict_arrival(req: ArrivalRequest):
    row = {
        "stop_sequence": req.stop_sequence, "hour_of_day": req.hour_of_day,
        "day_of_week": req.day_of_week, "is_weekend": req.is_weekend,
        "cumulative_delay_min": req.cumulative_delay_min, "speed_factor": req.speed_factor,
        "minutes_to_next_bus": req.minutes_to_next_bus,
        "weather_condition_enc": encode("weather_condition", req.weather_condition),
        "traffic_level_enc": encode("traffic_level", req.traffic_level),
        "stop_type_enc": encode("stop_type", req.stop_type),
        "time_bucket_enc": encode("time_bucket", req.time_bucket),
        "line_id_enc": encode("line_id", req.line_id),
    }
    X = pd.DataFrame([row])[features["delay"]]
    delay = max(float(model_delay.predict(X)[0]), 0)
    return {
        "line_id": req.line_id,
        "predicted_delay_min": round(delay, 1),
        "status": "On time" if delay <= 2 else f"{round(delay)} min late",
        "is_delayed": delay > 2,
    }

@app.post("/predict-crowd")
def predict_crowd(req: CrowdRequest):
    row = {
        "stop_sequence": req.stop_sequence, "hour_of_day": req.hour_of_day,
        "day_of_week": req.day_of_week, "is_weekend": req.is_weekend,
        "minutes_to_next_bus": req.minutes_to_next_bus, "dwell_time_min": req.dwell_time_min,
        "weather_condition_enc": encode("weather_condition", req.weather_condition),
        "traffic_level_enc": encode("traffic_level", req.traffic_level),
        "stop_type_enc": encode("stop_type", req.stop_type),
        "time_bucket_enc": encode("time_bucket", req.time_bucket),
        "line_id_enc": encode("line_id", req.line_id),
    }
    X = pd.DataFrame([row])[features["crowd"]]
    passengers = max(int(round(float(model_crowd.predict(X)[0]))), 0)
    crowding = crowding_label(passengers)
    return {
        "line_id": req.line_id,
        "predicted_passengers": passengers,
        "crowding_level": crowding["label"],
        "crowding_color": crowding["color"],
        "crowding_emoji": crowding["emoji"],
    }
