from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
import pandas as pd
import numpy as np
import os
import json
import redis
import joblib
import shap
from scipy.spatial import cKDTree
from circuitbreaker import circuit

app = FastAPI(title="Terminal Sivas Enterprise Intelligence API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# --- INITIALIZATION ---
print("Initializing Redis Cache...")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
try:
    cache = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
    cache.ping()
except:
    cache = None

print("Loading Spatial Index...")
try:
    stops_df = pd.read_csv(os.path.join(DATA_DIR, "bus_stops.csv"))
    spatial_index = cKDTree(stops_df[['latitude', 'longitude']].values)
except:
    spatial_index, stops_df = None, None

print("Loading ML Assets...")
sess_options = ort.SessionOptions()
sess_50 = ort.InferenceSession(os.path.join(MODEL_DIR, "model_delay_50_quant.onnx"), sess_options)
sess_90 = ort.InferenceSession(os.path.join(MODEL_DIR, "model_delay_90_quant.onnx"), sess_options)
sklearn_model = joblib.load(os.path.join(MODEL_DIR, "model_delay_sklearn.pkl"))

# --- REQUEST MODEL ---
class PredictionRequest(BaseModel):
    line_id: str
    stop_sequence: int
    stop_type: str
    time_bucket: str
    hour_of_day: int
    day_of_week: int
    is_weekend: int
    cumulative_delay_min: float
    speed_factor: float
    minutes_to_next_bus: float
    upstream_delay_min: float = 0.0
    rolling_delay_last_2_stops: float = 0.0

# --- CIRCUIT BREAKER ---
@circuit(failure_threshold=3, recovery_timeout=30)
def fetch_weather_api():
    # Simulated external call
    return {"impact": 1.5, "status": "Live Data"}

def get_weather():
    try: return fetch_weather_api()
    except Exception: return {"impact": 1.0, "status": "Historical Fallback"}

# --- HELPER ---
def build_onnx_inputs(sess, df):
    inputs = {}
    for inp in sess.get_inputs():
        if inp.name not in df.columns: continue
        val = df[inp.name].values
        if inp.type == 'tensor(float)': val = val.astype(np.float32)
        elif inp.type == 'tensor(int64)': val = val.astype(np.int64)
        elif inp.type == 'tensor(string)': val = val.astype(str).astype(object)
        inputs[inp.name] = val.reshape(-1, 1) if len(val.shape) == 1 else val
    return inputs

# --- ENDPOINTS ---
@app.post("/predict-arrival")
def predict_arrival(req: PredictionRequest):
    # 1. Check Redis Cache
    cache_key = f"delay:{req.line_id}:{req.stop_sequence}:{req.cumulative_delay_min}"
    if cache and cache.get(cache_key): return json.loads(cache.get(cache_key))

    # 2. Compile Data
    weather = get_weather()
    df = pd.DataFrame([req.dict()])
    df['weather_traffic_impact'] = weather['impact']
    
    # 3. ONNX Inference (Quantile Regression)
    onnx_inputs = build_onnx_inputs(sess_50, df)
    expected = float(sess_50.run(None, onnx_inputs)[0][0, 0])
    worst_case = float(sess_90.run(None, onnx_inputs)[0][0, 0])
    
    # 4. SHAP Explainable AI (The Transparent Oracle)
    X_trans = sklearn_model.named_steps['preprocessor'].transform(df)
    explainer = shap.TreeExplainer(sklearn_model.named_steps['regressor'])
    shap_vals = explainer.shap_values(X_trans)
    feat_names = sklearn_model.named_steps['preprocessor'].get_feature_names_out()
    
    contributions = sorted(list(zip(feat_names, shap_vals[0])), key=lambda x: abs(x[1]), reverse=True)
    top_factors = [f"{f.split('__')[-1]}: {round(v,1)}m" for f, v in contributions[:2] if abs(v) > 0.5]
    
    # 5. Ghost Bus Anomaly Detection
    is_ghost = (req.speed_factor < 0.1 and req.cumulative_delay_min > 15.0)

    response = {
        "line_id": req.line_id,
        "confidence_window": f"{max(0, round(expected, 1))} - {max(0, round(worst_case, 1))} mins",
        "ai_reasoning": f"Major factors: {', '.join(top_factors)}" if top_factors else "Standard traffic flow.",
        "status": "⚠️ ANOMALY: Ghost Bus suspected" if is_ghost else "Normal",
        "weather_source": weather['status']
    }

    if cache: cache.setex(cache_key, 30, json.dumps(response))
    return response

@app.get("/stops-near-me")
def get_stops_near_me(lat: float, lon: float, radius_km: float = 0.5):
    if spatial_index is None: raise HTTPException(status_code=500, detail="Index offline")
    radius_deg = radius_km / 111.0
    indices = spatial_index.query_ball_point([lat, lon], r=radius_deg)
    return {"stops": stops_df.iloc[indices][['stop_id', 'line_name']].to_dict(orient='records')}
