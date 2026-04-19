from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

app = FastAPI(title="Terminal Sivas Enterprise Intelligence API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

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
try:
    sess_50 = ort.InferenceSession(os.path.join(MODEL_DIR, "model_delay_50_quant.onnx"), sess_options)
    sess_90 = ort.InferenceSession(os.path.join(MODEL_DIR, "model_delay_90_quant.onnx"), sess_options)
    sklearn_model = joblib.load(os.path.join(MODEL_DIR, "model_delay_sklearn.pkl"))
except Exception as e:
    print(f"Warning: ML Models not found or failed to load. Run train.py first. Error: {e}")
    sess_50, sess_90, sklearn_model = None, None, None

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
    # Added fields from the frontend
    weather_condition: str = "clear"
    traffic_level: str = "low"
    dwell_time_min: float = 1.0

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
    if not sess_50:
        raise HTTPException(status_code=503, detail="ML Models are offline.")

    # 1. Check Redis Cache
    cache_key = f"delay:{req.line_id}:{req.stop_sequence}:{req.cumulative_delay_min}:{req.weather_condition}:{req.traffic_level}"
    if cache and cache.get(cache_key): return json.loads(cache.get(cache_key))

    # 2. Compile Data & Feature Engineering
    df = pd.DataFrame([req.dict()])
    
    # Calculate the weather/traffic interaction exactly as trained
    traffic_map = {'low': 1, 'moderate': 2, 'high': 3, 'congested': 4}
    traffic_num = traffic_map.get(req.traffic_level, 1)
    is_raining_or_snowing = 1 if req.weather_condition in ['rain', 'snow'] else 0
    df['weather_traffic_impact'] = is_raining_or_snowing * traffic_num
    
    # 3. ONNX Inference (Quantile Regression)
    onnx_inputs = build_onnx_inputs(sess_50, df)
    expected = float(sess_50.run(None, onnx_inputs)[0][0, 0])
    worst_case = float(sess_90.run(None, onnx_inputs)[0][0, 0])
    
    # 4. SHAP Explainable AI
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
        "predicted_delay_min": expected,
        "is_delayed": expected > 2.0 or is_ghost,
        "confidence_window": f"{max(0, round(expected, 1))} - {max(0, round(worst_case, 1))} mins",
        "ai_reasoning": f"Major factors: {', '.join(top_factors)}" if top_factors else "Standard traffic flow.",
        "status": "⚠️ ANOMALY: Ghost Bus suspected" if is_ghost else "Normal",
        "weather_source": "Live Model Inference"
    }

    if cache: cache.setex(cache_key, 30, json.dumps(response))
    return response

@app.post("/predict-crowd")
def predict_crowd(req: PredictionRequest):
    """New endpoint to satisfy the frontend's concurrent Promise.all request"""
    base_crowd = 15
    
    # Rush hour logic
    if (7 <= req.hour_of_day <= 9) or (16 <= req.hour_of_day <= 18):
        base_crowd += 35
        
    # Weather and traffic multipliers
    if req.weather_condition in ["rain", "snow"]:
        base_crowd = int(base_crowd * 1.3)
    if req.traffic_level in ["high", "congested"]:
        base_crowd = int(base_crowd * 1.2)
        
    variance = int(np.random.randint(-5, 10))
    
    return {
        "line_id": req.line_id,
        "predicted_passengers": max(0, base_crowd + variance)
    }

@app.get("/stops-near-me")
def get_stops_near_me(lat: float, lon: float, radius_km: float = 0.5):
    if spatial_index is None: raise HTTPException(status_code=500, detail="Index offline")
    radius_deg = radius_km / 111.0
    indices = spatial_index.query_ball_point([lat, lon], r=radius_deg)
    return {"stops": stops_df.iloc[indices][['stop_id', 'line_name']].to_dict(orient='records')}

# --- FRONTEND MOUNT ---
# This ensures the API serves your index.html natively at http://localhost:8000/
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
