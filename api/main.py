from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Terminal Sivas Prediction API (ONNX Powered)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

# Load Highly Optimized ONNX Runtime Sessions
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

print("Loading ONNX Execution Providers...")
sess_delay = ort.InferenceSession(os.path.join(MODEL_DIR, "model_delay_quant.onnx"), sess_options)
sess_crowd = ort.InferenceSession(os.path.join(MODEL_DIR, "model_crowd_quant.onnx"), sess_options)

class PredictionRequest(BaseModel):
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
    dwell_time_min: float = 1.0

def build_onnx_inputs(session: ort.InferenceSession, df: pd.DataFrame) -> dict:
    """Dynamically casts Pandas data to strictly typed ONNX tensors."""
    inputs = {}
    for inp in session.get_inputs():
        col_name = inp.name
        if col_name not in df.columns:
            continue
            
        val = df[col_name].values
        # Match ONNX strict type expectations
        if inp.type == 'tensor(float)':
            val = val.astype(np.float32)
        elif inp.type == 'tensor(double)':
            val = val.astype(np.float64)
        elif inp.type == 'tensor(int64)':
            val = val.astype(np.int64)
        elif inp.type == 'tensor(string)':
            val = val.astype(str).astype(object)
            
        inputs[col_name] = val.reshape(-1, 1) if len(val.shape) == 1 else val
    return inputs

@app.post("/predict-arrival")
def predict_arrival(req: PredictionRequest):
    df = pd.DataFrame([req.dict()])
    onnx_inputs = build_onnx_inputs(sess_delay, df)
    
    # Run the C++ computation graph
    predicted_delay = float(sess_delay.run(None, onnx_inputs)[0][0, 0])
    
    return {
        "line_id": req.line_id,
        "predicted_delay_min": max(0.0, predicted_delay),
        "is_delayed": predicted_delay > 2.0
    }

@app.post("/predict-crowd")
def predict_crowd(req: PredictionRequest):
    df = pd.DataFrame([req.dict()])
    onnx_inputs = build_onnx_inputs(sess_crowd, df)
    
    # Run the C++ computation graph
    predicted_passengers = float(sess_crowd.run(None, onnx_inputs)[0][0, 0])
    
    return {
        "line_id": req.line_id,
        "predicted_passengers": max(0, int(round(predicted_passengers)))
    }
