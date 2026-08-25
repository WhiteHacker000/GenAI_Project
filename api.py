import os
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from agent import run_agentic_workflow

load_dotenv() # Load environment variables from .env

app = FastAPI(title="EV Charging AI Backend")

frontend_urls = os.getenv("FRONTEND_URL", "")
allowed_origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    *[url.strip().rstrip("/") for url in frontend_urls.split(",") if url.strip()],
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(allowed_origins)),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Load model artifacts once on startup
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BASE_DIR / "model_artifacts.pkl"
if ARTIFACT_PATH.is_file():
    artifacts = joblib.load(ARTIFACT_PATH)
    rf_model = artifacts['model']
    station_avg_map = artifacts['station_avg_map']
    feature_columns = artifacts['feature_columns']
else:
    raise RuntimeError(f"Model artifacts not found at {ARTIFACT_PATH}. Run training first.")

# API Key for LangGraph (loaded from .env)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "EV Charging AI Backend"}

class PredictRequest(BaseModel):
    station_name: str
    hour: int
    weekday: int
    weekday_name: str

class PlanRequest(BaseModel):
    station_name: str
    time_context: str
    predicted_energy: float
    avg_load: float

@app.get("/stations")
async def get_stations():
    return sorted(list(station_avg_map.keys()))

@app.post("/predict")
async def predict_energy(req: PredictRequest):
    try:
        if req.station_name not in station_avg_map:
            raise HTTPException(status_code=404, detail="Station not found")

        # Prepare features exactly like Milestone 1 logic
        input_data = {
            'Hour': req.hour,
            'Weekday': req.weekday
        }
        input_data['Peak_Hour'] = 1 if (6 <= req.hour <= 10) or (17 <= req.hour <= 21) else 0
        input_data['Is_Weekend'] = 1 if req.weekday >= 5 else 0
        input_data['Station_Avg_Load'] = station_avg_map.get(req.station_name, 0)
        input_data['Hour_Weekend_Interaction'] = input_data['Hour'] * input_data['Is_Weekend']

        # Create model input DataFrame matching feature columns, pre-filled with floats
        model_input = pd.DataFrame(0.0, index=[0], columns=feature_columns)
        for col in ['Hour', 'Weekday', 'Peak_Hour', 'Is_Weekend', 'Station_Avg_Load', 'Hour_Weekend_Interaction']:
            if col in feature_columns:
                model_input.at[0, col] = input_data[col]

        station_col = f"Station Name_{req.station_name}"
        if station_col in feature_columns:
            model_input.at[0, station_col] = 1

        model_input = model_input.infer_objects()
        prediction = rf_model.predict(model_input)[0]

        return {
            "prediction": float(prediction),
            "station_avg": float(station_avg_map[req.station_name]),
            "is_high_load": bool(prediction > station_avg_map[req.station_name])
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plan")
async def generate_plan(req: PlanRequest):
    try:
        if not GROQ_API_KEY:
            raise HTTPException(
                status_code=503,
                detail="GROQ_API_KEY is not configured on the backend.",
            )
        plan = run_agentic_workflow(
            station=req.station_name,
            time_context=req.time_context,
            predicted=req.predicted_energy,
            avg=req.avg_load,
            groq_key=GROQ_API_KEY
        )
        return plan
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
