import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Equipment ML Model API",
    version="2.0"
)

# --------------------------------------------------
# Load model artifact once at startup
# --------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    _artifact = joblib.load(MODEL_PATH)
    _model = _artifact["model"]
    _feature_names = _artifact["feature_names"]
    _model_version = _artifact.get("model_version", "unknown")
except Exception as e:
    _model = None
    _feature_names = None
    _model_version = "unavailable"
    _load_error = str(e)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class TelemetryRequest(BaseModel):
    equipmentId: int

    # Telemetry values
    temperature: float
    pressure: float
    flowRate: float
    runtimeHours: float
    errorCount: int

    # Equipment thresholds
    minTemperature: float
    maxTemperature: float

    minPressure: float
    maxPressure: float

    minFlowRate: float
    maxFlowRate: float

    maxErrorCount: int


# --------------------------------------------------
# Response model
# --------------------------------------------------

class PredictionResponse(BaseModel):
    equipmentId: int
    isAnomaly: bool
    anomalyScore: float
    riskLevel: str
    modelVersion: str


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Equipment ML Model API is running",
        "modelLoaded": _model is not None,
        "modelVersion": _model_version,
    }


# --------------------------------------------------
# Risk bucketing (kept as thresholds on the model's
# continuous score, same cut points as before)
# --------------------------------------------------

def score_to_risk(anomaly_score: float):
    if anomaly_score < 0.05:
        return "LOW", False
    elif anomaly_score <= 0.25:
        return "MEDIUM", True
    elif anomaly_score <= 0.50:
        return "HIGH", True
    else:
        return "CRITICAL", True


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict(data: TelemetryRequest):

    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {_load_error}"
        )

    # Build a single-row DataFrame in the exact feature order the model
    # was trained on (order matters for tree-based sklearn models).
    row = {name: getattr(data, name) for name in _feature_names}
    X = pd.DataFrame([row], columns=_feature_names)

    anomaly_score = float(_model.predict(X)[0])
    anomaly_score = max(0.0, min(1.0, anomaly_score))  # clip to [0, 1]

    risk_level, is_anomaly = score_to_risk(anomaly_score)

    return PredictionResponse(
        equipmentId=data.equipmentId,
        isAnomaly=is_anomaly,
        anomalyScore=round(anomaly_score, 4),
        riskLevel=risk_level,
        modelVersion=_model_version
    )
