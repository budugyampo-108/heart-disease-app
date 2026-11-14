"""
backend/main.py - FastAPI service for the heart-disease classifier.

Endpoints
---------
GET  /api/health    service status (is the model loaded?)
GET  /api/metrics   evaluation metrics, confusion matrix, ROC curve, form schema
POST /api/predict   patient features in -> prediction + class probabilities out

Run locally (from the project root):
    uvicorn backend.main:app --reload
    -> dashboard at http://localhost:8000 , API docs at http://localhost:8000/api/docs

On Vercel the dashboard is served as static files and only /api/* reaches this
file (see vercel.json).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model.pkl"
METRICS_PATH = ROOT / "metrics.json"
WEB_DIR = ROOT / "web"

# Probability cut-offs used only to label the risk level on the dashboard.
# The class prediction itself uses the standard 0.5 threshold.
RISK_LOW, RISK_HIGH = 0.30, 0.60

app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts heart-disease probability from clinical features.",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url=None,
)

# The API is read-only and has no cookies, so open CORS is safe and lets the
# dashboard be hosted on a different origin during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Loading the trained artefacts (cached after the first request)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl not found. Run `python train.py` first.")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        raise FileNotFoundError("metrics.json not found. Run `python train.py` first.")
    return json.loads(METRICS_PATH.read_text())


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
Category = Union[str, int, float, bool]


class PatientFeatures(BaseModel):
    """One patient's clinical record. Category fields accept e.g. "Yes"/"No"."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 58,
                "sex": "Male",
                "blood_pressure": 145,
                "cholesterol": 240,
                "bmi": 31.5,
                "smoker": "Yes",
                "diabetes": "No",
                "physical_activity": 1,
                "family_history": "Yes",
                "stress_level": 7,
            }
        },
    )

    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: Category
    blood_pressure: float = Field(..., ge=40, le=300, description="Blood pressure (mmHg)")
    cholesterol: float = Field(..., ge=50, le=700, description="Cholesterol (mg/dL)")
    bmi: float = Field(
        ..., ge=8, le=80, description="Body mass index",
        validation_alias=AliasChoices("bmi", "BMI"),
    )
    smoker: Category
    diabetes: Category
    physical_activity: float = Field(..., ge=0, description="Physical activity level")
    family_history: Category
    stress_level: float = Field(..., ge=0, description="Stress level")


class Probabilities(BaseModel):
    no_heart_disease: float
    heart_disease: float


class PredictionResponse(BaseModel):
    prediction: int = Field(description="0 = no heart disease, 1 = heart disease")
    label: str
    probability: float = Field(description="Probability of heart disease (class 1)")
    probabilities: Probabilities
    risk_level: str = Field(description="low, moderate or high")
    threshold: float
    model: str


# --------------------------------------------------------------------------- #
# Input checking: match what the client sent to the categories seen in training
# --------------------------------------------------------------------------- #
TRUE_WORDS = {"1", "yes", "y", "true"}
FALSE_WORDS = {"0", "no", "n", "false"}
SEX_ALIASES = {"m": "male", "f": "female"}


def match_category(field: str, value: Any, options: list[str]) -> str:
    """Return the training-time spelling of `value`, or raise a clear 422."""
    lookup = {opt.lower(): opt for opt in options}

    if isinstance(value, bool):
        text = str(int(value))
    elif isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
    text = text.lower()

    candidates = [text, SEX_ALIASES.get(text, text)]
    if text in TRUE_WORDS:
        candidates += sorted(TRUE_WORDS)
    elif text in FALSE_WORDS:
        candidates += sorted(FALSE_WORDS)

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    raise HTTPException(
        status_code=422,
        detail=f"'{field}' must be one of {options}, got {value!r}.",
    )


def build_frame(patient: PatientFeatures, schema: dict[str, Any]) -> pd.DataFrame:
    """One-row DataFrame in the exact column order the model was trained on."""
    raw = patient.model_dump()
    row: dict[str, Any] = {}
    for name in schema["features"]:
        if name in schema["categorical"]:
            row[name] = match_category(name, raw[name], schema["categorical"][name])
        else:
            try:
                row[name] = float(raw[name])
            except (TypeError, ValueError):
                raise HTTPException(422, f"'{name}' must be a number, got {raw[name]!r}.")
    return pd.DataFrame([row], columns=schema["features"])


def risk_level(p: float) -> str:
    return "low" if p < RISK_LOW else "moderate" if p < RISK_HIGH else "high"


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict[str, Any]:
    """Quick status check, safe to call as often as you like."""
    model_ready = metrics_ready = True
    try:
        get_model()
    except Exception:
        model_ready = False
    try:
        get_metrics()
    except Exception:
        metrics_ready = False
    return {
        "status": "ok" if model_ready and metrics_ready else "degraded",
        "model_loaded": model_ready,
        "metrics_loaded": metrics_ready,
        "version": app.version,
    }


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    """Evaluation results from train.py, plus the form schema for the dashboard."""
    try:
        return get_metrics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures) -> PredictionResponse:
    """Predict heart-disease probability for one patient."""
    try:
        model = get_model()
        info = get_metrics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # e.g. scikit-learn version mismatch when unpickling
        raise HTTPException(status_code=500, detail=f"Could not load the model: {exc}")

    frame = build_frame(patient, info["schema"])
    proba = model.predict_proba(frame)[0]
    classes = list(model.classes_)
    p_yes = float(proba[classes.index(1)])
    p_no = float(proba[classes.index(0)])
    threshold = float(info["model"].get("threshold", 0.5))
    predicted = int(p_yes >= threshold)

    return PredictionResponse(
        prediction=predicted,
        label="Heart disease likely" if predicted else "Heart disease unlikely",
        probability=round(p_yes, 4),
        probabilities=Probabilities(
            no_heart_disease=round(p_no, 4), heart_disease=round(p_yes, 4)
        ),
        risk_level=risk_level(p_yes),
        threshold=threshold,
        model=info["model"]["name"],
    )


# --------------------------------------------------------------------------- #
# Local development only: also serve the dashboard from this same server.
# (On Vercel the static files are served by the platform, not by Python.)
# --------------------------------------------------------------------------- #
if not os.getenv("VERCEL") and WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
