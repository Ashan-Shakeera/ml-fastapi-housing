from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
    title="House Price Model API",
    description="FastAPI for ML model inference (House Prices)"
)

# ----------- Load model on startup (as required) -----------
# model.pkl contains {"model": pipeline, "metadata": {...}}
_model_bundle = joblib.load("model.pkl")
_pipe = _model_bundle["model"] if isinstance(_model_bundle, dict) and "model" in _model_bundle else _model_bundle
_metadata = _model_bundle.get("metadata", {}) if isinstance(_model_bundle, dict) else {}

_FEATURES = _metadata.get("features", [
    "area","bedrooms","bathrooms","stories","parking",
    "mainroad","guestroom","basement","hotwaterheating",
    "airconditioning","prefarea","furnishingstatus"
])

# ----------- Pydantic schemas -----------
class PredictionInput(BaseModel):
    # numeric
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    parking: int
    # categoricalpandas
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    prefarea: str
    furnishingstatus: str

class PredictionOutput(BaseModel):
    prediction: float
    confidence: Optional[float] = None  # optional std-based uncertainty

# Helper: normalize and validate categorical inputs
_ALLOWED_YESNO = {"yes","no"}
_ALLOWED_FURNISH = {"furnished","semi-furnished","unfurnished"}

def _normalize_row(d: dict) -> dict:
    out = d.copy()
    # lowercase strings
    for k in ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea","furnishingstatus"]:
        out[k] = str(out[k]).strip().lower()
    # basic validation
    for k in ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]:
        if out[k] not in _ALLOWED_YESNO:
            raise HTTPException(status_code=422, detail=f"'{k}' must be one of {_ALLOWED_YESNO}, got '{out[k]}'")
    if out["furnishingstatus"] not in _ALLOWED_FURNISH:
        raise HTTPException(status_code=422, detail=f"'furnishingstatus' must be one of {_ALLOWED_FURNISH}, got '{out['furnishingstatus']}'")
    return out

def _rf_std_for_single(input_df: pd.DataFrame) -> Optional[float]:
    """
    If the underlying model is RandomForestRegressor, estimate uncertainty
    as the std dev across trees' predictions (1-sample).
    """
    try:
        pre = _pipe.named_steps["preprocess"]
        rf  = _pipe.named_steps["model"]
        Xp  = pre.transform(input_df)
        tree_preds = np.array([t.predict(Xp)[0] for t in rf.estimators_], dtype=float)
        return float(np.std(tree_preds))
    except Exception:
        return None

# ------------------- Endpoints -------------------

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "ML Model API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        row = _normalize_row(input_data.dict())
        df = pd.DataFrame([row], columns=_FEATURES)
        yhat = float(_pipe.predict(df)[0])

        conf = _rf_std_for_single(df)  # may be None for non-RF models
        return PredictionOutput(prediction=yhat, confidence=conf)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": type(_pipe.named_steps.get("model", _pipe)).__name__,
        "problem_type": _metadata.get("problem_type", "regression"),
        "features": _FEATURES,
        "metrics": _metadata.get("metrics", {}),
        "trained_at": _metadata.get("trained_at", None),
    }

# -------- Bonus: batch prediction (optional extra credit) --------
class BatchPredictionInput(BaseModel):
    items: List[PredictionInput]

@app.post("/predict-batch")
def predict_batch(batch: BatchPredictionInput):
    try:
        rows = [_normalize_row(item.dict()) for item in batch.items]
        df = pd.DataFrame(rows, columns=_FEATURES)
        yhat = _pipe.predict(df).tolist()
        return {"predictions": yhat}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
