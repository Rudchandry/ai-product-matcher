import os
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib


SEP = " [SEP] "

MODEL_DIR = os.environ.get("MODEL_DIR", "artifacts/v1")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

app = FastAPI(title="pair-matcher-model")

# Globals populated at startup
MODEL = None
LABEL_ENCODER = None


class Pair(BaseModel):
    text_a: Optional[str] = None
    text_b: Optional[str] = None
    text: Optional[str] = None


class PredictRequest(BaseModel):
    pairs: Optional[List[Pair]] = None
    pair: Optional[Pair] = None


class PredictResponse(BaseModel):
    predictions: List[str]
    probabilities: Optional[List[Optional[float]]] = None
    meta: Optional[Dict[str, Any]] = None


@app.on_event("startup")
def load_model():
    global MODEL, LABEL_ENCODER
    try:
        MODEL = joblib.load(MODEL_PATH)
    except Exception as e:
        MODEL = None
        app.state.load_error = f"Could not load model from {MODEL_PATH}: {e}"
    try:
        LABEL_ENCODER = joblib.load(LE_PATH)
    except Exception as e:
        LABEL_ENCODER = None
        app.state.le_error = f"Could not load label encoder from {LE_PATH}: {e}"


@app.get("/health")
def health():
    ok = True
    details = {}
    if getattr(app.state, "load_error", None):
        ok = False
        details["model_error"] = app.state.load_error
    if getattr(app.state, "le_error", None):
        ok = False
        details["le_error"] = app.state.le_error
    details["model_path"] = MODEL_PATH
    details["label_encoder_path"] = LE_PATH
    return {"status": "ok" if ok else "error", "details": details}


def _to_feature(p: Pair) -> str:
    if p.text is not None:
        return str(p.text)
    a = p.text_a or ""
    b = p.text_b or ""
    return f"{a}{SEP}{b}" if a or b else ""


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs / /health.")
    inputs: List[str] = []
    if req.pairs:
        inputs = [_to_feature(Pair(**p.dict())) for p in req.pairs]
    elif req.pair:
        inputs = [_to_feature(req.pair)]
    else:
        raise HTTPException(status_code=400, detail="Request must include 'pair' or 'pairs'.")

    if not any(inputs):
        raise HTTPException(status_code=400, detail="No valid text provided in inputs.")

    try:
        preds = MODEL.predict(inputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    if LABEL_ENCODER is not None:
        try:
            decoded = LABEL_ENCODER.inverse_transform(preds)
            decoded = [str(x) for x in decoded]
        except Exception:
            decoded = [str(int(p)) if hasattr(p, "__int__") else str(p) for p in preds]
    else:
        decoded = [str(p) for p in preds]

    probs = None
    if hasattr(MODEL, "predict_proba"):
        try:
            proba = MODEL.predict_proba(inputs)
            if proba.shape[1] == 2:
                probs = [float(p[1]) for p in proba]
            else:
                probs = [None for _ in range(len(inputs))]
        except Exception:
            probs = None

    return {"predictions": decoded, "probabilities": probs, "meta": {"n": len(inputs)}}
