"""FastAPI REST server for real-time IDS predictions.

Run:
    uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /predict   — classify one or more network connections
    GET  /health    — health check
    GET  /models    — list loaded models
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# sys.path must be set before importing local src modules
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.models import EnsembleIDS, NeuralNetworkIDS, RandomForestIDS, XGBoostIDS  # noqa: E402
from src.preprocessing import NSLKDDPreprocessor  # noqa: E402

app = FastAPI(
    title="AI Intrusion Detection System API",
    description="Classify network traffic as normal or attack using trained ML models.",
    version="1.0.0",
)

# ------------------------------------------------------------------
# Model registry — lazy-loaded on first request
# ------------------------------------------------------------------

_prep: NSLKDDPreprocessor | None = None
_models: dict = {}


def _load_prep() -> NSLKDDPreprocessor:
    global _prep
    if _prep is None:
        path = "models/saved/preprocessor.pkl"
        if not Path(path).exists():
            raise HTTPException(503, "Preprocessor not found. Run training first.")
        _prep = NSLKDDPreprocessor.load(path)
    return _prep


def _load_model(key: str, task: str):
    cache_key = f"{key}_{task}"
    if cache_key not in _models:
        ext = ".pt" if key == "nn" else ".pkl"
        path = f"models/saved/{key}_{task}{ext}"
        if not Path(path).exists():
            raise HTTPException(404, f"Model not found: {path}. Run training first.")
        if key == "rf":
            _models[cache_key] = RandomForestIDS.load(path)
        elif key == "xgb":
            _models[cache_key] = XGBoostIDS.load(path)
        elif key == "nn":
            _models[cache_key] = NeuralNetworkIDS.load(path)
        elif key == "ensemble":
            _models[cache_key] = EnsembleIDS.from_saved(task)
    return _models[cache_key]


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------

class ConnectionRecord(BaseModel):
    duration: float = 0
    protocol_type: str = "tcp"
    service: str = "http"
    flag: str = "SF"
    src_bytes: float = 0
    dst_bytes: float = 0
    land: int = 0
    wrong_fragment: int = 0
    urgent: int = 0
    hot: int = 0
    num_failed_logins: int = 0
    logged_in: int = 0
    num_compromised: int = 0
    root_shell: int = 0
    su_attempted: int = 0
    num_root: int = 0
    num_file_creations: int = 0
    num_shells: int = 0
    num_access_files: int = 0
    num_outbound_cmds: int = 0
    is_host_login: int = 0
    is_guest_login: int = 0
    count: float = 0
    srv_count: float = 0
    serror_rate: float = 0.0
    srv_serror_rate: float = 0.0
    rerror_rate: float = 0.0
    srv_rerror_rate: float = 0.0
    same_srv_rate: float = 0.0
    diff_srv_rate: float = 0.0
    srv_diff_host_rate: float = 0.0
    dst_host_count: float = 0
    dst_host_srv_count: float = 0
    dst_host_same_srv_rate: float = 0.0
    dst_host_diff_srv_rate: float = 0.0
    dst_host_same_src_port_rate: float = 0.0
    dst_host_srv_diff_host_rate: float = 0.0
    dst_host_serror_rate: float = 0.0
    dst_host_srv_serror_rate: float = 0.0
    dst_host_rerror_rate: float = 0.0
    dst_host_srv_rerror_rate: float = 0.0


class PredictRequest(BaseModel):
    connections: list[ConnectionRecord] = Field(..., min_length=1)
    model: Literal["rf", "xgb", "nn", "ensemble"] = "rf"
    task: Literal["binary", "multiclass"] = "multiclass"


class PredictionResult(BaseModel):
    index: int
    predicted_class: str
    confidence: float
    is_attack: bool
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    model: str
    task: str
    total: int
    attacks_detected: int
    results: list[PredictionResult]


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "AI-IDS"}


@app.get("/models")
def list_models():
    available = []
    for key in ["rf", "xgb", "nn", "ensemble"]:
        for task in ["binary", "multiclass"]:
            ext = ".pt" if key == "nn" else ".pkl"
            path = f"models/saved/{key}_{task}{ext}"
            if key == "ensemble":
                exists = all(
                    Path(f"models/saved/{k}_{task}.{'pt' if k == 'nn' else 'pkl'}").exists()
                    for k in ["rf", "xgb", "nn"]
                )
            else:
                exists = Path(path).exists()
            if exists:
                available.append({"model": key, "task": task})
    return {"available_models": available, "total": len(available)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    prep = _load_prep()
    model = _load_model(req.model, req.task)

    df = pd.DataFrame([c.model_dump() for c in req.connections])
    X = prep.transform(df)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    classes = (
        prep.le_multi.classes_ if req.task == "multiclass" else prep.le_binary.classes_
    )
    pred_labels = classes[preds]

    results = []
    for i, (label, prob_row) in enumerate(zip(pred_labels, probs)):
        confidence = float(np.max(prob_row))
        prob_dict = {cls: float(p) for cls, p in zip(classes, prob_row)}
        results.append(PredictionResult(
            index=i,
            predicted_class=str(label),
            confidence=confidence,
            is_attack=str(label) != "normal",
            probabilities=prob_dict,
        ))

    attacks = sum(1 for r in results if r.is_attack)
    return PredictResponse(
        model=req.model,
        task=req.task,
        total=len(results),
        attacks_detected=attacks,
        results=results,
    )
