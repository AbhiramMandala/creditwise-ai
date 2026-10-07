"""FastAPI backend exposing the ML modules with validation + logging."""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()

from app.schemas import (  # noqa: E402
    AnomalyResponse,
    CustomerRecord,
    HealthResponse,
    LoanRequest,
    RiskResponse,
    SegmentResponse,
)
from src import config  # noqa: E402
from src.anomaly import score_one as anomaly_score_one  # noqa: E402
from src.credit_risk import predict_one  # noqa: E402
from src.loan_recommender import recommend_loan  # noqa: E402
from src.logging_utils import get_logger  # noqa: E402

log = get_logger("api")
app = FastAPI(title=config.APP_NAME, version=config.MODEL_VERSION)


def _load(path: Path):
    if not path.exists():
        raise HTTPException(500, f"Model artifact missing: {path.name}. Run scripts/train.py.")
    return joblib.load(path)


@app.get("/")
def root():
    return {
        "name": "AI-Powered Credit Risk & Financial Decision Support API",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model_version": config.MODEL_VERSION}


@app.get("/model-metrics")
def model_metrics():
    if not config.RISK_METRICS_PATH.exists():
        raise HTTPException(500, "Metrics not found. Run scripts/train.py.")
    return json.loads(config.RISK_METRICS_PATH.read_text())


@app.post("/predict-risk", response_model=RiskResponse)
def predict_risk(rec: CustomerRecord):
    pipe = _load(config.RISK_MODEL_PATH)
    try:
        return predict_one(pipe, rec.model_dump())
    except Exception as exc:  # noqa: BLE001
        log.exception("predict-risk failed")
        raise HTTPException(400, str(exc)) from exc


@app.post("/detect-anomaly", response_model=AnomalyResponse)
def detect_anomaly(rec: CustomerRecord):
    pipe = _load(config.ANOMALY_MODEL_PATH)
    try:
        return anomaly_score_one(pipe, rec.model_dump())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc


@app.post("/customer-segment", response_model=SegmentResponse)
def customer_segment(rec: CustomerRecord):
    import pandas as pd

    pipe = _load(config.CLUSTER_MODEL_PATH)
    try:
        label = int(pipe.predict(pd.DataFrame([rec.model_dump()]))[0])
        profiles = {}
        if config.CLUSTER_SUMMARY_PATH.exists():
            profiles = json.loads(config.CLUSTER_SUMMARY_PATH.read_text()).get("profiles", {})
        return {"cluster": label, "profile": profiles.get(str(label), {})}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc


@app.post("/recommend-loan")
def recommend_loan_ep(req: LoanRequest):
    try:
        return recommend_loan(**req.model_dump())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
