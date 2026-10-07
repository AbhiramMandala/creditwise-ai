"""Anomaly detection with IsolationForest.

Language rule enforced everywhere: we report "anomaly — needs review", NEVER
"fraud confirmed". Scores come from the model decision function.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from src import config
from src.logging_utils import get_logger
from src.preprocessing import build_preprocessor

log = get_logger(__name__)


def build_anomaly_pipeline(
    contamination: float = config.ANOMALY_CONTAMINATION,
) -> Pipeline:
    return Pipeline(
        [
            ("pre", build_preprocessor()),
            (
                "iso",
                IsolationForest(
                    contamination=contamination,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )


def fit_anomaly(
    df: pd.DataFrame, contamination: float = config.ANOMALY_CONTAMINATION
) -> tuple[Pipeline, pd.DataFrame]:
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    pipe = build_anomaly_pipeline(contamination)
    preds = pipe.fit_predict(X)  # 1 normal, -1 anomaly
    scores = -pipe.decision_function(X)  # higher = more anomalous
    out = df.copy()
    out["anomaly"] = (preds == -1).astype(int)
    out["anomaly_score"] = scores
    log.info("Flagged %d anomalies (%.1f%%)", int(out["anomaly"].sum()), 100 * out["anomaly"].mean())
    return pipe, out


def score_one(pipe: Pipeline, record: dict) -> dict:
    row = pd.DataFrame([record])
    pred = int(pipe.predict(row)[0])
    score = float(-pipe.decision_function(row)[0])
    is_anomaly = pred == -1
    return {
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": round(score, 4),
        "verdict": (
            "Anomaly detected — needs human review (NOT confirmed fraud)."
            if is_anomaly
            else "No anomaly detected."
        ),
    }


def save_anomaly(pipe: Pipeline) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, config.ANOMALY_MODEL_PATH)
