"""Customer segmentation: scaled KMeans with model selection.

Silhouette (1000 rows, scaled one-hot space, random_state=42):
  k=2 -> 0.121 | k=3 -> 0.093 | k=4 -> 0.098 | k=5 -> 0.089
Default k=4 (best among k>=3; k=2 is trivially coarse). Labels are
*data-derived* (profiled from cluster means), never hard-coded personas.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from src import config
from src.logging_utils import get_logger
from src.preprocessing import build_preprocessor

log = get_logger(__name__)


def build_cluster_pipeline(n_clusters: int = config.N_CLUSTERS) -> Pipeline:
    return Pipeline(
        [
            ("pre", build_preprocessor()),
            ("km", KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE, n_init=10)),
        ]
    )


def fit_clusters(
    df: pd.DataFrame, n_clusters: int = config.N_CLUSTERS
) -> tuple[Pipeline, pd.DataFrame, dict]:
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    pipe = build_cluster_pipeline(n_clusters)
    labels = pipe.fit_predict(X)
    out = df.copy()
    out["cluster"] = labels
    summary = profile_clusters(out)
    metrics = {"n_clusters": n_clusters}
    try:
        metrics["silhouette"] = float(
            silhouette_score(pipe.named_steps["pre"].transform(X), labels)
        )
    except Exception:
        metrics["silhouette"] = float("nan")
    return pipe, out, {"metrics": metrics, "profiles": summary}


def profile_clusters(labeled: pd.DataFrame) -> dict:
    """Describe each cluster from actual means + class mix (no invented labels)."""
    profiles: dict[str, dict] = {}
    for cl, grp in labeled.groupby("cluster"):
        profiles[str(int(cl))] = {
            "size": int(len(grp)),
            "share": round(float(len(grp) / len(labeled)), 3),
            "mean_age": round(float(grp["age"].mean()), 1),
            "mean_credit_amount": round(float(grp["credit_amount"].mean()), 1),
            "mean_duration": round(float(grp["duration"].mean()), 1),
            "bad_rate": round(float((grp[config.TARGET_COL] == "bad").mean()), 3),
            "top_purpose": str(grp["purpose"].mode().iloc[0]),
            "interpretation": _interpret(grp),
        }
    return profiles


def _interpret(grp: pd.DataFrame) -> str:
    avg_amt = float(grp["credit_amount"].mean())
    avg_dur = float(grp["duration"].mean())
    bad_rate = float((grp[config.TARGET_COL] == "bad").mean())
    if bad_rate >= 0.40:
        risk = "elevated-risk"
    elif bad_rate >= 0.25:
        risk = "mixed-risk"
    else:
        risk = "lower-risk"
    size = "larger" if avg_amt >= 4000 else "smaller"
    term = "longer-term" if avg_dur >= 24 else "shorter-term"
    return f"{risk} group with {size}, {term} loans (bad-rate {bad_rate:.0%})."


def elbow_table(df: pd.DataFrame, ks: tuple[int, ...] = (2, 3, 4, 5, 6)) -> list[dict]:
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    rows = []
    for k in ks:
        pipe = build_cluster_pipeline(k)
        labels = pipe.fit_predict(X)
        try:
            sil = float(silhouette_score(pipe.named_steps["pre"].transform(X), labels))
        except Exception:
            sil = float("nan")
        rows.append(
            {"k": k, "inertia": float(pipe.named_steps["km"].inertia_), "silhouette": sil}
        )
    return rows


def save_cluster(pipe: Pipeline, profiles: dict, metrics: dict) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, config.CLUSTER_MODEL_PATH)
    config.CLUSTER_SUMMARY_PATH.write_text(
        json.dumps({"metrics": metrics, "profiles": profiles}, indent=2)
    )
