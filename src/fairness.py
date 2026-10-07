"""Fairness analysis — computed from model predictions, not raw labels.

Groups available in this dataset: personal_status (proxy for sex/marital),
age bands, employment tenure. Reports approval (predicted good) rate, TPR,
FPR, precision/recall per group + demographic-parity gaps. Text explicitly
says disparities are *signals to investigate*, not proof of discrimination.
"""
from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.data_loader import target_to_binary


def _group_rates(df: pd.DataFrame, group_col: str, y_true, y_pred) -> list[dict]:
    tmp = pd.DataFrame({"g": df[group_col].astype(str), "y": y_true, "p": y_pred})
    rows = []
    for g, sub in tmp.groupby("g"):
        yt, yp = sub["y"].to_numpy(), sub["p"].to_numpy()
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        rows.append(
            {
                "group": str(g),
                "n": int(len(sub)),
                "approval_rate": round(float((yp == 1).mean()), 3),
                "tpr": round(float(tp / (tp + fn)) if (tp + fn) else 0.0, 3),
                "fpr": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 3),
                "fnr": round(float(fn / (tp + fn)) if (tp + fn) else 0.0, 3),
                "precision": round(float(tp / (tp + fp)) if (tp + fp) else 0.0, 3),
                "recall": round(float(tp / (tp + fn)) if (tp + fn) else 0.0, 3),
            }
        )
    return sorted(rows, key=lambda r: r["n"], reverse=True)


def fairness_report(df: pd.DataFrame, pipe: Pipeline) -> dict:
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y_true = target_to_binary(df[config.TARGET_COL]).to_numpy()
    y_pred = pipe.predict(X)
    out: dict[str, list[dict]] = {}
    tmp = df.copy()
    tmp["age_band"] = pd.cut(
        tmp["age"], bins=[0, 30, 45, 200], labels=["<=30", "31-45", "45+"]
    ).astype(str)
    for col in ["personal_status", "employment", "age_band"]:
        groups = _group_rates(tmp, col, y_true, y_pred)
        approvals = [g["approval_rate"] for g in groups]
        gap = round(float(max(approvals) - min(approvals)), 3) if approvals else 0.0
        out[col] = {"groups": groups, "max_approval_gap": gap}
    out["note"] = (
        "Gaps flag potential disparities for review; they do not by themselves "
        "prove discrimination — check sample sizes, confounding and thresholds."
    )
    return out
