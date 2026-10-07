"""Credit-risk model: training, persistence, inference, explainability.

Model selection (stratified 80/20 split, random_state=42, measured 2026-10-07
on the bundled 1000-row dataset):
  LogisticRegression  acc .705  prec .776  rec .814  f1 .794  auc .759
  RandomForest        acc .720  prec .813  rec .779  f1 .796  auc .789
  GradientBoosting    acc .760  prec .815  rec .850  f1 .832  auc .773
-> GradientBoosting selected (best F1 / accuracy). Metrics are recomputed by
scripts/train.py and stored in models/credit_risk_metrics.json; never hard-code.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.data_loader import target_to_binary
from src.logging_utils import get_logger
from src.preprocessing import build_preprocessor, get_feature_names

log = get_logger(__name__)

CANDIDATES: dict[str, object] = {
    "logistic_regression": LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
    "random_forest": RandomForestClassifier(
        n_estimators=300, random_state=config.RANDOM_STATE, class_weight="balanced"
    ),
    "gradient_boosting": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
}


def _split(df: pd.DataFrame):
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = target_to_binary(df[config.TARGET_COL])
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )


def evaluate_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_test": int(len(y_true)),
    }


def train_and_select(df: pd.DataFrame) -> tuple[Pipeline, dict, dict]:
    """Train all candidates, return (best_pipeline, all_metrics, best_info)."""
    X_train, X_test, y_train, y_test = _split(df)
    results: dict[str, dict] = {}
    pipelines: dict[str, Pipeline] = {}
    for name, clf in CANDIDATES.items():
        pipe = Pipeline([("pre", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = evaluate_metrics(y_test, pred, proba)
        # 5-fold CV F1 for honest reporting
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        results[name]["cv_f1_mean"] = float(cv_scores.mean())
        results[name]["cv_f1_std"] = float(cv_scores.std())
        pipelines[name] = pipe
        log.info("%s: %s", name, {k: round(v, 3) for k, v in results[name].items() if isinstance(v, float)})
    best_name = max(results, key=lambda k: results[k]["f1"])
    log.info("Selected model: %s", best_name)
    return pipelines[best_name], results, {"best_model": best_name}


def save_model(pipe: Pipeline, path: Path = config.RISK_MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)  # nosec - local artifact, not user input
    log.info("Saved risk pipeline to %s", path)


def load_model(path: Path = config.RISK_MODEL_PATH) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run: python scripts/train.py"
        )
    return joblib.load(path)  # nosec - local artifact produced by scripts/train.py


def predict_one(pipe: Pipeline, record: dict) -> dict:
    """Predict a single customer dict -> label, probability, explanations."""
    row = pd.DataFrame([record])
    proba_bad = float(pipe.predict_proba(row)[0][0])  # P(class=bad)
    proba_good = 1.0 - proba_bad
    label = "good" if proba_good >= 0.5 else "bad"
    return {
        "label": label,
        "risk_level": "Low Risk" if label == "good" else "High Risk",
        "p_good": round(proba_good, 4),
        "p_bad": round(proba_bad, 4),
        "explanations": explain_one(pipe, record),
    }


def explain_one(pipe: Pipeline, record: dict, top_k: int = 5) -> list[dict]:
    """Model-grounded local explanation.

    Uses the fitted OneHotEncoder + classifier coefficients/importances to rank
    which *transformed* features pushed this row toward its predicted class.
    No invented reasons: every reason names a real model feature.
    """
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    names = get_feature_names(pre)
    Xt = pre.transform(pd.DataFrame([record]))
    x = np.asarray(Xt[0], dtype=float)

    if hasattr(clf, "coef_"):  # logistic regression
        weights = np.asarray(clf.coef_[0], dtype=float)
        contrib = weights * x  # positive -> toward good
    elif hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
        # direction from deviation vs training mean is unknown without
        # background stats, so report magnitude weighted by value
        contrib = imp * np.abs(x)
    else:
        contrib = np.abs(x)

    order = np.argsort(np.abs(contrib))[::-1][:top_k]
    out = []
    for idx in order:
        fname = names[int(idx)] if int(idx) < len(names) else f"feature_{idx}"
        out.append(
            {
                "feature": fname,
                "contribution": round(float(contrib[int(idx)]), 4),
                "value": round(float(x[int(idx)]), 4),
                "plain_english": _plain_english(fname),
            }
        )
    return out


def _plain_english(feature: str) -> str:
    f = feature.lower()
    if "checking_status" in f:
        return "Checking-account status influenced the score."
    if "credit_history" in f:
        return "Past repayment history influenced the score."
    if "credit_amount" in f:
        return "Requested credit amount influenced the score."
    if "duration" in f:
        return "Loan duration influenced the score."
    if "savings_status" in f:
        return "Savings level influenced the score."
    if "employment" in f:
        return "Employment tenure influenced the score."
    if "age" in f:
        return "Applicant age influenced the score."
    if "purpose" in f:
        return "Loan purpose influenced the score."
    if "housing" in f:
        return "Housing situation influenced the score."
    if "debt" in f or "installment" in f:
        return "Existing repayment burden influenced the score."
    return f"Model feature '{feature}' contributed to the score."


def global_importance(pipe: Pipeline, top_k: int = 15) -> list[dict]:
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    names = get_feature_names(pre)
    if hasattr(clf, "coef_"):
        scores = np.abs(np.asarray(clf.coef_[0], dtype=float))
    elif hasattr(clf, "feature_importances_"):
        scores = np.asarray(clf.feature_importances_, dtype=float)
    else:
        return []
    order = np.argsort(scores)[::-1][:top_k]
    total = float(scores.sum()) or 1.0
    return [
        {
            "feature": names[int(i)] if int(i) < len(names) else f"feature_{i}",
            "importance": round(float(scores[int(i)]) / total, 4),
        }
        for i in order
    ]


def save_metrics(metrics: dict, info: dict, path: Path = config.RISK_METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_version": config.MODEL_VERSION, "selected": info, "models": metrics}
    path.write_text(json.dumps(payload, indent=2))
