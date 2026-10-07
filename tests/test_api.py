"""API tests use real trained artifacts; train first via scripts/train.py."""

from fastapi.testclient import TestClient


def _record():
    from src import config
    from src.data_loader import load_data

    row = load_data().iloc[0]
    return {c: (float(row[c]) if c in config.NUMERIC_FEATURES else str(row[c]))
            for c in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES}


def test_endpoints():
    import joblib  # noqa: F401
    from pathlib import Path

    from src import config as cfg

    if not cfg.RISK_MODEL_PATH.exists():
        import pytest

        pytest.skip("Run scripts/train.py first")
    from app.api import app

    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.get("/model-metrics").status_code == 200
    rec = _record()
    assert c.post("/predict-risk", json=rec).status_code == 200
    assert c.post("/detect-anomaly", json=rec).status_code == 200
    assert c.post("/customer-segment", json=rec).status_code == 200
    bad = dict(rec)
    bad["age"] = 5  # violates ge=18
    assert c.post("/predict-risk", json=bad).status_code == 422
