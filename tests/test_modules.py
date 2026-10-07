from src.anomaly import fit_anomaly, score_one
from src.clustering import fit_clusters
from src.loan_recommender import recommend_loan


def test_clustering(df):
    _, labeled, info = fit_clusters(df, 4)
    assert set(labeled["cluster"].unique()) <= {0, 1, 2, 3}
    assert len(info["profiles"]) == 4


def test_anomaly_language(df, sample_record):
    _, scored = fit_anomaly(df)
    assert 50 <= int(scored["anomaly"].sum()) <= 150  # ~10% contamination
    r = score_one.__wrapped__ if hasattr(score_one, "__wrapped__") else score_one
    from src import config
    import joblib  # noqa

    from src.anomaly import build_anomaly_pipeline

    pipe = build_anomaly_pipeline()
    pipe.fit(df[[*config.NUMERIC_FEATURES, *config.CATEGORICAL_FEATURES]])
    out = score_one(pipe, sample_record)
    assert "NOT confirmed fraud" in out["verdict"] or "No anomaly" in out["verdict"]


def test_loan_rules():
    ok = recommend_loan(0.85, 6000, 3000, 10000, 24)
    assert ok["decision"] == "recommend"
    bad = recommend_loan(0.2, 2000, 1900, 50000, 12)
    assert bad["decision"] == "not_recommended"
    assert "not" in bad["disclaimer"].lower() or "support" in bad["disclaimer"].lower()
    edge = recommend_loan(0.6, 0, 0, 1000)
    assert edge["decision"] == "insufficient_data"


def test_invalid_loan_input_rejected():
    from fastapi.testclient import TestClient

    from app.api import app

    c = TestClient(app)
    r = c.post("/recommend-loan", json={"p_good": 2.0, "monthly_income": -5,
                                        "monthly_expenses": 0, "requested_amount": 1})
    assert r.status_code == 422
