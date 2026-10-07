from src.credit_risk import explain_one, global_importance, predict_one


def test_predict_and_explain(trained, sample_record):
    pipe, metrics, info = trained
    out = predict_one(pipe, sample_record)
    assert out["label"] in ("good", "bad")
    assert 0 <= out["p_good"] <= 1
    assert len(out["explanations"]) == 5
    assert all("feature" in e for e in out["explanations"])
    assert len(global_importance(pipe)) > 0


def test_metrics_are_real(trained):
    _, metrics, info = trained
    best = metrics[info["best_model"]]
    # Honest bounds for this dataset — guards against fabricated 99% claims
    assert 0.60 <= best["accuracy"] <= 0.90
    assert 0.60 <= best["roc_auc"] <= 0.95
    assert best["confusion_matrix"][0][0] + best["confusion_matrix"][0][1] > 0
