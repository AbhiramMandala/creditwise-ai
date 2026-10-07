from src import config
from src.preprocessing import build_preprocessor


def test_preprocessor_handles_unseen_category(df):
    import pandas as pd

    pre = build_preprocessor()
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    pre.fit(X)
    row = X.iloc[[0]].copy()
    row["purpose"] = "UNSEEN_PURPOSE_XYZ"  # must not crash (handle_unknown=ignore)
    out = pre.transform(row)
    assert out.shape[0] == 1 and out.shape[1] > len(config.NUMERIC_FEATURES)
