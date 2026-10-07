"""Regression test for the dashboard statistics operation.

pandas 3.0 removed the ``numeric_only`` keyword from ``DataFrame.describe``,
which crashed the Streamlit overview tab. ``numeric_summary`` uses explicit
numeric-column selection instead and is covered here against the real dataset.
"""
from src.data_loader import numeric_summary


def test_numeric_summary_on_real_data(df):
    stats = numeric_summary(df)
    # Only numeric columns are described
    assert list(stats.columns) == list(df.select_dtypes(include="number").columns)
    assert len(stats.columns) >= 7
    # Standard describe rows are present
    for row in ("count", "mean", "min", "max"):
        assert row in stats.index
    # Values are sane for the known dataset
    assert float(stats.loc["count", "age"]) == 1000
    assert float(stats.loc["min", "age"]) >= 18
