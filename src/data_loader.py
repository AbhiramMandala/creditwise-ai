"""Data loading + validation. Fixes the two bugs in the legacy data_loader.py:

1. Legacy used ``if not data:`` on a DataFrame (raises ValueError: ambiguous
   truth value) and a hard-coded absolute Windows path.
2. Legacy ``preprocess_data`` used bare ``pd.get_dummies`` which breaks on
   unseen categories at inference time.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.logging_utils import get_logger

log = get_logger(__name__)

EXPECTED_COLUMNS = set(
    config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES + [config.TARGET_COL]
)


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the German-credit-style CSV. Never crashes on missing file silently."""
    if path is not None:
        csv_path = Path(path)
    else:
        csv_path = config.DATA_RAW
        # Back-compat: original repo file had spaces/parens in the name.
        if not csv_path.exists():
            legacy = config.PROJECT_ROOT / "credit_customers (DS).csv"
            if legacy.exists():
                csv_path = legacy
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Place credit_customers.csv under data/raw/."
        )
    df = pd.read_csv(csv_path)
    log.info("Loaded %s rows x %s cols from %s", *df.shape, csv_path)
    return df


def validate_data(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable data-quality problems (empty = OK)."""
    problems: list[str] = []
    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        problems.append(f"Missing columns: {sorted(missing_cols)}")
    if df.empty:
        problems.append("Dataset is empty.")
        return problems
    n_null = int(df.isna().sum().sum())
    if n_null:
        problems.append(f"{n_null} missing values found.")
    n_dup = int(df.duplicated().sum())
    if n_dup:
        problems.append(f"{n_dup} duplicate rows found.")
    if config.TARGET_COL in df.columns:
        classes = set(df[config.TARGET_COL].unique())
        if classes - {"good", "bad"}:
            problems.append(f"Unexpected target labels: {sorted(classes)}")
    for col in config.NUMERIC_FEATURES:
        if col in df.columns and (df[col] < 0).any():
            problems.append(f"Negative values in numeric column '{col}'.")
    return problems


def target_to_binary(series: pd.Series) -> pd.Series:
    """Map 'good'->1 (will repay), 'bad'->0."""
    return (series == config.POSITIVE_LABEL).astype(int)


def basic_profile(df: pd.DataFrame) -> dict:
    """JSON-serialisable dataset summary for dashboard / API."""
    class_counts = (
        df[config.TARGET_COL].value_counts().to_dict() if config.TARGET_COL in df else {}
    )
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": {
            c: {
                "mean": float(df[c].mean()),
                "median": float(df[c].median()),
                "min": float(df[c].min()),
                "max": float(df[c].max()),
            }
            for c in config.NUMERIC_FEATURES
            if c in df.columns
        },
    }


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive stats for numeric columns only.

    Uses explicit ``select_dtypes`` instead of ``describe(numeric_only=True)``
    because pandas 3.0 removed the ``numeric_only`` keyword from
    ``DataFrame.describe``.
    """
    numeric_df = df.select_dtypes(include="number")
    return numeric_df.describe()
