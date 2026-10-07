"""Reproducible preprocessing: one ColumnTransformer used by train AND inference.

Numeric -> StandardScaler, Categorical -> OneHotEncoder(handle_unknown='ignore').
This replaces legacy ``pd.get_dummies`` which produced mismatched columns for
single-row user input and silently dropped unseen categories.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Human-readable output feature names after one-hot encoding."""
    try:
        names = preprocessor.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return []
