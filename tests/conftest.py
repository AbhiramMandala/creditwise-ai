import pandas as pd
import pytest

from src import config


@pytest.fixture(scope="session")
def df():
    from src.data_loader import load_data

    return load_data()


@pytest.fixture(scope="session")
def sample_record(df):
    row = df.iloc[0]
    return {c: (float(row[c]) if c in config.NUMERIC_FEATURES else str(row[c])) for c in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES}


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    from src.credit_risk import train_and_select
    from src.data_loader import load_data

    pipe, metrics, info = train_and_select(load_data())
    return pipe, metrics, info
