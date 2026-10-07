from src.data_loader import basic_profile, load_data, validate_data


def test_load_and_validate(df):
    assert len(df) == 1000
    assert set(["class"]).issubset(df.columns)
    assert df["class"].value_counts().to_dict() == {"good": 700, "bad": 300}
    assert validate_data(df) == []


def test_invalid_path_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_data("nonexistent.csv")


def test_profile(df):
    p = basic_profile(df)
    assert p["n_rows"] == 1000 and p["missing_values"] == 0
