import pandas as pd
import pytest

from src.schema import SchemaError, validate_clients
from tests.conftest import make_clients


def test_valid_df_passes(df_valid):
    out = validate_clients(df_valid)
    assert len(out) == len(df_valid)


def test_missing_required_column(df_valid):
    with pytest.raises(SchemaError, match="Валидация"):
        validate_clients(df_valid.drop(columns=["age"]))


def test_invalid_month(df_valid):
    df = df_valid.copy()
    df.loc[0, "month"] = "januar"
    with pytest.raises(SchemaError):
        validate_clients(df)


def test_invalid_age_range(df_valid):
    df = df_valid.copy()
    df.loc[0, "age"] = 500
    with pytest.raises(SchemaError):
        validate_clients(df)


def test_extra_columns_allowed(df_valid):
    df = df_valid.copy()
    df["some_extra"] = "x"
    assert len(validate_clients(df)) == len(df)


def test_target_column_ignored(df_valid):
    df = df_valid.copy()
    df["y"] = 1
    assert len(validate_clients(df)) == len(df)
