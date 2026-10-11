"""Строгая валидация входных данных клиентов (pandera).

Схема описывает сырой CSV до feature engineering: наличие обязательных
колонок, типы и допустимые домены значений. `id` и `y` опциональны.
Лишние колонки разрешены (strict=False) — они игнорируются пайплайном.
"""

import numpy as np
import pandas as pd
import pandera.pandas as pa

_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
           "jul", "aug", "sep", "oct", "nov", "dec"]
_YES_NO = ["yes", "no"]

CLIENT_SCHEMA = pa.DataFrameSchema(
    columns={
        "age": pa.Column(int, checks=pa.Check.in_range(0, 120)),
        "job": pa.Column(str),
        "marital": pa.Column(str),
        "education": pa.Column(str),
        "default": pa.Column(str, checks=pa.Check.isin(_YES_NO + ["unknown"])),
        "balance": pa.Column(int),
        "housing": pa.Column(str, checks=pa.Check.isin(_YES_NO)),
        "loan": pa.Column(str, checks=pa.Check.isin(_YES_NO)),
        "contact": pa.Column(str),
        "day": pa.Column(int, checks=pa.Check.in_range(1, 31)),
        "month": pa.Column(str, checks=pa.Check.isin(_MONTHS)),
        "duration": pa.Column(int, checks=pa.Check.ge(0)),
        "campaign": pa.Column(int, checks=pa.Check.ge(0)),
        "pdays": pa.Column(int),
        "previous": pa.Column(int, checks=pa.Check.ge(0)),
    },
    strict=False,
    coerce=True,
)


class SchemaError(ValueError):
    """Ошибка валидации входного CSV с человекочитаемым сообщением."""


def validate_clients(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяет датафрейм по схеме; бросает SchemaError с деталями."""
    try:
        return CLIENT_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        cases = e.failure_cases[["column", "check", "failure_case"]].head(10)
        raise SchemaError(
            "Валидация входных данных не пройдена: "
            + "; ".join(
                f"колонка {r['column']}: {r['check']} (значение {r['failure_case']!r})"
                for _, r in cases.iterrows()
            )
        )
    except pa.errors.SchemaError as e:
        raise SchemaError(f"Валидация входных данных не пройдена: {e}")
