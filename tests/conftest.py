import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_clients(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "id": range(n),
        "age": rng.randint(18, 90, n),
        "job": rng.choice(["technician", "blue-collar", "management", "unknown"], n),
        "marital": rng.choice(["married", "single", "divorced"], n),
        "education": rng.choice(["primary", "secondary", "tertiary", "unknown"], n),
        "default": rng.choice(["yes", "no"], n),
        "balance": rng.randint(-1000, 10000, n),
        "housing": rng.choice(["yes", "no"], n),
        "loan": rng.choice(["yes", "no"], n),
        "contact": rng.choice(["cellular", "telephone", "unknown"], n),
        "day": rng.randint(1, 31, n),
        "month": rng.choice(["jan", "feb", "mar", "apr", "may"], n),
        "duration": rng.randint(0, 1000, n),
        "campaign": rng.randint(0, 10, n),
        "pdays": rng.randint(-1, 100, n),
        "previous": rng.randint(0, 5, n),
        "poutcome": rng.choice(["unknown", "success", "failure"], n),
    })


@pytest.fixture(scope="session")
def df_valid():
    return make_clients()


@pytest.fixture(scope="session")
def trained_model(df_valid):
    """Маленькая реальная модель для тестов — без обращения к MLflow."""
    from src.pipeline import build_pipeline

    df = df_valid.copy()
    y = ((df["balance"] > 2000) | (df["duration"] > 500)).astype(int)
    model = build_pipeline(n_estimators=10, max_depth=3)
    model.fit(df, y)
    return model


@pytest.fixture()
def client(trained_model, monkeypatch):
    """TestClient с подменённой загрузкой модели (без MLflow)."""
    from src.api import main as api_main

    monkeypatch.setattr(api_main.model_manager, "load", lambda: None)
    api_main.model_manager._model = trained_model
    api_main.model_manager._drift_monitor = None
    api_main.model_manager.version = "test"

    with TestClient(api_main.app) as c:
        yield c


@pytest.fixture()
def csv_bytes(df_valid) -> bytes:
    buf = io.StringIO()
    df_valid.to_csv(buf, index=False)
    return buf.getvalue().encode()
