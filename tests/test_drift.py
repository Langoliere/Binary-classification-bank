import pandas as pd

from src.drift import DriftMonitor, build_baseline
from tests.conftest import make_clients


def test_no_drift_same_distribution():
    df = make_clients(500, seed=1)
    monitor = DriftMonitor(build_baseline(df))
    report = monitor.score_batch(make_clients(500, seed=2))
    assert report["max_psi"] < 0.25
    assert report["status"] in ("ok", "warn")


def test_drift_detected_on_shift():
    df = make_clients(500, seed=1)
    monitor = DriftMonitor(build_baseline(df))
    shifted = make_clients(500, seed=3)
    shifted["age"] = 95  # все клиенты стали стариками
    shifted["month"] = "may"
    report = monitor.score_batch(shifted)
    assert report["max_psi"] > 0.25
    assert report["status"] == "alert"
