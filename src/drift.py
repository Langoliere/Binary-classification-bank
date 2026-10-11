"""Мониторинг дрейфа входных данных через PSI (Population Stability Index).

Базовое распределение строится на обучающих данных и сохраняется как
JSON-артефакт в MLflow вместе с моделью. Сервис сравнивает распределение
каждого прогнанного батча с базовым.

Интерпретация PSI: < 0.1 — дрейфа нет; 0.1–0.25 — умеренный; > 0.25 — сильный.
"""

import json

import numpy as np
import pandas as pd

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features

EPS = 1e-4
N_BINS = 10
PSI_WARN = 0.1
PSI_ALERT = 0.25


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, EPS, None)
    actual = np.clip(actual, EPS, None)
    return float(np.sum((expected - actual) * np.log(expected / actual)))


def build_baseline(df: pd.DataFrame) -> dict:
    """Строит baseline-распределения по обработанным фичам тренировочных данных."""
    features = build_features(df)
    baseline = {"numeric": {}, "categorical": {}}

    for col in NUMERIC_FEATURES:
        s = features[col].astype(float)
        edges = np.unique(np.quantile(s, np.linspace(0, 1, N_BINS + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        hist, _ = np.histogram(s, bins=edges)
        baseline["numeric"][col] = {
            "edges": edges.tolist(),
            "proportions": (hist / hist.sum()).tolist(),
        }

    for col in CATEGORICAL_FEATURES:
        freq = features[col].astype(str).value_counts(normalize=True)
        baseline["categorical"][col] = freq.to_dict()

    return baseline


def save_baseline(baseline: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False)


def load_baseline(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class DriftMonitor:
    """Считает PSI для входных батчей относительно baseline."""

    def __init__(self, baseline: dict):
        self.baseline = baseline
        self.last_report: dict | None = None
        self.history: list[dict] = []

    def score_batch(self, raw_df: pd.DataFrame) -> dict:
        """Возвращает PSI по каждому признаку + максимум по батчу."""
        try:
            features = build_features(raw_df)
        except Exception:
            return {}

        report = {}
        for col, base in self.baseline["numeric"].items():
            if col not in features:
                continue
            hist, _ = np.histogram(features[col].astype(float), bins=base["edges"])
            actual = hist / max(hist.sum(), 1)
            report[col] = _psi(np.array(base["proportions"]), actual)

        for col, base in self.baseline["categorical"].items():
            if col not in features:
                continue
            actual = features[col].astype(str).value_counts(normalize=True)
            cats = set(base) | set(actual.index)
            expected = np.array([base.get(c, 0.0) for c in cats])
            observed = np.array([actual.get(c, 0.0) for c in cats])
            report[col] = _psi(expected, observed)

        max_psi = max(report.values()) if report else 0.0
        result = {
            "psi": report,
            "max_psi": max_psi,
            "status": "alert" if max_psi > PSI_ALERT else ("warn" if max_psi > PSI_WARN else "ok"),
        }
        self.last_report = result
        self.history.append(result)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        return result
