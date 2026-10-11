"""ModelManager: потокобезопасная загрузка/перезагрузка модели из MLflow.

Поддерживает graceful reload без рестарта: новая модель и её drift-baseline
грузятся и валидируются под lock, затем указатель переключается атомарно.
"""

import threading

import mlflow
import mlflow.artifacts
import mlflow.sklearn
import pandas as pd

from src.api.observability import MODEL_INFO, get_logger
from src.drift import DriftMonitor, load_baseline

log = get_logger("model_manager")


class ModelManager:
    def __init__(self, model_name: str, model_uri: str | None, alias: str = "champion"):
        self.model_name = model_name
        self.model_uri = model_uri
        self.alias = alias
        self._lock = threading.RLock()
        self._model = None
        self._drift_monitor: DriftMonitor | None = None
        self.version: str | None = None

    # ---------- загрузка ----------

    def _resolve(self) -> tuple[str, str | None]:
        """Возвращает (model_uri, run_id версии)."""
        if self.model_uri:
            return self.model_uri, None
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version_by_alias(self.model_name, self.alias)
        return f"models:/{self.model_name}@{self.alias}", mv.run_id

    def _load(self):
        uri, run_id = self._resolve()
        model = mlflow.sklearn.load_model(uri)
        monitor = None
        if run_id:
            try:
                baseline_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path="drift_baseline.json"
                )
                monitor = DriftMonitor(load_baseline(baseline_path))
            except Exception as e:
                log.warning("drift_baseline не найден, мониторинг дрейфа выключен", error=str(e))
        # Проверка: загруженный объект поддерживает predict_proba
        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Загруженный объект не поддерживает predict_proba")
        return model, monitor

    def load(self) -> None:
        """Первичная загрузка (падает при ошибке — сервис не должен стартовать без модели)."""
        model, monitor = self._load()
        with self._lock:
            self._model, self._drift_monitor = model, monitor
            self.version = self._current_version()
        MODEL_INFO.labels(self.model_name, self.version or "?", self.alias).set(1)
        log.info("model_loaded", model=self.model_name, version=self.version, alias=self.alias)

    def reload(self) -> dict:
        """Graceful reload: при ошибке новой модели остаётся старая."""
        try:
            model, monitor = self._load()
        except Exception as e:
            log.error("model_reload_failed", error=str(e))
            return {"reloaded": False, "error": str(e), "version": self.version}
        with self._lock:
            self._model, self._drift_monitor = model, monitor
            self.version = self._current_version()
        MODEL_INFO.clear()
        MODEL_INFO.labels(self.model_name, self.version or "?", self.alias).set(1)
        log.info("model_reloaded", version=self.version)
        return {"reloaded": True, "version": self.version}

    def _current_version(self) -> str | None:
        try:
            mv = mlflow.tracking.MlflowClient().get_model_version_by_alias(
                self.model_name, self.alias)
            return mv.version
        except Exception:
            return None

    # ---------- инференс ----------

    def predict_proba_chunk(self, df: pd.DataFrame) -> tuple[object, object]:
        """Возвращает (proba, drift_report) для чанка."""
        with self._lock:
            model, monitor = self._model, self._drift_monitor
        if model is None:
            raise RuntimeError("Модель не загружена")
        proba = model.predict_proba(df)[:, 1]
        drift = monitor.score_batch(df) if monitor else None
        return proba, drift

    @property
    def drift_monitor(self) -> DriftMonitor | None:
        with self._lock:
            return self._drift_monitor

    @property
    def ready(self) -> bool:
        return self._model is not None
