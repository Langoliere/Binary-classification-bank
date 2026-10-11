"""Конфигурация сервиса через переменные окружения."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("MODEL_NAME", "bank-deposit-classifier")
    model_uri: str | None = os.getenv("MODEL_URI")  # переопределяет реестр
    model_alias: str = os.getenv("MODEL_ALIAS", "champion")
    threshold: float = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

    # Лимиты
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
    predict_chunk_size: int = int(os.getenv("PREDICT_CHUNK_SIZE", "20000"))

    # Rate limiting: N запросов за окно на IP
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
    rate_limit_window_sec: int = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))

    # API-ключ для админ-эндпоинтов (reload модели)
    admin_api_key: str | None = os.getenv("ADMIN_API_KEY")

    # Асинхронные batch-джобы
    job_dir: str = os.getenv("JOB_DIR", "/tmp/bank_jobs")
    job_ttl_sec: int = int(os.getenv("JOB_TTL_SEC", "3600"))

    # Наблюдаемость
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    otlp_endpoint: str | None = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")


settings = Settings()
