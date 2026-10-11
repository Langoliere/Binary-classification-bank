"""Observability: структурные логи (structlog), Prometheus-метрики, OTEL-трейсинг."""

import logging
import sys

import structlog
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

# ---------- Метрики Prometheus ----------
PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Количество скорингованных клиентов", ["status"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Время скоринга одного чанка",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
)
DRIFT_PSI = Gauge("drift_psi", "Последний PSI по признаку", ["feature"])
DRIFT_MAX_PSI = Gauge("drift_max_psi", "Максимальный PSI последнего батча")
MODEL_INFO = Gauge("model_info", "Информация о загруженной модели",
                   ["model_name", "version", "alias"])
JOBS_TOTAL = Counter("async_jobs_total", "Асинхронные batch-джобы", ["status"])


def setup_logging(level: str = "INFO") -> None:
    """JSON-логи в stdout — формат для сбора (Loki/ELK)."""
    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=getattr(logging, level.upper(), logging.INFO)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    return structlog.get_logger(name)


def setup_tracing(app, otlp_endpoint: str | None) -> None:
    """OpenTelemetry-трейсинг FastAPI (активен только если задан OTLP endpoint)."""
    if not otlp_endpoint:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": "bank-deposit-api"}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
    except Exception as e:  # трейсинг не должен ломать сервис
        logging.getLogger(__name__).warning("OTEL tracing disabled: %s", e)


def instrument_metrics(app) -> None:
    """Стандартные HTTP-метрики FastAPI -> /metrics."""
    Instrumentator().instrument(app).expose(app, include_in_schema=False)
