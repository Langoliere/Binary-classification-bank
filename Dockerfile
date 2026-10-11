FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Устанавливаем зафиксированные версии из lock-файла для воспроизводимости
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY src/ ./src/

# Непривилегированный пользователь
RUN useradd -m appuser && mkdir -p /app/mlruns /tmp/bank_jobs && chown -R appuser /app /tmp/bank_jobs
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
