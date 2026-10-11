# Bank Deposit Classifier

Production-сервис бинарной классификации: предсказывает, заключит ли клиент
договор на вклад. Построен на основе `notebooks/binary-classification.ipynb`.

## Архитектура

```
data/data.csv                     — обучающие данные (750k строк)
notebooks/                        — исследовательский ноутбук
src/
  features.py                     — feature engineering (единый код train/inference)
  pipeline.py                     — sklearn Pipeline: features → RobustScaler/OHE → XGBoost
  schema.py                       — pandera-схема входных данных (валидация CSV)
  drift.py                        — PSI-мониторинг дрейфа (baseline — артефакт MLflow)
  train.py                        — обучение: Optuna (nested runs) + MLflow + registry aliases
  api/
    main.py                       — FastAPI: sync/async скоринг, reload, drift report
    config.py                     — конфигурация через env
    model_manager.py              — потокобезопасная загрузка/graceful reload модели
    ratelimit.py                  — sliding-window rate limiter по IP
    jobs.py                       — in-memory хранилище async batch-джобов
    observability.py              — structlog, Prometheus, OpenTelemetry
    static/index.html             — UI загрузки CSV
tests/                            — pytest: схема, дрейф, API
Dockerfile, docker-compose.yml    — контейнеризация (api + mlflow + prometheus)
.github/workflows/ci.yml          — CI: тесты + smoke-train + docker build
```

Модель — единый sklearn `Pipeline` (FunctionTransformer + ColumnTransformer +
XGBoost), сериализуется целиком в MLflow: inference принимает сырой CSV.

## Эндпоинты

| Метод | Путь | Описание |
|---|---|---|
| GET | `/` | UI загрузки CSV |
| GET | `/health` | liveness/readiness + версия модели |
| POST | `/predict` | Синхронный скоринг CSV → CSV (чнковая обработка, стриминг ответа) |
| POST | `/predict/batch` | Асинхронный скоринг → `job_id` |
| GET | `/jobs/{id}` | Статус джобы |
| GET | `/jobs/{id}/result` | Скачать результат |
| POST | `/admin/reload-model` | Graceful reload модели (заголовок `X-API-Key`) |
| GET | `/drift/report` | PSI-отчёт мониторинга дрейфа |
| GET | `/metrics` | Prometheus |

## Модельный цикл

- **Champion/challenger**: обучение регистрирует версию с alias
  (`--alias champion|challenger`); сервис резолвит `models:/name@champion`.
  Canary-выкат challenger'а — через отдельный инстанс с `MODEL_ALIAS=challenger`.
- **Graceful reload**: `POST /admin/reload-model` атомарно переключает модель
  без рестарта; при ошибке новой версии остаётся старая.
- **Мониторинг дрейфа**: на обучении строится PSI-baseline (артефакт рана);
  каждый скоринг-батч сравнивается с ним — результат в `/drift/report` и
  метриках `drift_psi`/`drift_max_psi` (warn > 0.1, alert > 0.25).
- **Optuna → MLflow**: каждый trial — nested run с параметрами и ROC-AUC;
  сводка trials логируется CSV-артефактом.

## Защита и надёжность

- Валидация входа: pandera-схема (колонки, типы, домены значений) → 422.
- Лимит размера загрузки (`MAX_UPLOAD_MB`, по умолчанию 50) — файл стримится
  во временное хранилище, а не в память; скоринг чанками `PREDICT_CHUNK_SIZE`.
- Rate limiting: sliding window по IP на `/predict*` (`RATE_LIMIT_*`).
- Админ-эндпоинты защищены `ADMIN_API_KEY` (если ключ не задан — закрыты).

## Observability

- **Логи**: structlog → JSON в stdout (уровень `LOG_LEVEL`).
- **Метрики**: `/metrics` — HTTP (instrumentator) + кастомные
  (`predictions_total`, `prediction_latency_seconds`, `drift_psi`, `async_jobs_total`, `model_info`).
- **Трейсинг**: OpenTelemetry FastAPI → OTLP endpoint (`OTEL_EXPORTER_OTLP_ENDPOINT`).

## Запуск локально

```
pip install -r requirements-dev.txt   # prod-зависимости + pytest/httpx
# либо строго зафиксированные версии: pip install -r requirements.lock

# Обучение (регистрация с alias champion)
python -m src.train --data data/data.csv --n-trials 20 --alias champion

# Сервис
uvicorn src.api.main:app --port 8000
```

MLflow UI экспериментов: `mlflow ui --port 5000`.

## Docker

```
docker compose up --build        # api:8000, mlflow:5000, prometheus:9090
ADMIN_API_KEY=secret docker compose up --build
```

## Тесты и CI

```
pytest tests -v
```

CI (GitHub Actions): lint (compileall) → pytest → smoke-train на сэмпле → docker build.

## Проверенные результаты

Smoke-прогон (3 trials, sample 60k): ROC-AUC test = 0.966, PR-AUC = 0.795, recall@0.5 = 0.91.
