"""FastAPI сервис скоринга вкладов.

Эндпоинты:
- GET  /                    — UI загрузки CSV
- GET  /health              — liveness/readiness
- POST /predict             — синхронный скоринг CSV -> CSV
- POST /predict/batch       — асинхронный скоринг (job_id)
- GET  /jobs/{id}           — статус джобы
- GET  /jobs/{id}/result    — скачать результат
- POST /admin/reload-model  — graceful reload модели (X-API-Key)
- GET  /drift/report        — отчёт мониторинга дрейфа
- GET  /metrics             — Prometheus
"""

import asyncio
import io
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # чтобы pickle нашёл src.features при загрузке модели

from src.api.config import settings
from src.api.jobs import JobStore
from src.api.model_manager import ModelManager
from src.api.observability import (
    DRIFT_MAX_PSI,
    DRIFT_PSI,
    JOBS_TOTAL,
    PREDICTION_LATENCY,
    PREDICTIONS_TOTAL,
    get_logger,
    instrument_metrics,
    setup_logging,
    setup_tracing,
)
from src.api.ratelimit import RateLimitMiddleware
from src.schema import SchemaError, validate_clients

setup_logging(settings.log_level)
log = get_logger("api")

model_manager = ModelManager(settings.model_name, settings.model_uri, settings.model_alias)
job_store = JobStore(settings.job_dir, settings.job_ttl_sec)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_manager.load()
    yield


app = FastAPI(title="Bank Deposit Classifier", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    RateLimitMiddleware,
    max_requests=settings.rate_limit_requests,
    window_sec=settings.rate_limit_window_sec,
)
instrument_metrics(app)
setup_tracing(app, settings.otlp_endpoint)

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = settings.max_upload_mb * 1024 * 1024


# ---------- вспомогательные функции ----------

async def spool_upload(file: UploadFile, dest) -> int:
    """Стримит upload во временный файл с лимитом размера. Возвращает размер."""
    size = 0
    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Файл больше лимита {settings.max_upload_mb} МБ")
        dest.write(chunk)
    return size


def _score_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Валидация + скоринг одного чанка + обновление метрик/дрейфа."""
    t0 = time.perf_counter()
    try:
        chunk = validate_clients(chunk)
        proba, drift = model_manager.predict_proba_chunk(chunk)
    except SchemaError:
        raise
    except Exception as e:
        raise RuntimeError(f"Ошибка скоринга: {e}")
    PREDICTION_LATENCY.observe(time.perf_counter() - t0)
    PREDICTIONS_TOTAL.labels("scored").inc(len(chunk))

    if drift:
        for feat, psi in drift["psi"].items():
            DRIFT_PSI.labels(feat).set(psi)
        DRIFT_MAX_PSI.set(drift["max_psi"])
        if drift["status"] != "ok":
            log.warning("drift_detected", status=drift["status"], max_psi=drift["max_psi"])

    return pd.DataFrame({
        "id": chunk["id"] if "id" in chunk.columns else chunk.index,
        "deposit_probability": proba.round(4),
        "prediction": (proba >= settings.threshold).astype(int),
    })


def score_csv(input_path: Path):
    """Чанковый скоринг CSV, генерирует текст результата (для StreamingResponse)."""
    header = True
    for chunk in pd.read_csv(input_path, chunksize=settings.predict_chunk_size):
        res = _score_chunk(chunk)
        yield res.to_csv(index=False, header=header, lineterminator="\n")
        header = False


def score_csv_to_file(input_path: Path, output_path: Path) -> int:
    """Чанковый скоринг CSV в файл (для async jobs). Возвращает число строк."""
    n_rows = 0
    with open(output_path, "w", encoding="utf-8", newline="") as out:
        for i, chunk in enumerate(pd.read_csv(input_path, chunksize=settings.predict_chunk_size)):
            res = _score_chunk(chunk)
            out.write(res.to_csv(index=False, header=(i == 0), lineterminator="\n"))
            n_rows += len(chunk)
    return n_rows


def require_admin(x_api_key: str | None = Header(default=None)):
    if not settings.admin_api_key or x_api_key != settings.admin_api_key:
        raise HTTPException(403, "Неверный или отсутствующий API-ключ")


# ---------- эндпоинты ----------

@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {
        "status": "ok" if model_manager.ready else "model_not_loaded",
        "model": {"name": settings.model_name, "version": model_manager.version,
                  "alias": settings.model_alias},
        "drift_monitoring": model_manager.drift_monitor is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Ожидается CSV-файл")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb")
    await spool_upload(file, tmp)
    tmp.close()
    input_path = Path(tmp.name)

    def stream():
        try:
            yield from score_csv(input_path)
        except SchemaError as e:
            PREDICTIONS_TOTAL.labels("validation_error").inc()
            # Стрим уже начался — сообщаем ошибку в теле
            yield f"\nerror,{e},\n"
        finally:
            input_path.unlink(missing_ok=True)

    return StreamingResponse(
        stream(), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


def _run_job(job_id: str):
    job = job_store.get(job_id)
    try:
        job.status = "running"
        result_path = job_store.dir / f"{job.id}_result.csv"
        n = score_csv_to_file(job.input_path, result_path)
        job.result_path = result_path
        job.n_rows = n
        job.status = "done"
        JOBS_TOTAL.labels("done").inc()
        log.info("job_done", job_id=job.id, rows=n)
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        JOBS_TOTAL.labels("failed").inc()
        log.error("job_failed", job_id=job.id, error=str(e))


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Асинхронный скоринг: возвращает job_id, результат — через /jobs/{id}."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Ожидается CSV-файл")
    job = await job_store.create()
    job.input_path = job_store.dir / f"{job.id}_input.csv"
    with open(job.input_path, "wb") as f:
        await spool_upload(file, f)
    JOBS_TOTAL.labels("created").inc()
    asyncio.get_running_loop().run_in_executor(None, _run_job, job.id)
    return {"job_id": job.id, "status_url": f"/jobs/{job.id}"}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Джоба не найдена")
    return {"id": job.id, "status": job.status, "error": job.error,
            "result_url": f"/jobs/{job.id}/result" if job.status == "done" else None}


@app.get("/jobs/{job_id}/result")
async def job_result(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Джоба не найдена")
    if job.status != "done" or not job.result_path:
        raise HTTPException(409, f"Результат недоступен: статус {job.status}")
    return FileResponse(job.result_path, media_type="text/csv",
                        filename=f"predictions_{job.id}.csv")


@app.post("/admin/reload-model", dependencies=[Depends(require_admin)])
async def reload_model():
    """Graceful reload модели из реестра (новая версия по alias) без рестарта."""
    return model_manager.reload()


@app.get("/drift/report")
async def drift_report():
    monitor = model_manager.drift_monitor
    if not monitor:
        return {"enabled": False, "detail": "drift baseline отсутствует у модели"}
    return {
        "enabled": True,
        "last_batch": monitor.last_report,
        "thresholds": {"warn": 0.1, "alert": 0.25},
        "history_size": len(monitor.history),
    }
