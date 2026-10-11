"""Асинхронные batch-джобы: файл скоринится в фоне, результат забирается позже.

Хранилище — in-memory + файлы в job_dir (per-инстанс). Для multi-replica
деплоя заменить на очередь (Celery/RQ) + объектное хранилище.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    id: str
    status: str = "pending"          # pending | running | done | failed (статусы джобы)
    created_at: float = field(default_factory=time.time)
    input_path: Path | None = None
    result_path: Path | None = None
    error: str | None = None
    n_rows: int = 0


class JobStore:
    def __init__(self, job_dir: str, ttl_sec: int = 3600):
        self.dir = Path(job_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_sec
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex[:12])
        async with self._lock:
            self._jobs[job.id] = job
        self._gc()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def update(self, job: Job, **fields) -> None:
        async with self._lock:
            for k, v in fields.items():
                setattr(job, k, v)

    def _gc(self) -> None:
        now = time.time()
        stale = [j for j in self._jobs.values() if now - j.created_at > self.ttl]
        for j in stale:
            for p in (j.input_path, j.result_path):
                if p and p.exists():
                    p.unlink(missing_ok=True)
            del self._jobs[j.id]
