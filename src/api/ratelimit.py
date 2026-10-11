"""Простой in-memory rate limiter (sliding window) по IP клиента.

Для горизонтального масштабирования в проде заменяется на Redis-backed
или внешний rate limiter на уровне API gateway / ingress.
"""

import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int, window_sec: int):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._hits: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        # Лимитируем только вычислительные эндпоинты
        if not request.url.path.startswith(("/predict",)):
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        hits = self._hits[ip]
        while hits and now - hits[0] > self.window_sec:
            hits.popleft()
        if len(hits) >= self.max_requests:
            return JSONResponse(
                {"detail": "Rate limit exceeded"}, status_code=429,
                headers={"Retry-After": str(int(self.window_sec - (now - hits[0])))},
            )
        hits.append(now)
        return await call_next(request)
