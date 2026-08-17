"""Retry policy + a minimal per-dependency circuit breaker.
See docs/06-harness.md.

`deadline_aware` is the part that matters: a blind retry is worse than none
under a latency SLO. We only retry when the remaining budget can plausibly
absorb another attempt.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

from api.harness.deadline import Deadline

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_attempts: int = 2
    base_ms: float = 50.0
    retry_on: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)

    async def run(
        self,
        fn: Callable[[], Awaitable[T]],
        deadline: Deadline,
        min_cost_ms: float,
    ) -> T:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            if not deadline.affords(min_cost_ms):
                break
            try:
                return await fn()
            except self.retry_on as exc:
                last_exc = exc
                if attempt >= self.max_attempts or not deadline.affords(min_cost_ms * 2):
                    break
                backoff = (self.base_ms * (2 ** (attempt - 1))) * (1 + random.random() * 0.2)
                await asyncio.sleep(backoff / 1000.0)
        assert last_exc is not None
        raise last_exc


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    window_s: float = 30.0
    cooldown_s: float = 60.0
    _failures: list[float] = field(default_factory=list)
    _opened_at: float | None = None

    def record_failure(self) -> None:
        now = time.time()
        self._failures = [t for t in self._failures if now - t < self.window_s]
        self._failures.append(now)
        if len(self._failures) >= self.failure_threshold:
            self._opened_at = now

    def record_success(self) -> None:
        self._failures.clear()
        self._opened_at = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.time() - self._opened_at >= self.cooldown_s:
            self._opened_at = None  # half-open: allow the next attempt through
            return False
        return True
