"""Timeouts, retries with backoff, a circuit breaker and failure classification for every outside call (LLM, MCP).

The pipeline never lets an outside failure crash a run: after the retries a call raises `CallFailed` with a
classified `kind`, and the node that made it degrades (heuristic fallback, or a validation issue) so the invoice
lands in the human-review queue instead of returning a 500.
"""
from __future__ import annotations

import asyncio
import random
import time
from enum import Enum
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class FailureKind(str, Enum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    UNAVAILABLE = "unavailable"  # connection refused / reset / 5xx
    INVALID_OUTPUT = "invalid_output"  # the model returned something that does not parse into the schema
    POLICY = "policy"  # rejected by our own tool policy (never retried)
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"


RETRYABLE = {FailureKind.TIMEOUT, FailureKind.RATE_LIMIT, FailureKind.UNAVAILABLE, FailureKind.INVALID_OUTPUT}


class CallFailed(Exception):
    def __init__(self, name: str, kind: FailureKind, attempts: int, detail: str = "") -> None:
        super().__init__(f"{name}: {kind.value} after {attempts} attempt(s) {detail}".strip())
        self.name, self.kind, self.attempts = name, kind, attempts


class PolicyViolation(CallFailed):
    def __init__(self, name: str, detail: str) -> None:
        super().__init__(name, FailureKind.POLICY, 0, detail)


def classify(exc: BaseException) -> FailureKind:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return FailureKind.TIMEOUT
    text = f"{type(exc).__name__} {exc}".lower()
    if "ratelimit" in text or "rate limit" in text or "429" in text:
        return FailureKind.RATE_LIMIT
    if isinstance(exc, (ConnectionError, OSError)) or any(
        s in text for s in ("connection", "unavailable", "overloaded", "502", "503", "504", "500")
    ):
        return FailureKind.UNAVAILABLE
    if any(s in text for s in ("validation", "parse", "outputparser", "json")):
        return FailureKind.INVALID_OUTPUT
    return FailureKind.UNKNOWN


class CircuitBreaker:
    """Opens after `threshold` consecutive failures; lets one probe through after `reset_after` seconds."""

    def __init__(self, threshold: int = 3, reset_after: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.reset_after, self._clock = threshold, reset_after, clock
        self.failures, self.opened_at = 0, None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if self._clock() - self.opened_at >= self.reset_after else "open"

    def allow(self) -> bool:
        return self.state != "open"

    def success(self) -> None:
        self.failures, self.opened_at = 0, None

    def failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = self._clock()  # a failed half-open probe re-opens for another full period


BREAKERS: dict[str, CircuitBreaker] = {}
STATS = {"calls": 0, "retries": 0, "failures": 0, "short_circuited": 0}
sleep = asyncio.sleep  # tests replace this to run instantly


def breaker(name: str) -> CircuitBreaker:
    return BREAKERS.setdefault(name, CircuitBreaker())


async def call(
    name: str,
    fn: Callable[[], Awaitable[T]],
    *,
    timeout: float = 20.0,
    retries: int = 2,
    base_delay: float = 0.5,
) -> T:
    """Run `fn` under a timeout, retrying transient failures with jittered exponential backoff.

    One breaker per `name`: while it is open, calls fail immediately instead of waiting out another timeout."""
    br = breaker(name)
    STATS["calls"] += 1
    if not br.allow():
        STATS["short_circuited"] += 1
        raise CallFailed(name, FailureKind.CIRCUIT_OPEN, 0)
    attempt = 0
    while True:
        attempt += 1
        try:
            result = await asyncio.wait_for(fn(), timeout)
            br.success()
            return result
        except CallFailed:
            raise
        except Exception as exc:  # noqa: BLE001 - classified below
            kind = classify(exc)
            if kind in RETRYABLE and attempt <= retries:
                STATS["retries"] += 1
                await sleep(base_delay * 2 ** (attempt - 1) * (0.5 + random.random()))
                continue
            br.failure()
            STATS["failures"] += 1
            raise CallFailed(name, kind, attempt, str(exc)[:120]) from exc


def snapshot() -> dict:
    return {**STATS, "breakers": {n: b.state for n, b in BREAKERS.items()}}
