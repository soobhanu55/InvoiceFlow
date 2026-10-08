"""Retry, timeout, circuit breaker and failure classification."""
import asyncio

import pytest

from agent import resilience as r


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    async def instant(_):
        pass

    monkeypatch.setattr(r, "sleep", instant)
    r.BREAKERS.clear()


def run(coro):
    return asyncio.run(coro)


def flaky(failures, exc):
    state = {"n": 0}

    async def fn():
        state["n"] += 1
        if state["n"] <= failures:
            raise exc
        return "ok"

    return fn, state


def test_transient_failures_are_retried_until_success():
    fn, state = flaky(2, ConnectionError("connection reset"))
    assert run(r.call("t.retry", fn, retries=2)) == "ok" and state["n"] == 3


def test_retries_are_bounded_and_the_failure_is_classified():
    fn, state = flaky(99, ConnectionError("connection refused"))
    with pytest.raises(r.CallFailed) as e:
        run(r.call("t.bound", fn, retries=2))
    assert e.value.kind is r.FailureKind.UNAVAILABLE and e.value.attempts == 3 and state["n"] == 3


def test_a_programming_error_is_not_retried():
    fn, state = flaky(99, ValueError("boom"))
    with pytest.raises(r.CallFailed) as e:
        run(r.call("t.bug", fn, retries=5))
    assert e.value.kind is r.FailureKind.UNKNOWN and state["n"] == 1


def test_timeout_is_enforced_and_classified():
    async def slow():
        await asyncio.sleep(5)

    with pytest.raises(r.CallFailed) as e:
        run(r.call("t.slow", slow, timeout=0.01, retries=0))
    assert e.value.kind is r.FailureKind.TIMEOUT


@pytest.mark.parametrize("exc,kind", [
    (RuntimeError("429 Too Many Requests"), r.FailureKind.RATE_LIMIT),
    (RuntimeError("Service Unavailable 503"), r.FailureKind.UNAVAILABLE),
    (ValueError("output parse error: bad json"), r.FailureKind.INVALID_OUTPUT),
])
def test_classify(exc, kind):
    assert r.classify(exc) is kind


def test_breaker_opens_short_circuits_and_recovers_after_the_reset_period():
    now = [0.0]
    br = r.CircuitBreaker(threshold=2, reset_after=30, clock=lambda: now[0])
    r.BREAKERS["t.cb"] = br
    fn, state = flaky(99, ConnectionError("connection refused"))
    for _ in range(2):
        with pytest.raises(r.CallFailed):
            run(r.call("t.cb", fn, retries=0))
    assert br.state == "open"
    calls = state["n"]
    with pytest.raises(r.CallFailed) as e:
        run(r.call("t.cb", fn, retries=0))
    assert e.value.kind is r.FailureKind.CIRCUIT_OPEN and state["n"] == calls  # the dependency was not touched

    now[0] = 31
    assert br.state == "half_open"
    good, _ = flaky(0, None)
    assert run(r.call("t.cb", good)) == "ok" and br.state == "closed"
