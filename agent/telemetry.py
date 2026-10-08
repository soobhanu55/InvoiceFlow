"""Tracing: one OpenTelemetry span per graph node, LLM call and MCP tool call, with latency, tokens and cost.

Finished spans are kept per invoice in a small in-process ring buffer (served at GET /traces/{invoice_id}), and are
also exported over OTLP when OTEL_EXPORTER_OTLP_ENDPOINT is set (Langfuse, Jaeger and Grafana Tempo all accept it).
LLM spans use the OpenTelemetry GenAI attribute names (gen_ai.request.model, gen_ai.usage.*).
"""
from __future__ import annotations

import contextvars
import functools
import json
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

MAX_TRACES = 200

# USD per 1M tokens (input, output): published list prices when this was written; override with
# LLM_PRICES_JSON='{"model": [in, out]}'. A model that is not listed is left unpriced rather than guessed.
PRICES: dict[str, tuple[float, float]] = {"heuristic": (0.0, 0.0), "gpt-4o-mini": (0.15, 0.60), "llama-3.3-70b-versatile": (0.59, 0.79)}

_invoice_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("invoice_id", default=None)


def llm_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    prices = {**PRICES, **{k: tuple(v) for k, v in json.loads(os.environ.get("LLM_PRICES_JSON", "{}")).items()}}
    if model not in prices:
        return None
    p_in, p_out = prices[model]
    return round((input_tokens * p_in + output_tokens * p_out) / 1e6, 6)


class RingExporter(SpanExporter):
    """Keeps finished spans grouped by invoice id; the oldest invoice is dropped past MAX_TRACES."""

    def __init__(self) -> None:
        self._by_invoice: OrderedDict[str, list[dict]] = OrderedDict()
        self._lock = threading.Lock()

    def export(self, spans) -> SpanExportResult:
        with self._lock:
            for s in spans:
                inv = (s.attributes or {}).get("invoice.id", "-")
                self._by_invoice.setdefault(inv, []).append({
                    "id": f"{s.context.span_id:016x}",
                    "parent": f"{s.parent.span_id:016x}" if s.parent else None,
                    "name": s.name,
                    "start_ns": s.start_time,
                    "ms": round((s.end_time - s.start_time) / 1e6, 2),
                    "error": s.status.status_code == trace.StatusCode.ERROR,
                    "attributes": {k: v for k, v in (s.attributes or {}).items() if k != "invoice.id"},
                })
            while len(self._by_invoice) > MAX_TRACES:
                self._by_invoice.popitem(last=False)
        return SpanExportResult.SUCCESS

    def spans(self, invoice_id: str) -> list[dict]:
        with self._lock:
            return sorted((dict(s) for s in self._by_invoice.get(invoice_id, [])), key=lambda s: s["start_ns"])

    def clear(self) -> None:
        with self._lock:
            self._by_invoice.clear()


ring = RingExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(ring))
if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):  # pragma: no cover - needs a collector
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
_tracer = _provider.get_tracer("invoiceflow")


@contextmanager
def span(name: str, **attrs: Any):
    """Open a span tagged with the current invoice id; an exception marks it as an error and propagates."""
    with _tracer.start_as_current_span(name) as sp:
        sp.set_attribute("invoice.id", _invoice_id.get() or "-")
        for k, v in attrs.items():
            sp.set_attribute(k, v)
        try:
            yield sp
        except Exception as exc:
            sp.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)[:200]))
            sp.set_attribute("error.type", type(exc).__name__)
            raise


def record_llm(sp, model: str, usage: dict | None) -> None:
    usage = usage or {}
    tin, tout = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    sp.set_attribute("gen_ai.request.model", model)
    sp.set_attribute("gen_ai.usage.input_tokens", tin)
    sp.set_attribute("gen_ai.usage.output_tokens", tout)
    cost = llm_cost(model, tin, tout)
    if cost is not None:
        sp.set_attribute("llm.cost_usd", cost)


def traced_node(name: str) -> Callable:
    """Decorator for a graph node: sets the invoice id for every span below it and times the node."""

    def wrap(fn):
        @functools.wraps(fn)
        async def inner(state, *a, **kw):
            token = _invoice_id.set(state.get("invoice_id"))
            try:
                with span(f"node.{name}"):
                    return await fn(state, *a, **kw)
            finally:
                _invoice_id.reset(token)

        return inner

    return wrap


def get_trace(invoice_id: str) -> dict:
    spans = ring.spans(invoice_id)
    by_id = {s["id"]: s for s in spans}
    for s in spans:
        depth, p = 0, s["parent"]
        while p in by_id:
            depth, p = depth + 1, by_id[p]["parent"]
        s["depth"] = depth
        s.pop("start_ns")
    llm = [s for s in spans if "gen_ai.request.model" in s["attributes"]]
    costs = [s["attributes"].get("llm.cost_usd") for s in llm]
    return {
        "invoice_id": invoice_id,
        "spans": spans,
        "summary": {
            "pipeline_ms": round(sum(s["ms"] for s in spans if s["name"].startswith("node.")), 2),
            "llm_calls": len(llm),
            "input_tokens": sum(s["attributes"]["gen_ai.usage.input_tokens"] for s in llm),
            "output_tokens": sum(s["attributes"]["gen_ai.usage.output_tokens"] for s in llm),
            # None when any call used a model with no known price
            "cost_usd": None if any(c is None for c in costs) else round(sum(costs), 6),
            "tool_calls": sum(1 for s in spans if s["name"].startswith("mcp.")),
            "errors": sum(1 for s in spans if s["error"]),
        },
    }
