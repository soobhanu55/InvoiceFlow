"""Every run leaves a span tree with latency, tokens and cost."""
import pytest

from agent import telemetry
from helpers import invoice_text, run_pipeline
from test_failure_modes import Router, install

pytestmark = pytest.mark.usefixtures("catalog_and_store")


def test_offline_run_traces_every_node_and_model_step(tmp_path, monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    run_pipeline(tmp_path, invoice_text(), "trace-1")
    t = telemetry.get_trace("trace-1")
    names = [s["name"] for s in t["spans"]]
    for node in ("intake", "classification", "extraction", "validation", "matching", "human_review"):
        assert f"node.{node}" in names
    assert "llm.classify" in names and "llm.extract" in names
    llm_span = next(s for s in t["spans"] if s["name"] == "llm.extract")
    assert llm_span["depth"] == 1 and llm_span["attributes"]["gen_ai.request.model"] == "heuristic"
    assert t["summary"]["errors"] == 0 and t["summary"]["pipeline_ms"] > 0 and t["summary"]["cost_usd"] == 0


def test_tokens_and_cost_come_from_the_model_response(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    text = invoice_text()
    install(monkeypatch, ("gpt-4o-mini", Router(text)))
    run_pipeline(tmp_path, text, "trace-2")
    s = telemetry.get_trace("trace-2")["summary"]
    assert s["llm_calls"] == 2 and s["input_tokens"] == 2000 and s["output_tokens"] == 400
    assert s["cost_usd"] == pytest.approx(2 * (1000 * 0.15 + 200 * 0.60) / 1e6)


def test_a_model_without_a_known_price_is_left_unpriced(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    text = invoice_text()
    install(monkeypatch, ("some-new-model", Router(text)))
    run_pipeline(tmp_path, text, "trace-3")
    assert telemetry.get_trace("trace-3")["summary"]["cost_usd"] is None


def test_prices_can_be_overridden(monkeypatch):
    monkeypatch.setenv("LLM_PRICES_JSON", '{"my-model": [1.0, 2.0]}')
    assert telemetry.llm_cost("my-model", 1_000_000, 500_000) == 2.0


def test_failed_calls_show_up_as_error_spans(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    text = invoice_text()
    install(monkeypatch, ("primary", Router(text, 99, TimeoutError())))
    run_pipeline(tmp_path, text, "trace-4")
    t = telemetry.get_trace("trace-4")
    assert t["summary"]["errors"] >= 2 and any(s["error"] and s["name"].startswith("llm.") for s in t["spans"])
