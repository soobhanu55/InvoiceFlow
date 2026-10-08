"""The pipeline degrades instead of crashing: model fallback, heuristic fallback, tool outages -> human review."""
import pytest

from agent import llm, resilience as r
from agent.nodes import matching, validation
from helpers import FakeModel, invoice_text, run_pipeline

pytestmark = pytest.mark.usefixtures("catalog_and_store")


@pytest.fixture(autouse=True)
def real_llm_path(monkeypatch):
    async def instant(_):
        pass

    monkeypatch.setattr(r, "sleep", instant)
    r.BREAKERS.clear()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")  # only switches on the model path; the chain is faked below
    monkeypatch.setenv("AUTO_APPROVE_CONFIDENCE", "0.85")


def classify_ok(schema):
    return schema(doc_type="invoice", vendor_name="Acme Office Supplies", confidence=0.95)


def extract_ok(text):
    return lambda schema: llm._mock_extract(text).model_copy(update={"extraction_confidence": 0.95})


def install(monkeypatch, *models):
    monkeypatch.setattr(llm, "_chain", lambda: [(name, m) for name, m in models])


class Router(FakeModel):
    """One model that answers classification and extraction calls differently."""

    def __init__(self, text, fail_first=0, exc=None):
        super().__init__([])
        self.text, self.fail_first, self.exc = text, fail_first, exc

    def with_structured_output(self, schema, include_raw=False):
        outer = self
        inner = FakeModel([classify_ok if schema.__name__ == "DocumentClassification" else extract_ok(self.text)])
        real = inner.with_structured_output(schema)

        class Runner:
            async def ainvoke(self, messages):
                outer.calls += 1
                if outer.calls <= outer.fail_first:
                    raise outer.exc
                return await real.ainvoke(messages)

        return Runner()


def test_a_transient_model_error_is_retried_and_the_invoice_still_auto_approves(tmp_path, monkeypatch):
    text = invoice_text()
    m = Router(text, fail_first=1, exc=ConnectionError("connection reset"))
    install(monkeypatch, ("primary", m))
    result, review = run_pipeline(tmp_path, text)
    assert result["status"] == "auto_approved" and review is None and m.calls == 3  # 1 failure + 2 successful calls


def test_when_the_primary_model_is_down_the_secondary_answers(tmp_path, monkeypatch):
    text = invoice_text()
    down = Router(text, fail_first=99, exc=ConnectionError("connection refused"))
    backup = Router(text)
    install(monkeypatch, ("primary", down), ("backup", backup))
    result, _ = run_pipeline(tmp_path, text)
    assert result["status"] == "auto_approved" and backup.calls == 2 and not result.get("degraded")


def test_when_every_model_is_down_the_heuristic_parser_runs_and_a_human_reviews(tmp_path, monkeypatch):
    text = invoice_text()
    install(monkeypatch, ("primary", Router(text, 99, TimeoutError())), ("backup", Router(text, 99, RuntimeError("503 unavailable"))))
    result, review = run_pipeline(tmp_path, text)
    assert result.get("status") != "auto_approved" and review is not None
    codes = {i["reason_code"] for i in review["validation_issues"]}
    assert "DEGRADED_PIPELINE" in codes
    assert any("unavailable" in line or "timeout" in line for line in result["audit_log"])


def test_unparseable_model_output_is_retried_then_degrades(tmp_path, monkeypatch):
    text = invoice_text()
    install(monkeypatch, ("primary", Router(text, 99, ValueError("output parse error: invalid JSON"))))
    result, review = run_pipeline(tmp_path, text)
    assert review is not None and any(i["reason_code"] == "DEGRADED_PIPELINE" for i in review["validation_issues"])


def test_an_erp_outage_routes_to_review_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")

    async def down(*_):
        raise r.CallFailed("mcp.lookup_po", r.FailureKind.UNAVAILABLE, 3)

    for mod, names in ((validation, ["lookup_po"]), (matching, ["lookup_po", "lookup_vendor", "get_catalog_item"])):
        for n in names:
            monkeypatch.setattr(mod, n, down)
    result, review = run_pipeline(tmp_path, invoice_text())
    assert review is not None
    codes = {i["reason_code"] for i in review["validation_issues"]}
    assert "PO_CHECK_FAILED" in codes and review["match_result"]["all_matched"] is False
