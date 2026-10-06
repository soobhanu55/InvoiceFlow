"""End-to-end runs of the real 6-node LangGraph (in-memory checkpointer, offline mock LLM, catalog served from a
seeded SQLite database) including the human-in-the-loop interrupt and resume paths."""
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent import store
from agent.graph import build_graph, pending_review
from helpers import invoice_text

pytestmark = pytest.mark.usefixtures("catalog_and_store")


@pytest.fixture(autouse=True)
def offline_llm(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AUTO_APPROVE_CONFIDENCE", "0.85")


def submit(tmp_path, text, invoice_id="inv-1", graph=None):
    path = tmp_path / f"{invoice_id}.txt"
    path.write_text(text, encoding="utf-8")
    graph = graph or build_graph(MemorySaver())
    cfg = {"configurable": {"thread_id": invoice_id}}
    import asyncio

    result = asyncio.run(graph.ainvoke({"invoice_id": invoice_id, "file_path": str(path)}, config=cfg))
    result["__review__"] = asyncio.run(pending_review(graph, cfg))  # None when the run finished
    return graph, cfg, result


def resume(graph, cfg, **decision):
    import asyncio

    return asyncio.run(graph.ainvoke(Command(resume=decision), config=cfg))


def test_clean_invoice_is_auto_approved_and_stored(catalog_and_store):
    _, _, result = submit(catalog_and_store, invoice_text())
    assert result["status"] == "auto_approved" and result["__review__"] is None
    assert result["match_result"]["all_matched"] and result["validation_passed"]
    row = store.get_one("inv-1")
    assert row["status"] == "auto_approved" and row["po_number"] == "PO-1001" and row["total"] == pytest.approx(148.5)


def test_price_mismatch_is_routed_to_a_human(catalog_and_store):
    _, _, result = submit(catalog_and_store, invoice_text(price_scale=1.2))
    payload = result["__review__"]
    assert payload is not None
    statuses = {li["status"] for li in payload["match_result"]["line_item_results"]}
    assert "price_mismatch" in statuses
    assert store.get_one("inv-1")["status"] == "needs_review"


def test_human_can_approve_after_review(catalog_and_store):
    graph, cfg, _ = submit(catalog_and_store, invoice_text(price_scale=1.2))
    final = resume(graph, cfg, decision="approve", corrections=None)
    assert final["status"] == "approved" and final["human_decision"] == "approve"
    assert store.get_one("inv-1")["status"] == "approved"


def test_human_can_reject(catalog_and_store):
    graph, cfg, _ = submit(catalog_and_store, invoice_text(qty_scale=3))
    final = resume(graph, cfg, decision="reject", corrections=None)
    assert final["status"] == "rejected" and store.get_one("inv-1")["status"] == "rejected"


def test_human_edit_applies_corrections(catalog_and_store):
    graph, cfg, _ = submit(catalog_and_store, invoice_text(tax_amount=99.0))
    final = resume(graph, cfg, decision="edit", corrections={"tax_amount": 13.5, "total": 148.5})
    assert final["status"] == "approved" and final["extracted"]["tax_amount"] == 13.5
    assert store.get_one("inv-1")["total"] == pytest.approx(148.5)


def test_unknown_po_is_flagged_with_reason_code(catalog_and_store):
    _, _, result = submit(catalog_and_store, invoice_text(po_number="PO-0000"))
    codes = {i["reason_code"] for i in result["__review__"]["validation_issues"]}
    assert "PO_NOT_FOUND" in codes


def test_tax_error_is_caught_by_validation(catalog_and_store):
    _, _, result = submit(catalog_and_store, invoice_text(tax_amount=50.0))
    codes = {i["reason_code"] for i in result["__review__"]["validation_issues"]}
    assert "TAX_MISCALCULATED" in codes


def test_credit_note_is_not_auto_approved(catalog_and_store):
    _, _, result = submit(catalog_and_store, invoice_text(doc="CREDIT NOTE"))
    assert result["__review__"] is not None
    codes = {i["reason_code"] for i in result["__review__"]["validation_issues"]}
    assert "DOC_TYPE_NOT_INVOICE" in codes


def test_high_threshold_forces_review_even_for_a_clean_invoice(catalog_and_store, monkeypatch):
    monkeypatch.setenv("AUTO_APPROVE_CONFIDENCE", "0.99")
    _, _, result = submit(catalog_and_store, invoice_text())
    assert result["__review__"] is not None


def test_two_invoices_have_independent_checkpoints(catalog_and_store):
    graph = build_graph(MemorySaver())
    _, cfg_a, _ = submit(catalog_and_store, invoice_text(price_scale=1.2), "a", graph)
    _, cfg_b, _ = submit(catalog_and_store, invoice_text(qty_scale=3), "b", graph)
    assert resume(graph, cfg_a, decision="approve", corrections=None)["status"] == "approved"
    assert store.get_one("b")["status"] == "needs_review"
    assert resume(graph, cfg_b, decision="reject", corrections=None)["status"] == "rejected"
    assert store.get_stats()["total_processed"] == 2
