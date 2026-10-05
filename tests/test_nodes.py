"""Offline unit tests for the rule-based nodes (no LLM, no MCP server, no API keys).

The MCP catalog calls are stubbed so the business rules are tested in isolation.
"""
import asyncio

import pytest

from agent.nodes import human_review, matching, validation
from agent.state import ExtractedInvoice, LineItem

PO = {
    "found": True, "status": "open", "vendor_id": "V1", "po_total": 200.0,
    "line_items": [{"sku": "A1", "qty": 2, "unit_price": 100.0}],
}


def run(coro):
    return asyncio.run(coro)


def invoice(**kw) -> dict:
    base = dict(
        invoice_number="INV-1", po_number="PO-1", vendor_name="Acme", due_date="2026-01-01",
        line_items=[LineItem(sku="A1", description="Widget", quantity=2, unit_price=100.0, line_total=200.0)],
        subtotal=200.0, tax_rate=0.1, tax_amount=20.0, total=220.0,
    )
    base.update(kw)
    return ExtractedInvoice(**base).model_dump()


@pytest.fixture(autouse=True)
def stub_mcp(monkeypatch):
    async def lookup_po(n): return PO if n == "PO-1" else {"found": False}
    async def lookup_vendor(n): return {"found": True, "vendor_id": "V1"}
    async def get_catalog_item(s): return {"found": s in ("A1", "B2")}
    for mod in (validation, matching):
        monkeypatch.setattr(mod, "lookup_po", lookup_po)
    monkeypatch.setattr(matching, "lookup_vendor", lookup_vendor)
    monkeypatch.setattr(matching, "get_catalog_item", get_catalog_item)


def validate(**kw):
    out = run(validation.validation_node({"extracted": invoice(**kw), "doc_type": "invoice"}))
    return out, {i["reason_code"] for i in out["validation_issues"]}


def test_clean_invoice_passes():
    out, codes = validate()
    assert out["validation_passed"] and not codes


@pytest.mark.parametrize("kw,code", [
    (dict(tax_amount=25.0), "TAX_MISCALCULATED"),
    (dict(total=999.0), "TOTAL_MISMATCH"),
    (dict(subtotal=150.0), "LINE_ITEMS_SUM_MISMATCH"),
    (dict(invoice_number=None), "MISSING_INVOICE_NUMBER"),
    (dict(po_number=None), "MISSING_PO_NUMBER"),
    (dict(po_number="PO-404"), "PO_NOT_FOUND"),
    (dict(line_items=[]), "NO_LINE_ITEMS"),
])
def test_validation_reason_codes(kw, code):
    out, codes = validate(**kw)
    assert code in codes


def test_missing_due_date_is_warning_only():
    out, codes = validate(due_date=None)
    assert "MISSING_DUE_DATE" in codes and out["validation_passed"]


def test_no_extraction_fails():
    out = run(validation.validation_node({}))
    assert out["validation_passed"] is False


def match(**kw):
    return run(matching.matching_node({"extracted": invoice(**kw)}))["match_result"]


def test_matching_clean():
    assert match()["all_matched"]


def test_matching_price_mismatch():
    li = LineItem(sku="A1", description="W", quantity=2, unit_price=120.0, line_total=240.0)
    r = match(line_items=[li], subtotal=240.0)
    assert r["line_item_results"][0]["status"] == "price_mismatch" and not r["all_matched"]


def test_matching_qty_mismatch():
    li = LineItem(sku="A1", description="W", quantity=5, unit_price=100.0, line_total=500.0)
    assert match(line_items=[li], subtotal=500.0)["line_item_results"][0]["status"] == "qty_mismatch"


def test_matching_sku_not_in_po_vs_not_in_catalog():
    mk = lambda sku: LineItem(sku=sku, description="W", quantity=1, unit_price=1.0, line_total=1.0)
    assert match(line_items=[mk("B2")])["line_item_results"][0]["status"] == "not_in_po"
    assert match(line_items=[mk("ZZ")])["line_item_results"][0]["status"] == "not_in_catalog"


def test_confidence_penalises_errors_and_mismatch():
    clean = {"classification_confidence": 0.9, "extraction_confidence": 0.9}
    assert human_review.compute_overall_confidence(clean) == 0.9
    bad = {**clean, "validation_issues": [{"severity": "error"}], "match_result": {"all_matched": False}}
    assert human_review.compute_overall_confidence(bad) == pytest.approx(0.55)
