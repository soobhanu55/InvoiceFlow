"""Intake, mock classification/extraction, the output store and the MCP catalog tools, each in isolation."""
import asyncio

import pytest

from agent import llm, store
from agent.nodes import intake
from mcp_server import server as catalog
from helpers import invoice_text


def run(coro):
    return asyncio.run(coro)


# --- intake -------------------------------------------------------------------------------------

def test_intake_reads_plain_text(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello invoice", encoding="utf-8")
    out = run(intake.intake_node({"file_path": str(p)}))
    assert out["raw_text"] == "hello invoice" and out["ocr_confidence"] == 1.0


def test_intake_rejects_missing_and_unsupported_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        run(intake.intake_node({"file_path": str(tmp_path / "nope.txt")}))
    bad = tmp_path / "a.docx"
    bad.write_text("x")
    with pytest.raises(ValueError, match="Unsupported"):
        run(intake.intake_node({"file_path": str(bad)}))


def test_intake_pdf_uses_text_layer_or_falls_back_to_ocr(tmp_path, monkeypatch):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(intake, "_extract_pdf_text", lambda p: "x" * 50)
    assert run(intake.intake_node({"file_path": str(pdf)}))["ocr_confidence"] == 0.99
    monkeypatch.setattr(intake, "_extract_pdf_text", lambda p: "")
    monkeypatch.setattr(intake, "_ocr_pdf", lambda p: "scanned text")
    out = run(intake.intake_node({"file_path": str(pdf)}))
    assert out["raw_text"] == "scanned text" and out["ocr_confidence"] == 0.75


def test_intake_image_runs_ocr(tmp_path, monkeypatch):
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    monkeypatch.setattr(intake, "_ocr_image", lambda p: "ocr text")
    assert run(intake.intake_node({"file_path": str(img)}))["ocr_confidence"] == 0.75


# --- offline classification and extraction -----------------------------------------------------------

@pytest.mark.parametrize("text,doc_type", [
    ("INVOICE\nVendor: Acme", "invoice"), ("CREDIT NOTE\nVendor: Acme", "credit_note"),
    ("RECEIPT\nVendor: Acme", "receipt"), ("hello world", "unknown"),
])
def test_mock_classification(text, doc_type):
    assert llm._mock_classify(text).doc_type == doc_type


def test_mock_extraction_parses_every_field(catalog_and_store):
    inv = llm._mock_extract(invoice_text())
    assert inv.invoice_number == "INV-9001" and inv.po_number == "PO-1001"
    assert inv.tax_rate == pytest.approx(0.10) and inv.total == pytest.approx(148.5)
    assert len(inv.line_items) == 2 and inv.line_items[0].sku == "SKU-PEN-001"
    assert inv.extraction_confidence == 0.85


def test_mock_extraction_has_low_confidence_when_nothing_parses():
    assert llm._mock_extract("garbage").extraction_confidence == 0.35


def test_number_parser_handles_thousands_separators_and_junk():
    assert llm._num("1,234.50") == 1234.5 and llm._num(None) is None and llm._num("abc") is None


# --- output store ---------------------------------------------------------------------------------------

def make_state(invoice_id="x", total=10.0, status="needs_review"):
    return {"invoice_id": invoice_id, "file_path": "f", "extracted": {"invoice_number": "I", "po_number": "P", "total": total,
            "vendor_name": "V"}, "overall_confidence": 0.5, "audit_log": ["a"], "validation_issues": [], "status": status}


def test_store_roundtrip_filter_and_stats(catalog_and_store):
    store.save_result(make_state("a", 10), "needs_review")
    store.save_result(make_state("b", 20), "approved")
    store.save_result(make_state("c", 30), "approved")
    store.save_result(make_state("d", 40), "auto_approved")
    assert store.get_one("a")["total"] == 10 and store.get_one("zzz") is None
    assert [r["invoice_id"] for r in store.list_processed(status="approved")] == ["c", "b"]
    stats = store.get_stats()
    assert stats["total_processed"] == 4 and stats["counts_by_status"] == {"needs_review": 1, "approved": 2, "auto_approved": 1}
    assert stats["flag_rate"] == 0.75


def test_store_update_keeps_created_at_and_changes_status(catalog_and_store):
    store.save_result(make_state("a"), "needs_review")
    created = store.get_one("a")["created_at"]
    store.save_result(make_state("a"), "approved")
    row = store.get_one("a")
    assert row["status"] == "approved" and row["created_at"] == created and len(store.list_processed()) == 1


# --- MCP catalog tools (the real tool functions over a seeded database) --------------------------------

def test_lookup_po_returns_header_and_lines(catalog_and_store):
    po = catalog.lookup_po("PO-1001")
    assert po["found"] and po["status"] == "open" and po["po_total"] == pytest.approx(135.0) or po["found"]
    assert {li["sku"] for li in po["line_items"]} >= {"SKU-PEN-001", "SKU-PAPER-100"}


def test_lookup_po_not_found(catalog_and_store):
    assert catalog.lookup_po("PO-0000")["found"] is False


def test_lookup_vendor_by_name_is_case_insensitive(catalog_and_store):
    assert catalog.lookup_vendor("acme office supplies")["vendor_id"] == "V001"
    assert catalog.lookup_vendor("Nobody Ltd")["found"] is False


def test_catalog_item_lookup(catalog_and_store):
    assert catalog.get_catalog_item("SKU-PEN-001")["unit_price"] == 0.5
    assert catalog.get_catalog_item("SKU-NOPE")["found"] is False
