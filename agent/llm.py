"""LLM abstraction used by the classification + extraction nodes.

Primary path: a real LangChain chat model (Anthropic Claude or OpenAI)
called with `.with_structured_output(<PydanticSchema>)`, i.e. genuine
function-calling / structured output -- no regex parsing of vendor
documents.

Fallback path: if neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set, we
fall back to a small deterministic heuristic parser so the *entire*
6-node pipeline (including this repo's test harness) still runs end to
end offline, without requiring API keys. This fallback is intentionally
simple and template-shaped -- it is not the generalizable path the spec
asks for, it exists purely so the graph is testable without network
access. Set an API key to exercise the real structured-output path.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

from agent.state import DocumentClassification, ExtractedInvoice, LineItem

CLASSIFICATION_SYSTEM_PROMPT = """You are a document classification assistant for an accounts-payable
intake pipeline. Given the raw OCR text of a scanned business document,
determine whether it is an "invoice", a "receipt", a "credit_note", or
"unknown", identify the vendor/supplier name as printed on the document,
and give a calibrated confidence score between 0 and 1."""

EXTRACTION_SYSTEM_PROMPT = """You are a data extraction assistant for an accounts-payable intake
pipeline. Given the raw OCR text of an invoice, extract every field of
the schema as precisely as possible: invoice_number, po_number,
vendor_name, invoice_date, due_date, currency, the full line_items table
(sku, description, quantity, unit_price, line_total), subtotal, tax_rate
(as a fraction, e.g. 0.08 for 8%), tax_amount, and total. If a field is
not present in the text, leave it null rather than guessing. Also return
your own extraction_confidence between 0 and 1 reflecting how legible /
unambiguous the source text was."""


def _has_real_llm() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def _get_chat_model():
    if os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"),
            temperature=0,
        )
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
    raise RuntimeError("No LLM API key configured")


async def classify_document(raw_text: str) -> DocumentClassification:
    if not _has_real_llm():
        return _mock_classify(raw_text)

    from langchain_core.messages import HumanMessage, SystemMessage

    model = _get_chat_model().with_structured_output(DocumentClassification)
    return await model.ainvoke(
        [
            SystemMessage(content=CLASSIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=raw_text),
        ]
    )


async def extract_invoice(raw_text: str) -> ExtractedInvoice:
    if not _has_real_llm():
        return _mock_extract(raw_text)

    from langchain_core.messages import HumanMessage, SystemMessage

    model = _get_chat_model().with_structured_output(ExtractedInvoice)
    return await model.ainvoke(
        [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=raw_text),
        ]
    )


# --------------------------------------------------------------------------
# Offline heuristic fallback (no API key configured)
# --------------------------------------------------------------------------

_FIELD_PATTERNS = {
    "vendor_name": r"^\s*Vendor:\s*(.+)$",
    "invoice_number": r"^\s*Invoice Number:\s*(.+)$",
    "po_number": r"^\s*PO Number:\s*(.+)$",
    "invoice_date": r"^\s*Invoice Date:\s*(.+)$",
    "due_date": r"^\s*Due Date:\s*(.+)$",
    "currency": r"^\s*Currency:\s*(.+)$",
    "subtotal": r"^\s*Subtotal:\s*\$?(-?[\d,]+\.?\d*)",
    "tax_rate": r"^\s*Tax Rate:\s*(-?[\d.]+)\s*%",
    "tax_amount": r"^\s*Tax Amount:\s*\$?(-?[\d,]+\.?\d*)",
    "total": r"^\s*Total:\s*\$?(-?[\d,]+\.?\d*)",
}

# Tolerant of both fixed-width (multi-space) layouts and PDF text
# extractors that collapse runs of whitespace to a single space (e.g.
# pdfplumber's default extract_text()) -- the description is captured
# non-greedily up to the last three purely-numeric whitespace-delimited
# tokens (qty, unit_price, line_total). Allows a leading '-' for credit
# notes / negative-quantity lines.
_LINE_ITEM_ROW = re.compile(
    r"^\s*(?P<sku>\S+)\s+(?P<description>.+?)\s+"
    r"(?P<qty>-?[\d.,]+)\s+(?P<unit_price>-?[\d.,]+)\s+(?P<line_total>-?[\d.,]+)\s*$"
)


def _mock_classify(raw_text: str) -> DocumentClassification:
    upper = raw_text.upper()
    if "CREDIT NOTE" in upper:
        doc_type = "credit_note"
    elif "RECEIPT" in upper and "INVOICE" not in upper:
        doc_type = "receipt"
    elif "INVOICE" in upper:
        doc_type = "invoice"
    else:
        doc_type = "unknown"

    vendor_match = re.search(_FIELD_PATTERNS["vendor_name"], raw_text, re.MULTILINE)
    vendor_name = vendor_match.group(1).strip() if vendor_match else None

    confidence = 0.9 if doc_type != "unknown" and vendor_name else 0.4
    return DocumentClassification(doc_type=doc_type, vendor_name=vendor_name, confidence=confidence)


def _num(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _mock_extract(raw_text: str) -> ExtractedInvoice:
    fields: dict[str, str] = {}
    for key, pattern in _FIELD_PATTERNS.items():
        m = re.search(pattern, raw_text, re.MULTILINE)
        if m:
            fields[key] = m.group(1).strip()

    line_items: list[LineItem] = []
    for line in raw_text.splitlines():
        m = _LINE_ITEM_ROW.match(line)
        if not m:
            continue
        try:
            line_items.append(
                LineItem(
                    sku=m.group("sku"),
                    description=m.group("description").strip(),
                    quantity=_num(m.group("qty")) or 0.0,
                    unit_price=_num(m.group("unit_price")) or 0.0,
                    line_total=_num(m.group("line_total")) or 0.0,
                )
            )
        except Exception:
            continue

    tax_rate_pct = _num(fields.get("tax_rate"))
    confidence = 0.85 if line_items and fields.get("total") else 0.35

    return ExtractedInvoice(
        invoice_number=fields.get("invoice_number"),
        po_number=fields.get("po_number"),
        vendor_name=fields.get("vendor_name"),
        invoice_date=fields.get("invoice_date"),
        due_date=fields.get("due_date"),
        currency=fields.get("currency", "USD"),
        line_items=line_items,
        subtotal=_num(fields.get("subtotal")),
        tax_rate=(tax_rate_pct / 100.0) if tax_rate_pct is not None else None,
        tax_amount=_num(fields.get("tax_amount")),
        total=_num(fields.get("total")),
        extraction_confidence=confidence,
    )
