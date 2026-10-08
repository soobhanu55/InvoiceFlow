"""Node 3: Extraction -- structured pull of line items/totals/tax/PO/etc.

Generalizes across vendor formats via LLM function-calling / structured
output (agent.llm.extract_invoice), not regex.
"""
from __future__ import annotations

from agent.llm import extract_safely
from agent.state import InvoiceState


async def extraction_node(state: InvoiceState) -> dict:
    extracted, degraded = await extract_safely(state["raw_text"])

    if not extracted.vendor_name and state.get("vendor_name"):
        extracted.vendor_name = state["vendor_name"]

    audit_log = list(state.get("audit_log", []))
    if degraded:
        audit_log.append(f"extraction: {degraded}")
    audit_log.append(
        f"extraction: invoice_number={extracted.invoice_number!r} po_number={extracted.po_number!r} "
        f"total={extracted.total} line_items={len(extracted.line_items)} "
        f"confidence={extracted.extraction_confidence:.2f}"
    )

    return {
        "extracted": extracted.model_dump(),
        "extraction_confidence": extracted.extraction_confidence,
        "degraded": list(state.get("degraded", [])) + ([degraded] if degraded else []),
        "audit_log": audit_log,
    }
