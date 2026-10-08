"""Node 2: Classification -- document type + vendor, via structured LLM output."""
from __future__ import annotations

from agent.llm import classify_safely
from agent.state import InvoiceState


async def classification_node(state: InvoiceState) -> dict:
    result, degraded = await classify_safely(state["raw_text"])
    audit_log = list(state.get("audit_log", []))
    if degraded:
        audit_log.append(f"classification: {degraded}")
    audit_log.append(
        f"classification: doc_type={result.doc_type} vendor={result.vendor_name!r} "
        f"confidence={result.confidence:.2f}"
    )
    return {
        "doc_type": result.doc_type,
        "vendor_name": result.vendor_name,
        "classification_confidence": result.confidence,
        "degraded": list(state.get("degraded", [])) + ([degraded] if degraded else []),
        "audit_log": audit_log,
    }
