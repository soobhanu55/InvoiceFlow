"""Node 2: Classification -- document type + vendor, via structured LLM output."""
from __future__ import annotations

from agent.llm import classify_document
from agent.state import InvoiceState


async def classification_node(state: InvoiceState) -> dict:
    result = await classify_document(state["raw_text"])
    audit_log = list(state.get("audit_log", []))
    audit_log.append(
        f"classification: doc_type={result.doc_type} vendor={result.vendor_name!r} "
        f"confidence={result.confidence:.2f}"
    )
    return {
        "doc_type": result.doc_type,
        "vendor_name": result.vendor_name,
        "classification_confidence": result.confidence,
        "audit_log": audit_log,
    }
