"""Node 6: Human-in-the-loop / output.

If validation passed and the PO/line-item match is clean and confidence
is high, auto-approve and write straight to the output store. Otherwise
pause the graph with `interrupt()` -- the checkpointer persists state so
a human can review the extracted fields side-by-side with the original
document (via the Streamlit queue / FastAPI review endpoints) and the
graph resumes exactly here with their decision.
"""
from __future__ import annotations

import os

from langgraph.types import interrupt

from agent import store
from agent.state import InvoiceState

DEFAULT_THRESHOLD = 0.85


def _threshold() -> float:
    return float(os.environ.get("AUTO_APPROVE_CONFIDENCE", DEFAULT_THRESHOLD))


def compute_overall_confidence(state: InvoiceState) -> float:
    classification_confidence = state.get("classification_confidence", 0.5)
    extraction_confidence = state.get("extraction_confidence", 0.5)
    score = (classification_confidence + extraction_confidence) / 2

    issues = state.get("validation_issues", [])
    error_count = sum(1 for i in issues if i.get("severity") == "error")
    warning_count = sum(1 for i in issues if i.get("severity") == "warning")
    score -= 0.15 * error_count
    score -= 0.05 * warning_count

    match_result = state.get("match_result")
    if match_result is not None and not match_result.get("all_matched"):
        score -= 0.2

    return max(0.0, min(1.0, round(score, 4)))


def _build_review_payload(state: InvoiceState, overall_confidence: float) -> dict:
    return {
        "invoice_id": state["invoice_id"],
        "file_path": state.get("file_path"),
        "doc_type": state.get("doc_type"),
        "extracted": state.get("extracted"),
        "validation_issues": state.get("validation_issues", []),
        "match_result": state.get("match_result"),
        "overall_confidence": overall_confidence,
        "audit_log": state.get("audit_log", []),
    }


async def human_review_node(state: InvoiceState) -> dict:
    overall_confidence = compute_overall_confidence(state)
    audit_log = list(state.get("audit_log", []))

    validation_passed = state.get("validation_passed", False)
    has_any_issue = len(state.get("validation_issues", [])) > 0
    match_result = state.get("match_result")
    all_matched = bool(match_result and match_result.get("all_matched"))

    # Auto-approve only when there is nothing at all to flag -- any
    # validation issue (even a warning like a missing due date or a closed
    # PO) or any PO/line-item mismatch routes to a human.
    auto_ok = (
        validation_passed
        and not has_any_issue
        and all_matched
        and overall_confidence >= _threshold()
    )

    if auto_ok:
        status = "auto_approved"
        audit_log.append(f"human_review: auto-approved at confidence {overall_confidence:.2f}")
        result_state = {**state, "status": status, "overall_confidence": overall_confidence}
        store.save_result(result_state, status)
        return {
            "status": status,
            "overall_confidence": overall_confidence,
            "audit_log": audit_log,
        }

    audit_log.append(
        f"human_review: routed to human review (confidence={overall_confidence:.2f}, "
        f"validation_passed={validation_passed}, all_matched={all_matched})"
    )
    flagged_state = {**state, "status": "needs_review", "overall_confidence": overall_confidence}
    store.save_result(flagged_state, "needs_review")

    decision = interrupt(_build_review_payload(state, overall_confidence))

    human_decision = decision.get("decision", "reject")
    corrections = decision.get("corrections")

    if human_decision == "reject":
        final_status = "rejected"
    else:
        final_status = "approved"

    audit_log.append(f"human_review: resolved as '{final_status}' by human ({human_decision})")

    extracted = dict(state.get("extracted") or {})
    if human_decision == "edit" and corrections:
        extracted.update(corrections)

    final_state = {
        **state,
        "extracted": extracted,
        "status": final_status,
        "human_decision": human_decision,
        "human_corrections": corrections,
        "overall_confidence": overall_confidence,
    }
    store.save_result(final_state, final_status)

    return {
        "extracted": extracted,
        "status": final_status,
        "overall_confidence": overall_confidence,
        "human_decision": human_decision,
        "human_corrections": corrections,
        "audit_log": audit_log,
    }
