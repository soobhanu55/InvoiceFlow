"""Streamlit human-in-the-loop dashboard for the invoice intake agent.

Shows the queue of invoices flagged by the LangGraph pipeline, a split
view (original document + editable extracted fields) for the selected
invoice, approve/edit/reject actions that resume the paused graph run,
and a running log of throughput / flag rate.
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("AGENT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Invoice Intake Review", layout="wide")


def api_get(path: str, **kwargs):
    r = requests.get(f"{API_URL}{path}", timeout=30, **kwargs)
    r.raise_for_status()
    return r.json()


def api_post(path: str, json: dict | None = None, **kwargs):
    r = requests.post(f"{API_URL}{path}", json=json, timeout=60, **kwargs)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Sidebar: throughput / flag-rate stats
# ---------------------------------------------------------------------------
st.sidebar.title("Invoice Intake Agent")
if st.sidebar.button("Refresh", use_container_width=True):
    st.rerun()

try:
    stats = api_get("/stats")
    counts = stats.get("counts_by_status", {})
    st.sidebar.metric("Total processed", stats.get("total_processed", 0))
    st.sidebar.metric("Flag rate", f"{stats.get('flag_rate', 0) * 100:.1f}%")
    st.sidebar.write("**By status**")
    for status, cnt in counts.items():
        st.sidebar.write(f"- {status}: {cnt}")
except requests.RequestException as exc:
    st.sidebar.error(f"Cannot reach agent API at {API_URL}\n\n{exc}")
    st.stop()

st.sidebar.divider()
st.sidebar.caption(f"Agent API: {API_URL}")


# ---------------------------------------------------------------------------
# Main: review queue
# ---------------------------------------------------------------------------
st.title("Human Review Queue")

pending = api_get("/review/pending")

if not pending:
    st.success("No invoices are currently flagged for review.")
    st.subheader("Recently processed")
    output = api_get("/output")[:20]
    if output:
        df = pd.DataFrame(
            [
                {
                    "invoice_id": o["invoice_id"],
                    "status": o["status"],
                    "vendor": o.get("vendor_name"),
                    "invoice_number": o.get("invoice_number"),
                    "po_number": o.get("po_number"),
                    "total": o.get("total"),
                    "confidence": o.get("overall_confidence"),
                    "updated_at": o.get("updated_at"),
                }
                for o in output
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    st.stop()

queue_labels = [
    f"{p['invoice_id']}  —  {p.get('vendor_name') or 'unknown vendor'}  "
    f"(conf {p.get('overall_confidence', 0):.2f})"
    for p in pending
]
selected_idx = st.selectbox(
    f"{len(pending)} invoice(s) awaiting review", range(len(pending)), format_func=lambda i: queue_labels[i]
)
item = pending[selected_idx]
invoice_id = item["invoice_id"]

extracted = item.get("extracted_json") or {}
validation_issues = item.get("validation_issues") or []
match_result = item.get("match_result") or {}

left, right = st.columns([1, 1.3])

# --- Left: original document ---------------------------------------------
with left:
    st.subheader("Original document")
    try:
        file_resp = requests.get(f"{API_URL}/invoices/{invoice_id}/file", timeout=30)
        file_resp.raise_for_status()
        content_type = file_resp.headers.get("content-type", "")
        file_path = item.get("file_path") or ""

        if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            st.image(file_resp.content, use_container_width=True)
        elif file_path.lower().endswith(".pdf"):
            try:
                from pdf2image import convert_from_bytes

                pages = convert_from_bytes(file_resp.content, first_page=1, last_page=1)
                st.image(pages[0], use_container_width=True)
            except Exception:
                st.download_button("Download PDF", file_resp.content, file_name=f"{invoice_id}.pdf")
        else:
            st.text_area("Document text", file_resp.content.decode("utf-8", errors="replace"), height=500)
    except requests.RequestException as exc:
        st.warning(f"Could not load original document: {exc}")

    st.subheader("Validation issues")
    if not validation_issues:
        st.info("No validation issues recorded.")
    for issue in validation_issues:
        if issue.get("severity") == "error":
            st.error(f"**{issue['reason_code']}** — {issue['message']}")
        else:
            st.warning(f"**{issue['reason_code']}** — {issue['message']}")

    st.subheader("PO / catalog match")
    st.write(
        f"PO found: **{match_result.get('po_found')}** &nbsp;|&nbsp; "
        f"Vendor verified: **{match_result.get('vendor_verified')}** &nbsp;|&nbsp; "
        f"All matched: **{match_result.get('all_matched')}**"
    )
    line_matches = match_result.get("line_item_results") or []
    if line_matches:
        st.dataframe(pd.DataFrame(line_matches), use_container_width=True, hide_index=True)

# --- Right: editable extracted fields -------------------------------------
with right:
    st.subheader("Extracted fields (editable)")

    with st.form("edit_form"):
        c1, c2 = st.columns(2)
        invoice_number = c1.text_input("Invoice number", extracted.get("invoice_number") or "")
        po_number = c2.text_input("PO number", extracted.get("po_number") or "")
        vendor_name = c1.text_input("Vendor name", extracted.get("vendor_name") or "")
        currency = c2.text_input("Currency", extracted.get("currency") or "USD")
        invoice_date = c1.text_input("Invoice date", extracted.get("invoice_date") or "")
        due_date = c2.text_input("Due date", extracted.get("due_date") or "")
        subtotal = c1.number_input("Subtotal", value=float(extracted.get("subtotal") or 0.0))
        tax_amount = c2.number_input("Tax amount", value=float(extracted.get("tax_amount") or 0.0))
        total = c1.number_input("Total", value=float(extracted.get("total") or 0.0))
        tax_rate = c2.number_input(
            "Tax rate (fraction)", value=float(extracted.get("tax_rate") or 0.0), format="%.4f"
        )

        st.markdown("**Line items**")
        line_items_df = pd.DataFrame(
            extracted.get("line_items")
            or [{"sku": "", "description": "", "quantity": 0.0, "unit_price": 0.0, "line_total": 0.0}]
        )
        edited_items = st.data_editor(line_items_df, num_rows="dynamic", use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        approve_clicked = col_a.form_submit_button("✅ Approve", use_container_width=True)
        edit_clicked = col_b.form_submit_button("✏️ Save edits & approve", use_container_width=True)
        reject_clicked = col_c.form_submit_button("❌ Reject", use_container_width=True)

    corrections = {
        "invoice_number": invoice_number or None,
        "po_number": po_number or None,
        "vendor_name": vendor_name or None,
        "currency": currency or None,
        "invoice_date": invoice_date or None,
        "due_date": due_date or None,
        "subtotal": subtotal,
        "tax_amount": tax_amount,
        "total": total,
        "tax_rate": tax_rate,
        "line_items": edited_items.to_dict(orient="records"),
    }

    decision = None
    payload_corrections = None
    if approve_clicked:
        decision = "approve"
    elif edit_clicked:
        decision = "edit"
        payload_corrections = corrections
    elif reject_clicked:
        decision = "reject"

    if decision:
        try:
            result = api_post(
                f"/review/{invoice_id}/resume",
                json={"decision": decision, "corrections": payload_corrections},
            )
            st.success(f"Invoice {invoice_id} resolved as **{result.get('status')}**")
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Failed to resume invoice: {exc}")

    with st.expander("Audit log"):
        for line in item.get("audit_log") or []:
            st.text(line)
