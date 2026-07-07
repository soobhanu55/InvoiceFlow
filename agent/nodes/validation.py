"""Node 4: Validation -- internal-consistency business rules.

Checks totals reconcile with line items, tax matches the stated rate,
required fields are present, and the PO number exists in the ERP (a real
MCP tool call to the standalone catalog server). Every failure is
recorded as a ValidationIssue with a specific reason_code so a human
reviewer (or an audit log) can see exactly why an invoice was flagged.

Deeper PO *line item* reconciliation (qty/price/SKU matching) happens in
the matching node -- this node only checks arithmetic consistency and PO
existence.
"""
from __future__ import annotations

from agent.mcp_client import lookup_po
from agent.state import ExtractedInvoice, InvoiceState, ValidationIssue

EPSILON = 0.02


def _issue(code: str, message: str, severity: str = "error") -> ValidationIssue:
    return ValidationIssue(reason_code=code, message=message, severity=severity)


async def validation_node(state: InvoiceState) -> dict:
    extracted_data = state.get("extracted")
    extracted = ExtractedInvoice(**extracted_data) if extracted_data else None
    issues: list[ValidationIssue] = []
    audit_log = list(state.get("audit_log", []))

    if extracted is None:
        issues.append(_issue("NO_EXTRACTION", "Extraction step produced no data"))
        return {
            "validation_issues": [i.model_dump() for i in issues],
            "validation_passed": False,
            "po_lookup": None,
            "audit_log": audit_log,
        }

    if state.get("doc_type") not in ("invoice",):
        issues.append(
            _issue(
                "DOC_TYPE_NOT_INVOICE",
                f"Document classified as '{state.get('doc_type')}', expected 'invoice'",
                severity="warning",
            )
        )

    if not extracted.invoice_number:
        issues.append(_issue("MISSING_INVOICE_NUMBER", "Invoice number could not be extracted"))

    if not extracted.due_date:
        issues.append(_issue("MISSING_DUE_DATE", "Due date could not be extracted", severity="warning"))

    if not extracted.line_items:
        issues.append(_issue("NO_LINE_ITEMS", "No line items were extracted from the document"))

    # Line items sum vs subtotal
    if extracted.line_items and extracted.subtotal is not None:
        computed_subtotal = round(sum(li.line_total for li in extracted.line_items), 2)
        if abs(computed_subtotal - extracted.subtotal) > EPSILON:
            issues.append(
                _issue(
                    "LINE_ITEMS_SUM_MISMATCH",
                    f"Line items sum to {computed_subtotal:.2f} but stated subtotal is "
                    f"{extracted.subtotal:.2f}",
                )
            )

    # Tax matches stated rate
    if (
        extracted.subtotal is not None
        and extracted.tax_rate is not None
        and extracted.tax_amount is not None
    ):
        expected_tax = round(extracted.subtotal * extracted.tax_rate, 2)
        if abs(expected_tax - extracted.tax_amount) > EPSILON:
            issues.append(
                _issue(
                    "TAX_MISCALCULATED",
                    f"Expected tax {expected_tax:.2f} at rate {extracted.tax_rate:.2%} of subtotal "
                    f"{extracted.subtotal:.2f}, but stated tax is {extracted.tax_amount:.2f}",
                )
            )

    # Subtotal + tax == total
    if (
        extracted.subtotal is not None
        and extracted.tax_amount is not None
        and extracted.total is not None
    ):
        expected_total = round(extracted.subtotal + extracted.tax_amount, 2)
        if abs(expected_total - extracted.total) > EPSILON:
            issues.append(
                _issue(
                    "TOTAL_MISMATCH",
                    f"Subtotal {extracted.subtotal:.2f} + tax {extracted.tax_amount:.2f} = "
                    f"{expected_total:.2f}, but stated total is {extracted.total:.2f}",
                )
            )

    # PO existence check via the MCP catalog/ERP tool
    po_lookup: dict | None = None
    if not extracted.po_number:
        issues.append(_issue("MISSING_PO_NUMBER", "No PO number could be extracted"))
    else:
        po_lookup = await lookup_po(extracted.po_number)
        audit_log.append(f"validation: lookup_po({extracted.po_number!r}) -> found={po_lookup.get('found')}")
        if not po_lookup.get("found"):
            issues.append(
                _issue(
                    "PO_NOT_FOUND",
                    f"PO number '{extracted.po_number}' does not exist in the ERP/order system",
                )
            )
        elif po_lookup.get("status") == "closed":
            issues.append(
                _issue(
                    "PO_CLOSED",
                    f"PO '{extracted.po_number}' exists but is closed",
                    severity="warning",
                )
            )

    validation_passed = not any(i.severity == "error" for i in issues)

    audit_log.append(
        f"validation: {len(issues)} issue(s) found "
        f"({sum(1 for i in issues if i.severity == 'error')} error, "
        f"{sum(1 for i in issues if i.severity == 'warning')} warning) -> passed={validation_passed}"
    )

    return {
        "validation_issues": [i.model_dump() for i in issues],
        "validation_passed": validation_passed,
        "po_lookup": po_lookup,
        "audit_log": audit_log,
    }
