"""Node 5: Matching -- reconcile the invoice against the referenced PO.

Calls the MCP catalog/ERP tools (lookup_po, lookup_vendor,
get_catalog_item) to compare each invoice line item against the PO's
line items, and to verify the vendor on the invoice matches the vendor
on record for that PO.
"""
from __future__ import annotations

from agent.mcp_client import get_catalog_item, lookup_po, lookup_vendor
from agent.state import ExtractedInvoice, InvoiceState, MatchLineItemResult, MatchResult

QTY_EPSILON = 0.001
PRICE_EPSILON = 0.01
TOTAL_EPSILON = 0.02


async def matching_node(state: InvoiceState) -> dict:
    extracted_data = state.get("extracted")
    extracted = ExtractedInvoice(**extracted_data) if extracted_data else None
    audit_log = list(state.get("audit_log", []))

    if extracted is None:
        result = MatchResult(po_found=False, vendor_verified=False, all_matched=False)
        return {"match_result": result.model_dump(), "audit_log": audit_log}

    po_lookup = state.get("po_lookup")
    if po_lookup is None and extracted.po_number:
        po_lookup = await lookup_po(extracted.po_number)

    po_found = bool(po_lookup and po_lookup.get("found"))

    vendor_verified = False
    vendor_name = extracted.vendor_name or state.get("vendor_name")
    if vendor_name:
        vendor_lookup = await lookup_vendor(vendor_name)
        audit_log.append(f"matching: lookup_vendor({vendor_name!r}) -> found={vendor_lookup.get('found')}")
        if vendor_lookup.get("found") and po_found:
            vendor_verified = vendor_lookup.get("vendor_id") == po_lookup.get("vendor_id")
        elif vendor_lookup.get("found") and not po_found:
            vendor_verified = True  # vendor is known, just no PO to cross-check against

    line_item_results: list[MatchLineItemResult] = []

    po_items_by_sku: dict[str, dict] = {}
    if po_found:
        for li in po_lookup.get("line_items", []):
            po_items_by_sku[li["sku"]] = li

    for item in extracted.line_items:
        sku = item.sku
        po_item = po_items_by_sku.get(sku) if sku else None

        if po_item is not None:
            qty_ok = abs(item.quantity - po_item["qty"]) <= QTY_EPSILON
            price_ok = abs(item.unit_price - po_item["unit_price"]) <= PRICE_EPSILON
            if qty_ok and price_ok:
                status = "match"
                detail = None
            elif not qty_ok and not price_ok:
                status = "qty_mismatch"
                detail = (
                    f"invoice qty {item.quantity} vs PO qty {po_item['qty']}; "
                    f"invoice price {item.unit_price} vs PO price {po_item['unit_price']}"
                )
            elif not qty_ok:
                status = "qty_mismatch"
                detail = f"invoice qty {item.quantity} vs PO qty {po_item['qty']}"
            else:
                status = "price_mismatch"
                detail = f"invoice price {item.unit_price} vs PO price {po_item['unit_price']}"

            line_item_results.append(
                MatchLineItemResult(
                    sku=sku,
                    description=item.description,
                    invoice_qty=item.quantity,
                    invoice_unit_price=item.unit_price,
                    po_qty=po_item["qty"],
                    po_unit_price=po_item["unit_price"],
                    status=status,
                    detail=detail,
                )
            )
        else:
            catalog_item = await get_catalog_item(sku) if sku else {"found": False}
            if catalog_item.get("found"):
                status = "not_in_po"
                detail = f"SKU '{sku}' exists in catalog but is not a line item on this PO"
            else:
                status = "not_in_catalog"
                detail = f"SKU '{sku}' is not a recognized catalog item"

            line_item_results.append(
                MatchLineItemResult(
                    sku=sku,
                    description=item.description,
                    invoice_qty=item.quantity,
                    invoice_unit_price=item.unit_price,
                    po_qty=None,
                    po_unit_price=None,
                    status=status,
                    detail=detail,
                )
            )

    # PO totals in the ERP are pre-tax, so compare against the invoice
    # subtotal (also pre-tax) rather than the tax-inclusive total.
    po_total = po_lookup.get("po_total") if po_found else None
    delta = None
    if po_total is not None and extracted.subtotal is not None:
        delta = round(extracted.subtotal - po_total, 2)

    all_matched = (
        po_found
        and vendor_verified
        and all(li.status == "match" for li in line_item_results)
        and (delta is not None and abs(delta) <= TOTAL_EPSILON)
    )

    match_result = MatchResult(
        po_found=po_found,
        po_status=po_lookup.get("status") if po_found else None,
        vendor_verified=vendor_verified,
        line_item_results=line_item_results,
        po_total=po_total,
        invoice_total_vs_po_delta=delta,
        all_matched=all_matched,
    )

    audit_log.append(
        f"matching: po_found={po_found} vendor_verified={vendor_verified} "
        f"line_items_matched={sum(1 for li in line_item_results if li.status == 'match')}/"
        f"{len(line_item_results)} all_matched={all_matched}"
    )

    return {"match_result": match_result.model_dump(), "audit_log": audit_log}
