"""Test helper: builds a plain-text invoice from the PO in the seeded catalog."""
from mcp_server import server as catalog


def invoice_text(po_number="PO-1001", tax_rate=10.0, qty_scale=1.0, price_scale=1.0, tax_amount=None,
                 doc="INVOICE", invoice_number="INV-9001"):
    """Plain-text invoice built from the PO in the catalog, optionally distorted (quantities, prices, tax)."""
    po = catalog.lookup_po(po_number)
    vendor = catalog.lookup_vendor(po["vendor_id"])["name"] if po.get("found") else "Acme Office Supplies"
    items = po["line_items"] if po.get("found") else [{"sku": "SKU-PEN-001", "description": "Pens", "qty": 10, "unit_price": 1.0}]
    rows, subtotal = [], 0.0
    for li in items:
        qty, price = li["qty"] * qty_scale, round(li["unit_price"] * price_scale, 2)
        line = round(qty * price, 2)
        subtotal += line
        rows.append(f"{li['sku']}  {li['description']}  {qty:g}  {price:.2f}  {line:.2f}")
    subtotal = round(subtotal, 2)
    tax = round(subtotal * tax_rate / 100, 2) if tax_amount is None else tax_amount
    return "\n".join([
        doc, f"Vendor: {vendor}", f"Invoice Number: {invoice_number}", f"PO Number: {po_number}",
        "Invoice Date: 2026-09-01", "Due Date: 2026-10-01", "Currency: USD", "",
        "SKU  Description  Qty  Unit Price  Line Total", *rows, "",
        f"Subtotal: {subtotal:.2f}", f"Tax Rate: {tax_rate:g}%", f"Tax Amount: {tax:.2f}",
        f"Total: {subtotal + tax:.2f}",
    ]) + "\n"
