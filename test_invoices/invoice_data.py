"""Data for the 20 synthetic test invoices (5 clean + 15 deliberately
malformed), plus the render-to-text logic shared by the PDF generator.

Each entry mirrors -- or deliberately deviates from -- the mock ERP/catalog
seeded in mcp_server/db.py (see PURCHASE_ORDERS / PO_LINE_ITEMS /
CATALOG_ITEMS there), so the validation and matching nodes have something
concrete to agree or disagree with.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LineItem:
    sku: str
    description: str
    qty: float
    unit_price: float

    @property
    def line_total(self) -> float:
        return round(self.qty * self.unit_price, 2)


@dataclass
class InvoiceCase:
    name: str
    category: str  # "clean" | "malformed"
    expected_status: str  # "auto_approved" | "needs_review"
    expected_reason_codes: list[str]
    doc_label: str = "INVOICE"
    vendor: str | None = "Acme Office Supplies"
    invoice_number: str | None = "INV-0000"
    po_number: str | None = "PO-1001"
    invoice_date: str | None = "2024-06-01"
    due_date: str | None = "2024-07-01"
    currency: str = "USD"
    line_items: list[LineItem] = field(default_factory=list)
    subtotal_override: float | None = None
    tax_rate: float | None = 0.08
    tax_amount_override: float | None = None
    total_override: float | None = None
    omit_line_items: bool = False
    notes: str = ""

    def computed_subtotal(self) -> float:
        if self.subtotal_override is not None:
            return self.subtotal_override
        return round(sum(li.line_total for li in self.line_items), 2)

    def computed_tax(self) -> float:
        if self.tax_amount_override is not None:
            return self.tax_amount_override
        if self.tax_rate is None:
            return 0.0
        return round(self.computed_subtotal() * self.tax_rate, 2)

    def computed_total(self) -> float:
        if self.total_override is not None:
            return self.total_override
        return round(self.computed_subtotal() + self.computed_tax(), 2)


CASES: list[InvoiceCase] = [
    # ---------------------------------------------------------------- clean
    InvoiceCase(
        name="clean_acme_pens_paper",
        category="clean",
        expected_status="auto_approved",
        expected_reason_codes=[],
        vendor="Acme Office Supplies",
        invoice_number="INV-1001",
        po_number="PO-1001",
        line_items=[
            LineItem("SKU-PEN-001", "Ballpoint Pens (box of 12)", 100, 0.50),
            LineItem("SKU-PAPER-100", "Copy Paper Ream (500 sheets)", 20, 4.25),
        ],
    ),
    InvoiceCase(
        name="clean_blueridge_monitors",
        category="clean",
        expected_status="auto_approved",
        expected_reason_codes=[],
        vendor="Blue Ridge Electronics",
        invoice_number="INV-1002",
        po_number="PO-1002",
        line_items=[
            LineItem("SKU-MON-27", "27in LED Monitor", 5, 210.00),
            LineItem("SKU-CBL-HDMI", "HDMI Cable 2m", 10, 8.00),
        ],
    ),
    InvoiceCase(
        name="clean_summit_freight",
        category="clean",
        expected_status="auto_approved",
        expected_reason_codes=[],
        vendor="Summit Logistics Inc",
        invoice_number="INV-1003",
        po_number="PO-1003",
        line_items=[LineItem("SKU-FRT-STD", "Standard Freight Shipment", 1, 1200.00)],
    ),
    InvoiceCase(
        name="clean_northwind_cardstock",
        category="clean",
        expected_status="auto_approved",
        expected_reason_codes=[],
        vendor="Northwind Paper Co",
        invoice_number="INV-1004",
        po_number="PO-1004",
        line_items=[LineItem("SKU-PAPER-500", "Cardstock Ream (250 sheets)", 50, 6.10)],
    ),
    InvoiceCase(
        name="clean_acme_chairs",
        category="clean",
        expected_status="auto_approved",
        expected_reason_codes=[],
        vendor="Acme Office Supplies",
        invoice_number="INV-1005",
        po_number="PO-1005",
        line_items=[LineItem("SKU-CHAIR-ERG", "Ergonomic Office Chair", 4, 175.00)],
    ),
    # ------------------------------------------------------------- malformed
    InvoiceCase(
        name="malformed_line_sum_mismatch",
        category="malformed",
        expected_status="needs_review",
        expected_reason_codes=["LINE_ITEMS_SUM_MISMATCH"],
        vendor="Acme Office Supplies",
        invoice_number="INV-2001",
        po_number="PO-1001",
        line_items=[
            LineItem("SKU-PEN-001", "Ballpoint Pens (box of 12)", 100, 0.50),
            LineItem("SKU-PAPER-100", "Copy Paper Ream (500 sheets)", 20, 4.25),
        ],
        subtotal_override=160.00,  # should be 135.00
        notes="Stated subtotal doesn't match the sum of line totals",
    ),
    InvoiceCase(
        name="malformed_tax_miscalculated",
        category="malformed",
        expected_status="needs_review",
        expected_reason_codes=["TAX_MISCALCULATED"],
        vendor="Blue Ridge Electronics",
        invoice_number="INV-2002",
        po_number="PO-1002",
        line_items=[
            LineItem("SKU-MON-27", "27in LED Monitor", 5, 210.00),
            LineItem("SKU-CBL-HDMI", "HDMI Cable 2m", 10, 8.00),
        ],
        tax_rate=0.08,
        tax_amount_override=200.00,  # should be ~90.40 at 8%
        total_override=1330.00,
        notes="Tax amount doesn't match subtotal * stated tax rate",
    ),
    InvoiceCase(
        name="malformed_total_mismatch",
        category="malformed",
        expected_status="needs_review",
        expected_reason_codes=["TOTAL_MISMATCH"],
        vendor="Northwind Paper Co",
        invoice_number="INV-2003",
        po_number="PO-1004",
        line_items=[LineItem("SKU-PAPER-500", "Cardstock Ream (250 sheets)", 50, 6.10)],
        total_override=999.00,
        notes="subtotal + tax does not equal the stated total",
    ),
    InvoiceCase(
        name="malformed_po_not_found",
        category="malformed",
        expected_status="needs_review",
        expected_reason_codes=["PO_NOT_FOUND"],
        vendor="Acme Office Supplies",
        invoice_number="INV-2004",
        po_number="PO-9999",
        line_items=[LineItem("SKU-PEN-001", "Ballpoint Pens (box of 12)", 100, 0.50)],
        notes="References a PO number that does not exist in the ERP",
    ),
    InvoiceCase(
        name="malformed_po_closed",
        category="malformed",
        expected_status="needs_review",
        expected_reason_codes=["PO_CLOSED"],
        vendor="Blue Ridge Electronics",
        invoice_number="INV-2005",
        po_number="PO-1006",
        line_items=[LineItem("SKU-LAP-14", "14in Business Laptop", 3, 899.00)],
        notes="PO exists and line items match exactly, but the PO is closed",
    ),
    InvoiceCase(
        name="malformed_sku_not_in_catalog",
        category="malformed",
        expected_status="needs_review",
        expected_reason_codes=[],
        vendor="Acme Office Supplies",
        invoice_number="INV-2006",
        po_number="PO-1001",
        line_items=[
            LineItem("SKU-PEN-001", "Ballpoint Pens (box of 12)", 100, 0.50),
            LineItem("SKU-GHOST-999", "Mystery Widget", 5, 12.00),
        ],
        notes="Second line item's SKU does not exist anywhere in the catalog (match status: not_in_catalog)",
    ),
    InvoiceCase(
        name="malformed_sku_not_on_po",
        category="malformed",
        expected_status="needs_review",
        expected_reason_codes=[],
        vendor="Acme Office Supplies",
        invoice_number="INV-2007",
        po_number="PO-1001",
        line_items=[
            LineItem("SKU-PEN-001", "Ballpoint Pens (box of 12)", 100, 0.50),
            LineItem("SKU-LAP-14", "14in Business Laptop", 1, 899.00),
        ],
        notes="Laptop SKU is a real catalog item but was never on PO-1001 (match status: not_in_po)",
    ),
    InvoiceCase(
        name="malformed_qty_mismatch",
        category="malformed",
        expected_status="needs_review",
        expected_reason_codes=[],
        vendor="Acme Office Supplies",
        invoice_number="INV-2008",
        po_number="PO-1001",
        line_items=[
            LineItem("SKU-PEN-001", "Ballpoint Pens (box of 12)", 500, 0.50),  # PO says 100
            LineItem("SKU-PAPER-100", "Copy Paper Ream (500 sheets)", 20, 4.25),
        ],
        notes="Invoiced quantity (500) doesn't match the PO quantity (100) for SKU-PEN-001",
    ),
    InvoiceCase(
        name="malformed_price_mismatch",
        category="malformed",
        expected_status="needs_review",
        expected_reason_codes=[],
        vendor="Blue Ridge Electronics",
        invoice_number="INV-2009",
        po_number="PO-1002",
        line_items=[
            LineItem("SKU-MON-27", "27in LED Monitor", 5, 260.00),  # PO says 210.00
            LineItem("SKU-CBL-HDMI", "HDMI Cable 2m", 10, 8.00),
        ],
        notes="Invoiced unit price ($260) doesn't match the PO price ($210) for SKU-MON-27",
    ),
    InvoiceCase(
        name="malformed_vendor_mismatch",
        category="malformed",
        expected_status="needs_review",
        expected_reason_codes=[],
        vendor="Acme Office Supplies",  # PO-1002 actually belongs to Blue Ridge Electronics
        invoice_number="INV-2010",
        po_number="PO-1002",
        line_items=[
            LineItem("SKU-MON-27", "27in LED Monitor", 5, 210.00),
            LineItem("SKU-CBL-HDMI", "HDMI Cable 2m", 10, 8.00),
        ],
        notes="Invoice vendor doesn't match the vendor on file for the referenced PO (vendor_verified=False)",
    ),
    InvoiceCase(
        name="malformed_missing_po_number",
        category="malformed",
        expected_status="needs_review",
        expected_reason_codes=["MISSING_PO_NUMBER"],
        vendor="Acme Office Supplies",
        invoice_number="INV-2011",
        po_number=None,
        line_items=[LineItem("SKU-PEN-001", "Ballpoint Pens (box of 12)", 100, 0.50)],
        notes="No PO number present anywhere on the document",
    ),
    InvoiceCase(
        name="malformed_missing_due_date",
        category="malformed",
        expected_status="needs_review",
        expected_reason_codes=["MISSING_DUE_DATE"],
        vendor="Acme Office Supplies",
        invoice_number="INV-2012",
        po_number="PO-1001",
        due_date=None,
        line_items=[
            LineItem("SKU-PEN-001", "Ballpoint Pens (box of 12)", 100, 0.50),
            LineItem("SKU-PAPER-100", "Copy Paper Ream (500 sheets)", 20, 4.25),
        ],
        notes="Otherwise-clean invoice but missing a due date",
    ),
    InvoiceCase(
        name="malformed_credit_note_misclassified",
        category="malformed",
        expected_status="needs_review",
        expected_reason_codes=["DOC_TYPE_NOT_INVOICE"],
        doc_label="CREDIT NOTE",
        vendor="Acme Office Supplies",
        invoice_number="CN-3001",
        po_number="PO-1001",
        line_items=[LineItem("SKU-PEN-001", "Ballpoint Pens (box of 12)", -20, 0.50)],
        notes="A credit note that should be classified as such, not as a standard invoice",
    ),
    InvoiceCase(
        name="malformed_missing_invoice_number",
        category="malformed",
        expected_status="needs_review",
        expected_reason_codes=["MISSING_INVOICE_NUMBER"],
        vendor="Northwind Paper Co",
        invoice_number=None,
        po_number="PO-1004",
        line_items=[LineItem("SKU-PAPER-500", "Cardstock Ream (250 sheets)", 50, 6.10)],
        notes="No invoice number present anywhere on the document",
    ),
    InvoiceCase(
        name="malformed_no_line_items",
        category="malformed",
        expected_status="needs_review",
        expected_reason_codes=["NO_LINE_ITEMS"],
        vendor="Summit Logistics Inc",
        invoice_number="INV-2013",
        po_number="PO-1003",
        line_items=[],
        omit_line_items=True,
        subtotal_override=1200.00,
        tax_amount_override=96.00,
        total_override=1296.00,
        notes="Line item table is missing entirely even though totals are stated",
    ),
    InvoiceCase(
        name="malformed_unknown_vendor",
        category="malformed",
        expected_status="needs_review",
        expected_reason_codes=[],
        vendor="Zylo Consulting Group",
        invoice_number="INV-2014",
        po_number="PO-1001",
        line_items=[LineItem("SKU-PEN-001", "Ballpoint Pens (box of 12)", 100, 0.50)],
        notes="Vendor is not recognized in the ERP's vendor master at all",
    ),
]


def render_invoice_text(case: InvoiceCase) -> str:
    lines = [case.doc_label, ""]

    def add(label: str, value: str | None) -> None:
        if value is not None:
            lines.append(f"{label}: {value}")

    add("Vendor", case.vendor)
    add("Invoice Number", case.invoice_number)
    add("PO Number", case.po_number)
    add("Invoice Date", case.invoice_date)
    add("Due Date", case.due_date)
    add("Currency", case.currency)
    lines.append("")

    if not case.omit_line_items and case.line_items:
        lines.append(
            f"{'SKU':<14}  {'Description':<32}  {'Qty':>6}  {'Unit Price':>10}  {'Line Total':>10}"
        )
        for li in case.line_items:
            lines.append(
                f"{li.sku:<14}  {li.description:<32}  {li.qty:>6.2f}  {li.unit_price:>10.2f}  "
                f"{li.line_total:>10.2f}"
            )
        lines.append("")

    lines.append(f"Subtotal: {case.computed_subtotal():.2f}")
    if case.tax_rate is not None:
        lines.append(f"Tax Rate: {case.tax_rate * 100:.0f}%")
    lines.append(f"Tax Amount: {case.computed_tax():.2f}")
    lines.append(f"Total: {case.computed_total():.2f}")

    return "\n".join(lines)
