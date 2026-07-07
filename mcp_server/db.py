"""Schema + seed data for the mock ERP / product-catalog SQLite database.

This stands in for a real ERP system. It is intentionally tiny: a handful
of vendors, purchase orders (with line items) and catalog SKUs, chosen so
that the synthetic test invoices in test_invoices/ can deliberately match
or mismatch against it.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DB_PATH = os.environ.get("CATALOG_DB_PATH", "data/catalog.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS vendors (
    vendor_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    tax_id TEXT,
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS purchase_orders (
    po_number TEXT PRIMARY KEY,
    vendor_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    currency TEXT NOT NULL DEFAULT 'USD',
    FOREIGN KEY (vendor_id) REFERENCES vendors(vendor_id)
);

CREATE TABLE IF NOT EXISTS po_line_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    po_number TEXT NOT NULL,
    sku TEXT NOT NULL,
    description TEXT,
    qty REAL NOT NULL,
    unit_price REAL NOT NULL,
    FOREIGN KEY (po_number) REFERENCES purchase_orders(po_number)
);

CREATE TABLE IF NOT EXISTS catalog_items (
    sku TEXT PRIMARY KEY,
    description TEXT,
    unit_price REAL NOT NULL,
    vendor_id TEXT NOT NULL,
    FOREIGN KEY (vendor_id) REFERENCES vendors(vendor_id)
);
"""

VENDORS = [
    ("V001", "Acme Office Supplies", "TAX-AC-001", "active"),
    ("V002", "Blue Ridge Electronics", "TAX-BR-002", "active"),
    ("V003", "Summit Logistics Inc", "TAX-SL-003", "active"),
    ("V004", "Northwind Paper Co", "TAX-NW-004", "active"),
    ("V005", "Legacy Tools Ltd", "TAX-LT-005", "inactive"),
]

PURCHASE_ORDERS = [
    ("PO-1001", "V001", "open", "USD"),
    ("PO-1002", "V002", "open", "USD"),
    ("PO-1003", "V003", "open", "USD"),
    ("PO-1004", "V004", "open", "USD"),
    ("PO-1005", "V001", "open", "USD"),
    ("PO-1006", "V002", "closed", "USD"),
]

PO_LINE_ITEMS = [
    ("PO-1001", "SKU-PEN-001", "Ballpoint Pens (box of 12)", 100, 0.50),
    ("PO-1001", "SKU-PAPER-100", "Copy Paper Ream (500 sheets)", 20, 4.25),
    ("PO-1002", "SKU-MON-27", "27in LED Monitor", 5, 210.00),
    ("PO-1002", "SKU-CBL-HDMI", "HDMI Cable 2m", 10, 8.00),
    ("PO-1003", "SKU-FRT-STD", "Standard Freight Shipment", 1, 1200.00),
    ("PO-1004", "SKU-PAPER-500", "Cardstock Ream (250 sheets)", 50, 6.10),
    ("PO-1005", "SKU-CHAIR-ERG", "Ergonomic Office Chair", 4, 175.00),
    ("PO-1006", "SKU-LAP-14", "14in Business Laptop", 3, 899.00),
]

CATALOG_ITEMS = [
    ("SKU-PEN-001", "Ballpoint Pens (box of 12)", 0.50, "V001"),
    ("SKU-PAPER-100", "Copy Paper Ream (500 sheets)", 4.25, "V001"),
    ("SKU-MON-27", "27in LED Monitor", 210.00, "V002"),
    ("SKU-CBL-HDMI", "HDMI Cable 2m", 8.00, "V002"),
    ("SKU-FRT-STD", "Standard Freight Shipment", 1200.00, "V003"),
    ("SKU-PAPER-500", "Cardstock Ream (250 sheets)", 6.10, "V004"),
    ("SKU-CHAIR-ERG", "Ergonomic Office Chair", 175.00, "V001"),
    ("SKU-LAP-14", "14in Business Laptop", 899.00, "V002"),
]


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | None = None, reset: bool = False) -> None:
    conn = get_connection(db_path)
    try:
        if reset:
            conn.executescript(
                "DROP TABLE IF EXISTS po_line_items;"
                "DROP TABLE IF EXISTS purchase_orders;"
                "DROP TABLE IF EXISTS catalog_items;"
                "DROP TABLE IF EXISTS vendors;"
            )
        conn.executescript(SCHEMA)

        if conn.execute("SELECT COUNT(*) FROM vendors").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO vendors (vendor_id, name, tax_id, status) VALUES (?, ?, ?, ?)",
                VENDORS,
            )
            conn.executemany(
                "INSERT INTO purchase_orders (po_number, vendor_id, status, currency) VALUES (?, ?, ?, ?)",
                PURCHASE_ORDERS,
            )
            conn.executemany(
                "INSERT INTO po_line_items (po_number, sku, description, qty, unit_price) VALUES (?, ?, ?, ?, ?)",
                PO_LINE_ITEMS,
            )
            conn.executemany(
                "INSERT INTO catalog_items (sku, description, unit_price, vendor_id) VALUES (?, ?, ?, ?)",
                CATALOG_ITEMS,
            )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db(reset=True)
    print(f"Seeded catalog DB at {DB_PATH}")
