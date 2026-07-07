"""Standalone MCP server exposing ERP / product-catalog lookup tools.

Run directly:
    python mcp_server/server.py                # stdio transport (default)
    python mcp_server/server.py --http          # SSE transport on :8765

This process owns the catalog SQLite database and is the ONLY thing that
talks to it. The LangGraph agent never imports this module or touches the
database directly -- it calls these tools over the MCP protocol (see
agent/mcp_client.py), so this is a genuinely separate process either way
(subprocess over stdio, or a standalone container over SSE/HTTP).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from mcp_server.db import get_connection, init_db

mcp = FastMCP("invoice-catalog")


def _row_to_dict(row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


@mcp.tool()
def lookup_po(po_number: str) -> dict[str, Any]:
    """Look up a purchase order by its PO number.

    Returns the PO header (vendor, status, currency) plus all of its line
    items (sku, description, qty, unit_price), or {"found": False} if no
    such PO exists in the ERP system.
    """
    conn = get_connection()
    try:
        po_row = conn.execute(
            "SELECT * FROM purchase_orders WHERE po_number = ?", (po_number,)
        ).fetchone()
        if po_row is None:
            return {"found": False, "po_number": po_number, "reason": "PO not found in ERP"}

        vendor_row = conn.execute(
            "SELECT * FROM vendors WHERE vendor_id = ?", (po_row["vendor_id"],)
        ).fetchone()

        line_item_rows = conn.execute(
            "SELECT sku, description, qty, unit_price FROM po_line_items WHERE po_number = ?",
            (po_number,),
        ).fetchall()

        line_items = [_row_to_dict(r) for r in line_item_rows]
        po_total = round(sum(li["qty"] * li["unit_price"] for li in line_items), 2)

        return {
            "found": True,
            "po_number": po_row["po_number"],
            "status": po_row["status"],
            "currency": po_row["currency"],
            "vendor_id": po_row["vendor_id"],
            "vendor_name": vendor_row["name"] if vendor_row else None,
            "line_items": line_items,
            "po_total": po_total,
        }
    finally:
        conn.close()


@mcp.tool()
def lookup_vendor(vendor: str) -> dict[str, Any]:
    """Look up a vendor by vendor_id OR by (case-insensitive) name.

    Returns the vendor record, or {"found": False} if unrecognized.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM vendors WHERE vendor_id = ? "
            "OR LOWER(name) = LOWER(?) "
            "OR LOWER(name) LIKE LOWER(?)",
            (vendor, vendor, f"%{vendor}%"),
        ).fetchone()
        if row is None:
            return {"found": False, "vendor": vendor, "reason": "Vendor not recognized"}
        return {"found": True, **_row_to_dict(row)}
    finally:
        conn.close()


@mcp.tool()
def get_catalog_item(sku: str) -> dict[str, Any]:
    """Look up a single catalog SKU.

    Returns description, canonical unit_price and owning vendor_id, or
    {"found": False} if the SKU does not exist in the product catalog.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM catalog_items WHERE sku = ?", (sku,)
        ).fetchone()
        if row is None:
            return {"found": False, "sku": sku, "reason": "SKU not found in catalog"}
        return {"found": True, **_row_to_dict(row)}
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="Serve over SSE/HTTP instead of stdio")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", 8765)))
    parser.add_argument("--reset-db", action="store_true", help="Drop and reseed the catalog DB on startup")
    args = parser.parse_args()

    init_db(reset=args.reset_db)

    if args.http:
        from mcp.server.transport_security import TransportSecuritySettings

        mcp.settings.host = os.environ.get("MCP_BIND_HOST", "0.0.0.0")
        mcp.settings.port = args.port

        # Extra allowed Host headers, e.g. the docker-compose service name
        # "mcp-server" -- by default the SDK's DNS-rebinding protection only
        # allows localhost/127.0.0.1, which would reject same-network
        # container-to-container calls.
        extra_hosts = [
            h.strip() for h in os.environ.get("MCP_ALLOWED_HOSTS", "mcp-server:*").split(",") if h.strip()
        ]
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*", *extra_hosts],
            allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"],
        )

        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
