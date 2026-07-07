"""Output store: where finalized (and in-flight-flagged) invoices land.

A tiny SQLite table used by the FastAPI layer for the /stats and /output
endpoints. This is separate from the LangGraph checkpointer (which holds
the *live, resumable* state of in-progress/interrupted runs) -- this store
is an append/update log of outcomes for reporting and for the Streamlit
dashboard's throughput view.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.state import InvoiceState

DB_PATH = os.environ.get("STORE_DB_PATH", "data/store.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS processed_invoices (
    invoice_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    file_path TEXT,
    vendor_name TEXT,
    invoice_number TEXT,
    po_number TEXT,
    total REAL,
    overall_confidence REAL,
    validation_issues TEXT,
    match_result TEXT,
    human_decision TEXT,
    extracted_json TEXT,
    audit_log TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_store(db_path: str | None = None) -> None:
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def save_result(state: InvoiceState, status: str) -> None:
    extracted = state.get("extracted") or {}
    match_result = state.get("match_result")
    validation_issues = state.get("validation_issues", [])
    now = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT created_at FROM processed_invoices WHERE invoice_id = ?",
            (state["invoice_id"],),
        ).fetchone()
        created_at = existing["created_at"] if existing else now

        conn.execute(
            """
            INSERT INTO processed_invoices (
                invoice_id, status, file_path, vendor_name, invoice_number, po_number, total,
                overall_confidence, validation_issues, match_result, human_decision,
                extracted_json, audit_log, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(invoice_id) DO UPDATE SET
                status=excluded.status,
                file_path=excluded.file_path,
                vendor_name=excluded.vendor_name,
                invoice_number=excluded.invoice_number,
                po_number=excluded.po_number,
                total=excluded.total,
                overall_confidence=excluded.overall_confidence,
                validation_issues=excluded.validation_issues,
                match_result=excluded.match_result,
                human_decision=excluded.human_decision,
                extracted_json=excluded.extracted_json,
                audit_log=excluded.audit_log,
                updated_at=excluded.updated_at
            """,
            (
                state["invoice_id"],
                status,
                state.get("file_path"),
                extracted.get("vendor_name") or state.get("vendor_name"),
                extracted.get("invoice_number"),
                extracted.get("po_number"),
                extracted.get("total"),
                state.get("overall_confidence"),
                json.dumps(validation_issues),
                json.dumps(match_result) if match_result else None,
                state.get("human_decision"),
                json.dumps(extracted) if extracted else None,
                json.dumps(state.get("audit_log", [])),
                created_at,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_one(invoice_id: str) -> dict[str, Any] | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM processed_invoices WHERE invoice_id = ?", (invoice_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_processed(status: str | None = None) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM processed_invoices WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM processed_invoices ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_stats() -> dict[str, Any]:
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM processed_invoices").fetchone()[0]
        by_status = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM processed_invoices GROUP BY status"
        ).fetchall()
        counts = {row["status"]: row["cnt"] for row in by_status}
        flagged = sum(
            counts.get(s, 0) for s in ("needs_review", "approved", "rejected")
        )
        flag_rate = (flagged / total) if total else 0.0
        return {
            "total_processed": total,
            "counts_by_status": counts,
            "flag_rate": round(flag_rate, 4),
        }
    finally:
        conn.close()
