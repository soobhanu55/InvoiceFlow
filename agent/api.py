"""FastAPI service exposing the 6-node LangGraph invoice pipeline.

This is the API n8n calls: a webhook receives a new invoice, forwards it
to POST /invoices/submit, and routes the response (auto-approved vs
needs_review) onward. The Streamlit dashboard talks to the /review/*
endpoints for the human-in-the-loop queue.
"""
from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from agent import store
from agent.graph import build_graph
from agent.mcp_client import get_mcp_client

INBOX_DIR = Path(os.environ.get("INBOX_DIR", "data/inbox"))
CHECKPOINT_DB_PATH = os.environ.get("CHECKPOINT_DB_PATH", "data/checkpoints.sqlite")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_store()
    INBOX_DIR.mkdir(parents=True, exist_ok=True)

    mcp_client = get_mcp_client()
    await mcp_client.connect()

    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB_PATH) as checkpointer:
        app.state.graph = build_graph(checkpointer)
        yield

    await mcp_client.close()


app = FastAPI(title="Invoice Intake Agent", lifespan=lifespan)


class ResumeRequest(BaseModel):
    decision: Literal["approve", "edit", "reject"]
    corrections: Optional[dict[str, Any]] = None


def _thread_config(invoice_id: str) -> dict:
    return {"configurable": {"thread_id": invoice_id}}


def _deserialize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for field in ("validation_issues", "match_result", "extracted_json", "audit_log"):
        if out.get(field):
            out[field] = json.loads(out[field])
    return out


async def _run_graph(invoice_id: str, file_path: str) -> dict[str, Any]:
    graph = app.state.graph
    config = _thread_config(invoice_id)
    result = await graph.ainvoke({"invoice_id": invoice_id, "file_path": file_path}, config=config)
    return _format_run_result(invoice_id, result)


def _format_run_result(invoice_id: str, result: dict[str, Any]) -> dict[str, Any]:
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return {"invoice_id": invoice_id, "status": "needs_review", "review": payload}
    return {
        "invoice_id": invoice_id,
        "status": result.get("status"),
        "overall_confidence": result.get("overall_confidence"),
        "extracted": result.get("extracted"),
        "validation_issues": result.get("validation_issues"),
        "match_result": result.get("match_result"),
    }


@app.get("/")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/invoices/submit")
async def submit_invoice(file: UploadFile) -> dict:
    invoice_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename or "").suffix or ".pdf"
    dest = INBOX_DIR / f"{invoice_id}{suffix}"
    dest.write_bytes(await file.read())

    try:
        return await _run_graph(invoice_id, str(dest))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class SubmitPathRequest(BaseModel):
    file_path: str
    invoice_id: Optional[str] = None


@app.post("/invoices/submit_path")
async def submit_invoice_path(req: SubmitPathRequest) -> dict:
    """Submit an invoice already on disk (used by the test harness / local dev)."""
    if not Path(req.file_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")
    invoice_id = req.invoice_id or uuid.uuid4().hex[:12]
    try:
        return await _run_graph(invoice_id, req.file_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/invoices/{invoice_id}")
async def get_invoice(invoice_id: str) -> dict:
    row = store.get_one(invoice_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return _deserialize_row(row)


@app.get("/invoices/{invoice_id}/file")
async def get_invoice_file(invoice_id: str) -> FileResponse:
    row = store.get_one(invoice_id)
    if row is None or not row.get("file_path"):
        raise HTTPException(status_code=404, detail="Invoice file not found")
    path = Path(row["file_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Invoice file no longer exists on disk")
    return FileResponse(path)


@app.get("/review/pending")
async def list_pending_review() -> list[dict]:
    rows = store.list_processed(status="needs_review")
    return [_deserialize_row(r) for r in rows]


@app.get("/review/{invoice_id}")
async def get_review_item(invoice_id: str) -> dict:
    row = store.get_one(invoice_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return _deserialize_row(row)


@app.post("/review/{invoice_id}/resume")
async def resume_review(invoice_id: str, body: ResumeRequest) -> dict:
    row = store.get_one(invoice_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Invoice not found")
    if row["status"] != "needs_review":
        raise HTTPException(
            status_code=409, detail=f"Invoice is not pending review (status={row['status']})"
        )

    graph = app.state.graph
    config = _thread_config(invoice_id)
    resume_value = {"decision": body.decision, "corrections": body.corrections}

    try:
        result = await graph.ainvoke(Command(resume=resume_value), config=config)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _format_run_result(invoice_id, result)


@app.get("/output")
async def list_output(status: Optional[str] = None) -> list[dict]:
    rows = store.list_processed(status=status)
    return [_deserialize_row(r) for r in rows]


@app.get("/stats")
async def stats() -> dict:
    return store.get_stats()
