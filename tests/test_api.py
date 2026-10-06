"""The FastAPI layer over the real graph (in-memory checkpointer, offline mock LLM, seeded catalog)."""
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from agent import api
from agent.graph import build_graph
from helpers import invoice_text


@pytest.fixture
def client(catalog_and_store, monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(api, "INBOX_DIR", catalog_and_store / "inbox")
    api.INBOX_DIR.mkdir()
    api.app.state.graph = build_graph(MemorySaver())
    return TestClient(api.app)  # no `with`: the lifespan (MCP subprocess, sqlite checkpointer) is not started


def write(tmp, name, text):
    p = tmp / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_health(client):
    assert client.get("/").json() == {"status": "ok"}


def test_clean_invoice_is_auto_approved(client, catalog_and_store):
    r = client.post("/invoices/submit_path", json={"file_path": write(catalog_and_store, "a.txt", invoice_text()), "invoice_id": "a"})
    assert r.status_code == 200 and r.json()["status"] == "auto_approved"
    assert client.get("/invoices/a").json()["status"] == "auto_approved"


def test_flagged_invoice_goes_to_the_review_queue_and_can_be_resumed(client, catalog_and_store):
    r = client.post("/invoices/submit_path", json={"file_path": write(catalog_and_store, "b.txt", invoice_text(price_scale=1.3)), "invoice_id": "b"})
    body = r.json()
    assert body["status"] == "needs_review" and body["review"]["match_result"]["all_matched"] is False
    assert [i["invoice_id"] for i in client.get("/review/pending").json()] == ["b"]
    assert client.get("/review/b").json()["status"] == "needs_review"

    done = client.post("/review/b/resume", json={"decision": "approve"})
    assert done.status_code == 200 and done.json()["status"] == "approved"
    assert client.get("/review/pending").json() == []
    assert [i["invoice_id"] for i in client.get("/output", params={"status": "approved"}).json()] == ["b"]


def test_resume_twice_is_a_conflict(client, catalog_and_store):
    client.post("/invoices/submit_path", json={"file_path": write(catalog_and_store, "c.txt", invoice_text(qty_scale=2)), "invoice_id": "c"})
    assert client.post("/review/c/resume", json={"decision": "reject"}).json()["status"] == "rejected"
    assert client.post("/review/c/resume", json={"decision": "reject"}).status_code == 409


def test_unknown_ids_and_files_return_404(client):
    assert client.get("/invoices/nope").status_code == 404
    assert client.get("/review/nope").status_code == 404
    assert client.post("/review/nope/resume", json={"decision": "approve"}).status_code == 404
    assert client.post("/invoices/submit_path", json={"file_path": "/does/not/exist.txt"}).status_code == 404


def test_invalid_decision_is_rejected_by_validation(client):
    assert client.post("/review/x/resume", json={"decision": "maybe"}).status_code == 422


def test_upload_endpoint_saves_the_file_and_runs_the_graph(client):
    r = client.post("/invoices/submit", files={"file": ("inv.txt", invoice_text().encode(), "text/plain")})
    assert r.status_code == 200 and r.json()["status"] == "auto_approved"
    assert len(list(api.INBOX_DIR.glob("*.txt"))) == 1


def test_stats_count_outcomes(client, catalog_and_store):
    client.post("/invoices/submit_path", json={"file_path": write(catalog_and_store, "d.txt", invoice_text()), "invoice_id": "d"})
    client.post("/invoices/submit_path", json={"file_path": write(catalog_and_store, "e.txt", invoice_text(price_scale=1.3)), "invoice_id": "e"})
    stats = client.get("/stats").json()
    assert stats["total_processed"] == 2 and stats["counts_by_status"] == {"auto_approved": 1, "needs_review": 1}
    assert stats["flag_rate"] == 0.5
