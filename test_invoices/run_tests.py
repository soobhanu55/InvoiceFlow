"""Test harness: runs every invoice in manifest.json through the actual
6-node LangGraph pipeline (in-process, no server needed) and asserts the
resulting status + validation reason codes match what each synthetic
case was designed to demonstrate.

    python test_invoices/generate_test_invoices.py   # (re)generate the PDFs
    python test_invoices/run_tests.py

Exits non-zero if any case doesn't match its expectation.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import MemorySaver

from agent import store
from agent.graph import build_graph
from agent.mcp_client import get_mcp_client

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"
BASE_DIR = Path(__file__).resolve().parent


async def run_case(graph, case: dict) -> tuple[str, list[str]]:
    invoice_id = case["name"]
    file_path = BASE_DIR / case["file"]
    config = {"configurable": {"thread_id": invoice_id}}

    result = await graph.ainvoke(
        {"invoice_id": invoice_id, "file_path": str(file_path)}, config=config
    )

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        actual_status = "needs_review"
        actual_codes = [i["reason_code"] for i in payload.get("validation_issues", [])]
    else:
        actual_status = result.get("status")
        actual_codes = [i["reason_code"] for i in result.get("validation_issues", [])]

    return actual_status, actual_codes


async def main() -> int:
    store.init_store()
    cases = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    graph = build_graph(checkpointer=MemorySaver())

    rows = []
    for case in cases:
        actual_status, actual_codes = await run_case(graph, case)
        status_ok = actual_status == case["expected_status"]
        codes_ok = set(case["expected_reason_codes"]).issubset(set(actual_codes))
        rows.append(
            {
                "name": case["name"],
                "category": case["category"],
                "expected_status": case["expected_status"],
                "actual_status": actual_status,
                "status_ok": status_ok,
                "expected_codes": case["expected_reason_codes"],
                "actual_codes": actual_codes,
                "codes_ok": codes_ok,
                "ok": status_ok and codes_ok,
            }
        )

    name_w = max(len(r["name"]) for r in rows) + 2
    print(f"{'CASE':<{name_w}}{'CATEGORY':<12}{'EXPECTED':<15}{'ACTUAL':<15}{'RESULT'}")
    print("-" * (name_w + 12 + 15 + 15 + 6))
    for r in rows:
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"{r['name']:<{name_w}}{r['category']:<12}{r['expected_status']:<15}{r['actual_status']:<15}{mark}")
        if not r["ok"]:
            print(f"    expected_reason_codes={r['expected_codes']} actual_reason_codes={r['actual_codes']}")

    passed = sum(1 for r in rows if r["ok"])
    failed = len(rows) - passed
    print(f"\n{passed}/{len(rows)} passed, {failed} failed")

    try:
        await get_mcp_client().close()
    except Exception:  # noqa: BLE001 -- best-effort subprocess teardown
        pass

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
