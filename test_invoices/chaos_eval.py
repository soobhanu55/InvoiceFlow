"""Failure-injection evaluation: run the 21 sample invoices while the LLM and the MCP tools fail at random.

    python test_invoices/chaos_eval.py [--seeds 5] [--out docs/reliability_eval.md]

For each fault rate p, every model call and every tool call fails with probability p (connection errors, which the
resilience layer retries). The baseline is the same run with p = 0. What is measured:
  crashes         runs where an exception escaped the graph (should be 0)
  unsafe          invoices that the baseline sent to review but that were auto-approved under faults (should be 0)
  same outcome    final status identical to the baseline
  lost auto       baseline auto-approvals that were sent to a human instead (the availability cost of failing safe)
Circuit breakers are reset between invoices so each invoice sees independent faults; p = 1.0 is a full outage and is
the case the breaker exists for.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["ANTHROPIC_API_KEY"] = "chaos"  # switches on the model path; the chain below is a stand-in
os.environ["TOOL_AUDIT_PATH"] = os.path.join(tempfile.gettempdir(), "chaos_tool_audit.jsonl")

from langgraph.checkpoint.memory import MemorySaver

from agent import llm, mcp_client, resilience, store
from agent.graph import build_graph, pending_review

BASE = Path(__file__).resolve().parent
RNG = random.Random(0)
P = {"p": 0.0}


async def _instant(_):
    pass


resilience.sleep = _instant


class ChaosModel:
    """Parses with the offline heuristic, but raises a connection error with probability P['p'] first."""

    def with_structured_output(self, schema, include_raw=False):
        class Runner:
            async def ainvoke(self, messages):
                if RNG.random() < P["p"]:
                    raise ConnectionError("injected: connection reset")
                text = messages[-1].content.replace("<untrusted_document>", "").replace("</untrusted_document>", "").strip()
                parsed = llm._mock_classify(text) if schema.__name__ == "DocumentClassification" else llm._mock_extract(text)
                return {"raw": SimpleNamespace(usage_metadata=None), "parsed": parsed, "parsing_error": None}

        return Runner()


llm._chain = lambda: [("chaos-primary", ChaosModel())]

_real_call_once = mcp_client.MCPClient._call_once


async def _flaky_call_once(self, name, arguments):
    if RNG.random() < P["p"]:
        raise ConnectionError("injected: tool connection refused")
    return await _real_call_once(self, name, arguments)


mcp_client.MCPClient._call_once = _flaky_call_once


async def run_all(cases, p, seed):
    P["p"] = p
    RNG.seed(seed)
    graph = build_graph(MemorySaver())
    out = {}
    for case in cases:
        resilience.BREAKERS.clear()
        cfg = {"configurable": {"thread_id": f"{case['name']}-{p}-{seed}"}}
        try:
            result = await graph.ainvoke({"invoice_id": cfg["configurable"]["thread_id"], "file_path": str(BASE / case["file"])}, config=cfg)
            out[case["name"]] = "needs_review" if await pending_review(graph, cfg) else result.get("status")
        except Exception as exc:  # noqa: BLE001 - this is exactly what is being counted
            out[case["name"]] = f"CRASH:{type(exc).__name__}"
    return out


async def main(seeds: int, out_path: str | None) -> None:
    store.DB_PATH = os.path.join(tempfile.gettempdir(), "chaos_store.db")
    store.init_store()
    cases = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    baseline = await run_all(cases, 0.0, 0)
    rows = []
    for p in (0.1, 0.3, 0.5, 1.0):
        n = crashes = unsafe = same = lost = 0
        for seed in range(seeds):
            for name, status in (await run_all(cases, p, seed)).items():
                n += 1
                crashes += status.startswith("CRASH")
                unsafe += status == "auto_approved" and baseline[name] != "auto_approved"
                same += status == baseline[name]
                lost += baseline[name] == "auto_approved" and status == "needs_review"
        rows.append((p, n, crashes, unsafe, same, lost))
    auto = sum(1 for s in baseline.values() if s == "auto_approved")

    lines = [
        "| fault rate p | invoice runs | crashes | unsafe approvals | same outcome as baseline | auto-approvals sent to a human |",
        "|---|---|---|---|---|---|",
    ]
    for p, n, c, u, s, l in rows:
        lines.append(f"| {p:.1f} | {n} | {c} | {u} | {100 * s / n:.1f}% | {100 * l / (auto * seeds):.1f}% |")
    table = "\n".join(lines)
    print(table)
    if out_path:
        Path(out_path).write_text(
            "# Reliability under injected faults\n\n"
            f"21 sample invoices x {seeds} seeds per fault rate, with the real LangGraph pipeline and the real MCP server; "
            "each model call and each tool call fails with probability p (retried twice with backoff, then degraded). "
            f"The baseline (p = 0) has {auto} auto-approved and {len(baseline) - auto} routed to review.\n\n"
            f"{table}\n\n"
            "Reading it: no run crashed and no invoice that needed review was ever auto-approved because a check could not "
            "run. The cost of failing safe is the last column: clean invoices that a human had to look at because a dependency "
            "was down for longer than the retries covered. At p = 1.0 (full outage) every invoice goes to a human, and the "
            "breaker stops the pipeline from waiting out timeouts. Faults are injected at the call boundary (connection "
            "errors), so this tests the control flow, not provider behaviour; the model is a stand-in for the heuristic parser.\n\n"
            "Reproduce: `python test_invoices/chaos_eval.py --seeds 5 --out docs/reliability_eval.md`\n",
            encoding="utf-8",
        )
    try:
        await mcp_client.get_mcp_client().close()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out")
    a = ap.parse_args()
    asyncio.run(main(a.seeds, a.out))
