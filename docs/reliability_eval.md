# Reliability under injected faults

21 sample invoices x 5 seeds per fault rate, with the real LangGraph pipeline and the real MCP server; each model call and each tool call fails with probability p (retried twice with backoff, then degraded). The baseline (p = 0) has 5 auto-approved and 16 routed to review.

| fault rate p | invoice runs | crashes | unsafe approvals | same outcome as baseline | auto-approvals sent to a human |
|---|---|---|---|---|---|
| 0.1 | 105 | 0 | 0 | 100.0% | 0.0% |
| 0.3 | 105 | 0 | 0 | 97.1% | 12.0% |
| 0.5 | 105 | 0 | 0 | 90.5% | 40.0% |
| 1.0 | 105 | 0 | 0 | 76.2% | 100.0% |

Reading it: no run crashed and no invoice that needed review was ever auto-approved because a check could not run. The cost of failing safe is the last column: clean invoices that a human had to look at because a dependency was down for longer than the retries covered. At p = 1.0 (full outage) every invoice goes to a human, and the breaker stops the pipeline from waiting out timeouts. Faults are injected at the call boundary (connection errors), so this tests the control flow, not provider behaviour; the model is a stand-in for the heuristic parser.

Reproduce: `python test_invoices/chaos_eval.py --seeds 5 --out docs/reliability_eval.md`
