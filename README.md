# Autonomous Invoice Intake Agent

A 6-node LangGraph pipeline that ingests invoices (PDF/image), extracts
and validates their data, reconciles them against purchase orders in a
mock ERP, and routes anything uncertain to a human reviewer. n8n sits in
front as the orchestration/webhook layer; Streamlit is the human-in-the-
loop review console.

## Architecture

```
                                   ┌───────────────────────┐
   email inbox / watched   POST    │        n8n            │
   folder / manual upload ───────▶ │  webhook trigger       │
                                   │  -> call agent API     │
                                   │  -> route on result    │──▶ notify reviewer
                                   └───────────┬───────────┘      (Slack webhook)
                                               │ HTTP
                                               ▼
                          ┌────────────────────────────────────┐
                          │           FastAPI (agent/api.py)     │
                          │  /invoices/submit  /review/*  /stats │
                          └───────────────────┬──────────────────┘
                                              │ graph.ainvoke() / Command(resume=...)
                                              ▼
        ┌───────────────────────────────────────────────────────────────────────┐
        │                     LangGraph StateGraph (agent/graph.py)              │
        │                                                                         │
        │  ┌────────┐   ┌────────────────┐   ┌────────────┐   ┌────────────┐     │
        │  │ intake │──▶│ classification │──▶│ extraction │──▶│ validation │──┐  │
        │  │ (OCR)  │   │   (LLM, doc     │   │ (LLM,      │   │ (business  │  │  │
        │  │        │   │   type+vendor)  │   │ structured)│   │  rules +   │  │  │
        │  └────────┘   └────────────────┘   └────────────┘   │  lookup_po)│  │  │
        │                                                      └────────────┘  │  │
        │                                                                       ▼  │
        │                                          ┌────────────┐   ┌──────────────┐│
        │                                          │  matching  │◀──│ (from above) ││
        │                                          │ (lookup_po,│   └──────────────┘│
        │                                          │ lookup_    │                   │
        │                                          │ vendor,    │                   │
        │                                          │ get_catalog│                   │
        │                                          │ _item)     │                   │
        │                                          └─────┬──────┘                   │
        │                                                 ▼                          │
        │                                     ┌────────────────────┐                 │
        │                                     │   human_review     │                 │
        │                                     │  clean+confident?  │                 │
        │                                     │  -> auto_approve   │                 │
        │                                     │  else interrupt()  │────┐            │
        │                                     └────────────────────┘    │            │
        └─────────────────────────────────────────────────────────────  │  ──────────┘
                                                                          │ paused, persisted
                                                                          │ via SQLite checkpointer
                                                                          ▼
                                                          ┌──────────────────────────┐
                                                          │   Streamlit review queue  │
                                                          │  doc + fields side-by-side│
                                                          │  approve / edit / reject  │
                                                          └────────────┬─────────────┘
                                                                       │ POST /review/{id}/resume
                                                                       ▼
                                                          graph resumes human_review node
                                                          -> writes final status to store


        ┌───────────────────────────────────┐
        │   MCP server (mcp_server/server.py) │◀── lookup_po / lookup_vendor / get_catalog_item
        │   standalone process, stdio or SSE   │    (real MCP protocol calls, cross-process)
        │   backed by data/catalog.db (mock ERP)│
        └───────────────────────────────────┘
```

### The 6 nodes (`agent/nodes/`)

1. **intake** — reads the source file. Native-text PDFs are read directly
   via `pdfplumber`; scanned PDFs/images are rasterized and OCR'd with
   `pytesseract`.
2. **classification** — LLM call with structured output
   (`DocumentClassification` Pydantic schema) -> doc type (invoice /
   receipt / credit_note / unknown) + vendor + confidence.
3. **extraction** — LLM call with structured output (`ExtractedInvoice`
   schema) -> line items, subtotal, tax, invoice/PO number, due date.
   Function-calling/structured output, not regex, so it generalizes
   across vendor layouts.
4. **validation** — pure business rules: line items must sum to the
   subtotal, tax must match the stated rate, subtotal+tax must equal the
   total, required fields must be present, and the PO number must exist
   in the ERP (one MCP tool call: `lookup_po`). Every failure gets a
   specific `reason_code`.
5. **matching** — calls `lookup_po`, `lookup_vendor`, `get_catalog_item`
   on the MCP server to reconcile each line item's SKU/qty/price against
   the referenced PO, and to verify the invoice's vendor is the vendor of
   record for that PO.
6. **human_review** — computes an overall confidence score. If validation
   passed, everything matched, and confidence clears the bar, it
   auto-approves and writes to the output store. Otherwise it calls
   LangGraph's `interrupt()`, which pauses the run (state persisted via a
   SQLite checkpointer) until a human resolves it through Streamlit;
   the resume value (`approve` / `edit` + corrections / `reject`) flows
   back into the same node, which then finalizes the record.

### Why `interrupt()` + a checkpointer

This is LangGraph's native human-in-the-loop primitive: the graph run
genuinely pauses mid-node (not just "return early and re-submit"), and
`graph.ainvoke(Command(resume=...), config={"configurable": {"thread_id": invoice_id}})`
resumes that exact paused node with the human's decision. The FastAPI
process owns an `AsyncSqliteSaver` checkpointer for the whole app
lifetime, so a run can be flagged, sit for hours, and still resume
correctly.

## Repo layout

```
mcp_server/         standalone MCP server (FastMCP) + mock ERP/catalog DB
agent/               LangGraph pipeline, FastAPI service, state/store/LLM code
  nodes/             the 6 node implementations
  graph.py           StateGraph wiring
  api.py             FastAPI app (submit / review / resume / stats)
  state.py           Pydantic schemas + the shared graph state
  llm.py             LLM abstraction (real Claude/GPT structured output, or offline mock)
  mcp_client.py      MCP client the nodes use to call the standalone server
  store.py           SQLite output/audit store
streamlit_app/       human-in-the-loop review dashboard
n8n/                 n8n workflow JSON export (webhook -> agent API -> route)
test_invoices/       synthetic invoice generator + manifest + test harness
data/                SQLite files (catalog, output store, checkpoints) -- gitignored
docker-compose.yml   wires mcp-server + agent-api + streamlit + n8n
```

## Running locally (no Docker)

```bash
python -m venv .venv
source .venv/Scripts/activate   # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# seed the mock ERP/catalog DB
python mcp_server/db.py

# generate the 20 synthetic test invoices (5 clean + 15 malformed)
python test_invoices/generate_test_invoices.py

# run the pipeline end-to-end against all of them, no server needed
python test_invoices/run_tests.py
```

To run the full stack (API + dashboard) locally:

```bash
# terminal 1
uvicorn agent.api:app --reload --port 8000

# terminal 2
AGENT_API_URL=http://localhost:8000 streamlit run streamlit_app/app.py

# terminal 3 (optional) -- submit a test invoice
curl -F "file=@test_invoices/invoices/malformed_po_not_found.pdf" \
     http://localhost:8000/invoices/submit
```

Open http://localhost:8501 to see it land in the review queue.

By default (no API key set) the classification/extraction nodes use a
deterministic offline heuristic parser (`agent/llm.py`) so the whole
pipeline — including the test harness — runs without any network access
or API key. **Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`** (see
`.env.example`) to exercise the real LLM structured-output path, which is
what actually generalizes across arbitrary vendor invoice layouts.

## Running with Docker Compose

```bash
cp .env.example .env    # optionally fill in ANTHROPIC_API_KEY / OPENAI_API_KEY
docker compose up --build
```

This starts four containers:

| service      | port | purpose                                            |
|--------------|------|-----------------------------------------------------|
| `mcp-server` | 8765 | standalone MCP server (SSE transport), mock ERP DB   |
| `agent-api`  | 8000 | FastAPI wrapping the LangGraph pipeline              |
| `streamlit`  | 8501 | human review dashboard                               |
| `n8n`        | 5678 | orchestration layer                                   |

`agent-api` talks to `mcp-server` over the network (`MCP_TRANSPORT=sse`),
so the two are genuinely separate processes/containers communicating via
the MCP protocol — not an in-process function call. For local (non-Docker)
dev, `agent-api` instead spawns `mcp_server/server.py` as a stdio
subprocess (`MCP_TRANSPORT=stdio`, the default), which is still a
separate OS process using the real MCP protocol, just simpler to run
without two terminals.

Then:

1. Open **http://localhost:8501** for the review dashboard.
2. Open **http://localhost:5678** for n8n. Import
   `n8n/invoice_intake_workflow.json` (Workflows -> Import from File),
   activate it, and point your email/watched-folder automation at its
   webhook (`POST /webhook/invoice-intake`, a `file` binary field). The
   workflow calls `agent-api`, and branches on `status == "auto_approved"`
   vs. routing a review notification.
3. Submit an invoice directly for a quick check:
   ```bash
   curl -F "file=@test_invoices/invoices/clean_acme_pens_paper.pdf" \
        http://localhost:8000/invoices/submit
   ```

## The mock ERP / MCP server

`mcp_server/server.py` is a standalone MCP server (built with
`mcp.server.fastmcp.FastMCP`) exposing three tools, backed by
`data/catalog.db` (seeded from `mcp_server/db.py`):

- `lookup_po(po_number)` — PO header + line items + computed PO total, or `{"found": false}`.
- `lookup_vendor(vendor)` — vendor record by id or (case-insensitive) name.
- `get_catalog_item(sku)` — canonical SKU description/price/owning vendor.

Run it standalone to poke at it directly:

```bash
python mcp_server/server.py --http --port 8765     # SSE transport
# or
python mcp_server/server.py                         # stdio transport
```

## Test invoices (`test_invoices/`)

`generate_test_invoices.py` renders 20 synthetic invoices as PDFs (via
reportlab) from the case definitions in `invoice_data.py`: 5 clean
invoices that should auto-approve, and 15 deliberately malformed ones,
each targeting a specific validation/matching failure:

| case                                 | what it demonstrates                                  |
|--------------------------------------|--------------------------------------------------------|
| `malformed_line_sum_mismatch`        | `LINE_ITEMS_SUM_MISMATCH` — line items don't sum to subtotal |
| `malformed_tax_miscalculated`        | `TAX_MISCALCULATED` — tax doesn't match stated rate     |
| `malformed_total_mismatch`           | `TOTAL_MISMATCH` — subtotal+tax != total                |
| `malformed_po_not_found`             | `PO_NOT_FOUND` — PO doesn't exist in the ERP            |
| `malformed_po_closed`                | `PO_CLOSED` — PO exists but is closed                   |
| `malformed_sku_not_in_catalog`       | matching: SKU unknown anywhere                          |
| `malformed_sku_not_on_po`            | matching: SKU real, but not on this PO                  |
| `malformed_qty_mismatch`             | matching: invoiced qty != PO qty                        |
| `malformed_price_mismatch`           | matching: invoiced price != PO price                    |
| `malformed_vendor_mismatch`          | matching: invoice vendor != PO's vendor of record        |
| `malformed_missing_po_number`        | `MISSING_PO_NUMBER`                                     |
| `malformed_missing_due_date`         | `MISSING_DUE_DATE` (warning)                            |
| `malformed_credit_note_misclassified`| `DOC_TYPE_NOT_INVOICE` — classification catches a credit note |
| `malformed_missing_invoice_number`   | `MISSING_INVOICE_NUMBER`                                |
| `malformed_no_line_items`            | `NO_LINE_ITEMS`                                         |
| `malformed_unknown_vendor`           | matching: vendor not in the ERP at all                  |

`run_tests.py` runs every case through the real 6-node graph in-process
(no server required) and asserts the resulting status + reason codes
match what each case is designed to prove:

```bash
python test_invoices/generate_test_invoices.py
python test_invoices/run_tests.py
# 21/21 passed, 0 failed
```

## Configuration

See `.env.example` for the full list. The important ones:

- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` — enable real LLM structured
  output for classification/extraction (otherwise falls back to the
  offline heuristic parser).
- `MCP_TRANSPORT` (`stdio` | `sse`) + `MCP_SERVER_URL` — how the agent
  reaches the MCP server.
- `AUTO_APPROVE_CONFIDENCE` — confidence threshold for auto-approval
  (default `0.85`). Auto-approval also requires zero validation issues
  (errors *or* warnings) and a full PO/line-item/vendor match — anything
  less routes to a human.
