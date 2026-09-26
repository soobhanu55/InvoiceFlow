# InvoiceFlow

A 6-node LangGraph agent that ingests invoices (PDF/image), extracts and validates their data, reconciles against purchase orders in a mock ERP via real MCP tool calls, and routes anything uncertain to a human reviewer.

![Test harness, terminal recording](docs/demo.gif)
![Human review dashboard walkthrough](docs/demo_ui.gif)

## How it works

```
intake (OCR) → classification (LLM) → extraction (LLM) → validation (rules + PO lookup)
             → matching (MCP: lookup_po / lookup_vendor / get_catalog_item)
             → human_review → auto-approve, or interrupt() and wait for a person
```

MCP server runs as a genuinely separate process (stdio or SSE), not an in-process function call. `human_review` uses LangGraph's native `interrupt()` — the run actually pauses, persisted via SQLite, and resumes exactly where it left off once a human decides.

## Results

- All 21 purpose-built test cases (5 clean, 16 deliberately malformed) → **21/21 correct reason codes**.
- **Known bug, disclosed:** the harness's reported pass/fail status has a real interrupt-propagation issue in this LangGraph/anyio version combo — the *logic* is 100% correct (reason codes all match), but the harness itself misreports 16 of those as "FAIL". Traced and documented rather than hidden behind a false "21/21 passing" headline. The API endpoints and the dashboard both read the correct persisted status regardless.

## Run it

```bash
pip install -r requirements.txt
python mcp_server/db.py                          # seed mock ERP
python test_invoices/generate_test_invoices.py    # generate the 21 test cases
python test_invoices/run_tests.py                 # run them end-to-end, no server needed
```

Full stack (API + dashboard + n8n):
```bash
docker compose up --build
# API: localhost:8000 · Dashboard: localhost:8501 · n8n: localhost:5678
```

No API key needed — classification/extraction fall back to a deterministic offline parser. Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` for the real LLM path.

## Use the MCP server with Claude Desktop

The ERP lookup server is a standalone MCP server, so any MCP client can use it, not just this agent. To give Claude Desktop the three tools (`lookup_po`, `lookup_vendor`, `get_catalog_item`), add this to `claude_desktop_config.json` (Settings → Developer → Edit Config) and restart Claude:

```json
{
  "mcpServers": {
    "invoice-catalog": {
      "command": "python",
      "args": ["/absolute/path/to/InvoiceFlow/mcp_server/server.py"]
    }
  }
}
```

Then ask Claude things like *"Is PO-1001 still open, and what's its total?"* and it answers from the ERP database through the tools. The database seeds itself on first start.

Check it works without Claude (spawns the server over stdio the same way Claude does, lists the tools, calls one):
```bash
python mcp_server/smoke_test.py
```

Full architecture diagram, all 21 test cases, and MCP server details in [`docs/DETAILS.md`](docs/DETAILS.md).
