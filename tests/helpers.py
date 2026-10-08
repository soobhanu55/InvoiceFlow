"""Test helper: builds a plain-text invoice from the PO in the seeded catalog."""
from mcp_server import server as catalog


def invoice_text(po_number="PO-1001", tax_rate=10.0, qty_scale=1.0, price_scale=1.0, tax_amount=None,
                 doc="INVOICE", invoice_number="INV-9001"):
    """Plain-text invoice built from the PO in the catalog, optionally distorted (quantities, prices, tax)."""
    po = catalog.lookup_po(po_number)
    vendor = catalog.lookup_vendor(po["vendor_id"])["name"] if po.get("found") else "Acme Office Supplies"
    items = po["line_items"] if po.get("found") else [{"sku": "SKU-PEN-001", "description": "Pens", "qty": 10, "unit_price": 1.0}]
    rows, subtotal = [], 0.0
    for li in items:
        qty, price = li["qty"] * qty_scale, round(li["unit_price"] * price_scale, 2)
        line = round(qty * price, 2)
        subtotal += line
        rows.append(f"{li['sku']}  {li['description']}  {qty:g}  {price:.2f}  {line:.2f}")
    subtotal = round(subtotal, 2)
    tax = round(subtotal * tax_rate / 100, 2) if tax_amount is None else tax_amount
    return "\n".join([
        doc, f"Vendor: {vendor}", f"Invoice Number: {invoice_number}", f"PO Number: {po_number}",
        "Invoice Date: 2026-09-01", "Due Date: 2026-10-01", "Currency: USD", "",
        "SKU  Description  Qty  Unit Price  Line Total", *rows, "",
        f"Subtotal: {subtotal:.2f}", f"Tax Rate: {tax_rate:g}%", f"Tax Amount: {tax:.2f}",
        f"Total: {subtotal + tax:.2f}",
    ]) + "\n"


def run_pipeline(tmp_path, text, invoice_id="inv-1"):
    """Run the real graph once (in-memory checkpointer); returns (final state, review payload or None)."""
    import asyncio

    from langgraph.checkpoint.memory import MemorySaver

    from agent.graph import build_graph, pending_review

    path = tmp_path / f"{invoice_id}.txt"
    path.write_text(text, encoding="utf-8")
    graph = build_graph(MemorySaver())
    cfg = {"configurable": {"thread_id": invoice_id}}

    async def go():
        result = await graph.ainvoke({"invoice_id": invoice_id, "file_path": str(path)}, config=cfg)
        return result, await pending_review(graph, cfg)

    return asyncio.run(go())


class FakeModel:
    """Stands in for a LangChain chat model: `outcomes` is a list of exceptions to raise or dicts to return, in order."""

    def __init__(self, outcomes, usage=None):
        self.outcomes, self.calls, self.usage = list(outcomes), 0, usage or {"input_tokens": 1000, "output_tokens": 200}

    def with_structured_output(self, schema, include_raw=False):
        model = self

        class Runner:
            async def ainvoke(self, messages):
                model.calls += 1
                outcome = model.outcomes.pop(0) if len(model.outcomes) > 1 else model.outcomes[0]
                if isinstance(outcome, Exception):
                    raise outcome
                from types import SimpleNamespace

                return {"raw": SimpleNamespace(usage_metadata=model.usage), "parsed": outcome(schema), "parsing_error": None}

        return Runner()
