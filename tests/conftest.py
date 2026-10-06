import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from agent import store  # noqa: E402
from agent.nodes import matching, validation  # noqa: E402
from mcp_server import db as catalog_db  # noqa: E402
from mcp_server import server as catalog  # noqa: E402


@pytest.fixture
def catalog_and_store(tmp_path, monkeypatch):
    """A seeded catalog database (the same tool functions the MCP server exposes, called in-process) and an empty
    output store, both in tmp_path, with the agent's MCP client calls routed to them."""
    monkeypatch.setattr(catalog_db, "DB_PATH", str(tmp_path / "catalog.db"))
    catalog_db.init_db(reset=True)
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "store.db"))
    store.init_store()

    async def lookup_po(n): return catalog.lookup_po(n)
    async def lookup_vendor(n): return catalog.lookup_vendor(n)
    async def get_catalog_item(s): return catalog.get_catalog_item(s)

    for mod in (validation, matching):
        monkeypatch.setattr(mod, "lookup_po", lookup_po)
    monkeypatch.setattr(matching, "lookup_vendor", lookup_vendor)
    monkeypatch.setattr(matching, "get_catalog_item", get_catalog_item)
    return tmp_path
