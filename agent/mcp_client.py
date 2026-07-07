"""Async client the LangGraph nodes use to call the standalone MCP server.

This talks to mcp_server/server.py over the real MCP protocol -- either by
spawning it as a stdio subprocess (MCP_TRANSPORT=stdio, the default for
local/dev use) or by connecting to an already-running SSE server
(MCP_TRANSPORT=sse, used in docker-compose where the MCP server is its own
container). Either way, tool calls are genuine cross-process MCP requests,
not in-process function calls.
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._session is not None:
            return
        async with self._lock:
            if self._session is not None:
                return
            stack = AsyncExitStack()
            transport = os.environ.get("MCP_TRANSPORT", "stdio")

            if transport == "sse":
                from mcp.client.sse import sse_client

                url = os.environ.get("MCP_SERVER_URL", "http://localhost:8765/sse")
                read, write = await stack.enter_async_context(sse_client(url))
            else:
                cmd = os.environ.get("MCP_SERVER_CMD", "python")
                args_str = os.environ.get("MCP_SERVER_ARGS", "mcp_server/server.py")
                params = StdioServerParameters(command=cmd, args=args_str.split())
                read, write = await stack.enter_async_context(stdio_client(params))

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            self._stack = stack
            self._session = session

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        await self.connect()
        assert self._session is not None
        result = await self._session.call_tool(name, arguments)
        if result.isError:
            raise RuntimeError(f"MCP tool {name} failed: {result.content}")
        text = result.content[0].text
        try:
            return json.loads(text)
        except (json.JSONDecodeError, AttributeError):
            return text

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._session = None
        self._stack = None


_client: MCPClient | None = None


def get_mcp_client() -> MCPClient:
    global _client
    if _client is None:
        _client = MCPClient()
    return _client


async def lookup_po(po_number: str) -> dict[str, Any]:
    return await get_mcp_client().call_tool("lookup_po", {"po_number": po_number})


async def lookup_vendor(vendor: str) -> dict[str, Any]:
    return await get_mcp_client().call_tool("lookup_vendor", {"vendor": vendor})


async def get_catalog_item(sku: str) -> dict[str, Any]:
    return await get_mcp_client().call_tool("get_catalog_item", {"sku": sku})
