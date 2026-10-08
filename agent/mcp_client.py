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
import time
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent import resilience, security
from agent.resilience import PolicyViolation
from agent.telemetry import span


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
            security.verify_server_tools({t.name for t in (await session.list_tools()).tools})

            self._stack = stack
            self._session = session

    async def _call_once(self, name: str, arguments: dict[str, Any]) -> Any:
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

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Policy check -> timeout/retry/breaker -> audit log. Raises CallFailed (PolicyViolation if refused)."""
        started = time.perf_counter()
        with span(f"mcp.{name}") as sp:
            reason = security.check_tool_call(name, arguments)
            if reason:
                sp.set_attribute("mcp.refused", reason)
                security.audit(name, arguments, "refused", 0.0, reason)
                raise PolicyViolation(name, reason)
            try:
                result = await resilience.call(
                    f"mcp.{name}",
                    lambda: self._call_once(name, arguments),
                    timeout=float(os.environ.get("MCP_TIMEOUT_SECONDS", 10)),
                )
                if not isinstance(result, dict):  # tool results are objects; anything else is not trusted
                    raise resilience.CallFailed(name, resilience.FailureKind.INVALID_OUTPUT, 1, "non-object result")
            except resilience.CallFailed as exc:
                sp.set_attribute("mcp.failure", exc.kind.value)
                security.audit(name, arguments, f"failed:{exc.kind.value}", (time.perf_counter() - started) * 1000)
                raise
            security.audit(name, arguments, "ok", (time.perf_counter() - started) * 1000)
            return result

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
