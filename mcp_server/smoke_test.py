# Smoke test: spawn the server over stdio exactly like Claude Desktop would, list tools, call one.
import asyncio, sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
async def main():
    p = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).with_name("server.py"))])
    async with stdio_client(p) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        tools = [t.name for t in (await s.list_tools()).tools]
        print("tools:", tools)
        res = await s.call_tool("lookup_vendor", {"vendor": "acme"})
        print(res.content[0].text[:200])
        assert set(tools) == {"lookup_po", "lookup_vendor", "get_catalog_item"}
asyncio.run(main())
