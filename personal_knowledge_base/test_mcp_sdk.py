import importlib.util
import socket
import threading
import time
import unittest

from django.test import SimpleTestCase, override_settings

from .mcp_client import MCPClient


@unittest.skipUnless(importlib.util.find_spec("mcp"), "MCP SDK is not installed")
class MCPSDKInteropTests(SimpleTestCase):
    def test_fastmcp_streamable_http_add_tool(self):
        import uvicorn
        from mcp.server.fastmcp import FastMCP

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]

        fastmcp = FastMCP("KnowAgent interop", host="127.0.0.1", port=port)

        @fastmcp.tool()
        def add(a: int, b: int) -> int:
            return a + b

        server_app = fastmcp.streamable_http_app()
        server = uvicorn.Server(uvicorn.Config(server_app, host="127.0.0.1", port=port, log_level="critical"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)

        client = MCPClient(f"http://127.0.0.1:{port}/mcp")
        try:
            with override_settings(MCP_ALLOW_PRIVATE_NETWORK=True):
                tools = client.list_tools()
                self.assertIn("add", [tool["name"] for tool in tools])
                text, is_error = client.call_tool("add", {"a": 2, "b": 3}, timeout=10)
                self.assertFalse(is_error)
                self.assertIn("5", text)
        finally:
            client.close()
            server.should_exit = True
            thread.join(timeout=5)
