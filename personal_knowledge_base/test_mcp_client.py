import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from django.test import SimpleTestCase, override_settings

from .mcp_client import MCPClient, MCPError


class FakeMCPState:
    def __init__(self):
        self.requests = []
        self.session_id = None
        self.initialize_count = 0
        self.protocol_version = "2025-06-18"
        self.sse = False
        self.sse_delay = 0
        self.paginate = False
        self.expire_next = False
        self.rpc_error = False
        self.large_response = False
        self.http_error = False


class FakeMCPServer:
    def __init__(self):
        state = FakeMCPState()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def _write(self, status, body=b"", content_type="application/json", headers=None):
                self.send_response(status)
                if content_type:
                    self.send_header("Content-Type", content_type)
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def _json(self, status, payload, headers=None):
                self._write(status, json.dumps(payload).encode(), headers=headers)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                message = json.loads(self.rfile.read(length))
                headers = {key.lower(): value for key, value in self.headers.items()}
                state.requests.append({"message": message, "headers": headers})
                method = message["method"]

                if method == "initialize":
                    state.initialize_count += 1
                    state.session_id = f"session-{state.initialize_count}"
                    self._json(
                        200,
                        {
                            "jsonrpc": "2.0",
                            "id": message["id"],
                            "result": {"protocolVersion": state.protocol_version, "capabilities": {"tools": {}}, "serverInfo": {"name": "fake", "version": "1"}},
                        },
                        {"Mcp-Session-Id": state.session_id},
                    )
                    return

                if headers.get("mcp-session-id") != state.session_id:
                    self._write(404, b"expired")
                    return
                if method != "notifications/initialized" and headers.get("mcp-protocol-version") != state.protocol_version:
                    self._write(400, b"missing protocol header")
                    return
                if method == "notifications/initialized":
                    self._write(202)
                    return
                if state.expire_next and method == "tools/list":
                    state.expire_next = False
                    state.session_id = None
                    self._write(404, b"session expired")
                    return

                if method == "tools/list":
                    if state.large_response:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(4 * 1024 * 1024 + 1))
                        self.end_headers()
                        return
                    if state.http_error:
                        self._write(500, b"x" * 400)
                        return
                    if state.rpc_error:
                        response = {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32000, "message": "fake MCP error"}}
                    elif state.paginate and message.get("params", {}).get("cursor") != "second":
                        response = {
                            "jsonrpc": "2.0",
                            "id": message["id"],
                            "result": {"tools": [{"name": "first", "inputSchema": {"type": "object"}}], "nextCursor": "second"},
                        }
                    elif state.paginate:
                        response = {
                            "jsonrpc": "2.0",
                            "id": message["id"],
                            "result": {"tools": [{"name": "second", "annotations": {"readOnlyHint": True}}]},
                        }
                    else:
                        response = {
                            "jsonrpc": "2.0",
                            "id": message["id"],
                            "result": {"tools": [{"name": "echo", "description": "Echo text", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}, "annotations": {"title": "Echo"}}]},
                        }
                elif method == "tools/call":
                    name = message["params"]["name"]
                    content = {
                        "is_error": [{"type": "text", "text": "tool failed"}],
                        "rich": [
                            {"type": "image", "mimeType": "image/png"},
                            {"type": "audio", "mimeType": "audio/wav"},
                            {"type": "resource", "resource": {"uri": "file://sample", "text": "resource text"}},
                            {"type": "resource_link", "uri": "https://example.test/resource"},
                        ],
                    }.get(name, [{"type": "text", "text": f"called {name}"}])
                    result = {"content": content}
                    if name == "is_error":
                        result["isError"] = True
                    if name == "structured":
                        result = {"structuredContent": {"answer": 42}}
                    response = {"jsonrpc": "2.0", "id": message["id"], "result": result}
                else:
                    self._write(400, b"unknown method")
                    return

                if state.sse:
                    events = [
                        {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}},
                        response,
                    ]
                    # 先发一个只有 id、data 为空的预热事件（2025-11-25 规范允许）
                    body = b": fake event stream\n\nid: prime-1\ndata:\n\n"
                    for event in events:
                        encoded = json.dumps(event, separators=(",", ":"))
                        split = encoded.find(',"')
                        if split > 0:
                            body += f"data: {encoded[:split + 1]}\ndata: {encoded[split + 1:]}\n\n".encode()
                        else:
                            body += f"data: {encoded}\n\n".encode()
                    if state.sse_delay:
                        trailing = b": stream remains open\n\n"
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Content-Length", str(len(body) + len(trailing)))
                        self.end_headers()
                        self.wfile.write(body)
                        self.wfile.flush()
                        time.sleep(state.sse_delay)
                        try:
                            self.wfile.write(trailing)
                            self.wfile.flush()
                        except OSError:
                            pass
                    else:
                        self._write(200, body, "text/event-stream")
                else:
                    self._json(200, response)

            def do_DELETE(self):
                headers = {key.lower(): value for key, value in self.headers.items()}
                state.requests.append({"message": None, "headers": headers, "delete": True})
                if headers.get("mcp-session-id") != state.session_id:
                    self._write(404, b"unknown session")
                    return
                state.session_id = None
                self._write(200)

        self.state = state
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/mcp"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class MCPClientTests(SimpleTestCase):
    def setUp(self):
        self.server = FakeMCPServer()
        self.settings_override = override_settings(MCP_ALLOW_PRIVATE_NETWORK=True)
        self.settings_override.enable()

    def tearDown(self):
        self.settings_override.disable()
        self.server.close()

    def test_initialize_handshake_and_session_headers(self):
        client = MCPClient(self.server.url, api_key="secret-key")
        client.initialize()
        tools = client.list_tools()

        self.assertEqual(tools[0]["name"], "echo")
        messages = [item["message"] for item in self.server.state.requests]
        self.assertEqual([item["method"] for item in messages], ["initialize", "notifications/initialized", "tools/list"])
        self.assertEqual(
            messages[0]["params"],
            {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "KnowAgent", "version": "1.0"}},
        )
        initialize_headers, initialized_headers, list_headers = [item["headers"] for item in self.server.state.requests]
        self.assertEqual(initialize_headers["content-type"], "application/json")
        self.assertEqual(initialize_headers["accept"], "application/json, text/event-stream")
        self.assertEqual(initialize_headers["authorization"], "Bearer secret-key")
        self.assertNotIn("mcp-session-id", initialize_headers)
        self.assertEqual(initialized_headers["mcp-session-id"], "session-1")
        self.assertEqual(initialized_headers["mcp-protocol-version"], "2025-06-18")
        self.assertEqual(list_headers["mcp-session-id"], "session-1")
        self.assertEqual(list_headers["mcp-protocol-version"], "2025-06-18")

    def test_sse_skips_interleaved_server_notification(self):
        self.server.state.sse = True
        client = MCPClient(self.server.url)
        self.assertEqual(client.list_tools()[0]["name"], "echo")

    def test_sse_returns_matching_response_before_stream_closes(self):
        self.server.state.sse = True
        self.server.state.sse_delay = 1.5
        started = time.monotonic()
        self.assertEqual(MCPClient(self.server.url).list_tools()[0]["name"], "echo")
        self.assertLess(time.monotonic() - started, 1)

    def test_json_rpc_error_becomes_mcp_error(self):
        self.server.state.rpc_error = True
        client = MCPClient(self.server.url)
        with self.assertRaises(MCPError) as error:
            client.list_tools()
        self.assertEqual(error.exception.code, -32000)
        self.assertEqual(str(error.exception), "fake MCP error")

    def test_http_error_body_is_limited_to_300_characters(self):
        self.server.state.http_error = True
        with self.assertRaises(MCPError) as error:
            MCPClient(self.server.url).list_tools()
        self.assertEqual(len(error.exception.message.split(": ", 1)[1]), 300)

    def test_response_body_is_capped_at_four_mib(self):
        self.server.state.large_response = True
        with self.assertRaises(MCPError) as error:
            MCPClient(self.server.url).list_tools()
        self.assertEqual(error.exception.code, "response_too_large")

    def test_list_tools_follows_cursors(self):
        self.server.state.paginate = True
        client = MCPClient(self.server.url)
        self.assertEqual([tool["name"] for tool in client.list_tools()], ["first", "second"])
        calls = [item["message"] for item in self.server.state.requests if item["message"]["method"] == "tools/list"]
        self.assertEqual(calls[1]["params"], {"cursor": "second"})

    def test_call_tool_flattens_results_and_reports_is_error(self):
        client = MCPClient(self.server.url)
        text, is_error = client.call_tool("is_error", {}, 3)
        self.assertEqual(text, "tool failed")
        self.assertTrue(is_error)
        rich, _ = client.call_tool("rich", {}, 3)
        self.assertIn("[image: image/png]", rich)
        self.assertIn("[audio: audio/wav]", rich)
        self.assertIn("file://sample", rich)
        self.assertIn("resource text", rich)
        self.assertIn("https://example.test/resource", rich)
        structured, _ = client.call_tool("structured", {}, 3)
        self.assertEqual(json.loads(structured), {"answer": 42})

    def test_expired_session_reinitializes_once_and_retries(self):
        self.server.state.expire_next = True
        client = MCPClient(self.server.url)
        self.assertEqual(client.list_tools()[0]["name"], "echo")
        self.assertEqual(self.server.state.initialize_count, 2)

    def test_unsupported_protocol_version_is_rejected(self):
        self.server.state.protocol_version = "2099-01-01"
        client = MCPClient(self.server.url)
        with self.assertRaises(MCPError):
            client.initialize()

    def test_private_address_is_rejected_by_default(self):
        self.settings_override.disable()
        with override_settings(MCP_ALLOW_PRIVATE_NETWORK=False):
            with self.assertRaisesRegex(MCPError, "MCP server address is not allowed"):
                MCPClient(self.server.url).initialize()
        self.settings_override.enable()
        self.assertEqual(self.server.state.requests, [])

    def test_close_deletes_the_session(self):
        client = MCPClient(self.server.url)
        client.initialize()
        client.close()
        deletion = self.server.state.requests[-1]
        self.assertTrue(deletion["delete"])
        self.assertEqual(deletion["headers"]["mcp-session-id"], "session-1")
        self.assertIsNone(client.session_id)
