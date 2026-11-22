import hashlib
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


FIXED_ANSWER = "Contract test answer."
_requests = []
_requests_lock = threading.Lock()


def received_requests():
    with _requests_lock:
        return list(_requests)


def reset_requests():
    with _requests_lock:
        _requests.clear()


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        return

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}

    def _send_json(self, payload, status=200):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        path = urlsplit(self.path).path
        payload = self._read_json()
        with _requests_lock:
            _requests.append({"path": path, "payload": payload})

        if path.endswith("/chat/completions"):
            if payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                words = ["Contract ", "test ", "answer."]
                for word in words:
                    chunk = {"choices": [{"delta": {"content": word}, "index": 0}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True
                return
            self._send_json(
                {
                    "id": "contract-chat",
                    "object": "chat.completion",
                    "model": payload.get("model", "contract-model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": FIXED_ANSWER},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 4, "total_tokens": 5},
                }
            )
            return

        if path.endswith("/embeddings"):
            values = payload.get("input", [])
            if isinstance(values, str):
                values = [values]
            rows = []
            for index, value in enumerate(values):
                seed = int.from_bytes(hashlib.sha256(str(value).encode("utf-8")).digest()[:8], "big")
                generator = random.Random(seed)
                vector = [generator.uniform(-1, 1) for _ in range(384)]
                rows.append({"object": "embedding", "index": index, "embedding": vector})
            self._send_json(
                {
                    "object": "list",
                    "data": rows,
                    "model": payload.get("model", "contract-embedding"),
                    "usage": {"prompt_tokens": len(values), "total_tokens": len(values)},
                }
            )
            return

        if path.endswith("/reranks"):
            documents = payload.get("documents") or payload.get("input") or []
            results = [
                {"index": index, "relevance_score": round(1 / (index + 1), 6)}
                for index in range(len(documents))
            ]
            self._send_json(
                {
                    "output": {"results": results},
                    "usage": {"total_tokens": len(documents)},
                }
            )
            return

        self._send_json({"error": {"message": "not found"}}, status=404)


class FakeLLM:
    def __init__(self):
        self.server = _Server(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self):
        host, port = self.server.server_address
        return f"http://{host}:{port}/v1"

    def start(self):
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
