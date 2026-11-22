import atexit
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

from contract_tests.fake_llm import FakeLLM


ROOT = Path(__file__).resolve().parents[1]
_lock = threading.Lock()
_admin = None
_temporary_directory = None
_server_process = None
_server_log = None
_fake_llm = None
_child_env = None


def _safe_child_environment():
    child = os.environ.copy()
    for key in list(child):
        if key.startswith("LLM_") or "API_KEY" in key.upper():
            child.pop(key, None)
    child.update(
        {
            "DJANGO_DEBUG": "false",
            "DJANGO_SETTINGS_MODULE": "config.settings",
            "DJANGO_SECRET_KEY": "contract-test",
            "KNOWAGENT_AUTO_LOGIN": "false",
            "LLM_API_KEY": "fake",
            "LLM_USE_ENV_EMBEDDING": "false",
            "NEO4J_URI": "bolt://127.0.0.1:1",
        }
    )
    return child


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _stop_servers():
    global _server_process, _server_log, _fake_llm
    if _server_process is not None and _server_process.poll() is None:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server_process.kill()
            _server_process.wait(timeout=5)
    if _fake_llm is not None:
        _fake_llm.stop()
        _fake_llm = None
    if _server_log is not None:
        _server_log.close()
        _server_log = None


def _start_local_server():
    global BASE_URL, _temporary_directory, _server_process, _server_log, _fake_llm, _child_env
    _temporary_directory = tempfile.TemporaryDirectory(prefix="knowagent-contract-")
    temporary_path = Path(_temporary_directory.name)
    (temporary_path / "media").mkdir()
    _fake_llm = FakeLLM()
    _fake_llm.start()

    _child_env = _safe_child_environment()
    _child_env.update(
        {
            "APP_DB_PATH": str(temporary_path / "db.sqlite3"),
            "APP_MEDIA_ROOT": str(temporary_path / "media"),
            "LLM_BASE_URL": _fake_llm.base_url,
        }
    )
    migrated = subprocess.run(
        [sys.executable, "manage.py", "migrate", "--noinput"],
        cwd=ROOT,
        env=_child_env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if migrated.returncode:
        raise RuntimeError(
            "Contract test database migration failed:\n"
            f"{migrated.stdout}\n{migrated.stderr}"
        )

    port = _free_port()
    command = os.environ.get(
        "CONTRACT_SERVER_CMD",
        "{python} manage.py runserver 127.0.0.1:{port} --noreload",
    ).format(python=sys.executable, port=port)
    _server_log = (temporary_path / "server.log").open("w+", encoding="utf-8")
    _server_process = subprocess.Popen(
        shlex.split(command),
        cwd=ROOT,
        env=_child_env,
        stdout=_server_log,
        stderr=subprocess.STDOUT,
    )
    BASE_URL = f"http://127.0.0.1:{port}"

    deadline = time.monotonic() + 60
    with httpx.Client(timeout=2) as client:
        while time.monotonic() < deadline:
            if _server_process.poll() is not None:
                _server_log.flush()
                _server_log.seek(0)
                raise RuntimeError(
                    "Contract test server exited before becoming healthy:\n"
                    + _server_log.read()
                )
            try:
                response = client.get(f"{BASE_URL}/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
    _server_log.flush()
    _server_log.seek(0)
    raise RuntimeError(
        "Contract test server did not become healthy:\n" + _server_log.read()
    )


BASE_URL = os.environ.get("CONTRACT_BASE_URL", "").rstrip("/")
if not BASE_URL:
    atexit.register(_stop_servers)
    try:
        _start_local_server()
    except Exception:
        _stop_servers()
        raise


def new_client(token=None, timeout=30):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.Client(base_url=BASE_URL, headers=headers, timeout=timeout)


def set_admin(token, refresh_token, password):
    global _admin
    with _lock:
        _admin = {
            "token": token,
            "refresh_token": refresh_token,
            "password": password,
        }


def admin():
    global _admin
    with _lock:
        if _admin is not None:
            return dict(_admin)
        with new_client() as client:
            setup = client.post("/api/v1/auth/auto-setup")
            if setup.status_code != 201:
                raise RuntimeError(
                    "Contract test admin was not initialized by the auth contract"
                )
            password = setup.json()["data"]["temp_password"]
            login = client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": password},
            )
            data = login.json()["data"]
            _admin = {
                "token": data["token"],
                "refresh_token": data["refresh_token"],
                "password": password,
            }
        return dict(_admin)
