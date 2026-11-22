import ipaddress
import socket
from urllib.parse import urlparse, urlunparse

from requests.adapters import HTTPAdapter
from requests.utils import get_environ_proxies, select_proxy


def resolve_public_address(hostname: str, port: int) -> str | None:
    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        addresses = [info[4][0].split("%")[0] for info in infos]
        if addresses and all(ipaddress.ip_address(address).is_global for address in addresses):
            return addresses[0]
    except (OSError, ValueError):
        pass
    return None


class _PinnedHostAdapter(HTTPAdapter):
    def __init__(self, hostname: str):
        self._tls_hostname = {"server_hostname": hostname, "assert_hostname": hostname}
        super().__init__()

    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **self._tls_hostname, **kwargs)

    def proxy_manager_for(self, proxy, **kwargs):
        return super().proxy_manager_for(proxy, **self._tls_hostname, **kwargs)


def prepare_public_url(url: str) -> tuple[str, str, str] | None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    try:
        hostname = parsed.hostname.encode("idna").decode("ascii")
        port = parsed.port
    except ValueError:
        return None
    address = resolve_public_address(hostname, port or (443 if parsed.scheme == "https" else 80))
    if not address:
        return None
    if parsed.scheme == "http" and select_proxy(url, get_environ_proxies(url)):
        return None
    host = f"[{address}]" if ":" in address else address
    pinned_url = urlunparse(parsed._replace(netloc=f"{host}:{port}" if port else host))
    host_header = f"{hostname}:{port}" if port else hostname
    return pinned_url, hostname, host_header
