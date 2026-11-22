import ipaddress
import socket
from urllib.parse import urlparse, urlunparse

from requests.adapters import HTTPAdapter
from requests.utils import get_environ_proxies, select_proxy


def resolve_public_address(hostname: str, port: int) -> str | None:
    """全部解析结果都是公网地址时返回其中一个 IP，否则返回 None；回环、内网、链路本地（云元数据）一律拒绝。"""
    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        addresses = [info[4][0].split("%")[0] for info in infos]
        if addresses and all(ipaddress.ip_address(address).is_global for address in addresses):
            return addresses[0]
    except (OSError, ValueError):
        pass
    return None


class _PinnedHostAdapter(HTTPAdapter):
    """连接到已校验的 IP，TLS 的 SNI 与证书校验仍使用原始主机名（直连和经代理都是）。"""

    def __init__(self, hostname: str):
        self._tls_hostname = {"server_hostname": hostname, "assert_hostname": hostname}
        super().__init__()

    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **self._tls_hostname, **kwargs)

    def proxy_manager_for(self, proxy, **kwargs):
        return super().proxy_manager_for(proxy, **self._tls_hostname, **kwargs)


def prepare_public_url(url: str) -> tuple[str, str, str] | None:
    """
    返回 (钉住 IP 的 URL, 原始主机名, Host 头)；地址不允许访问时返回 None。
    连接直接发往校验过的 IP，避免校验之后主机名被重新解析到内网（DNS rebinding）。
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    try:
        # 国际化域名转成 ASCII 形式，Host 头和 TLS 主机名才能正确发出
        hostname = parsed.hostname.encode("idna").decode("ascii")
        port = parsed.port
    except ValueError:
        return None
    address = resolve_public_address(hostname, port or (443 if parsed.scheme == "https" else 80))
    if not address:
        return None
    # 明文 HTTP 经正向代理时由代理自己解析主机名，无法钉住校验过的 IP，所以拒绝；HTTPS 走 CONNECT 到 IP，不受影响
    if parsed.scheme == "http" and select_proxy(url, get_environ_proxies(url)):
        return None
    host = f"[{address}]" if ":" in address else address
    pinned_url = urlunparse(parsed._replace(netloc=f"{host}:{port}" if port else host))
    host_header = f"{hostname}:{port}" if port else hostname
    return pinned_url, hostname, host_header
