"""
knowledge_service.fetch
=========================

Загрузка по URL для источников «Ссылка».

Защита от обращений во внутреннюю сеть: запрашиваются только http/https
адреса, разрешающиеся в публичные IP, — в том числе после каждого
редиректа; исключение — хосты из `KB_ALLOWED_HOSTS`. Заголовок
авторизации из секрета (`KB_SECRETS`) отправляется только на хосты,
привязанные к этому секрету.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import unquote, urlsplit

import httpx

from .formats import CONTENT_TYPES, extension_of

USER_AGENT = "Mozilla/5.0 (compatible; AgentsKnowledge/1.0)"


class FetchError(Exception):
    pass


@dataclass
class FetchResult:
    url: str
    data: bytes
    content_type: str
    charset: Optional[str]
    filename: str
    last_modified: Optional[str]


def host_matches(host: str, patterns: List[str]) -> bool:
    host = (host or "").lower().rstrip(".")
    return any(host == p or host.endswith("." + p) for p in patterns)


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise FetchError("не указан адрес")
    if "://" not in url:
        url = "https://" + url.lstrip("/")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError(f"недопустимый адрес {url!r}: поддерживаются только http и https")
    return url


def _default_resolver(host: str) -> List[str]:
    return [info[4][0] for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)]


class UrlFetcher:
    def __init__(self, allowed_hosts: List[str], secrets: Dict[str, List[str]], timeout: float = 30.0,
                 max_bytes: int = 50 * 1024 * 1024, transport: Optional[httpx.BaseTransport] = None,
                 resolver: Optional[Callable[[str], List[str]]] = None):
        self.allowed_hosts = [h.lower() for h in allowed_hosts]
        self.secrets = secrets
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.transport = transport
        self.resolver = resolver or _default_resolver

    def check_host(self, host: str) -> None:
        if host_matches(host, self.allowed_hosts):
            return
        try:
            addresses = self.resolver(host)
        except OSError as exc:
            raise FetchError(f"не удалось разрешить {host}: {exc}") from exc
        for addr in addresses:
            ip = ipaddress.ip_address(addr)
            if not ip.is_global:
                raise FetchError(f"адрес {host} ведёт во внутреннюю сеть ({ip}) — добавьте хост в KB_ALLOWED_HOSTS")

    def auth_header(self, auth: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Проверенное описание заголовка: {"header", "value", "hosts"}."""
        if not auth:
            return None
        name = str(auth.get("secret") or "")
        if name not in self.secrets:
            raise FetchError(f"секрет {name!r} не разрешён (KB_SECRETS)")
        value = os.environ.get(name)
        if not value:
            raise FetchError(f"переменная окружения {name} не задана")
        scheme = str(auth.get("scheme") if auth.get("scheme") is not None else "Bearer").strip()
        header = str(auth.get("header") or "Authorization")
        return {"header": header, "value": f"{scheme} {value}" if scheme else value, "hosts": self.secrets[name]}

    def fetch(self, url: str, auth: Optional[Dict[str, Any]] = None) -> FetchResult:
        url = normalize_url(url)
        auth_spec = self.auth_header(auth)
        if auth_spec and not host_matches(urlsplit(url).hostname or "", auth_spec["hosts"]):
            raise FetchError(f"секрет {auth.get('secret')!r} нельзя отправлять на {urlsplit(url).hostname}")

        def on_request(request: httpx.Request) -> None:
            if request.url.scheme not in ("http", "https"):
                raise FetchError(f"недопустимая схема адреса: {request.url.scheme}")
            self.check_host(request.url.host)
            if auth_spec:
                if host_matches(request.url.host, auth_spec["hosts"]):
                    request.headers[auth_spec["header"]] = auth_spec["value"]
                else:
                    request.headers.pop(auth_spec["header"], None)

        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True, max_redirects=5, transport=self.transport,
                              headers={"User-Agent": USER_AGENT}, event_hooks={"request": [on_request]}) as client:
                with client.stream("GET", url) as resp:
                    if resp.status_code in (401, 403):
                        raise FetchError(f"HTTP {resp.status_code}: нет доступа (нужна авторизация?)")
                    if resp.status_code >= 400:
                        raise FetchError(f"HTTP {resp.status_code}")
                    chunks, size = [], 0
                    for chunk in resp.iter_bytes():
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise FetchError(f"файл больше {self.max_bytes // (1024 * 1024)} МБ")
                        chunks.append(chunk)
                    final_url = str(resp.url)
                    ctype_full = resp.headers.get("content-type", "")
                    disposition = resp.headers.get("content-disposition", "")
                    last_modified = resp.headers.get("last-modified")
        except httpx.HTTPError as exc:
            raise FetchError(f"сетевая ошибка: {exc}") from exc
        ctype = ctype_full.split(";")[0].strip().lower()
        charset_match = re.search(r"charset=([\w-]+)", ctype_full, re.I)
        date = None
        if last_modified:
            try:
                date = parsedate_to_datetime(last_modified).date().isoformat()
            except (TypeError, ValueError):
                date = None
        return FetchResult(
            url=final_url, data=b"".join(chunks), content_type=ctype,
            charset=charset_match.group(1) if charset_match else None,
            filename=filename_for(final_url, ctype, disposition), last_modified=date,
        )


def filename_for(url: str, content_type: str, disposition: str = "") -> str:
    match = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", disposition or "", re.I)
    if match:
        name = unquote(match.group(1)).strip()
        if extension_of(name):
            return name
    path = unquote(urlsplit(url).path)
    base = path.rstrip("/").rsplit("/", 1)[-1] or (urlsplit(url).hostname or "page")
    ext = extension_of(base)
    by_type = CONTENT_TYPES.get(content_type)
    if content_type in ("text/html", "application/xhtml+xml") and ext not in (".html", ".htm"):
        return base + ".html"
    if not ext and by_type:
        return base + by_type
    return base
