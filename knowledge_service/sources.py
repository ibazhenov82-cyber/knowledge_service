"""
knowledge_service.sources
===========================

Источники контента: проверка описания и перечисление документов.

- `path` — файл, папка (рекурсивно) или glob (`/data/**/*.md`) внутри
  разрешённых корней `KB_ROOTS`; маски `include` / `exclude`;
- `url` — страница или файл по ссылке, необязательный обход ссылок
  (`crawl: {depth, same_domain, max_pages}`), авторизация секретом;
- `text` — текст, переданный в запросе.

Загрузка файлов (multipart) источника не создаёт — документы появляются
сразу (`source_type = "file"`).
"""

from __future__ import annotations

import glob
import os
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urldefrag, urljoin, urlsplit

from .fetch import FetchError, FetchResult, UrlFetcher, normalize_url
from .formats import FileTypes, html_blocks, parse_jsonl_options, resolve_kind

SOURCE_TYPES = {"path": "Путь / папка", "url": "Ссылка", "text": "Текст"}
PLANNED_SOURCE_TYPES = {"git": "Git-репозиторий", "s3": "S3", "gcs": "GCS"}
DEFAULT_EXCLUDE = ["**/.git/**", "**/node_modules/**", "**/__pycache__/**", "**/.venv/**", "**/.idea/**"]
_GLOB_CHARS = re.compile(r"[*?\[]")


class SourceError(Exception):
    """Ошибка описания или чтения источника (сообщение — для пользователя)."""


@dataclass
class Loaded:
    data: bytes
    charset: Optional[str] = None
    content_type: Optional[str] = None
    doc_date: Optional[str] = None
    uri: Optional[str] = None
    filename: Optional[str] = None


@dataclass
class SourceItem:
    key: str
    uri: str
    filename: str
    load: Callable[[], Loaded]


# ---------------------------------------------------------------------------
# Маски
# ---------------------------------------------------------------------------


def glob_to_regex(pattern: str) -> "re.Pattern[str]":
    out, i = [], 0
    while i < len(pattern):
        ch = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
            continue
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(ch))
        i += 1
    return re.compile("^" + "".join(out) + "$", re.I)


def matches_any(path: str, patterns: List[str]) -> bool:
    path = path.replace("\\", "/")
    for pattern in patterns:
        rx = glob_to_regex(pattern)
        # Маска без «/» (например, «*.md») сравнивается с именем файла.
        if rx.match(path) or ("/" not in pattern and rx.match(path.rsplit("/", 1)[-1])):
            return True
    return False


def _str_list(value: Any, name: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise SourceError(f"{name}: ожидается список строк")
    return [v.strip() for v in value if v.strip()]


# ---------------------------------------------------------------------------
# Проверка описания источника
# ---------------------------------------------------------------------------


def validate_source(body: Dict[str, Any], fetcher: UrlFetcher, roots: List[str],
                    max_pages: int, max_depth: int) -> Tuple[str, Dict[str, Any], Dict[str, Any], str]:
    """(тип, spec, options, заголовок) — spec: что читать; options: маски,
    обход, авторизация, JSONL."""
    source = body.get("source")
    if not isinstance(source, dict) or not source.get("type"):
        raise SourceError("source: ожидается объект с полем type")
    stype = source["type"]
    if stype in PLANNED_SOURCE_TYPES:
        raise SourceError(f"источник «{PLANNED_SOURCE_TYPES[stype]}» пока не поддерживается")
    if stype not in SOURCE_TYPES:
        raise SourceError(f"source.type: ожидается одно из {sorted(SOURCE_TYPES)}")
    options: Dict[str, Any] = {
        "include": _str_list(body.get("include"), "include"),
        "exclude": _str_list(body.get("exclude"), "exclude"),
    }
    if body.get("jsonl") is not None:
        try:
            text_fields, meta_fields = parse_jsonl_options(body["jsonl"])
        except ValueError as exc:
            raise SourceError(str(exc)) from exc
        options["jsonl"] = {"text_fields": text_fields, "metadata_fields": meta_fields}
    title = str(body.get("title") or "").strip()

    if stype == "path":
        path = str(source.get("path") or "").strip()
        if not path:
            raise SourceError("source.path: не указан путь")
        if not roots:
            raise SourceError("источники «путь» выключены: не задан KB_ROOTS")
        check_in_roots(glob_base(path), roots)
        return stype, {"type": "path", "path": path}, options, title or path

    if stype == "url":
        try:
            url = normalize_url(str(source.get("url") or ""))
            fetcher.auth_header(body.get("auth"))
        except FetchError as exc:
            raise SourceError(str(exc)) from exc
        if body.get("auth"):
            auth = body["auth"]
            options["auth"] = {k: auth[k] for k in ("secret", "header", "scheme") if k in auth}
        crawl = body.get("crawl") or {}
        if not isinstance(crawl, dict) or set(crawl) - {"depth", "same_domain", "max_pages"}:
            raise SourceError("crawl: ожидается {\"depth\", \"same_domain\", \"max_pages\"}")
        depth = int(crawl.get("depth") or 0)
        pages = int(crawl.get("max_pages") or (20 if depth else 1))
        if not 0 <= depth <= max_depth:
            raise SourceError(f"crawl.depth: от 0 до {max_depth}")
        if not 1 <= pages <= max_pages:
            raise SourceError(f"crawl.max_pages: от 1 до {max_pages}")
        options["crawl"] = {"depth": depth, "same_domain": bool(crawl.get("same_domain", True)), "max_pages": pages}
        return stype, {"type": "url", "url": url}, options, title or url

    # text
    content = source.get("content")
    if not isinstance(content, str) or not content.strip():
        raise SourceError("source.content: пустой текст")
    fmt = source.get("format") or "markdown"
    if fmt not in ("markdown", "text"):
        raise SourceError("source.format: markdown или text")
    text_title = str(source.get("title") or title or "").strip() or content.strip().split("\n", 1)[0][:80]
    return stype, {"type": "text", "title": text_title, "format": fmt}, options, text_title


# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------


def glob_base(path: str) -> str:
    """Часть пути до первого компонента с маской."""
    parts = path.replace("\\", "/").split("/")
    base = []
    for part in parts:
        if _GLOB_CHARS.search(part):
            break
        base.append(part)
    return "/".join(base) or "/"


def check_in_roots(path: str, roots: List[str]) -> str:
    """Реальный путь (с раскрытыми ссылками) внутри одного из корней."""
    real = os.path.realpath(path)
    for root in roots:
        root_real = os.path.realpath(root)
        if real == root_real or real.startswith(root_real.rstrip(os.sep) + os.sep):
            return real
    raise SourceError(f"путь {path} вне разрешённых каталогов KB_ROOTS")


def _file_date(path: str) -> Optional[str]:
    try:
        return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).date().isoformat()
    except OSError:
        return None


def read_path(path: str, roots: List[str], max_bytes: int) -> Loaded:
    real = check_in_roots(path, roots)
    if not os.path.isfile(real):
        raise SourceError(f"файл {path} не найден")
    size = os.path.getsize(real)
    if size > max_bytes:
        raise SourceError(f"файл больше {max_bytes // (1024 * 1024)} МБ")
    with open(real, "rb") as fh:
        return Loaded(data=fh.read(), doc_date=_file_date(real))


def enumerate_path(spec: Dict[str, Any], options: Dict[str, Any], roots: List[str], file_types: FileTypes,
                   max_bytes: int, max_docs: int, stats: Dict[str, Any]) -> Iterator[SourceItem]:
    path = spec["path"]
    include, exclude = options.get("include") or [], (options.get("exclude") or []) + DEFAULT_EXCLUDE
    if _GLOB_CHARS.search(path):
        base = check_in_roots(glob_base(path), roots)
        candidates = sorted(p for p in glob.glob(path, recursive=True) if os.path.isfile(p))
    else:
        real = check_in_roots(path, roots)
        if os.path.isfile(real):
            base, candidates = os.path.dirname(real), [real]
        elif os.path.isdir(real):
            base = real
            candidates = []
            for dirpath, dirnames, filenames in os.walk(real):
                dirnames.sort()
                for name in sorted(filenames):
                    candidates.append(os.path.join(dirpath, name))
        else:
            raise SourceError(f"путь {path} не найден")
    count = 0
    for candidate in candidates:
        try:
            real = check_in_roots(candidate, roots)
        except SourceError:
            stats["outside_roots"] = stats.get("outside_roots", 0) + 1
            continue
        rel = os.path.relpath(real, base).replace(os.sep, "/")
        if exclude and matches_any(rel, exclude):
            continue
        if include and not matches_any(rel, include):
            continue
        name = os.path.basename(real)
        if resolve_kind(name, file_types) is None:
            stats.setdefault("unsupported", []).append(rel)
            continue
        count += 1
        if count > max_docs:
            raise SourceError(f"в источнике больше {max_docs} документов (KB_MAX_SOURCE_DOCUMENTS)")
        yield SourceItem(key=real, uri=real, filename=name,
                         load=lambda real=real: read_path(real, roots, max_bytes))


# ---------------------------------------------------------------------------
# Ссылки
# ---------------------------------------------------------------------------


def _page_links(result: FetchResult) -> List[str]:
    if "html" not in result.content_type and not result.filename.endswith((".html", ".htm")):
        return []
    from .formats import decode_text

    parser = html_blocks(decode_text(result.data, result.charset))
    links = []
    for href in parser.links:
        absolute = urldefrag(urljoin(result.url, href.strip()))[0]
        if urlsplit(absolute).scheme in ("http", "https"):
            links.append(absolute)
    return links


def enumerate_url(spec: Dict[str, Any], options: Dict[str, Any], fetcher: UrlFetcher, file_types: FileTypes,
                  stats: Dict[str, Any], check_cancel: Callable[[], None]) -> Iterator[SourceItem]:
    crawl = options.get("crawl") or {"depth": 0, "same_domain": True, "max_pages": 1}
    include, exclude = options.get("include") or [], options.get("exclude") or []
    start = urldefrag(spec["url"])[0]
    start_host = urlsplit(start).hostname
    queue = deque([(start, 0)])
    seen = {start}
    fetched = 0
    while queue and fetched < crawl["max_pages"]:
        check_cancel()
        url, depth = queue.popleft()
        try:
            result = fetcher.fetch(url, options.get("auth"))
        except FetchError as exc:
            if url == start:
                raise SourceError(f"{url}: {exc}") from exc
            stats.setdefault("fetch_errors", []).append({"uri": url, "error": str(exc)})
            continue
        fetched += 1
        final = urldefrag(result.url)[0]
        seen.add(final)
        if resolve_kind(result.filename, file_types) is None:
            stats.setdefault("unsupported", []).append(final)
        else:
            loaded = Loaded(data=result.data, charset=result.charset, content_type=result.content_type,
                            doc_date=result.last_modified, uri=final, filename=result.filename)
            yield SourceItem(key=final, uri=final, filename=result.filename, load=lambda loaded=loaded: loaded)
        if depth >= crawl["depth"]:
            continue
        for link in _page_links(result):
            if link in seen:
                continue
            if crawl.get("same_domain", True) and urlsplit(link).hostname != start_host:
                continue
            if exclude and matches_any(link, exclude):
                continue
            if include and not matches_any(link, include):
                continue
            seen.add(link)
            queue.append((link, depth + 1))


def text_item(spec: Dict[str, Any], content: bytes) -> SourceItem:
    ext = ".md" if spec.get("format", "markdown") == "markdown" else ".txt"
    name = re.sub(r"[^\w\- .]+", "_", spec.get("title") or "text").strip()[:80] or "text"
    return SourceItem(key="text", uri=f"text:{spec.get('title', '')}", filename=name + ext,
                      load=lambda: Loaded(data=content))
