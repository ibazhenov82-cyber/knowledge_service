"""
knowledge_service.config
==========================

Настройки сервиса баз знаний (тот же стиль, что у остальных сервисов):
считываются один раз из переменных окружения либо файла `.env` в каталоге
запуска.

Сервис самостоятельный: не импортирует ничего из `agents_core`,
разворачивается отдельным процессом со своим хранилищем.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

#: Откуда загружен `.env` (None — файл не найден); печатается при старте.
DOTENV_PATH: Optional[Path] = None


def _load_dotenv(path: str = ".env") -> None:
    global DOTENV_PATH
    file = Path(path)
    if not file.exists():
        return
    DOTENV_PATH = file.resolve()
    for raw_line in file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _list_env(name: str) -> List[str]:
    raw = os.environ.get(name, "")
    return [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]


def parse_secrets(raw: str) -> Dict[str, List[str]]:
    """`KB_SECRETS`: «ИМЯ@хост1|хост2, ИМЯ2@хост» — какие переменные окружения
    можно указывать в `auth.secret` источника и на какие хосты их разрешено
    отправлять (иначе через ссылку на свой сайт можно было бы выманить
    любой ключ из окружения сервиса)."""
    result: Dict[str, List[str]] = {}
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        name, _, hosts = item.partition("@")
        result[name.strip()] = [h.strip().lower() for h in hosts.split("|") if h.strip()]
    return result


class KnowledgeConfig:
    """Читается один раз при импорте модуля."""

    HOST: str = os.environ.get("KNOWLEDGE_HOST", "0.0.0.0")
    PORT: int = _int_env("KNOWLEDGE_PORT", 8003)
    #: Каталог данных: база `knowledge.sqlite`, копии загруженных файлов и страниц.
    DATA_DIR: str = os.environ.get("KNOWLEDGE_DATA_DIR", "knowledge_data")
    #: Заготовка под аутентификацию (как ключи других сервисов): если задан,
    #: запросы к /api/v1 (кроме /health) должны нести `Authorization: Bearer <ключ>`.
    API_KEY: str = os.environ.get("KNOWLEDGE_API_KEY", "").strip()

    # --- Источники ---------------------------------------------------------------
    #: Разрешённые корни для источников «путь / папка / glob». Пусто — такие
    #: источники выключены.
    ROOTS: List[str] = _list_env("KB_ROOTS")
    #: Внутренние хосты, к которым можно обращаться по URL (иначе — только
    #: публичные адреса). Совпадение по имени хоста или его суффиксу.
    ALLOWED_HOSTS: List[str] = [h.lower() for h in _list_env("KB_ALLOWED_HOSTS")]
    #: Секреты для заголовков авторизации: «ИМЯ@хост1|хост2, ...».
    SECRETS: Dict[str, List[str]] = parse_secrets(os.environ.get("KB_SECRETS", ""))
    MAX_FILE_MB: int = _int_env("KB_MAX_FILE_MB", 50)
    #: Сколько документов может дать один источник (папка, обход ссылок).
    MAX_SOURCE_DOCUMENTS: int = _int_env("KB_MAX_SOURCE_DOCUMENTS", 5000)
    #: Предел страниц при обходе ссылок.
    CRAWL_MAX_PAGES: int = _int_env("KB_CRAWL_MAX_PAGES", 200)
    CRAWL_MAX_DEPTH: int = _int_env("KB_CRAWL_MAX_DEPTH", 3)
    HTTP_TIMEOUT: float = _float_env("KB_HTTP_TIMEOUT", 30.0)

    # --- Эмбеддинги ----------------------------------------------------------------
    OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434").strip().rstrip("/")
    DEFAULT_EMBEDDING_MODEL: str = os.environ.get(
        "KB_DEFAULT_EMBEDDING_MODEL", "ollama/qwen3-embedding:0.6b").strip()
    #: JSON-файл OpenAI-совместимых провайдеров эмбеддингов (см.
    #: embedding_providers.example.json). Пусто — только Ollama.
    EMBEDDING_PROVIDERS_FILE: str = os.environ.get("EMBEDDING_PROVIDERS_FILE", "").strip()
    EMBED_BATCH: int = _int_env("KB_EMBED_BATCH", 32)

    # --- Реранкинг ----------------------------------------------------------------------
    #: JSON-файл моделей-реранкеров (см. rerank_providers.example.json). Пусто —
    #: доступна только «Эвристика (без LLM)».
    RERANK_PROVIDERS_FILE: str = os.environ.get("RERANK_PROVIDERS_FILE", "").strip()
    #: Реранкер по умолчанию («провайдер/модель»); пусто — первый из файла.
    DEFAULT_RERANK_MODEL: str = os.environ.get("KB_DEFAULT_RERANK_MODEL", "").strip()
    #: Сколько ждать модель-реранкер; не дождались — эвристика (`rerank.fallback`).
    #: Должно быть заметно меньше тайм-аута поиска у клиента (KNOWLEDGE_SERVICE_TIMEOUT
    #: в AgentsCore), иначе клиент оборвёт запрос раньше, чем сработает откат.
    RERANK_TIMEOUT: float = _float_env("KB_RERANK_TIMEOUT", 20.0)
    #: Сколько символов фрагмента передавать модели-реранкеру (начало фрагмента
    #: с заголовком): cross-encoder на CPU работает со скоростью, пропорциональной
    #: длине текста, а для оценки релевантности начала обычно достаточно.
    RERANK_MAX_CHARS: int = _int_env("KB_RERANK_MAX_CHARS", 1500)
    EMBED_TIMEOUT: float = _float_env("KB_EMBED_TIMEOUT", 120.0)

    # --- Выполнение ------------------------------------------------------------------
    #: Сколько задач загрузки выполняется одновременно.
    WORKERS: int = _int_env("KB_WORKERS", 1)

    LOG_LEVEL: str = os.environ.get("KNOWLEDGE_LOG_LEVEL", "INFO").strip().upper() or "INFO"
    LOG_FILE: str = os.environ.get("KNOWLEDGE_LOG_FILE", "").strip()


from dataclasses import dataclass, field  # noqa: E402


@dataclass
class Settings:
    """Настройки, передаваемые в сервис явно (тесты создают свои)."""

    data_dir: str
    roots: List[str] = field(default_factory=list)
    allowed_hosts: List[str] = field(default_factory=list)
    secrets: Dict[str, List[str]] = field(default_factory=dict)
    max_file_bytes: int = 50 * 1024 * 1024
    max_source_documents: int = 5000
    crawl_max_pages: int = 200
    crawl_max_depth: int = 3
    http_timeout: float = 30.0
    default_embedding_model: str = "ollama/qwen3-embedding:0.6b"
    embed_batch: int = 32
    workers: int = 1
    rerank_max_chars: int = 1500

    @classmethod
    def from_config(cls) -> "Settings":
        c = KnowledgeConfig
        return cls(
            data_dir=c.DATA_DIR, roots=list(c.ROOTS), allowed_hosts=list(c.ALLOWED_HOSTS),
            secrets=dict(c.SECRETS), max_file_bytes=c.MAX_FILE_MB * 1024 * 1024,
            max_source_documents=c.MAX_SOURCE_DOCUMENTS, crawl_max_pages=c.CRAWL_MAX_PAGES,
            crawl_max_depth=c.CRAWL_MAX_DEPTH, http_timeout=c.HTTP_TIMEOUT,
            default_embedding_model=c.DEFAULT_EMBEDDING_MODEL, embed_batch=c.EMBED_BATCH, workers=c.WORKERS,
            rerank_max_chars=c.RERANK_MAX_CHARS,
        )
