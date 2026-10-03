"""Точка входа: `python -m knowledge_service`."""

from __future__ import annotations

import logging
import os

from . import config
from .app import build_registry, build_reranker, create_app
from .config import KnowledgeConfig, Settings
from .embeddings import EmbeddingError
from .fetch import UrlFetcher
from .formats import supported_formats
from .db import SchemaMismatchError
from .service import KnowledgeService


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, KnowledgeConfig.LOG_LEVEL, logging.INFO),
        filename=KnowledgeConfig.LOG_FILE or None,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    import uvicorn

    settings = Settings.from_config()
    registry = build_registry(settings)
    reranker = build_reranker()
    try:
        service = KnowledgeService(settings, registry, UrlFetcher(settings.allowed_hosts, settings.secrets,
                                                                  settings.http_timeout, settings.max_file_bytes),
                                   reranker=reranker)
    except SchemaMismatchError as exc:
        raise SystemExit(f"[knowledge] Ошибка: {exc}") from None
    print(f"[knowledge] .env: {config.DOTENV_PATH or 'не найден (только переменные окружения)'}")
    print(f"[knowledge] Данные: {os.path.abspath(settings.data_dir)}")
    print(f"[knowledge] Модель эмбеддингов по умолчанию: {settings.default_embedding_model}")
    for name, provider in registry.providers.items():
        line = f"[knowledge] Провайдер эмбеддингов {name}: {getattr(provider, 'base_url', '')}"
        try:
            ping = getattr(provider, "ping", None)
            if ping:
                ping()
                line += " — доступен"
        except (EmbeddingError, Exception) as exc:  # noqa: BLE001
            line += f" — недоступен ({exc})"
        print(line)
    rerank_models = [m["id"] + (" (по умолчанию)" if m["default"] else "") for m in reranker.models()]
    for model_id, seconds, error in reranker.probe():
        state = f"отвечает за {seconds:.1f} с на 2 коротких фрагмента" if error is None else f"НЕ отвечает: {error}"
        print(f"[knowledge] Реранкер {model_id}: {state}")
    print("[knowledge] Реранкинг: эвристика (без LLM)"
          + (f"; модели-реранкеры: {', '.join(rerank_models)}" if rerank_models else "; моделей-реранкеров нет"))
    roots = ", ".join(os.path.abspath(r) for r in settings.roots) if settings.roots else "не заданы (источники «путь» выключены)"
    print(f"[knowledge] KB_ROOTS: {roots}")
    if settings.allowed_hosts:
        print(f"[knowledge] Внутренние хосты для ссылок: {', '.join(settings.allowed_hosts)}")
    if settings.secrets:
        secrets = "; ".join(n + " → " + ", ".join(h) for n, h in settings.secrets.items())
        print(f"[knowledge] Секреты: {secrets}")
    formats = supported_formats()
    print(f"[knowledge] Форматы: {' '.join(formats['text'] + formats['documents'] + formats['structured'])}; "
          f"код: {' '.join(formats['code'])}")
    uvicorn.run(create_app(service), host=KnowledgeConfig.HOST, port=KnowledgeConfig.PORT)


if __name__ == "__main__":
    main()
