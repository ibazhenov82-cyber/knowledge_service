"""
knowledge_service.app
=======================

Сборка FastAPI-приложения: REST API `/api/v1`, рабочие потоки задач в
lifespan, единый формат ошибок `{"error": "..."}`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api import router
from .config import KnowledgeConfig, Settings
from .embeddings import EmbeddingError, EmbeddingRegistry, OllamaProvider, load_providers_file
from .fetch import UrlFetcher
from .rerank import RerankRegistry, load_rerank_providers
from .service import KnowledgeService, NotFoundError, ValidationError

DESCRIPTION = """
Сервис баз знаний: загрузка документов (файлы, путь или папка на сервере,
ссылки, текст), разбор, разбиение на фрагменты («По структуре» или
«Фиксированный размер»), эмбеддинги и поиск для RAG.

Базы знаний в API — коллекции (`/api/v1/collections`). Загрузка
асинхронная: ответ `202` с `job_id`, прогресс — `/api/v1/jobs/{id}`.
"""


def build_registry(settings: Settings) -> EmbeddingRegistry:
    registry = EmbeddingRegistry(batch=settings.embed_batch)
    registry.add(OllamaProvider(KnowledgeConfig.OLLAMA_URL, timeout=KnowledgeConfig.EMBED_TIMEOUT))
    if KnowledgeConfig.EMBEDDING_PROVIDERS_FILE:
        for provider in load_providers_file(KnowledgeConfig.EMBEDDING_PROVIDERS_FILE, KnowledgeConfig.EMBED_TIMEOUT):
            registry.add(provider)
    return registry


def build_reranker() -> RerankRegistry:
    registry = RerankRegistry(default_model=KnowledgeConfig.DEFAULT_RERANK_MODEL)
    if KnowledgeConfig.RERANK_PROVIDERS_FILE:
        for provider in load_rerank_providers(KnowledgeConfig.RERANK_PROVIDERS_FILE, KnowledgeConfig.RERANK_TIMEOUT):
            registry.add(provider)
    return registry


def create_app(service: Optional[KnowledgeService] = None, *, start_workers: bool = True,
               api_key: Optional[str] = None) -> FastAPI:
    if service is None:
        settings = Settings.from_config()
        service = KnowledgeService(settings, build_registry(settings),
                                   UrlFetcher(settings.allowed_hosts, settings.secrets, settings.http_timeout,
                                              settings.max_file_bytes),
                                   reranker=build_reranker())
    api_key = KnowledgeConfig.API_KEY if api_key is None else api_key

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_workers:
            service.start()
        try:
            yield
        finally:
            service.stop()

    app = FastAPI(title="Knowledge Service", description=DESCRIPTION, lifespan=lifespan)
    app.state.service = service
    app.include_router(router)

    if api_key:
        @app.middleware("http")
        async def check_key(request: Request, call_next):
            path = request.url.path
            if path.startswith("/api/v1") and path != "/api/v1/health":
                if request.headers.get("authorization", "") != f"Bearer {api_key}":
                    return JSONResponse(status_code=401, content={"error": "нужен ключ API (Authorization: Bearer)"})
            return await call_next(request)

    @app.exception_handler(NotFoundError)
    def _not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": str(exc)})

    @app.exception_handler(ValidationError)
    def _validation(request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.exception_handler(EmbeddingError)
    def _embedding(request: Request, exc: EmbeddingError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"error": str(exc)})

    @app.exception_handler(HTTPException)
    def _http(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})

    @app.exception_handler(RequestValidationError)
    def _request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', []) if p != 'body')}: {e.get('msg')}" for e in exc.errors())
        return JSONResponse(status_code=422, content={"error": details or "некорректный запрос"})

    return app
