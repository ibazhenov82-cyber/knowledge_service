"""
knowledge_service.api
=======================

REST API `/api/v1` — типовой набор RAG-сервиса: коллекции (базы знаний) →
источники и документы → фрагменты → поиск. Загрузка асинхронная: ответ
`202 Accepted` с `job_id`, прогресс — `GET /jobs/{id}` или SSE
`GET /jobs/{id}/events`. Ошибки — `{"error": "..."}`, списки —
`{"items", "total", "limit", "offset"}`.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .jobs import TERMINAL
from .service import KnowledgeService, ValidationError

router = APIRouter(prefix="/api/v1", tags=["knowledge"])


def _svc(request: Request) -> KnowledgeService:
    return request.app.state.service


def _json_field(raw: Optional[str], name: str) -> Optional[Any]:
    if raw is None or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise ValidationError(f"{name}: некорректный JSON") from exc


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CollectionCreate(_Model):
    name: str = Field(..., description="Название базы знаний")
    description: str = ""
    embedding_model: Optional[str] = Field(None, description="«провайдер/модель», по умолчанию ollama/qwen3-embedding:0.6b")
    chunking: Optional[Dict[str, Any]] = Field(
        None, description="method: structure («По структуре», по умолчанию) | fixed («Фиксированный размер»); "
                          "chunk_size_tokens, chunk_overlap_tokens, max_chunk_tokens, min_chunk_tokens")
    file_types: Optional[Dict[str, Any]] = Field(
        None, description="code_extensions, extra_text_extensions, jsonl: {text_fields, metadata_fields}")


class CollectionPatch(_Model):
    name: Optional[str] = None
    description: Optional[str] = None
    embedding_model: Optional[str] = Field(None, description="Смена модели запускает переиндексацию")
    chunking: Optional[Dict[str, Any]] = None
    file_types: Optional[Dict[str, Any]] = None


class SourceCreate(_Model):
    source: Dict[str, Any] = Field(..., description="{type: path|url|text, path | url | title+content+format}")
    title: Optional[str] = None
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    chunking: Optional[Dict[str, Any]] = None
    auth: Optional[Dict[str, Any]] = Field(None, description="{secret, header=Authorization, scheme=Bearer}")
    crawl: Optional[Dict[str, Any]] = Field(None, description="{depth, same_domain, max_pages}")
    jsonl: Optional[Dict[str, Any]] = None


class SourcePatch(_Model):
    title: Optional[str] = None
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    chunking: Optional[Dict[str, Any]] = None
    auth: Optional[Dict[str, Any]] = None
    crawl: Optional[Dict[str, Any]] = None
    jsonl: Optional[Dict[str, Any]] = None


class DocumentPatch(_Model):
    title: Optional[str] = None
    doc_date: Optional[str] = None
    doc_version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = Field(None, description="false — исключить из поиска без удаления")


class DocumentReindex(_Model):
    chunking: Optional[Dict[str, Any]] = Field(None, description="Разбиение для этого документа ({} — как у базы)")


class ChunkCreate(_Model):
    text: str
    section: str = ""
    page: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ChunkPatch(_Model):
    text: Optional[str] = Field(None, description="Новый текст — эмбеддинг пересчитывается")
    enabled: Optional[bool] = None


class RetrievalRequest(_Model):
    query: str
    collection_ids: List[str]
    top_k: int = Field(5, description="Фрагментов после фильтрации (1–50)")
    candidate_k: Optional[int] = Field(None, description="Кандидатов до фильтрации (по умолчанию max(20, top_k), до 200)")
    score_threshold: Optional[float] = Field(None, description="Этап 1: минимальное косинусное сходство")
    rerank: Optional[str] = Field(None, description="Этап 2: none | heuristic («Эвристика (без LLM)») | model («Модель-реранкер»)")
    rerank_model: Optional[str] = Field(None, description="Модель-реранкер «провайдер/модель» (по умолчанию — модель сервиса)")
    rerank_threshold: Optional[float] = Field(None, description="Минимальный балл после реранкинга (0..1)")
    filters: Optional[Dict[str, Any]] = Field(
        None, description="document_ids, source_ids, doc_types, source_types, languages, "
                          "doc_date: {gte, lte}, metadata: {ключ: значение | [значения]}")


def _dump(model: BaseModel) -> Dict[str, Any]:
    return model.model_dump(exclude_unset=True)


# ---------------------------------------------------------------------------
# Коллекции
# ---------------------------------------------------------------------------


@router.post("/collections", status_code=status.HTTP_201_CREATED)
def create_collection(body: CollectionCreate, request: Request) -> Dict[str, Any]:
    return _svc(request).create_collection(_dump(body))


@router.get("/collections")
def list_collections(request: Request) -> Dict[str, Any]:
    items = _svc(request).list_collections()
    return {"items": items, "total": len(items)}


@router.get("/collections/{collection_id}")
def get_collection(collection_id: str, request: Request) -> Dict[str, Any]:
    return _svc(request).get_collection(collection_id)


@router.patch("/collections/{collection_id}")
def update_collection(collection_id: str, body: CollectionPatch, request: Request) -> Dict[str, Any]:
    collection, job_id = _svc(request).update_collection(collection_id, _dump(body))
    return {**collection, "reindex_job_id": job_id}


@router.delete("/collections/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(collection_id: str, request: Request) -> Response:
    _svc(request).delete_collection(collection_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/collections/{collection_id}/reindex", status_code=status.HTTP_202_ACCEPTED)
def reindex_collection(collection_id: str, request: Request) -> Dict[str, Any]:
    return {"job_id": _svc(request).reindex_collection(collection_id)}


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------


@router.post("/collections/{collection_id}/documents", status_code=status.HTTP_202_ACCEPTED)
def upload_documents(
    collection_id: str, request: Request,
    file: List[UploadFile] = File(..., description="Один или несколько файлов"),
    metadata: Optional[str] = Form(None, description="JSON: метаданные документов"),
    chunking: Optional[str] = Form(None, description="JSON: разбиение для этих файлов"),
    jsonl: Optional[str] = Form(None, description="JSON: {text_fields, metadata_fields} для .jsonl"),
) -> Dict[str, Any]:
    files = [(f.filename or "file", f.file.read()) for f in file]
    return _svc(request).upload_documents(
        collection_id, files, _json_field(metadata, "metadata"), _json_field(chunking, "chunking"),
        _json_field(jsonl, "jsonl"))


@router.post("/collections/{collection_id}/sources", status_code=status.HTTP_202_ACCEPTED)
def add_source(collection_id: str, body: SourceCreate, request: Request) -> Dict[str, Any]:
    return _svc(request).add_source(collection_id, _dump(body))


@router.post("/sources/{source_id}/sync", status_code=status.HTTP_202_ACCEPTED)
def sync_source(source_id: str, request: Request) -> Dict[str, Any]:
    return _svc(request).sync_source(source_id)


@router.put("/documents/{document_id}/file", status_code=status.HTTP_202_ACCEPTED)
def replace_document_file(document_id: str, request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    return _svc(request).replace_file(document_id, file.filename or "", file.file.read())


@router.post("/documents/{document_id}/reindex", status_code=status.HTTP_202_ACCEPTED)
def reindex_document(document_id: str, request: Request, body: Optional[DocumentReindex] = None) -> Dict[str, Any]:
    chunking = body.chunking if body is not None and "chunking" in body.model_fields_set else None
    return _svc(request).reindex_document(document_id, chunking)


# ---------------------------------------------------------------------------
# Задачи
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> Dict[str, Any]:
    return _svc(request).get_job(job_id)


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    svc = _svc(request)
    await run_in_threadpool(svc.get_job, job_id)  # 404, если задачи нет

    async def stream():
        last = None
        while True:
            job = await run_in_threadpool(svc.get_job, job_id)
            payload = json.dumps(job, ensure_ascii=False)
            if payload != last:
                last = payload
                event = "done" if job["status"] in TERMINAL else "progress"
                yield f"event: {event}\ndata: {payload}\n\n"
            if job["status"] in TERMINAL or await request.is_disconnected():
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream; charset=utf-8",
                             headers={"Cache-Control": "no-cache"})


@router.get("/collections/{collection_id}/jobs")
def list_jobs(collection_id: str, request: Request, limit: int = Query(50, ge=1, le=200),
              offset: int = Query(0, ge=0)) -> Dict[str, Any]:
    return _svc(request).list_jobs(collection_id, limit, offset)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> Dict[str, Any]:
    return _svc(request).cancel_job(job_id)


# ---------------------------------------------------------------------------
# Источники и документы
# ---------------------------------------------------------------------------


@router.get("/collections/{collection_id}/sources")
def list_sources(collection_id: str, request: Request) -> Dict[str, Any]:
    return _svc(request).list_sources(collection_id)


@router.get("/sources/{source_id}")
def get_source(source_id: str, request: Request) -> Dict[str, Any]:
    return _svc(request).get_source(source_id)


@router.patch("/sources/{source_id}")
def update_source(source_id: str, body: SourcePatch, request: Request) -> Dict[str, Any]:
    return _svc(request).update_source(source_id, _dump(body))


@router.delete("/sources/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_source(source_id: str, request: Request) -> Response:
    _svc(request).delete_source(source_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/collections/{collection_id}/documents")
def list_documents(
    collection_id: str, request: Request,
    status_: Optional[str] = Query(None, alias="status", description="Статусы через запятую; processing — все в работе"),
    source_id: Optional[str] = None, doc_type: Optional[str] = None,
    q: Optional[str] = Query(None, description="Поиск по названию, имени файла, адресу"),
    limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    return _svc(request).list_documents(collection_id, status_, source_id, doc_type, q, limit, offset)


@router.get("/documents/{document_id}")
def get_document(document_id: str, request: Request) -> Dict[str, Any]:
    return _svc(request).get_document(document_id)


@router.get("/documents/{document_id}/content")
def get_document_content(document_id: str, request: Request) -> Dict[str, Any]:
    return _svc(request).get_document_content(document_id)


@router.get("/documents/{document_id}/file")
def get_document_file(document_id: str, request: Request) -> FileResponse:
    path, filename = _svc(request).document_file(document_id)
    return FileResponse(path, filename=filename)


@router.patch("/documents/{document_id}")
def update_document(document_id: str, body: DocumentPatch, request: Request) -> Dict[str, Any]:
    return _svc(request).update_document(document_id, _dump(body))


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, request: Request) -> Response:
    _svc(request).delete_document(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Фрагменты
# ---------------------------------------------------------------------------


@router.get("/documents/{document_id}/chunks")
def list_chunks(document_id: str, request: Request, limit: int = Query(100, ge=1, le=1000),
                offset: int = Query(0, ge=0)) -> Dict[str, Any]:
    return _svc(request).list_chunks(document_id, limit, offset)


@router.post("/documents/{document_id}/chunks", status_code=status.HTTP_201_CREATED)
def add_chunk(document_id: str, body: ChunkCreate, request: Request) -> Dict[str, Any]:
    return _svc(request).add_chunk(document_id, _dump(body))


@router.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: str, request: Request) -> Dict[str, Any]:
    return _svc(request).get_chunk(chunk_id)


@router.patch("/chunks/{chunk_id}")
def update_chunk(chunk_id: str, body: ChunkPatch, request: Request) -> Dict[str, Any]:
    return _svc(request).update_chunk(chunk_id, _dump(body))


@router.delete("/chunks/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chunk(chunk_id: str, request: Request) -> Response:
    _svc(request).delete_chunk(chunk_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Поиск и служебное
# ---------------------------------------------------------------------------


@router.post("/retrieval")
def retrieval(body: RetrievalRequest, request: Request) -> Dict[str, Any]:
    return _svc(request).retrieve(_dump(body) | {"top_k": body.top_k})


@router.get("/health")
def health(request: Request) -> Dict[str, Any]:
    return _svc(request).health()


@router.get("/info")
def info(request: Request) -> Dict[str, Any]:
    return _svc(request).info()


@router.get("/rerank-models")
def rerank_models(request: Request) -> Dict[str, Any]:
    return _svc(request).rerank_models()


@router.get("/embedding-models")
def embedding_models(request: Request) -> Dict[str, Any]:
    return _svc(request).embedding_models()


@router.get("/fs")
def browse(request: Request, path: Optional[str] = None, collection_id: Optional[str] = None) -> Dict[str, Any]:
    return _svc(request).browse(path, collection_id)
