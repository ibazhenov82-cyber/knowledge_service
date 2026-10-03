"""
knowledge_service.service
===========================

Логика сервиса баз знаний: коллекции, источники, документы, фрагменты,
задачи загрузки и поиск. REST API (`api.py`) — тонкая обёртка над ним.

Загрузка асинхронная: запрос создаёт задачу (`jobs`) и сразу возвращает её
идентификатор; рабочий поток (`jobs.JobRunner`) проводит каждый документ
через этапы: загрузка → разбор → разбиение → эмбеддинги → индекс.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .chunking import METHODS, ChunkingConfig, chunk_document
from .config import Settings
from .db import Database, dumps
from .embeddings import EmbeddingError, EmbeddingRegistry
from .fetch import FetchError, UrlFetcher
from .formats import ExtractError, FileTypes, extension_of, parse_document, resolve_kind, supported_formats
from .jobs import TERMINAL, JobRunner
from .rerank import METHODS as RERANK_METHODS
from .rerank import RerankError, RerankRegistry, heuristic_scores
from .sources import (
    PLANNED_SOURCE_TYPES, SOURCE_TYPES, Loaded, SourceError, SourceItem, check_in_roots, enumerate_path,
    enumerate_url, read_path, text_item, validate_source,
)
from .util import detect_language, estimate_tokens, new_id, normalized_hash, now_ts, sha256_bytes, sha256_text

log = logging.getLogger("knowledge_service")


class NotFoundError(Exception):
    pass


class ValidationError(Exception):
    pass


class JobCancelled(Exception):
    pass


DOC_STATUS_TITLES = {
    "queued": "В очереди", "fetching": "Загрузка", "parsing": "Разбор", "chunking": "Разбиение",
    "embedding": "Эмбеддинги", "indexed": "Готово", "failed": "Ошибка", "skipped": "Пропущен",
    "cancelled": "Отменён",
}
IN_PROGRESS = ("queued", "fetching", "parsing", "chunking", "embedding")
JOB_KIND_TITLES = {"index": "Загрузка файлов", "sync": "Синхронизация источника", "reindex": "Переиндексация"}
JOB_STAGE_TITLES = {
    "listing": "Поиск документов", "fetching": "Загрузка", "parsing": "Разбор", "chunking": "Разбиение",
    "embedding": "Эмбеддинги", "indexing": "Запись в индекс", "cleanup": "Удаление пропавших документов",
}
SOURCE_TYPE_TITLES = {"file": "Файл", **SOURCE_TYPES}
MAX_JOB_ERRORS = 100
RESERVED_METADATA = ("title", "doc_date", "doc_version")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _check_metadata(metadata: Any) -> Dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValidationError("metadata: ожидается объект")
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise ValidationError("metadata: ключи — непустые строки")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValidationError(f"metadata.{key}: допускаются строки, числа и логические значения")
        if key == "doc_date" and value and not _DATE_RE.match(str(value)):
            raise ValidationError("metadata.doc_date: дата в формате ГГГГ-ММ-ДД")
    return {k: v for k, v in metadata.items() if v is not None}


@dataclass
class _Index:
    model: str
    ids: List[str]
    matrix: np.ndarray


@dataclass
class _JobCtx:
    service: "KnowledgeService"
    job_id: str
    progress: Dict[str, Any] = field(default_factory=lambda: {
        "documents_total": 0, "documents_done": 0, "documents_failed": 0, "documents_skipped": 0,
        "documents_unchanged": 0, "documents_deleted": 0, "chunks_done": 0, "current": None,
    })
    errors: List[Dict[str, Any]] = field(default_factory=list)
    stage: Optional[str] = None
    #: Переиндексировать даже неизменившиеся документы.
    force: bool = False

    def save(self) -> None:
        self.service.db.update("jobs", self.job_id, {
            "stage": self.stage, "progress_json": dumps(self.progress), "errors_json": dumps(self.errors),
        })

    def set_stage(self, stage: Optional[str], current: Optional[str] = None) -> None:
        self.stage = stage
        if current is not None:
            self.progress["current"] = current
        self.save()

    def add_error(self, document_id: Optional[str], uri: str, error: str) -> None:
        if len(self.errors) < MAX_JOB_ERRORS:
            self.errors.append({"document_id": document_id, "uri": uri, "error": error})

    def check_cancel(self) -> None:
        flag = self.service.db.scalar("SELECT cancel_requested FROM jobs WHERE id = ?", (self.job_id,))
        if flag is None or flag:  # задачу удалили вместе с коллекцией или отменили
            raise JobCancelled()


class KnowledgeService:
    def __init__(self, settings: Settings, registry: EmbeddingRegistry, fetcher: Optional[UrlFetcher] = None,
                 db: Optional[Database] = None, reranker: Optional[RerankRegistry] = None):
        self.settings = settings
        #: Модели-реранкеры (второй этап поиска); пусто — доступна только эвристика.
        self.reranker = reranker or RerankRegistry()
        #: Сколько символов фрагмента уходит модели-реранкеру (0 — без ограничения).
        self.rerank_max_chars = settings.rerank_max_chars
        os.makedirs(settings.data_dir, exist_ok=True)
        self.files_dir = os.path.join(settings.data_dir, "files")
        os.makedirs(self.files_dir, exist_ok=True)
        self.db = db or Database(os.path.join(settings.data_dir, "knowledge.sqlite"))
        self.registry = registry
        self.registry.cache_get = self._cache_get
        self.registry.cache_put = self._cache_put
        self.fetcher = fetcher or UrlFetcher(settings.allowed_hosts, settings.secrets, settings.http_timeout,
                                             settings.max_file_bytes)
        self.runner = JobRunner(self, settings.workers)
        self._index: Dict[str, _Index] = {}
        self._index_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Восстановление после перезапуска и запуск рабочих потоков."""
        now = now_ts()
        interrupted = "прервано перезапуском сервиса"
        with self.db.connect() as conn:
            conn.execute("UPDATE jobs SET status = 'failed', error = ?, finished_at = ? WHERE status = 'running'",
                         (interrupted, now))
            conn.execute(
                f"UPDATE documents SET status = 'failed', error = ?, updated_at = ? "
                f"WHERE status IN ({','.join('?' for _ in IN_PROGRESS[1:])})",
                (interrupted, now, *IN_PROGRESS[1:]),
            )
        self.runner.start()
        for job in self.db.all("SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at"):
            self.runner.submit(job["id"])

    def stop(self) -> None:
        self.runner.stop()

    # ------------------------------------------------------------------
    # Кэш эмбеддингов
    # ------------------------------------------------------------------

    def _cache_get(self, model: str, keys: List[str]) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}
        with self.db.connect() as conn:
            for pos in range(0, len(keys), 500):
                part = keys[pos:pos + 500]
                rows = conn.execute(
                    f"SELECT text_hash, vector FROM embedding_cache WHERE model = ? AND text_hash IN "
                    f"({','.join('?' for _ in part)})", (model, *part)).fetchall()
                for row in rows:
                    result[row["text_hash"]] = np.frombuffer(row["vector"], dtype=np.float32)
        return result

    def _cache_put(self, model: str, vectors: Dict[str, np.ndarray]) -> None:
        with self.db.connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (model, text_hash, vector) VALUES (?, ?, ?)",
                [(model, key, np.asarray(vec, dtype=np.float32).tobytes()) for key, vec in vectors.items()],
            )

    # ------------------------------------------------------------------
    # Коллекции
    # ------------------------------------------------------------------

    def _collection_row(self, collection_id: str) -> Dict[str, Any]:
        row = self.db.one("SELECT * FROM collections WHERE id = ?", (collection_id,))
        if not row:
            raise NotFoundError(f"коллекция {collection_id} не найдена")
        return row

    def _collection_out(self, row: Dict[str, Any]) -> Dict[str, Any]:
        stats = self.db.one(
            """SELECT COUNT(*) AS documents,
                      SUM(status = 'indexed') AS indexed, SUM(status = 'failed') AS failed,
                      SUM(status = 'skipped') AS skipped,
                      SUM(status IN ('queued','fetching','parsing','chunking','embedding')) AS processing,
                      COALESCE(SUM(chunk_count), 0) AS chunks, COALESCE(SUM(size_bytes), 0) AS size_bytes,
                      MAX(indexed_at) AS last_indexed_at
               FROM documents WHERE collection_id = ?""", (row["id"],)) or {}
        stats = {k: (v or 0) if k != "last_indexed_at" else v for k, v in stats.items()}
        if stats["processing"]:
            status = "processing"
        elif not stats["documents"]:
            status = "empty"
        elif stats["failed"]:
            status = "has_errors"
        else:
            status = "ready"
        return {
            "id": row["id"], "name": row["name"], "description": row["description"],
            "embedding_model": row["embedding_model"], "dims": row["dims"],
            "chunking": row["chunking"], "file_types": FileTypes.from_dict(row["file_types"]).to_dict(),
            "status": status, "stats": {k: v for k, v in stats.items() if k != "last_indexed_at"},
            "last_indexed_at": stats.get("last_indexed_at"),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    def _file_types(self, collection: Dict[str, Any]) -> FileTypes:
        return FileTypes.from_dict(collection["file_types"])

    def create_collection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        name = str(data.get("name") or "").strip()
        if not name:
            raise ValidationError("name: не указано название")
        model = str(data.get("embedding_model") or self.settings.default_embedding_model).strip()
        try:
            self.registry.validate(model)
            chunking = ChunkingConfig.from_dict(data.get("chunking"))
            file_types = FileTypes.from_dict(data.get("file_types"))
        except (EmbeddingError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc
        now = now_ts()
        row = {
            "id": new_id("col"), "name": name, "description": str(data.get("description") or ""),
            "embedding_model": model, "dims": None, "chunking_json": dumps(chunking.to_dict()),
            "file_types_json": dumps(file_types.to_dict()), "created_at": now, "updated_at": now,
        }
        self.db.insert("collections", row)
        return self._collection_out(self._collection_row(row["id"]))

    def list_collections(self) -> List[Dict[str, Any]]:
        return [self._collection_out(r) for r in self.db.all("SELECT * FROM collections ORDER BY created_at")]

    def get_collection(self, collection_id: str) -> Dict[str, Any]:
        return self._collection_out(self._collection_row(collection_id))

    def update_collection(self, collection_id: str, patch: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        """(коллекция, id задачи переиндексации — если сменилась модель)."""
        row = self._collection_row(collection_id)
        values: Dict[str, Any] = {}
        try:
            if patch.get("name") is not None:
                if not str(patch["name"]).strip():
                    raise ValidationError("name: не указано название")
                values["name"] = str(patch["name"]).strip()
            if patch.get("description") is not None:
                values["description"] = str(patch["description"])
            if patch.get("chunking") is not None:
                base = ChunkingConfig.from_dict(row["chunking"])
                # Смена способа без параметров — параметры остаются прежними.
                values["chunking_json"] = dumps(ChunkingConfig.from_dict(patch["chunking"], base).to_dict())
            if patch.get("file_types") is not None:
                values["file_types_json"] = dumps(
                    FileTypes.from_dict(patch["file_types"], FileTypes.from_dict(row["file_types"])).to_dict())
            model_changed = False
            if patch.get("embedding_model") and patch["embedding_model"] != row["embedding_model"]:
                self.registry.validate(patch["embedding_model"])
                values["embedding_model"] = patch["embedding_model"]
                values["dims"] = None
                model_changed = True
        except (EmbeddingError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc
        if values:
            values["updated_at"] = now_ts()
            self.db.update("collections", collection_id, values)
        job_id = None
        if model_changed:
            self._invalidate(collection_id)
            job_id = self._create_job(collection_id, "reindex", {"collection_id": collection_id})
        return self.get_collection(collection_id), job_id

    def delete_collection(self, collection_id: str) -> None:
        self._collection_row(collection_id)
        self.db.execute("UPDATE jobs SET cancel_requested = 1 WHERE collection_id = ? AND status IN ('queued','running')",
                        (collection_id,))
        with self.db.connect() as conn:
            conn.execute("DELETE FROM chunks WHERE collection_id = ?", (collection_id,))
            conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
        shutil.rmtree(os.path.join(self.files_dir, collection_id), ignore_errors=True)
        self._invalidate(collection_id)

    def reindex_collection(self, collection_id: str) -> str:
        self._collection_row(collection_id)
        return self._create_job(collection_id, "reindex", {"collection_id": collection_id})

    # ------------------------------------------------------------------
    # Задачи
    # ------------------------------------------------------------------

    def _create_job(self, collection_id: str, kind: str, target: Dict[str, Any]) -> str:
        job_id = new_id("job")
        self.db.insert("jobs", {
            "id": job_id, "collection_id": collection_id, "kind": kind, "status": "queued", "stage": None,
            "target_json": dumps(target), "progress_json": dumps(_JobCtx(self, job_id).progress),
            "errors_json": "[]", "created_at": now_ts(),
        })
        self.runner.submit(job_id)
        return job_id

    def _job_out(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": row["id"], "collection_id": row["collection_id"], "kind": row["kind"],
            "kind_title": JOB_KIND_TITLES.get(row["kind"], row["kind"]), "status": row["status"],
            "stage": row["stage"], "stage_title": JOB_STAGE_TITLES.get(row["stage"] or "", None),
            "target": row["target"], "progress": row["progress"], "errors": row["errors"], "error": row["error"],
            "cancel_requested": bool(row["cancel_requested"]),
            "created_at": row["created_at"], "started_at": row["started_at"], "finished_at": row["finished_at"],
        }

    def get_job(self, job_id: str) -> Dict[str, Any]:
        row = self.db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if not row:
            raise NotFoundError(f"задача {job_id} не найдена")
        return self._job_out(row)

    def list_jobs(self, collection_id: str, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        self._collection_row(collection_id)
        rows = self.db.all("SELECT * FROM jobs WHERE collection_id = ? ORDER BY created_at DESC, rowid DESC "
                           "LIMIT ? OFFSET ?", (collection_id, limit, offset))
        total = self.db.scalar("SELECT COUNT(*) FROM jobs WHERE collection_id = ?", (collection_id,))
        return {"items": [self._job_out(r) for r in rows], "total": total, "limit": limit, "offset": offset}

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        job = self.get_job(job_id)
        if job["status"] == "queued":
            now = now_ts()
            self.db.update("jobs", job_id, {"status": "cancelled", "finished_at": now, "cancel_requested": 1})
            self.db.execute("UPDATE documents SET status = 'cancelled', updated_at = ? WHERE job_id = ? "
                            "AND status = 'queued'", (now, job_id))
        elif job["status"] == "running":
            self.db.update("jobs", job_id, {"cancel_requested": 1})
        return self.get_job(job_id)

    def run_job(self, job_id: str) -> None:
        row = self.db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if not row or row["status"] != "queued":
            return
        ctx = _JobCtx(self, job_id)
        ctx.progress = dict(row["progress"] or ctx.progress)
        ctx.force = bool(row["target"].get("force")) or row["kind"] == "reindex"
        self.db.update("jobs", job_id, {"status": "running", "started_at": now_ts()})
        status, error = "completed", None
        try:
            ctx.check_cancel()
            if row["kind"] == "index":
                self._run_documents(ctx, row["target"]["document_ids"])
            elif row["kind"] == "reindex":
                self._collection_row(row["collection_id"])
                ids = [d["id"] for d in self.db.all(
                    "SELECT id FROM documents WHERE collection_id = ? ORDER BY created_at", (row["collection_id"],))]
                self._run_documents(ctx, ids)
            elif row["kind"] == "sync":
                self._run_sync(ctx, row["target"]["source_id"])
            else:
                raise ValidationError(f"неизвестный вид задачи {row['kind']}")
            if ctx.progress["documents_failed"] and not ctx.progress["documents_done"] \
                    and not ctx.progress["documents_unchanged"]:
                status, error = "failed", "ни один документ не проиндексирован"
        except JobCancelled:
            status = "cancelled"
            self.db.execute("UPDATE documents SET status = 'cancelled', updated_at = ? WHERE job_id = ? "
                            "AND status = 'queued'", (now_ts(), job_id))
        except (NotFoundError, SourceError, ValidationError, EmbeddingError) as exc:
            status, error = "failed", str(exc)
        except Exception as exc:  # noqa: BLE001
            log.exception("задача %s", job_id)
            status, error = "failed", f"внутренняя ошибка: {exc}"
        ctx.progress["current"] = None
        if self.db.one("SELECT id FROM jobs WHERE id = ?", (job_id,)):
            self.db.update("jobs", job_id, {
                "status": status, "error": error, "stage": None, "finished_at": now_ts(),
                "progress_json": dumps(ctx.progress), "errors_json": dumps(ctx.errors),
            })

    def _run_documents(self, ctx: _JobCtx, document_ids: List[str]) -> None:
        ctx.progress["documents_total"] = len(document_ids)
        ctx.save()
        for doc_id in document_ids:
            ctx.check_cancel()
            doc = self.db.one("SELECT * FROM documents WHERE id = ?", (doc_id,))
            if not doc:
                continue
            if doc["status"] == "skipped" and doc["error"] and doc["error"].startswith("формат"):
                ctx.progress["documents_skipped"] += 1
                continue
            self._process_and_count(ctx, doc)

    def _process_and_count(self, ctx: _JobCtx, doc: Dict[str, Any], loaded: Optional[Loaded] = None) -> None:
        status = self._process_document(ctx, doc, loaded)
        if status == "indexed":
            ctx.progress["documents_done"] += 1
        elif status == "skipped":
            ctx.progress["documents_skipped"] += 1
        elif status == "unchanged":
            ctx.progress["documents_unchanged"] += 1
        else:
            ctx.progress["documents_failed"] += 1
        ctx.save()

    def _run_sync(self, ctx: _JobCtx, source_id: str) -> None:
        source = self.db.one("SELECT * FROM sources WHERE id = ?", (source_id,))
        if not source:
            raise NotFoundError(f"источник {source_id} не найден")
        collection = self._collection_row(source["collection_id"])
        file_types = self._file_types(collection).with_jsonl(source["options"].get("jsonl"))
        stats: Dict[str, Any] = {}
        ctx.set_stage("listing", source["title"])
        items: Iterable[SourceItem]
        if source["type"] == "path":
            items = enumerate_path(source["spec"], source["options"], self.settings.roots, file_types,
                                   self.settings.max_file_bytes, self.settings.max_source_documents, stats)
        elif source["type"] == "url":
            items = enumerate_url(source["spec"], source["options"], self.fetcher, file_types, stats, ctx.check_cancel)
        else:
            path = self._text_path(source)
            with open(path, "rb") as fh:
                content = fh.read()
            items = [text_item(source["spec"], content)]
        seen = set()
        for item in items:
            ctx.check_cancel()
            seen.add(item.key)
            ctx.progress["documents_total"] += 1
            doc = self.db.one("SELECT * FROM documents WHERE source_id = ? AND source_key = ?", (source_id, item.key))
            source_meta = {k: v for k, v in (source["metadata"] or {}).items() if k != "title"}
            if doc is None:
                doc = self._new_document(collection["id"], source_id, source["type"], item.key, item.uri,
                                         item.filename, ctx.job_id, metadata=source_meta)
            else:
                self.db.update("documents", doc["id"], {
                    "job_id": ctx.job_id, "metadata_json": dumps({**(doc["metadata"] or {}), **source_meta})})
            try:
                ctx.set_stage("fetching", item.uri)
                self._set_doc_status(doc["id"], "fetching")
                loaded = item.load()
            except (SourceError, FetchError) as exc:
                self._fail_document(ctx, doc, str(exc))
                ctx.progress["documents_failed"] += 1
                ctx.save()
                continue
            self._process_and_count(ctx, self.db.one("SELECT * FROM documents WHERE id = ?", (doc["id"],)), loaded)
        unsupported = stats.get("unsupported") or []
        if unsupported:
            ctx.progress["unsupported"] = len(unsupported)
            ctx.progress["unsupported_examples"] = unsupported[:20]
        for err in stats.get("fetch_errors") or []:
            ctx.add_error(None, err["uri"], err["error"])
            ctx.progress["documents_failed"] += 1
        # Документы, которых больше нет в источнике.
        ctx.set_stage("cleanup")
        for doc in self.db.all("SELECT id, source_key, raw_path FROM documents WHERE source_id = ?", (source_id,)):
            if doc["source_key"] not in seen:
                self._delete_document_row(doc)
                ctx.progress["documents_deleted"] += 1
        self.db.update("sources", source_id, {"last_synced_at": now_ts()})
        self._invalidate(collection["id"])
        ctx.save()

    # ------------------------------------------------------------------
    # Обработка документа
    # ------------------------------------------------------------------

    def _set_doc_status(self, doc_id: str, status: str) -> None:
        self.db.update("documents", doc_id, {"status": status, "updated_at": now_ts()})

    def _fail_document(self, ctx: Optional[_JobCtx], doc: Dict[str, Any], error: str, status: str = "failed") -> None:
        with self.db.connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ? AND origin != 'manual'", (doc["id"],))
            conn.execute("UPDATE documents SET status = ?, error = ?, updated_at = ?, "
                         "chunk_count = (SELECT COUNT(*) FROM chunks WHERE document_id = ?) WHERE id = ?",
                         (status, error, now_ts(), doc["id"], doc["id"]))
        self._invalidate(doc["collection_id"])
        if ctx is not None and status == "failed":
            ctx.add_error(doc["id"], doc["uri"], error)

    def _load_raw(self, doc: Dict[str, Any]) -> Loaded:
        if doc["source_type"] == "path":
            return read_path(doc["uri"], self.settings.roots, self.settings.max_file_bytes)
        if doc["source_type"] == "text" and doc["source_id"]:
            doc = {**doc, "raw_path": self._text_path({"id": doc["source_id"], "collection_id": doc["collection_id"]})}
        if not doc["raw_path"] or not os.path.exists(doc["raw_path"]):
            raise SourceError("исходный файл документа не найден — загрузите его заново")
        with open(doc["raw_path"], "rb") as fh:
            return Loaded(data=fh.read())

    def _effective_chunking(self, collection: Dict[str, Any], doc: Dict[str, Any]) -> ChunkingConfig:
        cfg = ChunkingConfig.from_dict(collection["chunking"])
        if doc["source_id"]:
            source = self.db.one("SELECT chunking_json FROM sources WHERE id = ?", (doc["source_id"],))
            if source and source["chunking"]:
                cfg = ChunkingConfig.from_dict(source["chunking"], cfg)
        if doc["chunking"]:
            cfg = ChunkingConfig.from_dict(doc["chunking"], cfg)
        return cfg

    def _doc_file_types(self, collection: Dict[str, Any], doc: Dict[str, Any]) -> FileTypes:
        file_types = self._file_types(collection)
        if doc["source_id"]:
            source = self.db.one("SELECT options_json FROM sources WHERE id = ?", (doc["source_id"],))
            if source:
                file_types = file_types.with_jsonl(source["options"].get("jsonl"))
        return file_types.with_jsonl((doc["metadata"] or {}).get("_jsonl"))

    def _process_document(self, ctx: _JobCtx, doc: Dict[str, Any], loaded: Optional[Loaded] = None) -> str:
        """Проиндексировать документ. Возвращает indexed / unchanged / skipped / failed."""
        try:
            collection = self._collection_row(doc["collection_id"])
        except NotFoundError:
            raise JobCancelled()
        title = doc["title"] or doc["filename"]
        try:
            if loaded is None:
                ctx.set_stage("fetching", title)
                self._set_doc_status(doc["id"], "fetching")
                loaded = self._load_raw(doc)
            if len(loaded.data) > self.settings.max_file_bytes:
                raise SourceError(f"файл больше {self.settings.max_file_bytes // (1024 * 1024)} МБ")
            raw_hash = sha256_bytes(loaded.data)
            chunking = self._effective_chunking(collection, doc)
            file_types = self._doc_file_types(collection, doc)
            if (not ctx.force and doc["indexed_at"] and not doc["error"] and doc["raw_hash"] == raw_hash
                    and doc["applied_chunking"] == chunking.to_dict()
                    and self._doc_model(doc["id"]) in (collection["embedding_model"], None)):
                self._set_doc_status(doc["id"], "indexed")
                return "unchanged"
            raw_path = doc["raw_path"]
            if doc["source_type"] == "url":
                raw_path = self._store_raw(doc, loaded.data)
            filename = loaded.filename or doc["filename"]

            ctx.set_stage("parsing", title)
            self._set_doc_status(doc["id"], "parsing")
            parsed = parse_document(loaded.data, filename, file_types, loaded.charset)

            ctx.set_stage("chunking", parsed.title)
            self._set_doc_status(doc["id"], "chunking")
            full_text, drafts = chunk_document(parsed, chunking)
            content_hash = sha256_text(full_text)
            dup = self.db.one(
                "SELECT id, title FROM documents WHERE collection_id = ? AND content_hash = ? AND id != ? "
                "AND status = 'indexed'", (collection["id"], content_hash, doc["id"]))
            if dup:
                self.db.update("documents", doc["id"], {"raw_hash": raw_hash, "content_hash": content_hash,
                                                        "title": doc["title"] or parsed.title})
                self._fail_document(ctx, doc, f"дубликат документа «{dup['title']}»", status="skipped")
                return "skipped"

            meta = doc["metadata"] or {}
            doc_title = str(meta.get("title") or parsed.title or doc["filename"])
            section_titles = [d.section for d in drafts]
            ctx.set_stage("embedding", doc_title)
            self._set_doc_status(doc["id"], "embedding")
            texts = [f"{doc_title}\n{s}\n\n{d.text}" if s else f"{doc_title}\n\n{d.text}"
                     for s, d in zip(section_titles, drafts)]
            model = collection["embedding_model"]
            vectors = self.registry.embed_documents(model, texts, check_cancel=ctx.check_cancel)

            ctx.set_stage("indexing", doc_title)
            now = now_ts()
            language = detect_language(full_text)
            with self.db.connect() as conn:
                conn.execute("DELETE FROM chunks WHERE document_id = ? AND origin != 'manual'", (doc["id"],))
                conn.executemany(
                    """INSERT INTO chunks (id, document_id, collection_id, seq, origin, text, section, page,
                           char_start, char_end, tokens, chunking_method, metadata_json, enabled, text_hash,
                           embedding_model, embedding, created_at, updated_at)
                       VALUES (?, ?, ?, ?, 'auto', ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)""",
                    [(f"{doc['id']}-{i + 1}", doc["id"], collection["id"], i + 1, d.text, d.section, d.page,
                      d.char_start, d.char_end, d.tokens, chunking.method, dumps(d.metadata),
                      normalized_hash(d.text), model, vectors[i].tobytes(), now, now)
                     for i, d in enumerate(drafts)],
                )
                chunk_count = conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?",
                                           (doc["id"],)).fetchone()[0]
                conn.execute(
                    """UPDATE documents SET status = 'indexed', error = NULL, warnings_json = ?, title = ?,
                           doc_type = ?, code_language = ?, raw_path = ?, raw_hash = ?, content_hash = ?,
                           size_bytes = ?, chunk_count = ?, char_count = ?, language = ?, doc_date = ?,
                           doc_version = ?, applied_chunking_json = ?, content_type = ?, text = ?, filename = ?,
                           properties_json = ?, updated_at = ?, indexed_at = ? WHERE id = ?""",
                    (dumps(parsed.warnings), doc_title, parsed.doc_type, parsed.code_language, raw_path, raw_hash,
                     content_hash, len(loaded.data), chunk_count, len(full_text), language,
                     meta.get("doc_date") or parsed.doc_date or loaded.doc_date or doc["doc_date"],
                     meta.get("doc_version") or parsed.doc_version, dumps(chunking.to_dict()),
                     loaded.content_type, full_text, filename, dumps(parsed.properties), now, now, doc["id"]),
                )
                if vectors.size:
                    conn.execute("UPDATE collections SET dims = ? WHERE id = ? AND (dims IS NULL OR dims != ?)",
                                 (int(vectors.shape[1]), collection["id"], int(vectors.shape[1])))
            ctx.progress["chunks_done"] += len(drafts)
            self._invalidate(collection["id"])
            return "indexed"
        except JobCancelled:
            self._set_doc_status(doc["id"], "cancelled" if not doc["indexed_at"] else "indexed")
            raise
        except (ExtractError, SourceError, FetchError, EmbeddingError, ValueError) as exc:
            status = "skipped" if isinstance(exc, ExtractError) and str(exc).startswith("формат") else "failed"
            self._fail_document(ctx, doc, str(exc), status=status)
            return status

    def _doc_model(self, doc_id: str) -> Optional[str]:
        return self.db.scalar("SELECT embedding_model FROM chunks WHERE document_id = ? AND origin = 'auto' LIMIT 1",
                              (doc_id,))

    # ------------------------------------------------------------------
    # Документы
    # ------------------------------------------------------------------

    def _doc_dir(self, collection_id: str, doc_id: str) -> str:
        path = os.path.join(self.files_dir, collection_id, doc_id)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _safe_name(filename: str) -> str:
        name = os.path.basename((filename or "").replace("\\", "/")).strip()
        name = re.sub(r"[\x00-\x1f/]", "_", name)
        return name or "file"

    def _store_raw(self, doc: Dict[str, Any], data: bytes, filename: Optional[str] = None) -> str:
        directory = self._doc_dir(doc["collection_id"], doc["id"])
        for old in os.listdir(directory):
            os.remove(os.path.join(directory, old))
        path = os.path.join(directory, self._safe_name(filename or doc["filename"]))
        with open(path, "wb") as fh:
            fh.write(data)
        self.db.update("documents", doc["id"], {"raw_path": path})
        return path

    def _text_path(self, source: Dict[str, Any]) -> str:
        return os.path.join(self.files_dir, source["collection_id"], "sources", f"{source['id']}.txt")

    def _new_document(self, collection_id: str, source_id: Optional[str], source_type: str, key: str, uri: str,
                      filename: str, job_id: Optional[str], metadata: Optional[Dict[str, Any]] = None,
                      chunking: Optional[Dict[str, Any]] = None, status: str = "queued",
                      error: Optional[str] = None) -> Dict[str, Any]:
        now = now_ts()
        meta = dict(metadata or {})
        row = {
            "id": new_id("doc"), "collection_id": collection_id, "source_id": source_id, "source_type": source_type,
            "source_key": key, "uri": uri, "filename": filename, "title": str(meta.get("title") or ""),
            "status": status, "error": error, "metadata_json": dumps(meta), "chunking_json": dumps(chunking),
            "doc_date": meta.get("doc_date"), "doc_version": meta.get("doc_version"),
            "job_id": job_id, "created_at": now, "updated_at": now,
        }
        self.db.insert("documents", row)
        return self.db.one("SELECT * FROM documents WHERE id = ?", (row["id"],))

    def upload_documents(self, collection_id: str, files: List[Tuple[str, bytes]],
                         metadata: Optional[Dict[str, Any]] = None, chunking: Optional[Dict[str, Any]] = None,
                         jsonl: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        collection = self._collection_row(collection_id)
        if not files:
            raise ValidationError("не передано ни одного файла (поле file)")
        metadata = _check_metadata(metadata)
        try:
            if chunking:
                ChunkingConfig.from_dict(chunking, ChunkingConfig.from_dict(collection["chunking"]))
            file_types = self._file_types(collection).with_jsonl(jsonl)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        for name, data in files:
            if len(data) > self.settings.max_file_bytes:
                raise ValidationError(f"{name}: файл больше {self.settings.max_file_bytes // (1024 * 1024)} МБ")
        docs, to_index = [], []
        for name, data in files:
            safe = self._safe_name(name)
            supported = resolve_kind(safe, file_types) is not None
            doc = self._new_document(
                collection_id, None, "file", f"file:{safe}", safe, safe, None, metadata, chunking,
                status="queued" if supported else "skipped",
                error=None if supported else f"формат {extension_of(safe) or 'без расширения'} не поддерживается",
            )
            self._store_raw(doc, data, safe)
            if jsonl:
                self._set_doc_jsonl(doc["id"], jsonl)
            if supported:
                to_index.append(doc["id"])
            docs.append(doc["id"])
        job_id = None
        if to_index:
            job_id = self._create_job(collection_id, "index", {"document_ids": to_index})
            self.db.execute(f"UPDATE documents SET job_id = ? WHERE id IN ({','.join('?' for _ in to_index)})",
                            (job_id, *to_index))
        return {"job_id": job_id, "documents": [self.get_document(d) for d in docs]}

    def _set_doc_jsonl(self, doc_id: str, jsonl: Dict[str, Any]) -> None:
        doc = self.db.one("SELECT metadata_json FROM documents WHERE id = ?", (doc_id,))
        # Настройки JSONL загруженного файла хранятся в скрытом поле метаданных.
        meta = dict(doc["metadata"] or {})
        meta["_jsonl"] = jsonl
        self.db.update("documents", doc_id, {"metadata_json": dumps(meta)})

    def replace_file(self, document_id: str, filename: str, data: bytes) -> Dict[str, Any]:
        doc = self._doc_row(document_id)
        if doc["source_type"] != "file":
            raise ValidationError("документ из источника обновляется синхронизацией источника")
        if len(data) > self.settings.max_file_bytes:
            raise ValidationError(f"файл больше {self.settings.max_file_bytes // (1024 * 1024)} МБ")
        collection = self._collection_row(doc["collection_id"])
        safe = self._safe_name(filename or doc["filename"])
        if resolve_kind(safe, self._file_types(collection)) is None:
            raise ValidationError(f"формат {extension_of(safe) or 'без расширения'} не поддерживается")
        self._store_raw(doc, data, safe)
        job_id = self._create_job(doc["collection_id"], "index", {"document_ids": [document_id]})
        self.db.update("documents", document_id, {"filename": safe, "uri": safe, "source_key": f"file:{safe}",
                                                  "status": "queued", "job_id": job_id, "updated_at": now_ts()})
        return {"job_id": job_id, "document": self.get_document(document_id)}

    def reindex_document(self, document_id: str, chunking: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        doc = self._doc_row(document_id)
        collection = self._collection_row(doc["collection_id"])
        values: Dict[str, Any] = {}
        if chunking is not None:
            try:
                ChunkingConfig.from_dict(chunking, ChunkingConfig.from_dict(collection["chunking"]))
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc
            values["chunking_json"] = dumps(chunking or None)
        job_id = self._create_job(doc["collection_id"], "index", {"document_ids": [document_id], "force": True})
        values.update({"job_id": job_id, "applied_chunking_json": None, "updated_at": now_ts()})
        self.db.update("documents", document_id, values)
        return {"job_id": job_id, "document": self.get_document(document_id)}

    def _doc_row(self, document_id: str) -> Dict[str, Any]:
        doc = self.db.one("SELECT * FROM documents WHERE id = ?", (document_id,))
        if not doc:
            raise NotFoundError(f"документ {document_id} не найден")
        return doc

    @staticmethod
    def _user_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in (doc["metadata"] or {}).items() if not k.startswith("_")}

    def _doc_metadata(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Метаданные документа: свойства, извлечённые из файла (автор, страницы,
        front matter и т. п.), поверх них — заданные пользователем."""
        return {**(doc.get("properties") or {}), **self._user_metadata(doc)}

    def _chunk_metadata(self, row: Dict[str, Any], doc: Dict[str, Any]) -> Dict[str, Any]:
        """Полные метаданные фрагмента: служебные поля (источник, документ,
        раздел, страница, позиция, разбиение, дата, версия, язык) + метаданные
        документа + собственные поля фрагмента (например, поля записи JSONL)."""
        system = {
            "chunk_id": row["id"], "collection_id": row["collection_id"], "document_id": row["document_id"],
            "source_id": doc["source_id"], "source_type": doc["source_type"], "source": doc["uri"],
            "title": doc["title"] or doc["filename"], "filename": doc["filename"], "section": row["section"] or None,
            "page": row["page"], "char_start": row["char_start"], "char_end": row["char_end"],
            "tokens": row["tokens"], "chunking_method": row["chunking_method"], "origin": row["origin"],
            "doc_type": doc["doc_type"], "code_language": doc["code_language"], "doc_date": doc["doc_date"],
            "doc_version": doc["doc_version"], "language": doc["language"], "content_hash": row["text_hash"],
        }
        system = {k: v for k, v in system.items() if v is not None}
        extra = {k: v for k, v in {**self._doc_metadata(doc), **(row["metadata"] or {})}.items() if k not in system}
        return {**system, **extra}

    def _document_out(self, doc: Dict[str, Any], collection_name: Optional[str] = None) -> Dict[str, Any]:
        meta = self._doc_metadata(doc)
        out = {
            "id": doc["id"], "collection_id": doc["collection_id"], "source_id": doc["source_id"],
            "source_type": doc["source_type"], "source_type_title": SOURCE_TYPE_TITLES.get(doc["source_type"]),
            "uri": doc["uri"], "filename": doc["filename"], "title": doc["title"] or doc["filename"],
            "doc_type": doc["doc_type"], "code_language": doc["code_language"], "status": doc["status"],
            "status_title": DOC_STATUS_TITLES.get(doc["status"], doc["status"]), "error": doc["error"],
            "warnings": doc["warnings"] or [], "size_bytes": doc["size_bytes"], "chunk_count": doc["chunk_count"],
            "char_count": doc["char_count"], "language": doc["language"], "doc_date": doc["doc_date"],
            "doc_version": doc["doc_version"], "metadata": meta, "user_metadata": self._user_metadata(doc),
            "properties": doc.get("properties") or {}, "chunking": doc["chunking"],
            "applied_chunking": doc["applied_chunking"], "enabled": bool(doc["enabled"]),
            "content_hash": doc["content_hash"], "job_id": doc["job_id"],
            "has_file": bool(doc["raw_path"]) or doc["source_type"] == "path",
            "created_at": doc["created_at"], "updated_at": doc["updated_at"], "indexed_at": doc["indexed_at"],
        }
        if collection_name is not None:
            out["collection_name"] = collection_name
        return out

    def get_document(self, document_id: str) -> Dict[str, Any]:
        return self._document_out(self._doc_row(document_id))

    def list_documents(self, collection_id: str, status: Optional[str] = None, source_id: Optional[str] = None,
                       doc_type: Optional[str] = None, q: Optional[str] = None, limit: int = 50,
                       offset: int = 0) -> Dict[str, Any]:
        self._collection_row(collection_id)
        where, params = ["collection_id = ?"], [collection_id]
        if status:
            statuses = status.split(",")
            if "processing" in statuses:
                statuses = [s for s in statuses if s != "processing"] + list(IN_PROGRESS)
            where.append(f"status IN ({','.join('?' for _ in statuses)})")
            params += statuses
        if source_id:
            where.append("source_id = ?")
            params.append(source_id)
        if doc_type:
            where.append("doc_type = ?")
            params.append(doc_type)
        if q:
            where.append("(title LIKE ? OR filename LIKE ? OR uri LIKE ?)")
            params += [f"%{q}%"] * 3
        sql_where = " AND ".join(where)
        rows = self.db.all(f"SELECT * FROM documents WHERE {sql_where} ORDER BY created_at DESC, rowid DESC "
                           f"LIMIT ? OFFSET ?", (*params, limit, offset))
        total = self.db.scalar(f"SELECT COUNT(*) FROM documents WHERE {sql_where}", params)
        return {"items": [self._document_out(r) for r in rows], "total": total, "limit": limit, "offset": offset}

    def get_document_content(self, document_id: str) -> Dict[str, Any]:
        doc = self._doc_row(document_id)
        return {"id": doc["id"], "title": doc["title"] or doc["filename"], "status": doc["status"],
                "text": doc["text"] or "", "char_count": doc["char_count"]}

    def document_file(self, document_id: str) -> Tuple[str, str]:
        """(путь к файлу, имя для скачивания)."""
        doc = self._doc_row(document_id)
        if doc["source_type"] == "path":
            path = check_in_roots(doc["uri"], self.settings.roots)
        else:
            path = doc["raw_path"] or ""
        if not path or not os.path.isfile(path):
            raise NotFoundError("исходный файл документа не сохранён")
        return path, doc["filename"]

    def update_document(self, document_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        doc = self._doc_row(document_id)
        meta = dict(doc["metadata"] or {})
        values: Dict[str, Any] = {}
        if patch.get("metadata") is not None:
            new_meta = _check_metadata(patch["metadata"])
            meta = {**{k: v for k, v in meta.items() if k.startswith("_") or k in RESERVED_METADATA}, **new_meta}
        for key in RESERVED_METADATA:
            if key in patch and patch[key] is not None:
                value = str(patch[key]).strip()
                if key == "doc_date" and value and not _DATE_RE.match(value):
                    raise ValidationError("doc_date: дата в формате ГГГГ-ММ-ДД")
                if value:
                    meta[key] = value
                else:
                    meta.pop(key, None)
                if key == "title":
                    values["title"] = value or doc["title"]
                else:
                    values[key] = value or None
        if patch.get("enabled") is not None:
            values["enabled"] = int(bool(patch["enabled"]))
        values["metadata_json"] = dumps(meta)
        values["updated_at"] = now_ts()
        self.db.update("documents", document_id, values)
        self._invalidate(doc["collection_id"])
        return self.get_document(document_id)

    def _delete_document_row(self, doc: Dict[str, Any]) -> None:
        with self.db.connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc["id"],))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc["id"],))
        directory = os.path.join(self.files_dir, doc.get("collection_id") or "", doc["id"])
        if doc.get("collection_id") and os.path.isdir(directory):
            shutil.rmtree(directory, ignore_errors=True)
        elif doc.get("raw_path") and doc["raw_path"].startswith(self.files_dir):
            shutil.rmtree(os.path.dirname(doc["raw_path"]), ignore_errors=True)

    def delete_document(self, document_id: str) -> None:
        doc = self._doc_row(document_id)
        self._delete_document_row(doc)
        self._invalidate(doc["collection_id"])

    # ------------------------------------------------------------------
    # Источники
    # ------------------------------------------------------------------

    def _source_row(self, source_id: str) -> Dict[str, Any]:
        row = self.db.one("SELECT * FROM sources WHERE id = ?", (source_id,))
        if not row:
            raise NotFoundError(f"источник {source_id} не найден")
        return row

    def _source_out(self, row: Dict[str, Any]) -> Dict[str, Any]:
        counts = self.db.one("SELECT COUNT(*) AS n, SUM(status = 'indexed') AS indexed, "
                             "SUM(status = 'failed') AS failed FROM documents WHERE source_id = ?", (row["id"],))
        options = row["options"] or {}
        return {
            "id": row["id"], "collection_id": row["collection_id"], "type": row["type"],
            "type_title": SOURCE_TYPE_TITLES.get(row["type"], row["type"]), "title": row["title"],
            "source": row["spec"], "include": options.get("include") or [], "exclude": options.get("exclude") or [],
            "crawl": options.get("crawl"), "auth": options.get("auth"), "jsonl": options.get("jsonl"),
            "metadata": row["metadata"] or {}, "chunking": row["chunking"],
            "documents": {"total": counts["n"] or 0, "indexed": counts["indexed"] or 0, "failed": counts["failed"] or 0},
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "last_synced_at": row["last_synced_at"], "last_job_id": row["last_job_id"],
        }

    def add_source(self, collection_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        collection = self._collection_row(collection_id)
        try:
            stype, spec, options, title = validate_source(body, self.fetcher, self.settings.roots,
                                                          self.settings.crawl_max_pages, self.settings.crawl_max_depth)
            chunking = body.get("chunking")
            if chunking:
                ChunkingConfig.from_dict(chunking, ChunkingConfig.from_dict(collection["chunking"]))
        except (SourceError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc
        metadata = _check_metadata(body.get("metadata"))
        now = now_ts()
        source_id = new_id("src")
        self.db.insert("sources", {
            "id": source_id, "collection_id": collection_id, "type": stype, "spec_json": dumps(spec),
            "options_json": dumps(options), "metadata_json": dumps(metadata), "chunking_json": dumps(chunking or None),
            "title": title, "created_at": now, "updated_at": now,
        })
        if stype == "text":
            path = self._text_path({"id": source_id, "collection_id": collection_id})
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(body["source"]["content"])
        job_id = self._create_job(collection_id, "sync", {"source_id": source_id})
        self.db.update("sources", source_id, {"last_job_id": job_id})
        return {"source": self.get_source(source_id), "job_id": job_id}

    def get_source(self, source_id: str) -> Dict[str, Any]:
        return self._source_out(self._source_row(source_id))

    def list_sources(self, collection_id: str) -> Dict[str, Any]:
        self._collection_row(collection_id)
        rows = self.db.all("SELECT * FROM sources WHERE collection_id = ? ORDER BY created_at", (collection_id,))
        return {"items": [self._source_out(r) for r in rows], "total": len(rows)}

    def update_source(self, source_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        row = self._source_row(source_id)
        collection = self._collection_row(row["collection_id"])
        options = dict(row["options"] or {})
        values: Dict[str, Any] = {}
        try:
            for key in ("include", "exclude"):
                if patch.get(key) is not None:
                    if not isinstance(patch[key], list) or not all(isinstance(v, str) for v in patch[key]):
                        raise ValidationError(f"{key}: ожидается список строк")
                    options[key] = [v.strip() for v in patch[key] if v.strip()]
            if row["type"] == "url" and (patch.get("crawl") is not None or patch.get("auth") is not None):
                merged = {"source": row["spec"], "crawl": patch.get("crawl", options.get("crawl")),
                          "auth": patch.get("auth", options.get("auth"))}
                _, _, new_options, _ = validate_source(merged, self.fetcher, self.settings.roots,
                                                       self.settings.crawl_max_pages, self.settings.crawl_max_depth)
                options["crawl"] = new_options.get("crawl")
                if "auth" in new_options:
                    options["auth"] = new_options["auth"]
                else:
                    options.pop("auth", None)
            if patch.get("jsonl") is not None:
                from .formats import parse_jsonl_options

                text_fields, meta_fields = parse_jsonl_options(patch["jsonl"])
                options["jsonl"] = {"text_fields": text_fields, "metadata_fields": meta_fields}
            if "chunking" in patch:
                if patch["chunking"]:
                    ChunkingConfig.from_dict(patch["chunking"], ChunkingConfig.from_dict(collection["chunking"]))
                values["chunking_json"] = dumps(patch["chunking"] or None)
            if patch.get("metadata") is not None:
                values["metadata_json"] = dumps(_check_metadata(patch["metadata"]))
            if patch.get("title") is not None and str(patch["title"]).strip():
                values["title"] = str(patch["title"]).strip()
        except (SourceError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc
        values["options_json"] = dumps(options)
        values["updated_at"] = now_ts()
        self.db.update("sources", source_id, values)
        return self.get_source(source_id)

    def delete_source(self, source_id: str) -> None:
        row = self._source_row(source_id)
        self.db.execute("UPDATE jobs SET cancel_requested = 1 WHERE status IN ('queued','running') "
                        "AND kind = 'sync' AND json_extract(target_json, '$.source_id') = ?", (source_id,))
        for doc in self.db.all("SELECT id, collection_id, raw_path FROM documents WHERE source_id = ?", (source_id,)):
            self._delete_document_row(doc)
        self.db.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        if row["type"] == "text":
            try:
                os.remove(self._text_path(row))
            except OSError:
                pass
        self._invalidate(row["collection_id"])

    def sync_source(self, source_id: str) -> Dict[str, Any]:
        row = self._source_row(source_id)
        active = self.db.one("SELECT id FROM jobs WHERE status IN ('queued','running') AND kind = 'sync' "
                             "AND json_extract(target_json, '$.source_id') = ?", (source_id,))
        if active:
            return {"job_id": active["id"], "source": self.get_source(source_id), "already_running": True}
        job_id = self._create_job(row["collection_id"], "sync", {"source_id": source_id})
        self.db.update("sources", source_id, {"last_job_id": job_id})
        return {"job_id": job_id, "source": self.get_source(source_id), "already_running": False}

    # ------------------------------------------------------------------
    # Фрагменты
    # ------------------------------------------------------------------

    def _chunk_out(self, row: Dict[str, Any], doc: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        doc = doc or self._doc_row(row["document_id"])
        return {
            "id": row["id"], "document_id": row["document_id"], "collection_id": row["collection_id"],
            "seq": row["seq"], "origin": row["origin"], "text": row["text"], "section": row["section"],
            "page": row["page"], "char_start": row["char_start"], "char_end": row["char_end"],
            "tokens": row["tokens"], "chunking_method": row["chunking_method"],
            "metadata": self._chunk_metadata(row, doc), "chunk_metadata": row["metadata"] or {},
            "enabled": bool(row["enabled"]), "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    def list_chunks(self, document_id: str, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        doc = self._doc_row(document_id)
        rows = self.db.all("SELECT * FROM chunks WHERE document_id = ? ORDER BY origin = 'manual', seq "
                           "LIMIT ? OFFSET ?", (document_id, limit, offset))
        total = self.db.scalar("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (document_id,))
        return {"items": [self._chunk_out(r, doc) for r in rows], "total": total, "limit": limit, "offset": offset}

    def _chunk_row(self, chunk_id: str) -> Dict[str, Any]:
        row = self.db.one("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        if not row:
            raise NotFoundError(f"фрагмент {chunk_id} не найден")
        return row

    def get_chunk(self, chunk_id: str) -> Dict[str, Any]:
        row = self._chunk_row(chunk_id)
        doc = self._doc_row(row["document_id"])
        out = self._chunk_out(row, doc)
        out["document"] = {"id": doc["id"], "title": doc["title"] or doc["filename"], "uri": doc["uri"],
                           "source_type": doc["source_type"], "doc_date": doc["doc_date"],
                           "doc_version": doc["doc_version"]}
        return out

    def _embed_chunk(self, doc: Dict[str, Any], text: str, section: str) -> Tuple[str, bytes]:
        collection = self._collection_row(doc["collection_id"])
        title = doc["title"] or doc["filename"]
        full = f"{title}\n{section}\n\n{text}" if section else f"{title}\n\n{text}"
        try:
            vector = self.registry.embed_documents(collection["embedding_model"], [full])[0]
        except EmbeddingError as exc:
            raise ValidationError(f"не удалось посчитать эмбеддинг: {exc}") from exc
        return collection["embedding_model"], vector.tobytes()

    def update_chunk(self, chunk_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        row = self._chunk_row(chunk_id)
        values: Dict[str, Any] = {}
        if patch.get("text") is not None:
            text = str(patch["text"]).strip()
            if not text:
                raise ValidationError("text: пустой фрагмент")
            doc = self._doc_row(row["document_id"])
            model, vector = self._embed_chunk(doc, text, row["section"])
            values.update({"text": text, "tokens": estimate_tokens(text), "text_hash": normalized_hash(text),
                           "embedding_model": model, "embedding": vector,
                           "origin": "manual" if row["origin"] == "manual" else "edited"})
        if patch.get("enabled") is not None:
            values["enabled"] = int(bool(patch["enabled"]))
        if values:
            values["updated_at"] = now_ts()
            self.db.update("chunks", chunk_id, values)
            self._invalidate(row["collection_id"])
        return self.get_chunk(chunk_id)

    def add_chunk(self, document_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        doc = self._doc_row(document_id)
        text = str(body.get("text") or "").strip()
        if not text:
            raise ValidationError("text: пустой фрагмент")
        section = str(body.get("section") or "").strip()
        metadata = _check_metadata(body.get("metadata"))
        model, vector = self._embed_chunk(doc, text, section)
        seq = (self.db.scalar("SELECT COALESCE(MAX(seq), 0) FROM chunks WHERE document_id = ? AND origin = 'manual'",
                              (document_id,)) or 0) + 1
        now = now_ts()
        chunk_id = f"{document_id}-m{seq}"
        self.db.insert("chunks", {
            "id": chunk_id, "document_id": document_id, "collection_id": doc["collection_id"], "seq": seq,
            "origin": "manual", "text": text, "section": section, "page": body.get("page"), "char_start": None,
            "char_end": None, "tokens": estimate_tokens(text), "chunking_method": None,
            "metadata_json": dumps(metadata), "enabled": 1, "text_hash": normalized_hash(text),
            "embedding_model": model, "embedding": vector, "created_at": now, "updated_at": now,
        })
        self.db.execute("UPDATE documents SET chunk_count = chunk_count + 1 WHERE id = ?", (document_id,))
        self._invalidate(doc["collection_id"])
        return self.get_chunk(chunk_id)

    def delete_chunk(self, chunk_id: str) -> None:
        row = self._chunk_row(chunk_id)
        self.db.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
        self.db.execute("UPDATE documents SET chunk_count = MAX(0, chunk_count - 1) WHERE id = ?",
                        (row["document_id"],))
        self._invalidate(row["collection_id"])

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------

    def _invalidate(self, collection_id: str) -> None:
        with self._index_lock:
            self._index.pop(collection_id, None)

    def _get_index(self, collection: Dict[str, Any]) -> _Index:
        with self._index_lock:
            cached = self._index.get(collection["id"])
            if cached and cached.model == collection["embedding_model"]:
                return cached
        rows = self.db.all(
            """SELECT c.id, c.embedding FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE c.collection_id = ? AND c.enabled = 1 AND d.enabled = 1 AND c.embedding IS NOT NULL
                 AND c.embedding_model = ? ORDER BY c.rowid""",
            (collection["id"], collection["embedding_model"]))
        if rows:
            matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        else:
            matrix = np.zeros((0, 0), dtype=np.float32)
        index = _Index(collection["embedding_model"], [r["id"] for r in rows], matrix)
        with self._index_lock:
            self._index[collection["id"]] = index
        return index

    def _filter_ids(self, collection_ids: List[str], filters: Dict[str, Any]) -> Optional[set]:
        if not filters:
            return None
        if not isinstance(filters, dict):
            raise ValidationError("filters: ожидается объект")
        where = [f"c.collection_id IN ({','.join('?' for _ in collection_ids)})"]
        params: List[Any] = list(collection_ids)
        for key, column in (("document_ids", "d.id"), ("source_ids", "d.source_id"), ("doc_types", "d.doc_type"),
                            ("source_types", "d.source_type"), ("languages", "d.language")):
            values = filters.get(key)
            if values:
                if not isinstance(values, list):
                    raise ValidationError(f"filters.{key}: ожидается список")
                where.append(f"{column} IN ({','.join('?' for _ in values)})")
                params += values
        date = filters.get("doc_date")
        if date:
            if not isinstance(date, dict) or set(date) - {"gte", "lte"}:
                raise ValidationError("filters.doc_date: ожидается {\"gte\": ..., \"lte\": ...}")
            if date.get("gte"):
                where.append("d.doc_date >= ?")
                params.append(str(date["gte"]))
            if date.get("lte"):
                where.append("d.doc_date <= ?")
                params.append(str(date["lte"]))
        metadata = filters.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValidationError("filters.metadata: ожидается объект")
        for key, value in metadata.items():
            if not re.fullmatch(r"[\w.-]+", key):
                raise ValidationError(f"filters.metadata: недопустимый ключ {key!r}")
            expr = (f"COALESCE(json_extract(c.metadata_json, '$.\"{key}\"'), "
                    f"json_extract(d.metadata_json, '$.\"{key}\"'), "
                    f"json_extract(d.properties_json, '$.\"{key}\"'))")
            values = value if isinstance(value, list) else [value]
            where.append(f"{expr} IN ({','.join('?' for _ in values)})")
            params += values
        unknown = set(filters) - {"document_ids", "source_ids", "doc_types", "source_types", "languages",
                                  "doc_date", "metadata"}
        if unknown:
            raise ValidationError(f"filters: неизвестные поля {sorted(unknown)}")
        rows = self.db.all(f"SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id "
                           f"WHERE {' AND '.join(where)}", params)
        return {r["id"] for r in rows}

    def retrieve(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Двухэтапный поиск.

        Этап 1: векторный поиск → `candidate_k` лучших кандидатов → порог
        сходства `score_threshold` → схлопывание почти одинаковых фрагментов.
        Этап 2: реранкинг (`rerank`: none | heuristic | model) → порог
        `rerank_threshold` → `top_k`. В ответе — счётчики этапов (`stages`)
        и оба балла у каждого фрагмента."""
        query = str(body.get("query") or "").strip()
        if not query:
            raise ValidationError("query: пустой запрос")
        collection_ids = body.get("collection_ids") or []
        if not isinstance(collection_ids, list) or not collection_ids:
            raise ValidationError("collection_ids: укажите хотя бы одну коллекцию")
        top_k = int(body.get("top_k") or 5)
        if not 1 <= top_k <= 50:
            raise ValidationError("top_k: от 1 до 50")
        candidate_k = int(body.get("candidate_k") or max(20, top_k))
        if not 1 <= candidate_k <= 200:
            raise ValidationError("candidate_k: от 1 до 200")
        if candidate_k < top_k:
            raise ValidationError("candidate_k (кандидатов до фильтрации) должно быть не меньше top_k")
        threshold = body.get("score_threshold")
        threshold = float(threshold) if threshold is not None else None
        method = body.get("rerank") or "none"
        if method not in RERANK_METHODS:
            raise ValidationError(f"rerank: ожидается одно из {sorted(RERANK_METHODS)}")
        rerank_threshold = body.get("rerank_threshold")
        rerank_threshold = float(rerank_threshold) if rerank_threshold is not None else None
        rerank_model = body.get("rerank_model") or None
        # Удалённые (или пересозданные) коллекции не роняют весь поиск: ищем по
        # оставшимся и сообщаем, каких нет; 404 — только если не нашлось ни одной.
        collections: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []
        for cid in dict.fromkeys(collection_ids):
            row = self.db.one("SELECT * FROM collections WHERE id = ?", (cid,))
            if row:
                collections[cid] = row
            else:
                missing.append(cid)
        if not collections:
            raise NotFoundError(
                f"базы знаний не найдены: {', '.join(missing)} — возможно, они удалены или сервис пересоздал "
                "базу; выберите базы знаний в настройках заново"
            )
        allowed = self._filter_ids(list(collections), body.get("filters") or {})

        # ---- Этап 1: векторный поиск ----
        query_vectors: Dict[str, np.ndarray] = {}
        found: List[Tuple[float, str, np.ndarray]] = []
        for collection in collections.values():
            index = self._get_index(collection)
            if not index.ids:
                continue
            model = collection["embedding_model"]
            if model not in query_vectors:
                try:
                    query_vectors[model] = self.registry.embed_query(model, query)
                except EmbeddingError as exc:
                    raise EmbeddingError(f"эмбеддинг запроса ({model}): {exc}") from exc
            q = query_vectors[model]
            if index.matrix.shape[1] != q.shape[0]:
                continue
            scores = index.matrix @ q
            if allowed is not None:
                mask = np.array([cid in allowed for cid in index.ids], dtype=bool)
                scores = np.where(mask, scores, -np.inf)
            order = np.argsort(-scores)[:candidate_k]
            for i in order:
                if np.isfinite(scores[i]):
                    found.append((float(scores[i]), index.ids[i], index.matrix[i]))
        found.sort(key=lambda c: -c[0])
        candidates = found[:candidate_k]
        stages: Dict[str, Any] = {"candidates": len(candidates)}
        if threshold is not None:
            candidates = [c for c in candidates if c[0] >= threshold]
        stages["after_threshold"] = len(candidates)

        items: List[Dict[str, Any]] = []
        picked_vectors: List[np.ndarray] = []
        picked_hashes = set()
        duplicates = 0
        for score, chunk_id, vector in candidates:
            row = self.db.one("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
            if not row:
                continue
            if row["text_hash"] in picked_hashes or any(float(vector @ v) > 0.97 for v in picked_vectors):
                duplicates += 1
                continue
            picked_hashes.add(row["text_hash"])
            picked_vectors.append(vector)
            doc = self.db.one("SELECT * FROM documents WHERE id = ?", (row["document_id"],))
            items.append({"row": row, "doc": doc, "vector_score": round(score, 4), "rerank_score": None})
        stages["after_dedup"] = len(items)

        # ---- Этап 2: реранкинг ----
        rerank_info: Dict[str, Any] = {"method": method, "model": None, "fallback": False, "error": None,
                                       "elapsed_ms": None}
        if method != "none" and items:
            texts = [self._rerank_text(it["row"], it["doc"]) for it in items]
            scores_2: Optional[List[float]] = None
            started = time.monotonic()
            if method == "model":
                limit = self.rerank_max_chars
                model_texts = [t if len(t) <= limit else t[:limit] for t in texts] if limit > 0 else texts
                try:
                    rerank_info["model"], scores_2 = self.reranker.rerank(rerank_model, query, model_texts)
                except RerankError as exc:
                    log.warning("реранкинг моделью не удался, применена эвристика: %s", exc)
                    rerank_info.update(fallback=True, error=str(exc))
            if scores_2 is None:
                scores_2 = heuristic_scores(query, texts, [it["vector_score"] for it in items])
            rerank_info["elapsed_ms"] = int((time.monotonic() - started) * 1000)
            for it, value in zip(items, scores_2):
                it["rerank_score"] = value
            items.sort(key=lambda it: -it["rerank_score"])
            if rerank_threshold is not None:
                items = [it for it in items if it["rerank_score"] >= rerank_threshold]
        stages["after_rerank"] = len(items)
        items = items[:top_k]
        stages["returned"] = len(items)

        results: List[Dict[str, Any]] = []
        for it in items:
            row, doc = it["row"], it["doc"]
            collection = collections[row["collection_id"]]
            final = it["rerank_score"] if it["rerank_score"] is not None else it["vector_score"]
            results.append({
                "chunk_id": row["id"], "score": final, "vector_score": it["vector_score"],
                "rerank_score": it["rerank_score"], "text": row["text"], "section": row["section"],
                "page": row["page"], "seq": row["seq"], "tokens": row["tokens"],
                "metadata": self._chunk_metadata(row, doc),
                "document": {
                    "id": doc["id"], "title": doc["title"] or doc["filename"], "source": doc["uri"],
                    "source_type": doc["source_type"], "source_id": doc["source_id"], "filename": doc["filename"],
                    "doc_type": doc["doc_type"], "doc_date": doc["doc_date"], "doc_version": doc["doc_version"],
                    "metadata": self._doc_metadata(doc),
                },
                "collection": {"id": collection["id"], "name": collection["name"]},
            })
        return {"query": query, "results": results, "stages": stages, "rerank": rerank_info,
                "duplicates_skipped": duplicates, "searched_collections": len(collections),
                "missing_collections": missing}

    @staticmethod
    def _rerank_text(row: Dict[str, Any], doc: Dict[str, Any]) -> str:
        """Текст фрагмента для реранкинга: название документа и раздел дают
        контекст коротким фрагментам."""
        head = " › ".join(p for p in (doc["title"] or doc["filename"], row["section"]) if p)
        return f"{head}\n{row['text']}" if head else row["text"]

    def rerank_models(self) -> Dict[str, Any]:
        return {"items": self.reranker.models(), "default": self.reranker.default_id()}

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        return {
            "source_types": [
                {"type": "file", "title": "Файлы", "enabled": True},
                {"type": "path", "title": SOURCE_TYPES["path"], "enabled": bool(self.settings.roots)},
                {"type": "url", "title": SOURCE_TYPES["url"], "enabled": True},
                {"type": "text", "title": SOURCE_TYPES["text"], "enabled": True},
            ] + [{"type": t, "title": title, "enabled": False} for t, title in PLANNED_SOURCE_TYPES.items()],
            "formats": supported_formats(),
            "chunking_methods": [{"id": k, "title": v} for k, v in METHODS.items()],
            "rerank_methods": [
                {"id": k, "title": v, "enabled": k != "model" or self.reranker.available}
                for k, v in RERANK_METHODS.items()
            ],
            "rerank_models": self.reranker.models(),
            "default_chunking": ChunkingConfig().to_dict(),
            "default_file_types": FileTypes().to_dict(),
            "default_embedding_model": self.settings.default_embedding_model,
            "roots": list(self.settings.roots),
            "secrets": [{"name": name, "hosts": hosts} for name, hosts in self.settings.secrets.items()],
            "limits": {
                "max_file_mb": self.settings.max_file_bytes // (1024 * 1024),
                "max_source_documents": self.settings.max_source_documents,
                "crawl_max_pages": self.settings.crawl_max_pages, "crawl_max_depth": self.settings.crawl_max_depth,
                "top_k_max": 50, "candidate_k_max": 200,
            },
        }

    def health(self) -> Dict[str, Any]:
        providers = {}
        for name, provider in self.registry.providers.items():
            ping = getattr(provider, "ping", None)
            if ping is None:
                providers[name] = {"reachable": None, "error": None}
                continue
            try:
                ping()
                providers[name] = {"reachable": True, "error": None}
            except Exception as exc:  # noqa: BLE001
                providers[name] = {"reachable": False, "error": str(exc)}
        return {
            "status": "ok", "embedding_providers": providers,
            "collections": self.db.scalar("SELECT COUNT(*) FROM collections"),
            "jobs_active": self.db.scalar("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')"),
        }

    def embedding_models(self) -> Dict[str, Any]:
        models, errors = self.registry.list_models()
        default = self.settings.default_embedding_model
        if not any(m["id"] == default for m in models):
            provider = default.split("/", 1)[0]
            models.insert(0, {"id": default, "provider": provider, "model": default.split("/", 1)[-1],
                              "embedding": True, "installed": False})
        for m in models:
            m.setdefault("installed", True)
            m["default"] = m["id"] == default
        models.sort(key=lambda m: (not m["default"], not m["embedding"], m["id"]))
        return {"items": models, "default": default, "errors": errors}

    def browse(self, path: Optional[str], collection_id: Optional[str] = None) -> Dict[str, Any]:
        roots = self.settings.roots
        if not roots:
            raise ValidationError("источники «путь» выключены: не задан KB_ROOTS")
        file_types = self._file_types(self._collection_row(collection_id)) if collection_id else FileTypes()
        if not path:
            return {"path": None, "parent": None, "items": [
                {"name": r, "path": os.path.realpath(r), "type": "dir", "size": None, "supported": None}
                for r in roots]}
        try:
            real = check_in_roots(path, roots)
        except SourceError as exc:
            raise ValidationError(str(exc)) from exc
        if not os.path.isdir(real):
            raise NotFoundError(f"каталог {path} не найден")
        items = []
        for entry in sorted(os.scandir(real), key=lambda e: (not e.is_dir(), e.name.lower()))[:1000]:
            if entry.name.startswith("."):
                continue
            is_dir = entry.is_dir()
            items.append({
                "name": entry.name, "path": os.path.join(real, entry.name), "type": "dir" if is_dir else "file",
                "size": None if is_dir else entry.stat().st_size,
                "supported": None if is_dir else resolve_kind(entry.name, file_types) is not None,
            })
        parent = os.path.dirname(real)
        try:
            check_in_roots(parent, roots)
        except SourceError:
            parent = None
        return {"path": real, "parent": parent if parent != real else None, "items": items}
