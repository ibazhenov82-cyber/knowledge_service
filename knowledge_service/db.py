"""
knowledge_service.db
======================

Хранилище на SQLite (`knowledge.sqlite` в каталоге данных): коллекции,
источники, документы, фрагменты с эмбеддингами (float32), задачи и кэш
эмбеддингов. Схема создаётся с нуля при первом запуске; миграций нет.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from typing import Any, Dict, Iterable, Iterator, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    embedding_model TEXT NOT NULL,
    dims INTEGER,
    chunking_json TEXT NOT NULL,
    file_types_json TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    options_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    chunking_json TEXT,
    title TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_synced_at INTEGER,
    last_job_id TEXT
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    source_id TEXT REFERENCES sources(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    source_key TEXT NOT NULL,
    uri TEXT NOT NULL,
    filename TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    doc_type TEXT,
    code_language TEXT,
    status TEXT NOT NULL,
    error TEXT,
    warnings_json TEXT NOT NULL DEFAULT '[]',
    raw_path TEXT,
    raw_hash TEXT,
    content_hash TEXT,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    char_count INTEGER NOT NULL DEFAULT 0,
    language TEXT,
    doc_date TEXT,
    doc_version TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    properties_json TEXT NOT NULL DEFAULT '{}',
    chunking_json TEXT,
    applied_chunking_json TEXT,
    content_type TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    text TEXT,
    job_id TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    indexed_at INTEGER
);
CREATE INDEX IF NOT EXISTS documents_by_collection ON documents(collection_id, created_at);
CREATE INDEX IF NOT EXISTS documents_by_source ON documents(source_id, source_key);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    origin TEXT NOT NULL DEFAULT 'auto',
    text TEXT NOT NULL,
    section TEXT NOT NULL DEFAULT '',
    page INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    tokens INTEGER NOT NULL,
    chunking_method TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    enabled INTEGER NOT NULL DEFAULT 1,
    text_hash TEXT NOT NULL,
    embedding_model TEXT,
    embedding BLOB,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_document ON chunks(document_id, seq);
CREATE INDEX IF NOT EXISTS chunks_by_collection ON chunks(collection_id);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    collection_id TEXT REFERENCES collections(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT,
    target_json TEXT NOT NULL DEFAULT '{}',
    progress_json TEXT NOT NULL DEFAULT '{}',
    errors_json TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    started_at INTEGER,
    finished_at INTEGER
);
CREATE INDEX IF NOT EXISTS jobs_by_collection ON jobs(collection_id, created_at);
CREATE TABLE IF NOT EXISTS embedding_cache (
    model TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (model, text_hash)
);
"""

JSON_FIELDS = {
    "chunking_json", "file_types_json", "spec_json", "options_json", "metadata_json", "warnings_json",
    "applied_chunking_json", "target_json", "progress_json", "errors_json", "properties_json",
}


def dumps(value: Any) -> Optional[str]:
    return None if value is None else json.dumps(value, ensure_ascii=False)


def row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    result: Dict[str, Any] = {}
    for key in row.keys():
        value = row[key]
        if key in JSON_FIELDS:
            result[key[:-5]] = json.loads(value) if value is not None else None
        else:
            result[key] = value
    return result


class Database:
    def __init__(self, path: str):
        self.path = path
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)

    @contextlib.contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def one(self, sql: str, params: Iterable[Any] = ()) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            return row_to_dict(conn.execute(sql, tuple(params)).fetchone())

    def all(self, sql: str, params: Iterable[Any] = ()) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            return [row_to_dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]

    def scalar(self, sql: str, params: Iterable[Any] = ()) -> Any:
        with self.connect() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
            return row[0] if row else None

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self.connect() as conn:
            return conn.execute(sql, tuple(params)).rowcount

    def insert(self, table: str, values: Dict[str, Any]) -> None:
        cols = list(values)
        with self.connect() as conn:
            conn.execute(
                f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                tuple(values[c] for c in cols),
            )

    def update(self, table: str, row_id: str, values: Dict[str, Any]) -> int:
        if not values:
            return 0
        cols = list(values)
        with self.connect() as conn:
            return conn.execute(
                f"UPDATE {table} SET {', '.join(f'{c} = ?' for c in cols)} WHERE id = ?",
                tuple(values[c] for c in cols) + (row_id,),
            ).rowcount
