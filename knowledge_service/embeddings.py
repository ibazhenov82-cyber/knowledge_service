"""
knowledge_service.embeddings
==============================

Модели эмбеддингов. Идентификатор модели — «<провайдер>/<модель>»:

- `ollama/<модель>` — любая модель Ollama, `POST /api/embed` (по умолчанию
  `ollama/qwen3-embedding:0.6b`);
- `<имя>/<модель>` — OpenAI-совместимый провайдер (`POST {base_url}/embeddings`)
  из файла `EMBEDDING_PROVIDERS_FILE`.

Векторы всегда нормируются по длине (L2): косинусное сходство тогда —
скалярное произведение. Для моделей, которые обучены с разными
префиксами запроса и документа (Qwen3-Embedding, nomic-embed-text, E5),
префиксы добавляются автоматически.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx
import numpy as np

log = logging.getLogger("knowledge_service.embeddings")

QUERY_INSTRUCTION = "Given a question, retrieve passages from the knowledge base that answer the question"


class EmbeddingError(Exception):
    pass


def split_model_id(model_id: str) -> Tuple[str, str]:
    provider, sep, model = (model_id or "").partition("/")
    if not sep or not provider or not model:
        raise EmbeddingError(f"модель эмбеддингов {model_id!r}: ожидается «провайдер/модель», например ollama/bge-m3")
    return provider, model


def query_text(model: str, text: str) -> str:
    name = model.lower()
    if "qwen3-embedding" in name:
        return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}"
    if "nomic-embed" in name:
        return f"search_query: {text}"
    if re.search(r"(^|[/_-])e5", name):
        return f"query: {text}"
    return text


def document_text(model: str, text: str) -> str:
    name = model.lower()
    if "nomic-embed" in name:
        return f"search_document: {text}"
    if re.search(r"(^|[/_-])e5", name):
        return f"passage: {text}"
    return text


def normalize(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise EmbeddingError("провайдер вернул векторы неверной формы")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class Provider:
    name: str
    title: str

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def list_models(self) -> List[Dict[str, Any]]:  # pragma: no cover - интерфейс
        raise NotImplementedError


def _with_retries(fn: Callable[[], Any], what: str, retries: int = 2) -> Any:
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            return fn()
        except (httpx.TransportError, _Retryable) as exc:
            if attempt == retries:
                raise EmbeddingError(f"{what}: {exc}") from exc
            log.warning("%s: %s, повтор через %.0f с", what, exc, delay)
            time.sleep(delay)
            delay *= 2


class _Retryable(Exception):
    pass


def _check(resp: httpx.Response, what: str) -> Any:
    if resp.status_code >= 500 or resp.status_code == 429:
        raise _Retryable(f"HTTP {resp.status_code}")
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("error")
            if isinstance(detail, dict):
                detail = detail.get("message")
        except ValueError:
            detail = resp.text[:300]
        raise EmbeddingError(f"{what}: HTTP {resp.status_code}: {detail}")
    try:
        return resp.json()
    except ValueError as exc:
        raise EmbeddingError(f"{what}: ответ не JSON") from exc


_EMBEDDING_NAME_HINTS = ("embed", "bge", "e5", "gte", "minilm", "mxbai", "arctic", "jina", "paraphrase", "labse")


class OllamaProvider(Provider):
    title = "Ollama"

    def __init__(self, base_url: str, timeout: float = 120.0, client: Optional[httpx.Client] = None):
        self.name = "ollama"
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        def call():
            resp = self.client.post(f"{self.base_url}/api/embed", json={"model": model, "input": texts})
            if resp.status_code == 404:
                raise EmbeddingError(f"модель {model!r} не найдена в Ollama — выполните `ollama pull {model}`")
            return _check(resp, f"Ollama /api/embed ({model})")

        data = _with_retries(call, f"Ollama {self.base_url}")
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingError("Ollama вернула неожиданный ответ /api/embed")
        return vectors

    def list_models(self) -> List[Dict[str, Any]]:
        resp = _with_retries(lambda: self.client.get(f"{self.base_url}/api/tags"), f"Ollama {self.base_url}", retries=0)
        data = _check(resp, "Ollama /api/tags")
        models = []
        for item in data.get("models") or []:
            name = item.get("name") or item.get("model")
            if not name:
                continue
            details = item.get("details") or {}
            families = " ".join(details.get("families") or [details.get("family") or ""]).lower()
            likely = any(h in name.lower() for h in _EMBEDDING_NAME_HINTS) or "bert" in families
            models.append({"id": f"ollama/{name}", "provider": "ollama", "model": name, "embedding": likely})
        return models

    def ping(self) -> None:
        _check(self.client.get(f"{self.base_url}/api/version", timeout=5.0), "Ollama")


class OpenAICompatibleProvider(Provider):
    def __init__(self, name: str, base_url: str, api_key: str = "", models: Optional[List[str]] = None,
                 title: str = "", timeout: float = 120.0, client: Optional[httpx.Client] = None):
        self.name = name
        self.title = title or name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.models = list(models or [])
        self.client = client or httpx.Client(timeout=timeout)

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        def call():
            resp = self.client.post(f"{self.base_url}/embeddings", json={"model": model, "input": texts},
                                    headers=headers)
            return _check(resp, f"{self.title} /embeddings ({model})")

        data = _with_retries(call, self.title)
        items = sorted(data.get("data") or [], key=lambda d: d.get("index", 0))
        vectors = [item.get("embedding") for item in items]
        if len(vectors) != len(texts) or any(not isinstance(v, list) for v in vectors):
            raise EmbeddingError(f"{self.title} вернул неожиданный ответ /embeddings")
        return vectors

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": f"{self.name}/{m}", "provider": self.name, "model": m, "embedding": True} for m in self.models]


_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_providers_file(path: str, timeout: float = 120.0) -> List[Provider]:
    """Файл провайдеров:

    {"openai": {"base_url": "https://api.openai.com/v1", "api_key": "${OPENAI_API_KEY}",
                "models": ["text-embedding-3-small"], "title": "OpenAI"}}
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError(f"EMBEDDING_PROVIDERS_FILE {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"EMBEDDING_PROVIDERS_FILE {path}: ожидается объект {{имя: описание}}")
    providers: List[Provider] = []
    for name, spec in data.items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or name == "ollama":
            raise ValueError(f"EMBEDDING_PROVIDERS_FILE: недопустимое имя провайдера {name!r}")
        if not isinstance(spec, dict) or not spec.get("base_url"):
            raise ValueError(f"EMBEDDING_PROVIDERS_FILE: у провайдера {name!r} нет base_url")

        def sub(value: str) -> str:
            def repl(m):
                if m.group(1) not in os.environ:
                    raise ValueError(f"EMBEDDING_PROVIDERS_FILE: не задана переменная окружения {m.group(1)}")
                return os.environ[m.group(1)]
            return _ENV_REF.sub(repl, value)

        providers.append(OpenAICompatibleProvider(
            name, sub(str(spec["base_url"])), sub(str(spec.get("api_key", ""))),
            [str(m) for m in spec.get("models") or []], str(spec.get("title") or name), timeout=timeout,
        ))
    return providers


@dataclass
class EmbeddingRegistry:
    providers: Dict[str, Provider] = field(default_factory=dict)
    batch: int = 32
    #: Кэш векторов: (модель, sha256 текста) -> вектор. Подключается сервисом
    #: (хранится в базе).
    cache_get: Optional[Callable[[str, List[str]], Dict[str, np.ndarray]]] = None
    cache_put: Optional[Callable[[str, Dict[str, np.ndarray]], None]] = None

    def add(self, provider: Provider) -> None:
        self.providers[provider.name] = provider

    def provider_for(self, model_id: str) -> Tuple[Provider, str]:
        name, model = split_model_id(model_id)
        provider = self.providers.get(name)
        if provider is None:
            raise EmbeddingError(f"провайдер эмбеддингов {name!r} не подключён; есть: {sorted(self.providers)}")
        return provider, model

    def validate(self, model_id: str) -> None:
        self.provider_for(model_id)

    def embed_documents(self, model_id: str, texts: List[str],
                        check_cancel: Optional[Callable[[], None]] = None,
                        on_batch: Optional[Callable[[int], None]] = None) -> np.ndarray:
        from .util import sha256_text

        provider, model = self.provider_for(model_id)
        prepared = [document_text(model, t) for t in texts]
        keys = [sha256_text(t) for t in prepared]
        cached = self.cache_get(model_id, keys) if self.cache_get else {}
        missing = [i for i, k in enumerate(keys) if k not in cached]
        fresh: Dict[str, np.ndarray] = {}
        for pos in range(0, len(missing), self.batch):
            if check_cancel:
                check_cancel()
            idx = missing[pos:pos + self.batch]
            vectors = normalize(provider.embed(model, [prepared[i] for i in idx]))
            for i, vec in zip(idx, vectors):
                fresh[keys[i]] = vec
            if on_batch:
                on_batch(len(idx))
        if fresh and self.cache_put:
            self.cache_put(model_id, fresh)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        all_vectors = {**cached, **fresh}
        return np.vstack([all_vectors[k] for k in keys]).astype(np.float32)

    def embed_query(self, model_id: str, text: str) -> np.ndarray:
        provider, model = self.provider_for(model_id)
        return normalize(provider.embed(model, [query_text(model, text)]))[0]

    def list_models(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        """(модели, ошибки провайдеров)."""
        models, errors = [], []
        for provider in self.providers.values():
            try:
                for item in provider.list_models():
                    models.append({**item, "provider_title": provider.title})
            except EmbeddingError as exc:
                errors.append({"provider": provider.name, "error": str(exc)})
        return models, errors
