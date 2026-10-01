"""Общие фейки для тестов сервиса баз знаний."""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from typing import Dict, List, Optional

import httpx
import numpy as np

from knowledge_service.config import Settings
from knowledge_service.embeddings import EmbeddingError, EmbeddingRegistry, Provider
from knowledge_service.fetch import UrlFetcher
from knowledge_service.service import KnowledgeService

DIMS = 256


def _stem(word: str) -> str:
    # Грубая «основа»: первые 5 символов — «ревью»/«ревьюер» совпадают.
    return word[:5]


def bow_vector(text: str) -> List[float]:
    """Детерминированный «эмбеддинг» — мешок слов, захешированный в DIMS
    измерений: тексты с общими словами близки по косинусу."""
    vec = np.zeros(DIMS, dtype=np.float32)
    # Служебные префиксы моделей (Instruct/Query) не должны влиять на сходство.
    text = re.sub(r"^Instruct:.*?\nQuery: ", "", text, flags=re.S)
    for word in re.findall(r"\w+", text.lower()):
        if len(word) < 3:
            continue
        h = int(hashlib.md5(_stem(word).encode()).hexdigest(), 16)
        vec[h % DIMS] += 1.0
    if not vec.any():
        vec[0] = 1.0
    return vec.tolist()


class FakeProvider(Provider):
    def __init__(self, name: str = "ollama", fail: bool = False):
        self.name = name
        self.title = name
        self.fail = fail
        self.calls: List[List[str]] = []
        self.models = ["qwen3-embedding:0.6b", "bge-m3"]

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        if self.fail:
            raise EmbeddingError("Ollama недоступна")
        self.calls.append(list(texts))
        return [bow_vector(t) for t in texts]

    def list_models(self):
        return [{"id": f"{self.name}/{m}", "provider": self.name, "model": m, "embedding": True} for m in self.models]


class ServiceFixture:
    """Сервис во временном каталоге с фейковыми эмбеддингами и HTTP."""

    def __init__(self, pages: Optional[Dict[str, tuple]] = None, roots: Optional[List[str]] = None,
                 workers: int = 1, secrets: Optional[Dict[str, List[str]]] = None, start: bool = True,
                 resolver=None, data_dir: Optional[str] = None):
        self.tmp = tempfile.mkdtemp(prefix="kb-test-")
        self.data_dir = data_dir or f"{self.tmp}/data"
        self.pages = pages or {}
        self.requests: List[httpx.Request] = []
        self.provider = FakeProvider()
        registry = EmbeddingRegistry(batch=4)
        registry.add(self.provider)
        registry.add(FakeProvider("local"))
        self.settings = Settings(data_dir=self.data_dir, roots=roots or [], secrets=secrets or {},
                                 workers=workers, crawl_max_pages=20)
        transport = httpx.MockTransport(self._handle)
        self.fetcher = UrlFetcher([], self.settings.secrets, transport=transport,
                                  resolver=resolver or (lambda host: ["93.184.216.34"]))
        self.service = KnowledgeService(self.settings, registry, self.fetcher)
        if start:
            self.service.start()

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url not in self.pages:
            return httpx.Response(404, text="not found")
        spec = self.pages[url]
        if spec[0] == "redirect":
            return httpx.Response(302, headers={"Location": spec[1]})
        content_type, body = spec[0], spec[1]
        headers = {"Content-Type": content_type}
        if len(spec) > 2:
            headers.update(spec[2])
        return httpx.Response(200, content=body if isinstance(body, bytes) else body.encode("utf-8"), headers=headers)

    def wait(self, job_id: str, timeout: float = 30.0) -> dict:
        return self.service.runner.wait(job_id, timeout)

    def close(self) -> None:
        self.service.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)
