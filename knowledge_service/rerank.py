"""
knowledge_service.rerank
==========================

Второй этап поиска — реранкинг кандидатов после векторного поиска:

- `heuristic` — «Эвристика (без LLM)»: итоговый балл = 0,6 × векторное
  сходство + 0,4 × лексическое совпадение (BM25 по кандидатам, нормированный
  к лучшему кандидату). Поднимает фрагменты, где есть слова вопроса, и
  опускает «похожие по смыслу, но не про то»;
- `model` — «Модель-реранкер» (cross-encoder, например bge-reranker-v2-m3):
  API `/rerank` в формате Jina/Cohere (llama.cpp server `--reranking`,
  Infinity, Jina, Cohere) или TEI. Модели перечислены в файле провайдеров
  `RERANK_PROVIDERS_FILE`; идентификатор — «провайдер/модель».

Баллы обоих способов — в диапазоне 0..1 (сырые логиты cross-encoder
переводятся сигмоидой), поэтому порог после реранкинга один для обоих.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

log = logging.getLogger("knowledge_service.rerank")

METHODS = {"none": "Нет", "heuristic": "Эвристика (без LLM)", "model": "Модель-реранкер"}
HEURISTIC_VECTOR_WEIGHT = 0.6

_WORD = re.compile(r"\w+", re.U)
_STOP = {
    "и", "в", "во", "на", "с", "со", "по", "к", "ко", "о", "об", "от", "до", "из", "за", "для", "не", "ли", "же",
    "а", "но", "или", "что", "как", "это", "то", "там", "тут", "у", "при", "про", "есть", "быть", "мне", "нам",
    "какой", "какая", "какие", "каких", "сколько", "где", "когда", "кто", "чем", "ли",
    "the", "a", "an", "of", "to", "in", "on", "for", "is", "are", "and", "or", "what", "how", "with", "by",
}


class RerankError(Exception):
    pass


def _terms(text: str) -> List[str]:
    """Слова без стоп-слов; грубая «основа» — первые 6 символов (русская
    морфология: «ревью»/«ревьюера», «сроки»/«сроков» совпадают)."""
    return [w[:6] for w in (m.group(0).lower() for m in _WORD.finditer(text)) if len(w) > 1 and w not in _STOP]


def bm25_scores(query: str, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """BM25 запроса по набору кандидатов (статистика — по самим кандидатам)."""
    q_terms = list(dict.fromkeys(_terms(query)))
    docs = [_terms(d) for d in documents]
    if not q_terms or not docs:
        return [0.0] * len(documents)
    n = len(docs)
    avg_len = sum(len(d) for d in docs) / n or 1.0
    df = Counter(t for d in docs for t in set(d))
    scores = []
    for d in docs:
        tf = Counter(d)
        score = 0.0
        for t in q_terms:
            if not tf[t]:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg_len))
        scores.append(score)
    return scores


def heuristic_scores(query: str, documents: Sequence[str], vector_scores: Sequence[float]) -> List[float]:
    lexical = bm25_scores(query, documents)
    best = max(lexical) if lexical else 0.0
    norm = [s / best if best > 0 else 0.0 for s in lexical]
    return [
        round(HEURISTIC_VECTOR_WEIGHT * max(0.0, v) + (1 - HEURISTIC_VECTOR_WEIGHT) * lx, 4)
        for v, lx in zip(vector_scores, norm)
    ]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class RerankProvider:
    name: str
    base_url: str
    models: List[str]
    api_key: str = ""
    title: str = ""
    #: "jina" — POST {base}/rerank {model, query, documents, top_n} → {results: [{index, relevance_score}]}
    #: (llama.cpp, Infinity, Jina, Cohere); "tei" — POST {base}/rerank {query, texts} → [{index, score}].
    api: str = "jina"
    timeout: float = 60.0
    client: Optional[httpx.Client] = None

    def rerank(self, model: str, query: str, documents: List[str]) -> List[float]:
        client = self.client or httpx.Client(timeout=self.timeout)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        if self.api == "tei":
            body: Dict[str, Any] = {"query": query, "texts": documents, "raw_scores": False}
        else:
            body = {"model": model, "query": query, "documents": documents, "top_n": len(documents)}
        try:
            resp = client.post(f"{self.base_url.rstrip('/')}/rerank", json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise RerankError(
                f"реранкер {self.title or self.name} не ответил за {self.timeout:g} с "
                f"({len(documents)} фрагментов) — уменьшите «Кандидатов до фильтрации» или KB_RERANK_MAX_CHARS"
            ) from exc
        except httpx.HTTPError as exc:
            raise RerankError(f"реранкер {self.title or self.name} недоступен: {exc}") from exc
        finally:
            if self.client is None:
                client.close()
        if resp.status_code >= 400:
            raise RerankError(f"реранкер {self.title or self.name}: HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise RerankError("реранкер вернул не JSON") from exc
        items = data if isinstance(data, list) else data.get("results") or data.get("data") or []
        scores: List[Optional[float]] = [None] * len(documents)
        for item in items:
            idx = item.get("index")
            value = item.get("relevance_score", item.get("score"))
            if isinstance(idx, int) and 0 <= idx < len(documents) and isinstance(value, (int, float)):
                scores[idx] = float(value)
        if any(s is None for s in scores):
            raise RerankError("реранкер вернул оценки не для всех фрагментов")
        values = [float(s) for s in scores]  # type: ignore[arg-type]
        # Сырые логиты cross-encoder (llama.cpp, TEI raw) — в 0..1 сигмоидой.
        if any(v < 0 or v > 1 for v in values):
            values = [_sigmoid(v) for v in values]
        return [round(v, 4) for v in values]


_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_rerank_providers(path: str, timeout: float = 60.0) -> List[RerankProvider]:
    """Файл реранкеров:

    {"local": {"title": "llama.cpp", "base_url": "http://localhost:8081/v1",
               "models": ["bge-reranker-v2-m3"], "api": "jina"}}
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError(f"RERANK_PROVIDERS_FILE {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"RERANK_PROVIDERS_FILE {path}: ожидается объект {{имя: описание}}")

    def sub(value: str) -> str:
        def repl(m):
            if m.group(1) not in os.environ:
                raise ValueError(f"RERANK_PROVIDERS_FILE: не задана переменная окружения {m.group(1)}")
            return os.environ[m.group(1)]
        return _ENV_REF.sub(repl, value)

    providers = []
    for name, spec in data.items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise ValueError(f"RERANK_PROVIDERS_FILE: недопустимое имя провайдера {name!r}")
        if not isinstance(spec, dict) or not spec.get("base_url") or not spec.get("models"):
            raise ValueError(f"RERANK_PROVIDERS_FILE: у провайдера {name!r} нужны base_url и models")
        api = str(spec.get("api") or "jina")
        if api not in ("jina", "tei"):
            raise ValueError(f"RERANK_PROVIDERS_FILE: api провайдера {name!r} — jina или tei")
        providers.append(RerankProvider(
            name=name, base_url=sub(str(spec["base_url"])), models=[str(m) for m in spec["models"]],
            api_key=sub(str(spec.get("api_key", ""))), title=str(spec.get("title") or name), api=api,
            timeout=timeout,
        ))
    return providers


@dataclass
class RerankRegistry:
    providers: Dict[str, RerankProvider] = field(default_factory=dict)
    default_model: str = ""

    def add(self, provider: RerankProvider) -> None:
        self.providers[provider.name] = provider

    def models(self) -> List[Dict[str, Any]]:
        items = []
        for provider in self.providers.values():
            for model in provider.models:
                model_id = f"{provider.name}/{model}"
                items.append({"id": model_id, "provider": provider.name, "model": model,
                              "provider_title": provider.title, "default": model_id == self.default_id()})
        return items

    def default_id(self) -> Optional[str]:
        if self.default_model:
            return self.default_model
        for provider in self.providers.values():
            if provider.models:
                return f"{provider.name}/{provider.models[0]}"
        return None

    @property
    def available(self) -> bool:
        return self.default_id() is not None

    def probe(self) -> List[Tuple[str, Optional[float], Optional[str]]]:
        """Проверка при старте: каждой модели — короткий запрос из двух
        фрагментов. Возвращает [(id, секунды или None, ошибка или None)] —
        медленный или зависший реранкер видно сразу, а не по тайм-аутам в чате."""
        import time

        sample = ["Ревью кода выполняется не позднее двух рабочих дней.", "Отпуск — 28 календарных дней."]
        out: List[Tuple[str, Optional[float], Optional[str]]] = []
        for provider in self.providers.values():
            for model in provider.models:
                started = time.monotonic()
                try:
                    provider.rerank(model, "Сколько дней на ревью кода?", sample)
                    out.append((f"{provider.name}/{model}", time.monotonic() - started, None))
                except RerankError as exc:
                    out.append((f"{provider.name}/{model}", None, str(exc)))
        return out

    def resolve(self, model_id: Optional[str]) -> Tuple[RerankProvider, str, str]:
        model_id = model_id or self.default_id()
        if not model_id:
            raise RerankError("модель-реранкер не настроена (RERANK_PROVIDERS_FILE)")
        name, _, model = model_id.partition("/")
        provider = self.providers.get(name)
        if provider is None or model not in provider.models:
            raise RerankError(f"модель-реранкер {model_id!r} не найдена; есть: {[m['id'] for m in self.models()]}")
        return provider, model, model_id

    def rerank(self, model_id: Optional[str], query: str, documents: List[str]) -> Tuple[str, List[float]]:
        provider, model, resolved = self.resolve(model_id)
        return resolved, provider.rerank(model, query, documents)
