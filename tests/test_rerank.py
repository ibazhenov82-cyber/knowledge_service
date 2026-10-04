"""Второй этап поиска: эвристика, модель-реранкер, пороги и топ-K до/после."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import httpx

from knowledge_service.rerank import (
    RerankError, RerankProvider, RerankRegistry, bm25_scores, heuristic_scores, load_rerank_providers,
)
from knowledge_service.service import ValidationError
from tests.fakes import ServiceFixture


def provider(handler, api="jina", models=("bge-reranker-v2-m3",)):
    return RerankProvider(name="local", base_url="http://rr:8081/v1", models=list(models), api=api,
                          client=httpx.Client(transport=httpx.MockTransport(handler)))


class HeuristicTests(unittest.TestCase):
    def test_bm25_prefers_documents_with_query_terms(self):
        docs = ["Сроки ревью кода — два рабочих дня.", "Отпуск составляет двадцать восемь дней.",
                "Ревьюер назначается автоматически."]
        scores = bm25_scores("какие сроки ревью", docs)
        self.assertEqual(max(range(3), key=scores.__getitem__), 0)
        self.assertEqual(scores[1], 0.0)

    def test_heuristic_combines_vector_and_lexical(self):
        docs = ["про отпуск и выходные", "сроки ревью кода"]
        # По вектору первый чуть выше, но слов вопроса в нём нет.
        scores = heuristic_scores("сроки ревью", docs, [0.62, 0.58])
        self.assertGreater(scores[1], scores[0])
        self.assertTrue(all(0 <= s <= 1 for s in scores))

    def test_stopwords_only_query(self):
        self.assertEqual(bm25_scores("что это", ["текст"]), [0.0])


class ProviderTests(unittest.TestCase):
    def test_jina_format_and_sigmoid_for_logits(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"results": [
                {"index": 1, "relevance_score": 3.2}, {"index": 0, "relevance_score": -2.0}]})

        scores = provider(handler).rerank("bge-reranker-v2-m3", "q", ["a", "b"])
        self.assertEqual(seen[0]["documents"], ["a", "b"])
        self.assertEqual(seen[0]["model"], "bge-reranker-v2-m3")
        self.assertLess(scores[0], 0.2)
        self.assertGreater(scores[1], 0.9)

    def test_tei_format(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body["texts"], ["a", "b"])
            return httpx.Response(200, json=[{"index": 0, "score": 0.1}, {"index": 1, "score": 0.8}])

        self.assertEqual(provider(handler, api="tei").rerank("m", "q", ["a", "b"]), [0.1, 0.8])

    def test_errors(self):
        with self.assertRaises(RerankError):
            provider(lambda r: httpx.Response(500, text="boom")).rerank("m", "q", ["a"])
        with self.assertRaises(RerankError):
            provider(lambda r: httpx.Response(200, json={"results": []})).rerank("m", "q", ["a"])

    def test_registry_and_file(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump({"local": {"base_url": "http://x/v1", "models": ["m1", "m2"], "title": "llama.cpp"},
                       "jina": {"base_url": "https://api.jina.ai/v1", "api_key": "${KB_T_JINA}", "models": ["j"]}}, fh)
        try:
            os.environ["KB_T_JINA"] = "k"
            providers = load_rerank_providers(path)
        finally:
            os.environ.pop("KB_T_JINA", None)
            os.unlink(path)
        registry = RerankRegistry()
        for p in providers:
            registry.add(p)
        self.assertEqual(registry.default_id(), "local/m1")
        self.assertEqual([m["id"] for m in registry.models()], ["local/m1", "local/m2", "jina/j"])
        self.assertEqual(registry.resolve("jina/j")[1], "j")
        with self.assertRaises(RerankError):
            registry.resolve("local/nope")
        self.assertFalse(RerankRegistry().available)

    def test_probe(self):
        registry = RerankRegistry()
        registry.add(provider(lambda r: httpx.Response(200, json={"results": [
            {"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.1}]})))
        registry.add(RerankProvider(name="down", base_url="http://x/v1", models=["m"],
                                    client=httpx.Client(transport=httpx.MockTransport(
                                        lambda r: httpx.Response(503, text="loading")))))
        (ok_id, seconds, error), (down_id, none, down_error) = registry.probe()
        self.assertEqual((ok_id, error), ("local/bge-reranker-v2-m3", None))
        self.assertGreaterEqual(seconds, 0)
        self.assertEqual((down_id, none), ("down/m", None))
        self.assertIn("503", down_error)


DOCS = {
    "review.md": "# Ревью\n\n## Сроки ревью\n\nРевью кода выполняется не позднее двух рабочих дней после запроса.\n",
    "deploy.md": "# Выкладка\n\n## Окна выкладки\n\nВыкладка в продакшен по вторникам и четвергам после ревью кода.\n",
    "vacation.md": "# Отпуска\n\n## Длительность\n\nЕжегодный отпуск — двадцать восемь календарных дней.\n",
    "oncall.md": "# Дежурства\n\n## Смена\n\nДежурный меняется по понедельникам, передача смены письменно.\n",
}


class RetrievalStagesTests(unittest.TestCase):
    def setUp(self):
        self.fx = ServiceFixture()
        self.svc = self.fx.service
        self.col = self.svc.create_collection({"name": "Регламенты", "chunking": {"min_chunk_tokens": 5}})
        result = self.svc.upload_documents(self.col["id"], [(n, t.encode()) for n, t in DOCS.items()])
        self.fx.wait(result["job_id"])

    def tearDown(self):
        self.fx.close()

    def search(self, **kw):
        return self.svc.retrieve({"query": "сроки ревью кода", "collection_ids": [self.col["id"]], **kw})

    def test_stages_without_rerank(self):
        found = self.search(top_k=2, candidate_k=3, score_threshold=0.05)
        stages = found["stages"]
        self.assertEqual(stages["candidates"], 3)
        self.assertLessEqual(stages["after_threshold"], 3)
        self.assertEqual(stages["returned"], 2)
        self.assertEqual(found["rerank"]["method"], "none")
        top = found["results"][0]
        self.assertIsNone(top["rerank_score"])
        self.assertEqual(top["score"], top["vector_score"])

    def test_heuristic_rerank_and_threshold(self):
        found = self.search(top_k=5, candidate_k=8, rerank="heuristic", rerank_threshold=0.5)
        self.assertEqual(found["rerank"]["method"], "heuristic")
        self.assertIn("двух рабочих дней", found["results"][0]["text"])
        scores = [r["rerank_score"] for r in found["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(s >= 0.5 for s in scores))
        self.assertLess(found["stages"]["after_rerank"], found["stages"]["after_dedup"])
        self.assertEqual(found["results"][0]["score"], found["results"][0]["rerank_score"])

    def test_model_rerank_and_fallback(self):
        calls = []

        def handler(request):
            body = json.loads(request.content)
            calls.append(body)
            # «Модель» ставит выше всего фрагмент про дежурства.
            return httpx.Response(200, json={"results": [
                {"index": i, "relevance_score": 0.95 if "Дежурный" in d else 0.1}
                for i, d in enumerate(body["documents"])]})

        self.svc.reranker.add(provider(handler))
        found = self.search(top_k=3, candidate_k=8, rerank="model")
        elapsed = found["rerank"].pop("elapsed_ms")
        self.assertIsInstance(elapsed, int)
        self.assertEqual(found["rerank"], {"method": "model", "model": "local/bge-reranker-v2-m3",
                                           "fallback": False, "error": None})
        self.assertIn("Дежурный", found["results"][0]["text"])
        self.assertTrue(calls[0]["documents"][0].startswith(("Ревью", "Выкладка", "Отпуска", "Дежурства")))
        self.assertEqual(self.svc.rerank_models()["default"], "local/bge-reranker-v2-m3")

        self.svc.reranker.providers["local"] = provider(lambda r: httpx.Response(503, text="down"))
        found = self.search(top_k=3, rerank="model")
        self.assertTrue(found["rerank"]["fallback"])
        self.assertIn("503", found["rerank"]["error"])
        self.assertIn("двух рабочих дней", found["results"][0]["text"])

    def test_model_timeout_falls_back_and_texts_are_truncated(self):
        sent = []

        def handler(request):
            sent.append(json.loads(request.content)["documents"])
            raise httpx.ReadTimeout("timed out", request=request)

        self.svc.reranker.add(provider(handler))
        self.svc.rerank_max_chars = 40
        found = self.search(top_k=3, candidate_k=8, rerank="model")
        self.assertTrue(found["rerank"]["fallback"])
        self.assertIn("не ответил за 60 с", found["rerank"]["error"])
        self.assertTrue(all(len(d) <= 40 for d in sent[0]))
        self.assertIn("двух рабочих дней", found["results"][0]["text"])

    def test_model_rerank_without_models_falls_back(self):
        found = self.search(rerank="model")
        self.assertTrue(found["rerank"]["fallback"])
        self.assertIn("не настроена", found["rerank"]["error"])

    def test_missing_collection_does_not_break_search(self):
        found = self.svc.retrieve({"query": "сроки ревью кода", "collection_ids": ["col_deleted", self.col["id"]]})
        self.assertEqual(found["missing_collections"], ["col_deleted"])
        self.assertIn("двух рабочих дней", found["results"][0]["text"])
        from knowledge_service.service import NotFoundError
        with self.assertRaises(NotFoundError) as ctx:
            self.svc.retrieve({"query": "сроки", "collection_ids": ["col_deleted"]})
        self.assertIn("выберите базы знаний в настройках заново", str(ctx.exception))

    def test_validation(self):
        with self.assertRaises(ValidationError):
            self.search(top_k=10, candidate_k=5)
        with self.assertRaises(ValidationError):
            self.search(rerank="llm")
        with self.assertRaises(ValidationError):
            self.search(candidate_k=500)

    def test_info_lists_rerank_methods(self):
        info = self.svc.info()
        self.assertEqual({m["id"]: m["enabled"] for m in info["rerank_methods"]},
                         {"none": True, "heuristic": True, "model": False})


if __name__ == "__main__":
    unittest.main()
