"""Провайдеры эмбеддингов и загрузка по URL (без сети: httpx.MockTransport)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import httpx
import numpy as np

from knowledge_service.config import parse_secrets
from knowledge_service.embeddings import (
    EmbeddingError, EmbeddingRegistry, OllamaProvider, OpenAICompatibleProvider, document_text, load_providers_file,
    query_text, split_model_id,
)
from knowledge_service.fetch import FetchError, UrlFetcher, filename_for


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


class EmbeddingTests(unittest.TestCase):
    def test_ollama_embed_and_models(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content or b"{}"))
            if request.url.path == "/api/embed":
                return httpx.Response(200, json={"embeddings": [[3.0, 4.0] for _ in seen[-1]["input"]]})
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [
                    {"name": "qwen3-embedding:0.6b", "details": {"family": "qwen3"}},
                    {"name": "qwen3:8b", "details": {"family": "qwen3"}},
                    {"name": "custom", "details": {"families": ["nomic-bert"]}}]})
            return httpx.Response(404)

        provider = OllamaProvider("http://ollama:11434", client=client(handler))
        registry = EmbeddingRegistry(batch=2)
        registry.add(provider)
        vectors = registry.embed_documents("ollama/qwen3-embedding:0.6b", ["a", "b", "c"])
        self.assertEqual(vectors.shape, (3, 2))
        np.testing.assert_allclose(vectors[0], [0.6, 0.8], rtol=1e-6)
        self.assertEqual([len(s["input"]) for s in seen], [2, 1])
        q = registry.embed_query("ollama/qwen3-embedding:0.6b", "вопрос")
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=5)
        self.assertTrue(seen[-1]["input"][0].startswith("Instruct: "))
        models, errors = registry.list_models()
        flags = {m["model"]: m["embedding"] for m in models}
        self.assertEqual(flags, {"qwen3-embedding:0.6b": True, "qwen3:8b": False, "custom": True})
        self.assertEqual(errors, [])

    def test_ollama_missing_model(self):
        provider = OllamaProvider("http://o", client=client(lambda r: httpx.Response(404, json={"error": "x"})))
        with self.assertRaises(EmbeddingError) as ctx:
            provider.embed("nope", ["a"])
        self.assertIn("ollama pull nope", str(ctx.exception))

    def test_openai_compatible_and_cache(self):
        calls = []

        def handler(request):
            body = json.loads(request.content)
            calls.append(body)
            self.assertEqual(request.headers["authorization"], "Bearer sk-1")
            return httpx.Response(200, json={"data": [{"index": i, "embedding": [1.0, float(i)]}
                                                      for i in reversed(range(len(body["input"])))]})

        provider = OpenAICompatibleProvider("openai", "https://api.x/v1/", "sk-1", ["te-3"], client=client(handler))
        store = {}
        registry = EmbeddingRegistry(batch=10, cache_get=lambda m, keys: {k: store[(m, k)] for k in keys if (m, k) in store},
                                     cache_put=lambda m, vecs: store.update({(m, k): v for k, v in vecs.items()}))
        registry.add(provider)
        first = registry.embed_documents("openai/te-3", ["x", "y"])
        np.testing.assert_allclose(first[1], np.array([1, 1]) / np.sqrt(2), rtol=1e-6)
        again = registry.embed_documents("openai/te-3", ["y", "x", "z"])
        self.assertEqual([len(c["input"]) for c in calls], [2, 1])
        np.testing.assert_allclose(again[0], first[1])

    def test_retry_then_error(self):
        attempts = []

        def handler(request):
            attempts.append(1)
            return httpx.Response(503)

        provider = OllamaProvider("http://o", client=client(handler))
        import knowledge_service.embeddings as emb

        original = emb.time.sleep
        emb.time.sleep = lambda s: None
        try:
            with self.assertRaises(EmbeddingError):
                provider.embed("m", ["a"])
        finally:
            emb.time.sleep = original
        self.assertEqual(len(attempts), 3)

    def test_model_ids_and_prefixes(self):
        self.assertEqual(split_model_id("ollama/hf.co/org/model:q4"), ("ollama", "hf.co/org/model:q4"))
        with self.assertRaises(EmbeddingError):
            split_model_id("bge-m3")
        self.assertEqual(query_text("nomic-embed-text", "q"), "search_query: q")
        self.assertEqual(document_text("nomic-embed-text", "d"), "search_document: d")
        self.assertEqual(query_text("multilingual-e5-large", "q"), "query: q")
        self.assertEqual(document_text("bge-m3", "d"), "d")
        registry = EmbeddingRegistry()
        with self.assertRaises(EmbeddingError):
            registry.validate("openai/te-3")

    def test_providers_file(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump({"openai": {"base_url": "https://api.openai.com/v1", "api_key": "${KB_TEST_KEY}",
                                  "models": ["text-embedding-3-small"], "title": "OpenAI"}}, fh)
        try:
            os.environ["KB_TEST_KEY"] = "sk-t"
            providers = load_providers_file(path)
            self.assertEqual(providers[0].api_key, "sk-t")
            self.assertEqual(providers[0].list_models()[0]["id"], "openai/text-embedding-3-small")
            del os.environ["KB_TEST_KEY"]
            with self.assertRaises(ValueError):
                load_providers_file(path)
        finally:
            os.unlink(path)


class FetchTests(unittest.TestCase):
    def fetcher(self, handler, resolver=lambda h: ["93.184.216.34"], allowed=(), secrets=None):
        return UrlFetcher(list(allowed), secrets or {}, transport=httpx.MockTransport(handler), resolver=resolver)

    def test_redirect_to_internal_blocked(self):
        def handler(request):
            if request.url.host == "public.example":
                return httpx.Response(302, headers={"Location": "http://internal.example/admin"})
            return httpx.Response(200, text="secret")

        resolver = lambda h: ["10.1.2.3"] if h == "internal.example" else ["93.184.216.34"]  # noqa: E731
        with self.assertRaises(FetchError) as ctx:
            self.fetcher(handler, resolver).fetch("https://public.example/")
        self.assertIn("внутреннюю сеть", str(ctx.exception))

    def test_allowed_internal_host(self):
        f = self.fetcher(lambda r: httpx.Response(200, text="ok", headers={"Content-Type": "text/plain"}),
                         resolver=lambda h: ["10.1.2.3"], allowed=["corp.local"])
        self.assertEqual(f.fetch("http://wiki.corp.local/page").data, b"ok")

    def test_secret_header_only_for_bound_host(self):
        seen = []

        def handler(request):
            seen.append((request.url.host, request.headers.get("authorization")))
            if request.url.host == "wiki.corp.com":
                return httpx.Response(302, headers={"Location": "https://cdn.other.com/file.pdf"})
            return httpx.Response(200, content=b"%PDF", headers={"Content-Type": "application/pdf"})

        os.environ["KB_T_TOKEN"] = "tok"
        try:
            f = self.fetcher(handler, secrets=parse_secrets("KB_T_TOKEN@corp.com"))
            result = f.fetch("https://wiki.corp.com/x", {"secret": "KB_T_TOKEN", "scheme": "Token"})
        finally:
            del os.environ["KB_T_TOKEN"]
        self.assertEqual(seen, [("wiki.corp.com", "Token tok"), ("cdn.other.com", None)])
        self.assertEqual(result.filename, "file.pdf")

    def test_size_limit_and_errors(self):
        f = UrlFetcher([], {}, max_bytes=10, resolver=lambda h: ["93.184.216.34"],
                       transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"x" * 100)))
        with self.assertRaises(FetchError):
            f.fetch("https://a.example/")
        f = self.fetcher(lambda r: httpx.Response(403))
        with self.assertRaises(FetchError) as ctx:
            f.fetch("a.example/page")
        self.assertIn("403", str(ctx.exception))
        with self.assertRaises(FetchError):
            f.fetch("ftp://a.example/")

    def test_filenames(self):
        self.assertEqual(filename_for("https://x.com/docs/", "text/html"), "docs.html")
        self.assertEqual(filename_for("https://x.com/a/index.php", "text/html"), "index.php.html")
        self.assertEqual(filename_for("https://x.com/d/report", "application/pdf"), "report.pdf")
        self.assertEqual(filename_for("https://x.com/d/f", "application/octet-stream",
                                      'attachment; filename="План.docx"'), "План.docx")
        self.assertEqual(filename_for("https://x.com/", "text/html"), "x.com.html")
        self.assertEqual(parse_secrets("A@h1|h2, B@x"), {"A": ["h1", "h2"], "B": ["x"]})


if __name__ == "__main__":
    unittest.main()
