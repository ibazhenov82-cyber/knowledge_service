"""REST API /api/v1 через TestClient. Требует fastapi (иначе пропускается)."""

from __future__ import annotations

import json
import unittest

try:
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FASTAPI_AVAILABLE = False

from tests.fakes import ServiceFixture

if FASTAPI_AVAILABLE:
    from knowledge_service.app import create_app

MD = "# Регламент ревью\n\n## Сроки\n\nРевью кода выполняется не позднее двух рабочих дней.\n"


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi не установлен в этом окружении")
class ApiTests(unittest.TestCase):
    api_key = ""

    def setUp(self):
        self.fx = ServiceFixture(start=False)
        self.client = TestClient(create_app(self.fx.service, api_key=self.api_key), base_url="http://localhost:8003")
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.fx.close()

    def wait(self, job_id):
        return self.fx.wait(job_id)

    def test_full_flow(self):
        col = self.client.post("/api/v1/collections", json={"name": "Регламенты"})
        self.assertEqual(col.status_code, 201, col.text)
        cid = col.json()["id"]
        up = self.client.post(f"/api/v1/collections/{cid}/documents",
                              files=[("file", ("review.md", MD.encode(), "text/markdown")),
                                     ("file", ("notes.txt", "Заметки о дежурствах.".encode(), "text/plain"))],
                              data={"metadata": json.dumps({"doc_version": "1.0"}),
                                    "chunking": json.dumps({"method": "fixed"})})
        self.assertEqual(up.status_code, 202, up.text)
        body = up.json()
        self.assertEqual(len(body["documents"]), 2)
        job = self.wait(body["job_id"])
        self.assertEqual(job["status"], "completed")
        self.assertEqual(self.client.get(f"/api/v1/jobs/{body['job_id']}").json()["status"], "completed")

        docs = self.client.get(f"/api/v1/collections/{cid}/documents", params={"status": "indexed"}).json()
        self.assertEqual(docs["total"], 2)
        doc_id = next(d["id"] for d in docs["items"] if d["filename"] == "review.md")
        self.assertEqual(self.client.get(f"/api/v1/documents/{doc_id}").json()["applied_chunking"]["method"], "fixed")
        self.assertIn("двух рабочих", self.client.get(f"/api/v1/documents/{doc_id}/content").json()["text"])
        file_resp = self.client.get(f"/api/v1/documents/{doc_id}/file")
        self.assertEqual(file_resp.content, MD.encode())
        chunks = self.client.get(f"/api/v1/documents/{doc_id}/chunks").json()
        self.assertGreaterEqual(chunks["total"], 1)
        chunk_id = chunks["items"][0]["id"]
        self.assertEqual(self.client.get(f"/api/v1/chunks/{chunk_id}").json()["document"]["id"], doc_id)

        found = self.client.post("/api/v1/retrieval", json={"query": "сроки ревью", "collection_ids": [cid]}).json()
        self.assertEqual(found["results"][0]["document"]["id"], doc_id)

        patched = self.client.patch(f"/api/v1/documents/{doc_id}", json={"title": "Ревью", "enabled": False}).json()
        self.assertEqual((patched["title"], patched["enabled"]), ("Ревью", False))
        added = self.client.post(f"/api/v1/documents/{doc_id}/chunks", json={"text": "Уточнение"})
        self.assertEqual(added.status_code, 201)
        self.assertEqual(self.client.delete(f"/api/v1/chunks/{added.json()['id']}").status_code, 204)

        reindex = self.client.post(f"/api/v1/documents/{doc_id}/reindex", json={"chunking": {}})
        self.assertEqual(reindex.status_code, 202)
        self.wait(reindex.json()["job_id"])
        self.assertEqual(self.client.get(f"/api/v1/documents/{doc_id}").json()["applied_chunking"]["method"],
                         "structure")
        jobs = self.client.get(f"/api/v1/collections/{cid}/jobs").json()
        self.assertEqual(jobs["total"], 2)
        col = self.client.get(f"/api/v1/collections/{cid}").json()
        self.assertEqual(col["stats"]["documents"], 2)
        self.assertEqual(self.client.delete(f"/api/v1/documents/{doc_id}").status_code, 204)
        self.assertEqual(self.client.delete(f"/api/v1/collections/{cid}").status_code, 204)
        self.assertEqual(self.client.get(f"/api/v1/collections/{cid}").status_code, 404)

    def test_sources_and_sse(self):
        cid = self.client.post("/api/v1/collections", json={"name": "Заметки"}).json()["id"]
        resp = self.client.post(f"/api/v1/collections/{cid}/sources",
                                json={"source": {"type": "text", "title": "Дежурства",
                                                 "content": "Дежурный меняется по понедельникам."}})
        self.assertEqual(resp.status_code, 202, resp.text)
        job_id = resp.json()["job_id"]
        with self.client.stream("GET", f"/api/v1/jobs/{job_id}/events") as stream:
            text = "".join(stream.iter_text())
        self.assertIn("event: done", text)
        last = json.loads(text.strip().split("data: ")[-1])
        self.assertEqual(last["status"], "completed")
        sources = self.client.get(f"/api/v1/collections/{cid}/sources").json()
        sid = sources["items"][0]["id"]
        self.assertEqual(sources["items"][0]["type_title"], "Текст")
        sync = self.client.post(f"/api/v1/sources/{sid}/sync")
        self.assertEqual(sync.status_code, 202)
        self.wait(sync.json()["job_id"])
        patched = self.client.patch(f"/api/v1/sources/{sid}", json={"metadata": {"team": "ops"}})
        self.assertEqual(patched.json()["metadata"], {"team": "ops"})
        self.assertEqual(self.client.delete(f"/api/v1/sources/{sid}").status_code, 204)

    def test_errors_format(self):
        resp = self.client.post("/api/v1/collections", json={"name": "x", "chunking": {"method": "semantic"}})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("chunking.method", resp.json()["error"])
        resp = self.client.post("/api/v1/collections", json={"title": "x"})
        self.assertEqual(resp.status_code, 422)
        self.assertIn("error", resp.json())
        self.assertEqual(self.client.get("/api/v1/documents/doc_nope").status_code, 404)
        self.assertEqual(self.client.get("/api/v1/documents/doc_nope").json(), {"error": "документ doc_nope не найден"})
        cid = self.client.post("/api/v1/collections", json={"name": "x"}).json()["id"]
        resp = self.client.post(f"/api/v1/collections/{cid}/documents", data={"metadata": "{bad"},
                                files=[("file", ("a.md", b"# a", "text/markdown"))])
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"], "metadata: некорректный JSON")
        resp = self.client.post(f"/api/v1/collections/{cid}/sources", json={"source": {"type": "git", "url": "x"}})
        self.assertEqual(resp.status_code, 400)

    def test_service_endpoints(self):
        self.assertEqual(self.client.get("/api/v1/health").json()["status"], "ok")
        info = self.client.get("/api/v1/info").json()
        self.assertIn(".pdf", info["formats"]["documents"])
        models = self.client.get("/api/v1/embedding-models").json()
        self.assertEqual(models["default"], "ollama/qwen3-embedding:0.6b")
        self.assertEqual(self.client.get("/api/v1/fs").status_code, 400)
        self.assertIn("/api/v1/retrieval", self.client.get("/openapi.json").json()["paths"])


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi не установлен в этом окружении")
class ApiKeyTests(ApiTests.__bases__[0]):
    def setUp(self):
        self.fx = ServiceFixture(start=False)
        self.client = TestClient(create_app(self.fx.service, api_key="k1"), base_url="http://localhost:8003")
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.fx.close()

    def test_key_required_except_health(self):
        self.assertEqual(self.client.get("/api/v1/collections").status_code, 401)
        self.assertEqual(self.client.get("/api/v1/health").status_code, 200)
        ok = self.client.get("/api/v1/collections", headers={"Authorization": "Bearer k1"})
        self.assertEqual(ok.status_code, 200)


if __name__ == "__main__":
    unittest.main()
