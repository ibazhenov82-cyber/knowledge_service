"""Сервис целиком: коллекции, загрузка всеми способами, задачи, поиск."""

from __future__ import annotations

import os
import tempfile
import time
import unittest

from knowledge_service.service import NotFoundError, ValidationError
from tests.fakes import ServiceFixture

REVIEW_MD = """# Регламент ревью

## Сроки ревью

Ревью кода выполняется не позднее двух рабочих дней после запроса.

## Оформление

Каждый пулл-реквест содержит описание изменений и ссылку на задачу.
"""

VACATION_MD = """# Отпуска

## Длительность

Ежегодный оплачиваемый отпуск составляет двадцать восемь календарных дней.
"""


class Base(unittest.TestCase):
    fixture_kwargs: dict = {}

    def setUp(self):
        self.fx = ServiceFixture(**self.fixture_kwargs)
        self.svc = self.fx.service

    def tearDown(self):
        self.fx.close()

    def collection(self, **kw):
        return self.svc.create_collection({"name": "Регламенты", **kw})

    def upload(self, col, files, **kw):
        result = self.svc.upload_documents(col["id"], files, **kw)
        job = self.fx.wait(result["job_id"])
        return result, job


class CollectionTests(Base):
    def test_create_defaults_and_validation(self):
        col = self.collection()
        self.assertEqual(col["embedding_model"], "ollama/qwen3-embedding:0.6b")
        self.assertEqual(col["chunking"]["method"], "structure")
        self.assertEqual(col["status"], "empty")
        self.assertIn(".kt", col["file_types"]["code_extensions"])
        with self.assertRaises(ValidationError):
            self.svc.create_collection({"name": " "})
        with self.assertRaises(ValidationError):
            self.svc.create_collection({"name": "x", "embedding_model": "openai/text-embedding-3-small"})
        with self.assertRaises(ValidationError):
            self.svc.create_collection({"name": "x", "chunking": {"method": "semantic"}})
        other = self.svc.create_collection({"name": "Локальная", "embedding_model": "local/bge-m3",
                                            "chunking": {"method": "fixed", "chunk_size_tokens": 300}})
        self.assertEqual(other["chunking"]["chunk_size_tokens"], 300)
        self.assertEqual(len(self.svc.list_collections()), 2)

    def test_patch_chunking_keeps_other_params_and_model_change_reindexes(self):
        col = self.collection(chunking={"chunk_size_tokens": 300})
        updated, job_id = self.svc.update_collection(col["id"], {"chunking": {"method": "fixed"}})
        self.assertEqual((updated["chunking"]["method"], updated["chunking"]["chunk_size_tokens"]), ("fixed", 300))
        self.assertIsNone(job_id)
        self.upload(col, [("review.md", REVIEW_MD.encode())])
        updated, job_id = self.svc.update_collection(col["id"], {"embedding_model": "local/bge-m3"})
        self.assertIsNotNone(job_id)
        job = self.fx.wait(job_id)
        self.assertEqual((job["status"], job["progress"]["documents_done"]), ("completed", 1))
        found = self.svc.retrieve({"query": "сроки ревью", "collection_ids": [col["id"]]})
        self.assertTrue(found["results"])

    def test_delete_collection_removes_files(self):
        col = self.collection()
        self.upload(col, [("review.md", REVIEW_MD.encode())])
        files_dir = os.path.join(self.fx.data_dir, "files", col["id"])
        self.assertTrue(os.path.isdir(files_dir))
        self.svc.delete_collection(col["id"])
        self.assertFalse(os.path.exists(files_dir))
        with self.assertRaises(NotFoundError):
            self.svc.get_collection(col["id"])
        self.assertEqual(self.svc.db.scalar("SELECT COUNT(*) FROM chunks"), 0)


class UploadAndSearchTests(Base):
    def test_upload_index_and_retrieve_with_metadata(self):
        col = self.collection(chunking={"min_chunk_tokens": 5})
        result, job = self.upload(col, [("review.md", REVIEW_MD.encode()), ("vacation.md", VACATION_MD.encode()),
                                        ("logo.png", b"\x89PNG")], metadata={"doc_version": "2.1", "team": "core"})
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["progress"]["documents_done"], 2)
        statuses = {d["filename"]: d["status"] for d in result["documents"]}
        self.assertEqual(statuses["logo.png"], "skipped")
        docs = self.svc.list_documents(col["id"])["items"]
        review = next(d for d in docs if d["filename"] == "review.md")
        self.assertEqual(review["status"], "indexed")
        self.assertEqual(review["title"], "Регламент ревью")
        self.assertEqual(review["doc_version"], "2.1")
        self.assertEqual(review["user_metadata"], {"doc_version": "2.1", "team": "core"})
        self.assertEqual(review["metadata"], {"sections": 3, "doc_version": "2.1", "team": "core"})
        self.assertEqual(review["properties"], {"sections": 3})
        self.assertEqual(review["language"], "ru")
        self.assertEqual(review["applied_chunking"]["method"], "structure")

        found = self.svc.retrieve({"query": "Сколько дней на ревью кода?", "collection_ids": [col["id"]], "top_k": 2})
        top = found["results"][0]
        self.assertIn("двух рабочих дней", top["text"])
        self.assertEqual(top["section"], "Регламент ревью › Сроки ревью")
        self.assertEqual(top["document"]["title"], "Регламент ревью")
        self.assertEqual(top["document"]["doc_version"], "2.1")
        self.assertEqual(top["collection"]["name"], "Регламенты")
        self.assertTrue(top["chunk_id"].startswith(review["id"]))

        chunks = self.svc.list_chunks(review["id"])["items"]
        self.assertEqual(chunks[0]["chunking_method"], "structure")
        meta = top["metadata"]
        for key in ("chunk_id", "document_id", "collection_id", "source", "title", "section", "char_start",
                    "char_end", "tokens", "chunking_method", "doc_type", "doc_version", "language", "team"):
            self.assertIn(key, meta)
        self.assertEqual((meta["source"], meta["source_type"], meta["title"]), ("review.md", "file", "Регламент ревью"))
        self.assertEqual(chunks[1]["metadata"]["section"], "Регламент ревью › Сроки ревью")
        content = self.svc.get_document_content(review["id"])
        self.assertIn("Каждый пулл-реквест", content["text"])
        for chunk in chunks:
            self.assertEqual(content["text"][chunk["char_start"]:chunk["char_end"]], chunk["text"])

    def test_threshold_filters_and_disabled_documents(self):
        col = self.collection()
        _, _ = self.upload(col, [("review.md", REVIEW_MD.encode())], metadata={"team": "core"})
        _, _ = self.upload(col, [("vacation.md", VACATION_MD.encode())], metadata={"team": "hr"})
        found = self.svc.retrieve({"query": "отпуск дней", "collection_ids": [col["id"]],
                                   "filters": {"metadata": {"team": "core"}}})
        self.assertTrue(all(r["document"]["metadata"]["team"] == "core" for r in found["results"]))
        found = self.svc.retrieve({"query": "квантовая хромодинамика", "collection_ids": [col["id"]],
                                   "score_threshold": 0.3})
        self.assertEqual(found["results"], [])
        vacation = self.svc.list_documents(col["id"], q="vacation")["items"][0]
        self.svc.update_document(vacation["id"], {"enabled": False})
        found = self.svc.retrieve({"query": "отпуск календарных дней", "collection_ids": [col["id"]]})
        self.assertFalse(any(r["document"]["id"] == vacation["id"] for r in found["results"]))
        with self.assertRaises(ValidationError):
            self.svc.retrieve({"query": "x", "collection_ids": [col["id"]], "filters": {"bogus": 1}})
        with self.assertRaises(NotFoundError):
            self.svc.retrieve({"query": "x", "collection_ids": ["col_nope"]})

    def test_duplicate_document_skipped_and_near_duplicate_chunks_collapsed(self):
        col = self.collection()
        _, job = self.upload(col, [("a.md", REVIEW_MD.encode())])
        _, job = self.upload(col, [("copy-of-a.md", REVIEW_MD.encode())])
        self.assertEqual(job["progress"]["documents_skipped"], 1)
        copy = self.svc.list_documents(col["id"], q="copy")["items"][0]
        self.assertEqual(copy["status"], "skipped")
        self.assertIn("дубликат", copy["error"])
        # Почти одинаковые фрагменты в разных документах — в выдаче один.
        self.upload(col, [("b.md", REVIEW_MD.replace("Ревью кода", "ревью  кода").encode())])
        found = self.svc.retrieve({"query": "не позднее двух рабочих дней ревью", "collection_ids": [col["id"]]})
        texts = [r["text"] for r in found["results"]]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertGreaterEqual(found["duplicates_skipped"], 1)

    def test_chunking_override_on_upload_and_reindex_document(self):
        col = self.collection()
        long_text = "# Длинный\n\n" + "\n\n".join(f"Абзац {i}. " + "текст " * 60 for i in range(10))
        result, _ = self.upload(col, [("long.md", long_text.encode())],
                                chunking={"method": "fixed", "chunk_size_tokens": 200, "chunk_overlap_tokens": 20})
        doc = self.svc.get_document(result["documents"][0]["id"])
        self.assertEqual(doc["applied_chunking"]["method"], "fixed")
        fixed_count = doc["chunk_count"]
        reindex = self.svc.reindex_document(doc["id"], {})
        self.fx.wait(reindex["job_id"])
        doc = self.svc.get_document(doc["id"])
        self.assertEqual(doc["applied_chunking"]["method"], "structure")
        self.assertNotEqual(doc["chunk_count"], fixed_count)

    def test_replace_file_and_reindex_collection(self):
        col = self.collection()
        result, _ = self.upload(col, [("review.md", REVIEW_MD.encode())])
        doc_id = result["documents"][0]["id"]
        replaced = self.svc.replace_file(doc_id, "review.md", REVIEW_MD.replace("двух", "трёх").encode())
        self.fx.wait(replaced["job_id"])
        self.assertIn("трёх", self.svc.get_document_content(doc_id)["text"])
        job = self.fx.wait(self.svc.reindex_collection(col["id"]))
        self.assertEqual(job["progress"]["documents_done"], 1)
        path, name = self.svc.document_file(doc_id)
        self.assertEqual(name, "review.md")
        with open(path, encoding="utf-8") as fh:
            self.assertIn("трёх", fh.read())

    def test_jsonl_upload_with_fields(self):
        col = self.collection()
        data = '{"id": "T-1", "subject": "Не работает вход", "status": "closed"}\n' \
               '{"id": "T-2", "subject": "Долго грузится отчёт", "status": "open"}\n'
        result, _ = self.upload(col, [("tickets.jsonl", data.encode())],
                                jsonl={"text_fields": ["subject"], "metadata_fields": ["status"]})
        doc = self.svc.get_document(result["documents"][0]["id"])
        self.assertEqual(doc["chunk_count"], 2)
        self.assertNotIn("_jsonl", doc["metadata"])
        found = self.svc.retrieve({"query": "отчёт грузится", "collection_ids": [col["id"]],
                                   "filters": {"metadata": {"status": "open"}}})
        meta = found["results"][0]["metadata"]
        self.assertEqual(len(found["results"]), 1)
        self.assertEqual((meta["status"], meta["id"], meta["section"], meta["filename"]),
                         ("open", "T-2", "Запись T-2", "tickets.jsonl"))
        chunk = self.svc.get_chunk(found["results"][0]["chunk_id"])
        self.assertEqual(chunk["chunk_metadata"], {"status": "open", "id": "T-2"})
        self.assertEqual(doc["properties"]["records"], 2)

    def test_manual_chunks_survive_reindex_and_edit_reembeds(self):
        col = self.collection()
        result, _ = self.upload(col, [("review.md", REVIEW_MD.encode())])
        doc_id = result["documents"][0]["id"]
        manual = self.svc.add_chunk(doc_id, {"text": "Ревью на выходных не проводится.", "section": "Уточнение"})
        self.assertEqual(manual["origin"], "manual")
        self.fx.wait(self.svc.reindex_document(doc_id)["job_id"])
        ids = [c["id"] for c in self.svc.list_chunks(doc_id)["items"]]
        self.assertIn(manual["id"], ids)
        first = self.svc.list_chunks(doc_id)["items"][0]
        edited = self.svc.update_chunk(first["id"], {"text": "Про кофемашину на кухне."})
        self.assertEqual(edited["origin"], "edited")
        found = self.svc.retrieve({"query": "кофемашина кухня", "collection_ids": [col["id"]], "top_k": 1})
        self.assertEqual(found["results"][0]["chunk_id"], first["id"])
        self.svc.update_chunk(first["id"], {"enabled": False})
        found = self.svc.retrieve({"query": "кофемашина кухня", "collection_ids": [col["id"]], "top_k": 1})
        self.assertNotEqual(found["results"][0]["chunk_id"], first["id"])
        self.svc.delete_chunk(manual["id"])
        with self.assertRaises(NotFoundError):
            self.svc.get_chunk(manual["id"])

    def test_embedding_failure_fails_document_and_job(self):
        col = self.collection()
        self.fx.provider.fail = True
        _, job = self.upload(col, [("review.md", REVIEW_MD.encode())])
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["errors"][0]["error"], "Ollama недоступна")
        doc = self.svc.list_documents(col["id"])["items"][0]
        self.assertEqual((doc["status"], doc["error"]), ("failed", "Ollama недоступна"))
        self.assertEqual(self.svc.get_collection(col["id"])["status"], "has_errors")

    def test_embedding_cache_reused(self):
        col = self.collection()
        self.upload(col, [("review.md", REVIEW_MD.encode())])
        calls = len(self.fx.provider.calls)
        self.fx.wait(self.svc.reindex_collection(col["id"]))
        self.assertEqual(len(self.fx.provider.calls), calls)


class PathSourceTests(Base):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kb-root-")
        self.fixture_kwargs = {"roots": [self.root]}
        super().setUp()
        os.makedirs(os.path.join(self.root, "docs", "drafts"))
        os.makedirs(os.path.join(self.root, "docs", ".git"))
        self.write("docs/review.md", REVIEW_MD)
        self.write("docs/vacation.md", VACATION_MD)
        self.write("docs/drafts/draft.md", "# Черновик\n\nНе индексировать.")
        self.write("docs/.git/config.md", "# git")
        self.write("docs/app/Main.kt", "class Main {\n    fun run() = 1\n}\n")
        self.write("docs/image.png", "png")

    def write(self, rel, text):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def test_folder_with_masks_then_sync_changes(self):
        col = self.collection()
        added = self.svc.add_source(col["id"], {"source": {"type": "path", "path": os.path.join(self.root, "docs")},
                                                "exclude": ["drafts/**"], "metadata": {"team": "core"}})
        job = self.fx.wait(added["job_id"])
        self.assertEqual(job["status"], "completed", job)
        self.assertEqual(job["progress"]["documents_done"], 3)
        self.assertEqual(job["progress"]["unsupported"], 1)
        names = sorted(d["filename"] for d in self.svc.list_documents(col["id"])["items"])
        self.assertEqual(names, ["Main.kt", "review.md", "vacation.md"])
        kt = self.svc.list_documents(col["id"], q="Main")["items"][0]
        self.assertEqual((kt["doc_type"], kt["code_language"]), ("code", "kotlin"))
        self.assertIsNotNone(kt["doc_date"])

        # Изменили один файл, удалили другой, добавили третий.
        self.write("docs/review.md", REVIEW_MD.replace("двух", "трёх"))
        os.remove(os.path.join(self.root, "docs", "vacation.md"))
        self.write("docs/new.txt", "Новый документ о дежурствах.")
        synced = self.svc.sync_source(added["source"]["id"])
        job = self.fx.wait(synced["job_id"])
        progress = job["progress"]
        self.assertEqual((progress["documents_done"], progress["documents_unchanged"], progress["documents_deleted"]),
                         (2, 1, 1))
        source = self.svc.get_source(added["source"]["id"])
        self.assertEqual(source["documents"]["total"], 3)
        self.assertIsNotNone(source["last_synced_at"])
        found = self.svc.retrieve({"query": "ревью трёх рабочих дней", "collection_ids": [col["id"]], "top_k": 1})
        self.assertIn("трёх", found["results"][0]["text"])
        self.assertEqual(found["results"][0]["document"]["metadata"]["team"], "core")

    def test_glob_and_include(self):
        col = self.collection()
        added = self.svc.add_source(col["id"], {"source": {"type": "path", "path": f"{self.root}/**/*.md"}})
        self.fx.wait(added["job_id"])
        names = sorted(d["filename"] for d in self.svc.list_documents(col["id"])["items"])
        self.assertEqual(names, ["draft.md", "review.md", "vacation.md"])  # .git исключён по умолчанию
        col2 = self.svc.create_collection({"name": "Код"})
        added = self.svc.add_source(col2["id"], {"source": {"type": "path", "path": self.root}, "include": ["*.kt"]})
        self.fx.wait(added["job_id"])
        self.assertEqual([d["filename"] for d in self.svc.list_documents(col2["id"])["items"]], ["Main.kt"])

    def test_outside_roots_and_symlink_escape(self):
        col = self.collection()
        with self.assertRaises(ValidationError):
            self.svc.add_source(col["id"], {"source": {"type": "path", "path": "/etc"}})
        with self.assertRaises(ValidationError):
            self.svc.add_source(col["id"], {"source": {"type": "path", "path": f"{self.root}/../etc"}})
        os.symlink("/etc", os.path.join(self.root, "docs", "etc-link"))
        with self.assertRaises(ValidationError):
            self.svc.add_source(col["id"], {"source": {"type": "path", "path": f"{self.root}/docs/etc-link"}})
        # Ссылка внутри папки-источника на внешний каталог пропускается при обходе.
        added = self.svc.add_source(col["id"], {"source": {"type": "path", "path": f"{self.root}/docs"}})
        self.fx.wait(added["job_id"])
        uris = [d["uri"] for d in self.svc.list_documents(col["id"])["items"]]
        self.assertTrue(all(u.startswith(os.path.realpath(self.root)) for u in uris), uris)

    def test_browse(self):
        top = self.svc.browse(None)
        self.assertEqual(top["items"][0]["path"], os.path.realpath(self.root))
        listing = self.svc.browse(os.path.join(self.root, "docs"))
        names = [i["name"] for i in listing["items"]]
        self.assertEqual(names[:2], ["app", "drafts"])
        self.assertNotIn(".git", names)
        png = next(i for i in listing["items"] if i["name"] == "image.png")
        self.assertFalse(png["supported"])
        with self.assertRaises(ValidationError):
            self.svc.browse("/etc")

    def test_delete_source_removes_documents(self):
        col = self.collection()
        added = self.svc.add_source(col["id"], {"source": {"type": "path", "path": os.path.join(self.root, "docs")}})
        self.fx.wait(added["job_id"])
        self.svc.delete_source(added["source"]["id"])
        self.assertEqual(self.svc.list_documents(col["id"])["total"], 0)
        self.assertEqual(self.svc.retrieve({"query": "ревью", "collection_ids": [col["id"]]})["results"], [])


class PathDisabledTests(Base):
    def test_path_source_requires_roots(self):
        col = self.collection()
        with self.assertRaises(ValidationError) as ctx:
            self.svc.add_source(col["id"], {"source": {"type": "path", "path": "/data"}})
        self.assertIn("KB_ROOTS", str(ctx.exception))
        for stype in ("git", "s3", "gcs"):
            with self.assertRaises(ValidationError) as ctx:
                self.svc.add_source(col["id"], {"source": {"type": stype, "url": "x"}})
            self.assertIn("пока не поддерживается", str(ctx.exception))


PAGES = {
    "https://wiki.example.com/review": ("text/html; charset=utf-8", """<html><head><title>Ревью</title></head><body>
        <main><h1>Ревью</h1><p>Ревью кода выполняется не позднее двух рабочих дней после запроса ревью.</p>
        <a href="/review/deadlines">Сроки ревью подробно</a> <a href="https://other.com/x">Чужой сайт</a>
        <a href="/files/policy.txt">Политика</a> <a href="/img.png">Картинка</a></main></body></html>""",
                                        {"Last-Modified": "Tue, 12 May 2026 10:00:00 GMT"}),
    "https://wiki.example.com/review/deadlines": ("text/html", "<h1>Сроки</h1><p>Срочные исправления в тот же день.</p>"),
    "https://wiki.example.com/files/policy.txt": ("text/plain", "Политика хранения данных: пять лет."),
    "https://wiki.example.com/img.png": ("image/png", b"\x89PNG"),
    "https://wiki.example.com/old": ("redirect", "https://wiki.example.com/review"),
}


class UrlSourceTests(Base):
    fixture_kwargs = {"pages": PAGES, "secrets": {"WIKI_TOKEN": ["wiki.example.com"]}}

    def test_single_page(self):
        col = self.collection()
        added = self.svc.add_source(col["id"], {"source": {"type": "url", "url": "https://wiki.example.com/old"}})
        job = self.fx.wait(added["job_id"])
        self.assertEqual(job["status"], "completed", job)
        doc = self.svc.list_documents(col["id"])["items"][0]
        self.assertEqual(doc["uri"], "https://wiki.example.com/review")
        self.assertEqual(doc["doc_date"], "2026-05-12")
        self.assertEqual(doc["title"], "Ревью")
        # Переиндексация берёт сохранённую копию страницы, без сети.
        before = len(self.fx.requests)
        self.fx.wait(self.svc.reindex_collection(col["id"]))
        self.assertEqual(len(self.fx.requests), before)
        self.assertEqual(self.svc.get_document(doc["id"])["status"], "indexed")

    def test_crawl_same_domain_and_auth_header(self):
        os.environ["WIKI_TOKEN"] = "secret-token"
        self.addCleanup(os.environ.pop, "WIKI_TOKEN", None)
        col = self.collection()
        added = self.svc.add_source(col["id"], {
            "source": {"type": "url", "url": "https://wiki.example.com/review"},
            "crawl": {"depth": 1, "max_pages": 10}, "auth": {"secret": "WIKI_TOKEN"}})
        job = self.fx.wait(added["job_id"])
        uris = sorted(d["uri"] for d in self.svc.list_documents(col["id"])["items"])
        self.assertEqual(uris, ["https://wiki.example.com/files/policy.txt", "https://wiki.example.com/review",
                                "https://wiki.example.com/review/deadlines"])
        self.assertEqual(job["progress"]["unsupported"], 1)
        self.assertTrue(all(r.url.host == "wiki.example.com" for r in self.fx.requests))
        self.assertTrue(all(r.headers.get("authorization") == "Bearer secret-token" for r in self.fx.requests))
        source = self.svc.get_source(added["source"]["id"])
        self.assertEqual(source["auth"], {"secret": "WIKI_TOKEN"})

    def test_secret_rules(self):
        col = self.collection()
        with self.assertRaises(ValidationError):
            self.svc.add_source(col["id"], {"source": {"type": "url", "url": "https://wiki.example.com/review"},
                                            "auth": {"secret": "OPENAI_API_KEY"}})
        os.environ["WIKI_TOKEN"] = "t"
        self.addCleanup(os.environ.pop, "WIKI_TOKEN", None)
        added = self.svc.add_source(col["id"], {"source": {"type": "url", "url": "https://evil.example.org/"},
                                                "auth": {"secret": "WIKI_TOKEN"}})
        job = self.fx.wait(added["job_id"])
        self.assertEqual(job["status"], "failed")
        self.assertIn("нельзя отправлять", job["error"])
        self.assertEqual(self.fx.requests, [])

    def test_not_found_page(self):
        col = self.collection()
        added = self.svc.add_source(col["id"], {"source": {"type": "url", "url": "https://wiki.example.com/missing"}})
        job = self.fx.wait(added["job_id"])
        self.assertEqual(job["status"], "failed")
        self.assertIn("HTTP 404", job["error"])


class SsrfTests(Base):
    fixture_kwargs = {"pages": {"http://intranet.local/": ("text/html", "<p>секрет</p>")},
                      "resolver": lambda host: ["10.0.0.5"]}

    def test_internal_address_blocked(self):
        col = self.collection()
        added = self.svc.add_source(col["id"], {"source": {"type": "url", "url": "http://intranet.local/"}})
        job = self.fx.wait(added["job_id"])
        self.assertEqual(job["status"], "failed")
        self.assertIn("внутреннюю сеть", job["error"])
        self.assertEqual(self.fx.requests, [])


class TextSourceAndJobsTests(Base):
    def test_text_source_and_resync_unchanged(self):
        col = self.collection()
        added = self.svc.add_source(col["id"], {"source": {"type": "text", "title": "Дежурства",
                                                           "content": "# Дежурства\n\nДежурный меняется по понедельникам."}})
        job = self.fx.wait(added["job_id"])
        self.assertEqual(job["progress"]["documents_done"], 1)
        doc = self.svc.list_documents(col["id"])["items"][0]
        self.assertEqual((doc["source_type"], doc["filename"]), ("text", "Дежурства.md"))
        job = self.fx.wait(self.svc.sync_source(added["source"]["id"])["job_id"])
        self.assertEqual(job["progress"]["documents_unchanged"], 1)
        job = self.fx.wait(self.svc.reindex_collection(col["id"]))
        self.assertEqual(job["progress"]["documents_done"], 1)
        with self.assertRaises(ValidationError):
            self.svc.add_source(col["id"], {"source": {"type": "text", "content": "  "}})

    def test_cancel_queued_job(self):
        self.fx.close()
        self.fx = ServiceFixture(start=False)
        self.svc = self.fx.service
        col = self.collection()
        result = self.svc.upload_documents(col["id"], [("review.md", REVIEW_MD.encode())])
        job = self.svc.cancel_job(result["job_id"])
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(self.svc.get_document(result["documents"][0]["id"])["status"], "cancelled")

    def test_cancel_running_job_between_documents(self):
        col = self.collection()
        original = self.fx.provider.embed

        def slow(model, texts):
            time.sleep(0.2)
            return original(model, texts)

        self.fx.provider.embed = slow
        files = [(f"doc{i}.md", f"# Документ {i}\n\nТекст номер {i} про разное.".encode()) for i in range(10)]
        result = self.svc.upload_documents(col["id"], files)
        time.sleep(0.3)
        self.svc.cancel_job(result["job_id"])
        job = self.fx.wait(result["job_id"])
        self.assertEqual(job["status"], "cancelled")
        statuses = [d["status"] for d in self.svc.list_documents(col["id"])["items"]]
        self.assertIn("cancelled", statuses)
        self.assertIn("indexed", statuses)

    def test_restart_recovery(self):
        col = self.collection()
        self.fx.service.stop()
        db = self.svc.db
        db.insert("jobs", {"id": "job_x", "collection_id": col["id"], "kind": "index", "status": "running",
                           "target_json": '{"document_ids": []}', "progress_json": "{}", "errors_json": "[]",
                           "created_at": 1})
        from knowledge_service.service import KnowledgeService

        again = KnowledgeService(self.fx.settings, self.svc.registry, self.fx.fetcher)
        again.start()
        try:
            job = again.get_job("job_x")
            self.assertEqual(job["status"], "failed")
            self.assertIn("перезапуском", job["error"])
        finally:
            again.stop()

    def test_list_jobs_and_info(self):
        col = self.collection()
        self.upload(col, [("review.md", REVIEW_MD.encode())])
        jobs = self.svc.list_jobs(col["id"])
        self.assertEqual(jobs["total"], 1)
        self.assertEqual(jobs["items"][0]["kind_title"], "Загрузка файлов")
        info = self.svc.info()
        self.assertEqual([m["id"] for m in info["chunking_methods"]], ["structure", "fixed"])
        types = {t["type"]: t["enabled"] for t in info["source_types"]}
        self.assertEqual(types, {"file": True, "path": False, "url": True, "text": True,
                                 "git": False, "s3": False, "gcs": False})
        models = self.svc.embedding_models()
        self.assertEqual(models["items"][0]["id"], "ollama/qwen3-embedding:0.6b")
        self.assertTrue(models["items"][0]["default"])


if __name__ == "__main__":
    unittest.main()
