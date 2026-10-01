"""Разбиение на фрагменты: «По структуре» и «Фиксированный размер»."""

from __future__ import annotations

import unittest

from knowledge_service.chunking import ChunkingConfig, chunk_document, split_spans
from knowledge_service.formats import FileTypes, Parsed, Unit, parse_document
from knowledge_service.util import estimate_tokens


def para(n: int, word: str = "слово") -> str:
    return " ".join(f"{word}{i}" for i in range(n)) + "."


class ConfigTests(unittest.TestCase):
    def test_defaults_and_override(self):
        cfg = ChunkingConfig.from_dict(None)
        self.assertEqual(cfg.method, "structure")
        fixed = ChunkingConfig.from_dict({"method": "fixed", "chunk_size_tokens": 300}, cfg)
        self.assertEqual((fixed.method, fixed.chunk_size_tokens, fixed.chunk_overlap_tokens), ("fixed", 300, 80))

    def test_validation(self):
        for bad in ({"method": "semantic"}, {"chunk_size_tokens": 10}, {"chunk_overlap_tokens": 400},
                    {"min_chunk_tokens": 2000}, {"foo": 1}, {"max_chunk_tokens": "big"}):
            with self.assertRaises(ValueError, msg=bad):
                ChunkingConfig.from_dict(bad)


class StructureTests(unittest.TestCase):
    def check_offsets(self, full, drafts):
        for d in drafts:
            self.assertEqual(full[d.char_start:d.char_end], d.text)

    def test_sections_become_chunks_and_small_ones_merge(self):
        parsed = Parsed(title="T", doc_type="markdown", units=[
            Unit(text="# A\n" + para(150), section="A"),
            Unit(text="## B\nкоротко", section="A › B"),
            Unit(text="## C\n" + para(150), section="A › C"),
        ])
        full, drafts = chunk_document(parsed, ChunkingConfig(max_chunk_tokens=1000, min_chunk_tokens=100))
        self.check_offsets(full, drafts)
        # Короткий раздел B склеен с предыдущим, C — отдельный фрагмент.
        self.assertEqual([d.section for d in drafts], ["A", "A › C"])
        self.assertIn("коротко", drafts[0].text)

    def test_long_section_split_by_paragraphs_keeps_code_fence(self):
        code = "```\n" + "\n\n".join(f"line{i} = {i}" for i in range(30)) + "\n```"
        text = "\n\n".join([para(120), code, para(120), para(120)])
        parsed = Parsed(title="T", doc_type="markdown", units=[Unit(text=text, section="S")])
        full, drafts = chunk_document(parsed, ChunkingConfig(max_chunk_tokens=400, min_chunk_tokens=10))
        self.check_offsets(full, drafts)
        self.assertTrue(all(d.tokens <= 400 for d in drafts))
        self.assertTrue(all(d.section == "S" for d in drafts))
        fenced = [d for d in drafts if "```" in d.text]
        self.assertEqual(len(fenced), 1)
        self.assertEqual(fenced[0].text.count("```"), 2)

    def test_single_huge_paragraph_is_split_by_sentences_then_words(self):
        text = " ".join(para(30) for _ in range(40)) + " " + "длинноеслово" * 400
        spans = split_spans(text, 0, len(text), 200)
        self.assertTrue(all(estimate_tokens(text[s:e]) <= 200 for s, e in spans))
        self.assertGreater(len(spans), 5)

    def test_jsonl_records_not_merged(self):
        parsed = parse_document("\n".join(f'{{"id": {i}, "text": "коротко {i}"}}' for i in range(5)).encode(),
                                "t.jsonl", FileTypes())
        _, drafts = chunk_document(parsed, ChunkingConfig())
        self.assertEqual(len(drafts), 5)
        self.assertEqual(drafts[2].metadata, {"id": 2})
        self.assertEqual(drafts[2].section, "Запись 2")

    def test_pages_and_code_sections(self):
        src = "\n\n".join(f"def f{i}():\n    return {i}\n" + "    # комментарий\n" * 40 for i in range(4))
        parsed = parse_document(src.encode(), "m.py", FileTypes())
        _, drafts = chunk_document(parsed, ChunkingConfig(max_chunk_tokens=300, min_chunk_tokens=50))
        self.assertEqual([d.section for d in drafts], ["f0", "f1", "f2", "f3"])


class FixedTests(unittest.TestCase):
    def test_size_and_overlap(self):
        text = " ".join(f"Предложение номер {i} о правилах ревью." for i in range(300))
        parsed = Parsed(title="T", doc_type="text", units=[Unit(text=text)])
        cfg = ChunkingConfig(method="fixed", chunk_size_tokens=200, chunk_overlap_tokens=40)
        full, drafts = chunk_document(parsed, cfg)
        self.assertGreater(len(drafts), 5)
        for d in drafts:
            self.assertEqual(full[d.char_start:d.char_end], d.text)
            self.assertLessEqual(d.tokens, 210)
        for prev, cur in zip(drafts, drafts[1:]):
            self.assertLess(cur.char_start, prev.char_end, "окна перекрываются")
            self.assertGreater(cur.char_start, prev.char_start)
            # Окно начинается с начала предложения.
            self.assertTrue(cur.text.startswith("Предложение"), cur.text[:30])
        self.assertTrue(drafts[-1].text.endswith("ревью."))

    def test_no_overlap_and_sections_from_units(self):
        parsed = Parsed(title="T", doc_type="markdown", units=[
            Unit(text=para(200, "альфа"), section="Первый", page=1),
            Unit(text=para(200, "бета"), section="Второй", page=2),
        ])
        full, drafts = chunk_document(parsed, ChunkingConfig(method="fixed", chunk_size_tokens=100,
                                                             chunk_overlap_tokens=0))
        for prev, cur in zip(drafts, drafts[1:]):
            self.assertGreaterEqual(cur.char_start, prev.char_end)
        self.assertEqual(drafts[0].section, "Первый")
        self.assertEqual(drafts[-1].section, "Второй")
        self.assertEqual(drafts[-1].page, 2)
        covered = "".join(full[d.char_start:d.char_end] for d in drafts).replace(" ", "").replace("\n", "")
        self.assertEqual(covered, full.replace(" ", "").replace("\n", ""))


if __name__ == "__main__":
    unittest.main()
