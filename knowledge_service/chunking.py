"""
knowledge_service.chunking
============================

Разбиение документа на фрагменты двумя способами:

- `structure` — «По структуре» (по умолчанию): структурные единицы документа
  (разделы, страницы, слайды, объявления кода, записи JSONL); единица длиннее
  `max_chunk_tokens` делится по абзацам → строкам → предложениям, единицы
  короче `min_chunk_tokens` склеиваются с соседними;
- `fixed` — «Фиксированный размер»: окно `chunk_size_tokens` с перекрытием
  `chunk_overlap_tokens`, граница сдвигается к концу абзаца или предложения.

Смещения фрагментов (`char_start`/`char_end`) — в полном тексте документа:
единицы, склеенные через пустую строку.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .formats import Parsed, Unit
from .util import chars_per_token, estimate_tokens

METHODS = {"structure": "По структуре", "fixed": "Фиксированный размер"}
UNIT_SEPARATOR = "\n\n"


@dataclass
class ChunkingConfig:
    method: str = "structure"
    chunk_size_tokens: int = 600
    chunk_overlap_tokens: int = 80
    max_chunk_tokens: int = 1000
    min_chunk_tokens: int = 100

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], base: Optional["ChunkingConfig"] = None) -> "ChunkingConfig":
        """Значения поверх `base` (или умолчаний) с проверкой."""
        result = cls(**asdict(base)) if base else cls()
        if not data:
            return result
        if not isinstance(data, dict):
            raise ValueError("chunking должен быть объектом")
        unknown = set(data) - set(asdict(result))
        if unknown:
            raise ValueError(f"chunking: неизвестные поля {sorted(unknown)}")
        for key, value in data.items():
            if value is None:
                continue
            if key == "method":
                if value not in METHODS:
                    raise ValueError(f"chunking.method: ожидается одно из {sorted(METHODS)}")
                result.method = value
            else:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"chunking.{key}: ожидается целое число")
                setattr(result, key, value)
        result.validate()
        return result

    def validate(self) -> None:
        if not 50 <= self.chunk_size_tokens <= 8000:
            raise ValueError("chunking.chunk_size_tokens: от 50 до 8000")
        if not 0 <= self.chunk_overlap_tokens <= self.chunk_size_tokens // 2:
            raise ValueError("chunking.chunk_overlap_tokens: от 0 до половины chunk_size_tokens")
        if not 50 <= self.max_chunk_tokens <= 8000:
            raise ValueError("chunking.max_chunk_tokens: от 50 до 8000")
        if not 0 <= self.min_chunk_tokens < self.max_chunk_tokens:
            raise ValueError("chunking.min_chunk_tokens: от 0 и меньше max_chunk_tokens")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkDraft:
    text: str
    section: str
    page: Optional[int]
    char_start: int
    char_end: int
    tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Piece:
    start: int
    end: int
    unit: int
    atomic: bool


def assemble(parsed: Parsed) -> Tuple[str, List[int]]:
    """Полный текст документа и смещения начала каждой единицы."""
    starts, parts, pos = [], [], 0
    for i, unit in enumerate(parsed.units):
        if i:
            parts.append(UNIT_SEPARATOR)
            pos += len(UNIT_SEPARATOR)
        starts.append(pos)
        parts.append(unit.text)
        pos += len(unit.text)
    return "".join(parts), starts


# ---------------------------------------------------------------------------
# Деление длинного текста на части
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*(```|~~~)")
_SENTENCE = re.compile(r"[^.!?…\n]+(?:[.!?…]+[\"»”)]*|$)")


def _strip_span(text: str, s: int, e: int) -> Optional[Tuple[int, int]]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return (s, e) if e > s else None


def _paragraph_spans(text: str, s: int, e: int) -> List[Tuple[int, int]]:
    spans, start, pos, in_fence = [], s, s, False
    for line in text[s:e].split("\n"):
        line_end = pos + len(line)
        if _FENCE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            span = _strip_span(text, start, pos)
            if span:
                spans.append(span)
            start = line_end + 1
        pos = line_end + 1
    span = _strip_span(text, start, e)
    if span:
        spans.append(span)
    return spans


def _line_spans(text: str, s: int, e: int) -> List[Tuple[int, int]]:
    spans, pos = [], s
    for line in text[s:e].split("\n"):
        span = _strip_span(text, pos, pos + len(line))
        if span:
            spans.append(span)
        pos += len(line) + 1
    return spans


def _sentence_spans(text: str, s: int, e: int) -> List[Tuple[int, int]]:
    spans = []
    for m in _SENTENCE.finditer(text, s, e):
        span = _strip_span(text, m.start(), m.end())
        if span:
            spans.append(span)
    return spans


def _hard_spans(text: str, s: int, e: int, max_tokens: int) -> List[Tuple[int, int]]:
    width = max(20, int(max_tokens * chars_per_token(text[s:e]) * 0.95))
    spans, pos = [], s
    while pos < e:
        end = min(e, pos + width)
        if end < e:
            space = text.rfind(" ", pos + width // 2, end)
            if space > pos:
                end = space
        span = _strip_span(text, pos, end)
        if span:
            spans.append(span)
        pos = end
    return spans


def split_spans(text: str, s: int, e: int, max_tokens: int, level: int = 0) -> List[Tuple[int, int]]:
    """Части [s, e) не длиннее `max_tokens`: абзацы, при необходимости строки,
    предложения, в крайнем случае — окна по словам."""
    if level == 0:
        spans = _paragraph_spans(text, s, e)
    elif level == 1:
        spans = _line_spans(text, s, e)
    elif level == 2:
        spans = _sentence_spans(text, s, e)
    else:
        return _hard_spans(text, s, e, max_tokens)
    out: List[Tuple[int, int]] = []
    cur: Optional[Tuple[int, int]] = None
    for span in spans:
        if estimate_tokens(text[span[0]:span[1]]) > max_tokens:
            if cur:
                out.append(cur)
                cur = None
            out.extend(split_spans(text, span[0], span[1], max_tokens, level + 1))
        elif cur is None:
            cur = span
        elif estimate_tokens(text[cur[0]:span[1]]) <= max_tokens:
            cur = (cur[0], span[1])
        else:
            out.append(cur)
            cur = span
    if cur:
        out.append(cur)
    return out


# ---------------------------------------------------------------------------
# Способы разбиения
# ---------------------------------------------------------------------------


def _structure(parsed: Parsed, full: str, starts: List[int], cfg: ChunkingConfig) -> List[_Piece]:
    pieces: List[_Piece] = []
    for i, unit in enumerate(parsed.units):
        s, e = starts[i], starts[i] + len(unit.text)
        if estimate_tokens(unit.text) <= cfg.max_chunk_tokens:
            pieces.append(_Piece(s, e, i, unit.atomic))
        else:
            for ps, pe in split_spans(full, s, e, cfg.max_chunk_tokens):
                pieces.append(_Piece(ps, pe, i, unit.atomic))
    merged: List[_Piece] = []
    for piece in pieces:
        if merged and not piece.atomic and not merged[-1].atomic:
            prev = merged[-1]
            prev_tokens = estimate_tokens(full[prev.start:prev.end])
            cur_tokens = estimate_tokens(full[piece.start:piece.end])
            if (prev_tokens < cfg.min_chunk_tokens or cur_tokens < cfg.min_chunk_tokens) \
                    and estimate_tokens(full[prev.start:piece.end]) <= cfg.max_chunk_tokens:
                prev.end = piece.end
                continue
        merged.append(piece)
    return merged


def _boundary(full: str, lo: int, hi: int) -> int:
    """Последняя удобная граница в (lo, hi]: абзац, строка, предложение, пробел."""
    window = full[lo:hi]
    for pattern in ("\n\n", "\n"):
        idx = window.rfind(pattern)
        if idx >= 0:
            return lo + idx + len(pattern)
    ends = [m.end() for m in re.finditer(r"[.!?…][\"»”)]*\s", window)]
    if ends:
        return lo + ends[-1]
    idx = window.rfind(" ")
    return lo + idx + 1 if idx >= 0 else hi


def _align_start(full: str, pos: int, limit: int) -> int:
    """Начало следующего окна: ближайшее начало предложения, иначе слова."""
    m = re.compile(r"[.!?…\n][\"»”)]*\s+").search(full, pos, limit)
    if m:
        return m.end()
    space = full.find(" ", pos, limit)
    return space + 1 if space >= 0 else pos


def _fixed(parsed: Parsed, full: str, starts: List[int], cfg: ChunkingConfig) -> List[_Piece]:
    cpt = chars_per_token(full)
    size = max(50, int(cfg.chunk_size_tokens * cpt))
    overlap = int(cfg.chunk_overlap_tokens * cpt)
    n = len(full)
    pieces: List[_Piece] = []
    start = 0
    while start < n and full[start].isspace():
        start += 1
    while start < n:
        end = min(n, start + size)
        if end < n:
            end = _boundary(full, start + int(size * 0.6), end)
        span = _strip_span(full, start, end)
        if span:
            unit = max(0, bisect.bisect_right(starts, span[0]) - 1)
            pieces.append(_Piece(span[0], span[1], unit, False))
        if end >= n:
            break
        nxt = end - overlap
        if overlap and nxt > start:
            nxt = _align_start(full, nxt, end)
        if nxt <= start:
            nxt = end
        start = nxt
        while start < n and full[start].isspace():
            start += 1
    return pieces


def chunk_document(parsed: Parsed, cfg: ChunkingConfig) -> Tuple[str, List[ChunkDraft]]:
    """Полный текст документа и его фрагменты."""
    full, starts = assemble(parsed)
    pieces = _structure(parsed, full, starts, cfg) if cfg.method == "structure" else _fixed(parsed, full, starts, cfg)
    drafts = []
    for piece in pieces:
        unit: Unit = parsed.units[piece.unit]
        text = full[piece.start:piece.end]
        drafts.append(ChunkDraft(
            text=text, section=unit.section, page=unit.page, char_start=piece.start, char_end=piece.end,
            tokens=estimate_tokens(text), metadata=dict(unit.metadata),
        ))
    return full, drafts
