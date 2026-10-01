"""Мелкие общие функции: идентификаторы, время, оценка токенов, хеши."""

from __future__ import annotations

import hashlib
import math
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

_CYR = re.compile(r"[А-Яа-яЁё]")
_WS = re.compile(r"\s+")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_ts() -> int:
    return int(time.time())


def iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def estimate_tokens(text: str) -> int:
    """Грубая оценка числа токенов без токенизатора: кириллица ~2,5 символа
    на токен, остальное ~4 символа (типично для BPE-токенизаторов)."""
    if not text:
        return 0
    cyr = len(_CYR.findall(text))
    return max(1, math.ceil(cyr / 2.5 + (len(text) - cyr) / 4))


def chars_per_token(text: str) -> float:
    tokens = estimate_tokens(text)
    return len(text) / tokens if tokens else 4.0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_hash(text: str) -> str:
    """Хеш текста без учёта регистра и пробелов — для поиска дублей."""
    return sha256_text(_WS.sub(" ", text).strip().lower())


def detect_language(text: str) -> str:
    sample = text[:5000]
    letters = sum(1 for ch in sample if ch.isalpha())
    if not letters:
        return ""
    return "ru" if len(_CYR.findall(sample)) / letters > 0.3 else "en"


def iso_date(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.date().isoformat()
