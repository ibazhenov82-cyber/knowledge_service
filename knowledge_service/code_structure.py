"""
knowledge_service.code_structure
==================================

Разбиение исходного кода на структурные единицы: Python — по классам,
методам и функциям (`ast`); Kotlin, Java, JavaScript/TypeScript, Go — по
объявлениям верхнего уровня и членам классов (регулярные выражения);
SQL — по операторам; YAML, JSON, TOML — по ключам/секциям верхнего уровня;
прочие расширения — файл целиком (дальше делится по абзацам).

Комментарии и аннотации непосредственно над объявлением относятся к нему.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from .formats import Unit

SEP = " › "

LANGUAGES = {
    ".py": "python", ".pyi": "python", ".kt": "kotlin", ".kts": "kotlin", ".java": "java",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".go": "go", ".sql": "sql", ".yaml": "yaml", ".yml": "yaml",
    ".json": "json", ".toml": "toml", ".xml": "xml", ".gradle": "gradle", ".properties": "properties",
    ".sh": "shell",
}

_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "new", "else", "when", "try", "do", "function",
             "throw", "super", "this", "synchronized"}


def _unit(text: str, section: str = ""):
    from .formats import Unit

    return Unit(text=text.strip("\n"), section=section)


def _lines_with_comments_start(lines: List[str], start: int, comment_prefixes: Tuple[str, ...]) -> int:
    """Начало объявления с учётом комментариев и аннотаций прямо над ним."""
    i = start
    while i > 0:
        prev = lines[i - 1].strip()
        if prev and prev.startswith(comment_prefixes):
            i -= 1
        else:
            break
    return i


def _cut(lines: List[str], bounds: List[Tuple[int, str]]) -> List["Unit"]:
    """bounds: [(номер строки начала, раздел)] по возрастанию; всё до первой
    границы — заголовок файла (импорты и т. п.)."""
    units = []
    starts = [b for b in bounds if 0 <= b[0] < len(lines)]
    if not starts or starts[0][0] > 0:
        starts = [(0, "")] + starts
    for idx, (start, section) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        text = "\n".join(lines[start:end])
        if text.strip():
            units.append(_unit(text, section))
    return units


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def _split_python(src: str) -> Optional[List["Unit"]]:
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return None
    lines = src.split("\n")
    bounds: List[Tuple[int, str]] = []

    def node_start(node) -> int:
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
        return _lines_with_comments_start(lines, first, ("#",))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bounds.append((node_start(node), node.name))
        elif isinstance(node, ast.ClassDef):
            bounds.append((node_start(node), node.name))
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    bounds.append((node_start(member), f"{node.name}{SEP}{member.name}"))
        elif bounds and bounds[-1][1]:
            # Код модуля после объявлений (константы, `if __name__ ...`) —
            # отдельная единица без раздела.
            bounds.append((_lines_with_comments_start(lines, node.lineno - 1, ("#",)), ""))
    return _cut(lines, sorted(bounds))


# ---------------------------------------------------------------------------
# Kotlin / Java / JS / TS / Go
# ---------------------------------------------------------------------------

_KOTLIN_TOP = re.compile(
    r"^(?:@[\w.]+(?:\([^)]*\))?\s+)*"
    r"(?:(?:public|private|internal|protected|open|abstract|sealed|data|enum|inline|value|suspend|override|"
    r"annotation|inner|const|actual|expect|operator|infix|tailrec|external|companion|fun(?=\s+interface))\s+)*"
    r"(class|interface|object|fun|typealias)\b\s*(?:<[^>]*>\s*)?(?:[\w.]+\.)?(`[^`]+`|\w+)?"
)
_JAVA_TYPE = re.compile(
    r"^(?:@\w+(?:\([^)]*\))?\s+)*(?:(?:public|private|protected|static|final|abstract|sealed|non-sealed|strictfp)\s+)*"
    r"(class|interface|enum|record|@interface)\s+(\w+)"
)
_JAVA_METHOD = re.compile(
    r"^(?:@\w+(?:\([^)]*\))?\s+)*(?:(?:public|private|protected|static|final|abstract|synchronized|native|default)\s+)*"
    r"(?:<[^>]+>\s+)?[\w<>\[\],.?]+(?:\s*<[^>]*>)?\s+(\w+)\s*\([^;]*$"
)
_JS_TOP = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?(?:async\s+)?"
    r"(function\*?|class|interface|type|enum|const|let|var|namespace)\s+([\w$]+)"
)
_JS_MEMBER = re.compile(
    r"^(?:(?:public|private|protected|static|readonly|async|override|abstract|get|set)\s+)*"
    r"\*?([\w$]+)\s*(?:<[^>]*>)?\s*\([^)]*\)?\s*(?::\s*[^={;]+)?\s*\{?\s*$"
)
_GO_FUNC = re.compile(r"^func\s+(?:\(\s*\w*\s*\*?(\w+)[^)]*\)\s*)?(\w+)")
_GO_TYPE = re.compile(r"^type\s+(\w+)")


def _indent(line: str) -> int:
    expanded = line.replace("\t", "    ")
    return len(expanded) - len(expanded.lstrip(" "))


def _split_c_like(src: str, language: str) -> List["Unit"]:
    lines = src.split("\n")
    bounds: List[Tuple[int, str]] = []
    comments = ("//", "/*", "*", "*/", "@", "#[")
    top_name = ""
    top_is_type = False
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        ind = _indent(line)
        stripped = line.strip()
        if ind > 4 or stripped.startswith(comments[:4]):
            continue
        name: Optional[str] = None
        is_type = False
        member = False
        if language == "kotlin":
            m = _KOTLIN_TOP.match(stripped)
            if m and (ind == 0 or m.group(1) in ("fun", "object", "class", "interface")):
                name = (m.group(2) or ("companion" if m.group(1) == "object" else m.group(1))).strip("`")
                is_type = m.group(1) in ("class", "interface", "object")
                member = ind > 0
        elif language == "java":
            m = _JAVA_TYPE.match(stripped)
            if m:
                name, is_type, member = m.group(2), True, ind > 0
            elif ind > 0 and top_is_type:
                m = _JAVA_METHOD.match(stripped)
                if m and m.group(1) not in _KEYWORDS:
                    name, member = m.group(1), True
        elif language in ("javascript", "typescript"):
            if ind == 0:
                m = _JS_TOP.match(stripped)
                if m:
                    name, is_type = m.group(2), m.group(1) in ("class", "interface", "namespace")
            elif top_is_type:
                m = _JS_MEMBER.match(stripped)
                if m and m.group(1) not in _KEYWORDS and stripped.endswith("{"):
                    name, member = m.group(1), True
        elif language == "go" and ind == 0:
            m = _GO_FUNC.match(stripped)
            if m:
                name = f"{m.group(1)}.{m.group(2)}" if m.group(1) else m.group(2)
            else:
                m = _GO_TYPE.match(stripped)
                if m:
                    name, is_type = m.group(1), True
        if name is None:
            continue
        start = _lines_with_comments_start(lines, i, comments)
        if member and top_name:
            bounds.append((start, f"{top_name}{SEP}{name}"))
        else:
            top_name, top_is_type = name, is_type
            bounds.append((start, name))
    # Аннотации, отнесённые к объявлению, могли дать дубли начала.
    uniq: List[Tuple[int, str]] = []
    for b in bounds:
        if uniq and uniq[-1][0] == b[0]:
            continue
        uniq.append(b)
    return _cut(lines, uniq)


# ---------------------------------------------------------------------------
# SQL, YAML, JSON, TOML
# ---------------------------------------------------------------------------


def _split_sql(src: str) -> List["Unit"]:
    lines = src.split("\n")
    bounds: List[Tuple[int, str]] = []
    start: Optional[int] = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if start is None and stripped and not stripped.startswith("--"):
            start = _lines_with_comments_start(lines, i, ("--",))
            words = re.findall(r"[\w.\"`*]+", " ".join(lines[i:i + 3]))
            section = " ".join(w.strip('"`') for w in words[:4])
            head = section.split(" ")[0].upper() if section else ""
            if head in ("CREATE", "ALTER", "DROP"):
                section = " ".join(words[:3] if "OR" not in [w.upper() for w in words[:3]] else words[:5])
            else:
                section = " ".join(words[:2])
            bounds.append((start, section.upper() if len(section) < 60 else section[:60]))
        if start is not None and stripped.endswith(";"):
            start = None
    return _cut(lines, bounds)


_YAML_KEY = re.compile(r"^([A-Za-z_\"'][^:#\n]*?)\s*:(?:\s|$)")


def _split_yaml(src: str) -> List["Unit"]:
    lines = src.split("\n")
    bounds: List[Tuple[int, str]] = []
    doc = 0
    for i, line in enumerate(lines):
        if line.strip() == "---":
            doc += 1
            continue
        m = _YAML_KEY.match(line)
        if m:
            key = m.group(1).strip("\"'")
            bounds.append((_lines_with_comments_start(lines, i, ("#",)), key if doc < 2 else f"документ {doc}{SEP}{key}"))
    return _cut(lines, bounds)


def _split_json(src: str) -> Optional[List["Unit"]]:
    try:
        data = json.loads(src)
    except ValueError:
        return None
    if isinstance(data, dict) and data:
        return [_unit(json.dumps({k: v}, ensure_ascii=False, indent=2), str(k)) for k, v in data.items()]
    if isinstance(data, list) and data:
        return [_unit(json.dumps(v, ensure_ascii=False, indent=2), f"[{i}]") for i, v in enumerate(data)]
    return None


_TOML_TABLE = re.compile(r"^\[\[?\s*([^\]]+?)\s*\]\]?\s*$")


def _split_toml(src: str) -> List["Unit"]:
    lines = src.split("\n")
    bounds = [(_lines_with_comments_start(lines, i, ("#",)), m.group(1))
              for i, line in enumerate(lines) if (m := _TOML_TABLE.match(line.strip())) and not line.startswith(" ")]
    return _cut(lines, bounds)


def split_code(text: str, filename: str) -> Tuple[str, List["Unit"]]:
    """(язык, единицы)."""
    src = text.replace("\r\n", "\n").replace("\r", "\n")
    ext = PurePosixPath(filename.replace("\\", "/")).suffix.lower()
    language = LANGUAGES.get(ext, ext.lstrip(".") or "text")
    units: Optional[List["Unit"]] = None
    if language == "python":
        units = _split_python(src)
    elif language in ("kotlin", "java", "javascript", "typescript", "go"):
        units = _split_c_like(src, language)
    elif language == "sql":
        units = _split_sql(src)
    elif language == "yaml":
        units = _split_yaml(src)
    elif language == "json":
        units = _split_json(src)
    elif language == "toml":
        units = _split_toml(src)
    if not units:
        units = [_unit(src)] if src.strip() else []
    return language, units
