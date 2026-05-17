# prompt_parser_patched_superhybrid.py
from __future__ import annotations

import re
import os
import sys
import random
import hashlib
import math
import py_compile
import logging
from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Sequence

import lark
logger = logging.getLogger(__name__)  # не настраиваем basicConfig в библиотеке

class PromptSyntaxError(ValueError):
    """Ошибка синтаксиса промпта с дополнительным контекстом.

    Используется для выдачи человекочитаемых сообщений (например, для reverse ranges).
    """

    def __init__(self, message: str, *, kind: str | None = None, token: str | None = None, full: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.token = token
        self.full = full

import threading as _threading
torch = None
_torch_import_lock = _threading.Lock()


def _ensure_torch():
    """Lazily import torch to avoid heavy import at module load time.
    Защищён threading.Lock от гонки при одновременном первом вызове из нескольких потоков.
    """
    global torch
    if torch is not None:
        return torch
    with _torch_import_lock:
        # Повторная проверка под локом — другой поток мог уже завершить импорт
        if torch is not None:
            return torch
        try:
            import torch as _torch
        except ImportError as exc:
            raise ImportError(
                "Torch is required for conditioning reconstruction but is not installed."
            ) from exc
        torch = _torch
    return torch

# ──────────────────────────────────────────────────────────────────────────────
# Фиче-флаги (переопределяемые через env)
# ──────────────────────────────────────────────────────────────────────────────
# Множители для скобок внимания
ROUND_BRACKET_MULTIPLIER = 1.1      # Вес для (...)
SQUARE_BRACKET_MULTIPLIER = 1 / 1.1  # Вес для [...]
# Строковые представления для использования в emphasized/weighted (не хардкодим "1.1")
_RB_WEIGHT_STR  = f"{ROUND_BRACKET_MULTIPLIER:.1f}"          # "1.1"
_RB_WEIGHT_STRS = (_RB_WEIGHT_STR, f"{ROUND_BRACKET_MULTIPLIER:.2f}")  # ("1.1","1.10")


def _env_bool(name: str, default: str = "0") -> bool:
    v = str(os.getenv(name, default)).strip().lower()
    return v not in ("0", "", "false", "no", "off")

def _env_int(name: str, default: int) -> int:
    """Безопасно прочитать целочисленное значение из окружения.

    При некорректном значении возвращает ``default`` и пишет предупреждение
    в лог, чтобы не падать при импорте модуля.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid value for %s=%r, using default %d", name, raw, default)
        return default

SAFE_EMPTY = " "

ALLOW_EMPTY_ALTERNATE          = _env_bool("ALLOW_EMPTY_ALTERNATE", "1")
EXPAND_ALTERNATE_PER_STEP      = _env_bool("EXPAND_ALTERNATE_PER_STEP", "1")
GROUP_COMBO_LIMIT              = _env_int("GROUP_COMBO_LIMIT", 100)
DEDUP_SCHEDULE_STEPS           = _env_bool("DEDUP_SCHEDULE_STEPS", "0")
GROUP_COMBO_FALLBACK           = os.getenv("GROUP_COMBO_FALLBACK", "truncate").strip().lower()  # "truncate"|"literal"|"sample"
SUPPRESS_STANDALONE_COLON      = _env_bool("SUPPRESS_STANDALONE_COLON", "1")
CACHE_SIZE                     = _env_int("PROMPT_PARSER_CACHE_SIZE", 4096)
RECURSION_LIMIT                = _env_int("PROMPT_PARSER_RECURSION_LIMIT", 0)

if RECURSION_LIMIT > 0:
    try:
        sys.setrecursionlimit(RECURSION_LIMIT)
    except (ValueError, RecursionError):
        logger.warning("Failed to set recursion limit to %d", RECURSION_LIMIT)


# ──────────────────────────────────────────────────────────────────────────────
# Общие числовые шаблоны / регэкспы (чтобы не дублировать)
# ──────────────────────────────────────────────────────────────────────────────
NUMERIC_RE = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
# «Без знака» — удобно для шаблонов вида [+|-]<число>
NUMERIC_NOSIGN_RE = r"(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
RE_NUMERIC = re.compile(NUMERIC_RE)
RE_NUMERIC_FULL = re.compile(rf"^(?:{NUMERIC_RE})$")
# Добавить константу в начало файла (после imports)
ATTENTION_AND_OPERATOR = "&"  # Внутреннее представление оператора AND
# Private-use char — cannot appear in real user prompts, so no collision possible.
# Previously was "__PROMPT_PARSER_ESCAPED_AMP__" (a string that could appear in user text).
ESCAPED_AMP_PLACEHOLDER = "\uE004"
# Placeholder (1 char) used to temporarily replace commas inside postfix scheduled blocks
# like "[...]:N ..." so Lark can parse comma-separated tags inside brackets.
# Using a single Unicode Private-Use char keeps string positions stable for linting/UX.
SCHEDULE_COMMA_PLACEHOLDER = "\uE000"
# Escaped literal placeholders keep "\(", "\[", "\:" etc. opaque until parsing finishes.
# This prevents escaped syntax markers from accidentally participating in grammar/attention.
_ESCAPED_LITERAL_SINGLE_PLACEHOLDERS = {
    "\\": "\uE110",
    ":": "\uE111",
    "&": "\uE112",
    "[": "\uE113",
    "]": "\uE114",
    "(": "\uE115",
    ")": "\uE116",
    "{": "\uE117",
    "}": "\uE118",
    "|": "\uE119",
    "!": "\uE11A",
}
_ESCAPED_LITERAL_SINGLE_RESTORE = {
    placeholder: literal
    for literal, placeholder in _ESCAPED_LITERAL_SINGLE_PLACEHOLDERS.items()
}
_ESCAPED_LITERAL_SINGLE_SOURCE_RESTORE = {
    placeholder: (r"\\" if literal == "\\" else f"\\{literal}")
    for literal, placeholder in _ESCAPED_LITERAL_SINGLE_PLACEHOLDERS.items()
}
_ESCAPED_LITERAL_SPAN_DELIMS = {"[": "]", "(": ")", "{": "}"}
_ESCAPED_LITERAL_SPAN_BASE = 0xE200

# ── Attention interpolation ───────────────────────────────────────────────────
# Syntax: (body:w0->w1)  — e.g. (cat:1.0->2.0), (red eyes:0.8->1.4)
#
# Architecture (v21): surface pre-pass BEFORE Lark.
#   1) _placeholderize_attention_interpolations() finds (body:w0->w1) patterns
#      in the raw prompt string and replaces them with private-use marker strings.
#   2) Lark + all visitors/transformers/fast-paths carry the marker as plain text.
#      No grammar changes needed — Lark never sees "->".
#   3) _expand_attention_interpolations() post-pass (called in _get_schedule_impl
#      just before _apply_scheduling_mode) expands each marker per active step,
#      making the weight interpolation contextual-aware (local segment, not global).
#
# Marker format: OPEN body SEP w0 SEP w1 SEP mode CLOSE
#   mode is one of: linear | ease-in | ease-out | ease-in-out  (default: linear)
# Private-use chars survive _collapse_spaces() and _unescape_literals() unchanged.
ATTN_INTERP_OPEN  = "\uE001"  # marker start
ATTN_INTERP_SEP   = "\uE002"  # field separator  body | w0 | w1 | mode
ATTN_INTERP_CLOSE = "\uE003"  # marker end

# Valid easing mode names.
_EASING_MODES = frozenset({"linear", "ease-in", "ease-out", "ease-in-out"})
_EASING_DEFAULT = "linear"

# Matches serialized marker in schedule text (used by post-pass).
# mode field is optional for backward-compat with markers written without it.
RE_ATTN_INTERP_LITERAL = re.compile(
    rf"{re.escape(ATTN_INTERP_OPEN)}(.*?)"
    rf"{re.escape(ATTN_INTERP_SEP)}({NUMERIC_RE})"
    rf"{re.escape(ATTN_INTERP_SEP)}({NUMERIC_RE})"
    rf"(?:{re.escape(ATTN_INTERP_SEP)}([A-Za-z][A-Za-z0-9_-]*))?"
    rf"{re.escape(ATTN_INTERP_CLOSE)}"
)

# Tail of a raw interpolation spec after last top-level ':'.
# Supports optional easing suffix: 'w0 -> w1' or 'w0 -> w1 ~ mode'
# mode token is case-insensitive; unknown values fall back to linear at runtime.
_RE_ATT_INTERP_TAIL = re.compile(
    rf'^\s*({NUMERIC_RE})\s*->\s*({NUMERIC_RE})'
    rf'(?:\s*~\s*([A-Za-z][A-Za-z0-9_-]*))?\s*$'
)

def _serialize_att_interp(body: str, w0: float, w1: float,
                           mode: str = _EASING_DEFAULT) -> str:
    """Pack (body, w0, w1, mode) into a single private-use marker string."""
    safe = (body or "").replace(ATTN_INTERP_OPEN, "").replace(ATTN_INTERP_SEP, "").replace(ATTN_INTERP_CLOSE, "")
    m = mode if mode in _EASING_MODES else _EASING_DEFAULT
    return f"{ATTN_INTERP_OPEN}{safe}{ATTN_INTERP_SEP}{w0}{ATTN_INTERP_SEP}{w1}{ATTN_INTERP_SEP}{m}{ATTN_INTERP_CLOSE}"

def _apply_easing(t: float, mode: str) -> float:
    """Map linear t∈[0,1] through easing curve."""
    if mode == "ease-in":
        return t * t
    if mode == "ease-out":
        return t * (2.0 - t)
    if mode == "ease-in-out":
        return t * t * (3.0 - 2.0 * t)
    return t  # linear

def _format_interp_weight(x: float) -> str:
    """Format interpolated weight: always at least one decimal, strip trailing zeros.

    1.0 -> '1.0',  1.25 -> '1.25',  2.0 -> '2.0',  1.5000 -> '1.5'
    """
    s = f"{float(x):.4f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогалки по тексту/пробелам для единообразия вывода
# ──────────────────────────────────────────────────────────────────────────────

_re_ws_collapse = re.compile(r"[ \t\r\n]+")

def _collapse_spaces(s: str, keep_edges: bool = False) -> str:
    """Сжать повторяющиеся пробелы/переводы строк в один пробел.

    Если ``keep_edges`` истинно, ведущие и хвостовые пробелы сохраняются,
    иначе результат подрезается по краям.
    """
    collapsed = _re_ws_collapse.sub(" ", s).replace(SCHEDULE_COMMA_PLACEHOLDER, ",")
    return collapsed if keep_edges else collapsed.strip()

def _unescape_literals(s: str) -> str:
    """
    Разэкранировать самые частые литералы, которые встречаются в промптах:
    '\\:' -> ':', '\\[' -> '[', '\\]' -> ']', '\\(' -> '(', '\\)' -> ')', '\\\\' -> '\\'
    Важно: сначала обрабатываем специализированные пары, потом общий '\\\\'.
    """
    if not s:
        return s
    s = (s.replace(r"\:", ":")
           .replace(r"\&", "&")
           .replace(r"\[", "[")
           .replace(r"\]", "]")
           .replace(r"\(", "(")
           .replace(r"\)", ")")
           .replace(r"\{", "{")
           .replace(r"\}", "}")
           .replace(r"\|", "|")
           .replace(r"\!", "!"))
    s = s.replace(SCHEDULE_COMMA_PLACEHOLDER, ",")
    # двойной бэкслеш в конце
    s = s.replace(r"\\", "\\")
    return s


@dataclass(frozen=True)
class _BoundarySpec:
    """Typed scheduler boundary keeps absolute steps distinct from fractions and percents."""

    kind: str
    value: float


def _make_boundary_spec(
    value: float | _BoundarySpec,
    *,
    is_percent: bool = False,
    assume_absolute: bool = False,
) -> _BoundarySpec:
    """Preserve boundary meaning instead of guessing it later from a raw float."""
    if isinstance(value, _BoundarySpec):
        return value

    value_f = float(value)
    if is_percent:
        return _BoundarySpec("percent", value_f)
    if assume_absolute or not (0.0 < value_f < 1.0):
        return _BoundarySpec("absolute", value_f)
    return _BoundarySpec("fraction", value_f)


def _protect_escaped_literals(text: str) -> str:
    """Replace escaped single-char literals with private-use placeholders."""
    if not text or "\\" not in text:
        return text

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "\\" and i + 1 < n:
            placeholder = _ESCAPED_LITERAL_SINGLE_PLACEHOLDERS.get(text[i + 1])
            if placeholder is not None:
                out.append(placeholder)
                i += 2
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _protect_escaped_literal_spans(text: str) -> tuple[str, dict[str, str]]:
    """Protect full escaped blocks like ``\\[...\\]`` or ``\\(...\\)`` as plain text."""
    if not text or "\\" not in text:
        return text, {}

    out: list[str] = []
    restore: dict[str, str] = {}
    i = 0
    n = len(text)

    while i < n:
        if text[i] == "\\" and i + 1 < n:
            opener = text[i + 1]
            closer = _ESCAPED_LITERAL_SPAN_DELIMS.get(opener)
            if closer is not None:
                start = i
                j = i + 2
                depth = 1
                matched_end: int | None = None
                while j < n:
                    if text[j] == "\\" and j + 1 < n:
                        escaped = text[j + 1]
                        if escaped == opener:
                            depth += 1
                            j += 2
                            continue
                        if escaped == closer:
                            depth -= 1
                            j += 2
                            if depth == 0:
                                matched_end = j
                                break
                            continue
                    j += 1
                if matched_end is not None:
                    placeholder = chr(_ESCAPED_LITERAL_SPAN_BASE + len(restore))
                    restore[placeholder] = _unescape_literals(text[start:matched_end])
                    out.append(placeholder)
                    i = matched_end
                    continue
        out.append(text[i])
        i += 1

    return "".join(out), restore


def _protect_escaped_literal_spans_for_source(text: str) -> tuple[str, dict[str, str]]:
    """Protect escaped blocks but restore them back to their original source text."""
    if not text or "\\" not in text:
        return text, {}

    out: list[str] = []
    restore: dict[str, str] = {}
    i = 0
    n = len(text)

    while i < n:
        if text[i] == "\\" and i + 1 < n:
            opener = text[i + 1]
            closer = _ESCAPED_LITERAL_SPAN_DELIMS.get(opener)
            if closer is not None:
                start = i
                j = i + 2
                depth = 1
                matched_end: int | None = None
                while j < n:
                    if text[j] == "\\" and j + 1 < n:
                        escaped = text[j + 1]
                        if escaped == opener:
                            depth += 1
                            j += 2
                            continue
                        if escaped == closer:
                            depth -= 1
                            j += 2
                            if depth == 0:
                                matched_end = j
                                break
                            continue
                    j += 1
                if matched_end is not None:
                    placeholder = chr(_ESCAPED_LITERAL_SPAN_BASE + len(restore))
                    restore[placeholder] = text[start:matched_end]
                    out.append(placeholder)
                    i = matched_end
                    continue
        out.append(text[i])
        i += 1

    return "".join(out), restore


def _restore_escaped_literals(text: str, span_restore: dict[str, str] | None = None) -> str:
    """Restore protected escaped literal placeholders back to visible text."""
    if not text:
        return text

    if span_restore:
        text = "".join(span_restore.get(ch, ch) for ch in text)
    if any(ch in text for ch in _ESCAPED_LITERAL_SINGLE_RESTORE):
        text = "".join(_ESCAPED_LITERAL_SINGLE_RESTORE.get(ch, ch) for ch in text)
    return text


def _restore_escaped_literal_source(text: str, span_restore: dict[str, str] | None = None) -> str:
    """Restore protected placeholders back to their original escaped source text."""
    if not text:
        return text

    if span_restore:
        text = "".join(span_restore.get(ch, ch) for ch in text)
    if any(ch in text for ch in _ESCAPED_LITERAL_SINGLE_SOURCE_RESTORE):
        text = "".join(_ESCAPED_LITERAL_SINGLE_SOURCE_RESTORE.get(ch, ch) for ch in text)
    return text

def _normalize_and_operators_for_parse(text: str) -> str:
    """Нормализовать логический оператор до внутреннего символа, не трогая экранированный '&'."""
    if not text:
        return text
    text = text.replace(r"\&", ESCAPED_AMP_PLACEHOLDER)
    text = re.sub(r'(?<!\\)(?<=\w)&(?=\w)', ESCAPED_AMP_PLACEHOLDER, text)
    text = re.sub(rf'(?<![\\\w_])AND(?![\w_])', ATTENTION_AND_OPERATOR, text)
    return re.sub(rf'\s*{re.escape(ATTENTION_AND_OPERATOR)}\s*', ATTENTION_AND_OPERATOR, text)

def _strip_outer_parens_once(s: str) -> str:
    """Снять ОДИН раз внешние круглые скобки, если они действительно обрамляют весь текст."""
    if not s:
        return s
    t = s.strip()
    if len(t) >= 2 and t[0] == "(" and t[-1] == ")":
        # быстрая проверка баланса на одном уровне
        depth = 0
        for i, ch in enumerate(t):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(t) - 1:
                    # есть ещё символы после закрытия — это не внешняя пара
                    return s
        if depth == 0:
            return t[1:-1].strip()
    return s

def _norm_join(*parts: str) -> str:
    """Аккуратно склеить префикс/контент/суффикс → одинарные пробелы, края подрезать."""
    return _collapse_spaces("".join(parts))

def _norm_join_keep_edges(*parts: str) -> str:
    """Как _norm_join, но без .strip() по краям (используется локально при сборке)."""
    return _collapse_spaces("".join(parts), keep_edges=True)

# ──────────────────────────────────────────────────────────────────────────────
# Склейка префикса/ядра/суффикса с корректной обработкой пробелов
# ──────────────────────────────────────────────────────────────────────────────
def _smart_space_trim(s: str) -> str:
    return _re_ws_collapse.sub(" ", s)


def _concat_prefix_text_suffix(prefix: str, text: str, suffix: str) -> str:
    """
    Склейка префикса/ядра/суффикса c улучшенной обработкой Unicode и эмодзи.
    """
    def _is_cjk(ch: str) -> bool:
        """Проверить, является ли символ CJK (китайский/японский/корейский)."""
        cp = ord(ch)
        return (
            0x4E00 <= cp <= 0x9FFF    # CJK Unified Ideographs (основной блок)
            or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
            or 0x20000 <= cp <= 0x2A6DF # CJK Extension B
            or 0x3000 <= cp <= 0x303F  # CJK Symbols and Punctuation
            or 0x3040 <= cp <= 0x309F  # Hiragana
            or 0x30A0 <= cp <= 0x30FF  # Katakana
            or 0xAC00 <= cp <= 0xD7AF  # Hangul Syllables
            or 0xFF00 <= cp <= 0xFFEF  # Halfwidth/Fullwidth Forms
        )

    def _need_space(a: str, b: str) -> bool:
        if not a or not b:
            return False
        a_clean = a.strip()
        b_clean = b.strip()
        if not a_clean or not b_clean:
            return False

        last_char = a[-1]
        first_char = b[0]

        # Если уже есть пробел
        if last_char.isspace() or first_char.isspace():
            return False

        # CJK символы не нуждаются в пробелах между собой
        if _is_cjk(last_char) and _is_cjk(first_char):
            return False
        # CJK + латиница: пробел нужен
        # (оставляем как есть — return True ниже)

        # FIX #6: Более безопасная логика склейки.
        # Ставим пробел практически всегда, кроме случаев, когда знаки препинания "обнимают" слово.
        # Например: "word" + "," -> "word," (без пробела)
        # Но: "word" + "(" -> "word (" (с пробелом)
        # И: "smile" + ":)" -> "smile :)" (с пробелом, чтобы не ломать токен)

        if first_char in ",.;?!:":
            return False
        # Opening brackets already "touch" their content — no space needed
        if last_char in "([{":
            return False

        return True

    def _strip_edge_and_left(s: str) -> tuple[str, bool]:
        stripped = s.lstrip()
        if stripped.startswith(ATTENTION_AND_OPERATOR):
            return stripped[1:].lstrip(), True
        return s, False

    def _strip_edge_and_right(s: str) -> tuple[str, bool]:
        stripped = s.rstrip()
        if stripped.endswith(ATTENTION_AND_OPERATOR):
            return stripped[:-1].rstrip(), True
        return s, False

    # Пустая центральная часть
    if text.strip() == "":
        left, left_has_and = _strip_edge_and_right(prefix)
        right, right_has_and = _strip_edge_and_left(suffix)

        if left.strip() and right.strip() and (left_has_and or right_has_and):
            return _smart_space_trim(f"{left} {ATTENTION_AND_OPERATOR} {right}")
        if left.strip() and right.strip():
            if _need_space(left, right):
                return _smart_space_trim(left + " " + right)
            return _smart_space_trim(left + right)
        if left.strip():
            return _smart_space_trim(left)
        if right.strip():
            return _smart_space_trim(right)
        return ""

    # Непустая
    left = prefix
    mid = text
    right = suffix

    if _need_space(left, mid):
        left += " "
    if _need_space(mid, right):
        mid += " "

    return _smart_space_trim(left + mid + right)


def _cleanup_adjunct_base_prompt_text(text: str) -> str:
    cleaned = _collapse_spaces(str(text or ""))
    cleaned = re.sub(r"\s*,\s*,+\s*", ", ", cleaned)
    cleaned = re.sub(r"^\s*,\s*", "", cleaned)
    cleaned = re.sub(r"\s*,\s*$", "", cleaned)
    return _collapse_spaces(cleaned)


def _contains_top_level_multicond(prompt: str) -> bool:
    return len(_split_top_level_multicond(prompt)) > 1


def _normalize_preview_fragment(text: str) -> str:
    normalized = _collapse_spaces(str(text or "")).strip()
    if normalized == SAFE_EMPTY:
        return ""
    return normalized

# ──────────────────────────────────────────────────────────────────────────────
# Предкомпилированные regex'ы для fast-path
# ──────────────────────────────────────────────────────────────────────────────

# [ ... ] : N [reverse]  с префиксом/суффиксом
RE_BRACKET_AFTER = re.compile(
    rf'(?s)^(.*)\[(.*?)\]\s*:\s*({NUMERIC_RE}%?)\s*(?:(?P<rev>reverse)\b)?\s*(?P<post>.*)$'
)
# Явные диапазоны: "[a:b]:10 1-4,5-7 [reverse]"
RE_BRACKET_AFTER_WITH_RANGES = re.compile(rf'(?s)^\s*\[(?P<inner>.*?)\]\s*:\s*(?P<num>{NUMERIC_RE})\s+'
    r'(?P<ranges>(?:\d+%?\s*-\s*\d+%?(?:\s*,\s*)?)+)'
    r'(?:\s*(?P<rev>reverse))?\s*$'
)

# Общий префикс для reverse в постфиксе
RE_REVERSE_PREFIX = re.compile(r'^\s*reverse\b\s*')
# Reverse в КОНЦЕ постфикса: "photo [cat:dog]:10 hd reverse" → reverse в конце post.
# Используем только полное слово 'reverse' (не 'r') во избежание ложных срабатываний.
RE_REVERSE_SUFFIX = re.compile(r'(?s)(?:^|\s)\breverse\b\s*$')
LITERAL_REVERSE_TOKEN = "\uE005"  # private-use char, avoids collision with any user text

# Normalization helpers for scheduler surface syntax:
# 1) canonicalize "] : N" / "]: N" to "]:N" for parser portability.
# 2) keep literal "reverse" text after range-blocks (e.g. "... 1-3 reverse shot").
_RE_POSTFIX_STEP_COLON_SPACING = re.compile(
    rf'\]\s*:\s*(?P<num>{NUMERIC_RE})'
)
_RE_LITERAL_REVERSE_AFTER_RANGES = re.compile(
    rf'(?P<head>\]\s*:\s*{NUMERIC_RE}\s+(?:\d+%?\s*-\s*\d+%?(?:\s*,\s*)?)+)\s+reverse(?=\s+\w|,)'
)


def _find_last_top_level_colon_index(s: str) -> int:
    """Return the index of the last top-level ':' in s, or -1."""
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    last = -1
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            i += 2 if i + 1 < len(s) else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        elif ch == "[":
            depth_brack += 1
        elif ch == "]":
            depth_brack = max(0, depth_brack - 1)
        elif ch == ":" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            last = i
        i += 1
    return last


CHUNK_KEYWORD = "CHUNK"
CHUNK_PREVIEW_SEPARATOR = " BREAK "
CHUNK_CROSSATTN_KEYS = frozenset({"crossattn", "c_crossattn", "open_clip_projected"})
SDXL_SPLITTABLE_CROSS_KEYS = frozenset({"crossattn", "c_crossattn"})
CHUNK_SHARED_MODES = frozenset({"share-pooled", "share-cross"})
POOL_KEYWORD = "POOL"
POOL_PREVIEW_PREFIX = "POOL<"
_RE_CHUNK_MARKER = re.compile(rf"(?<![\w\\]){re.escape(CHUNK_KEYWORD)}(?:\s*\[[^\]]*\])?\s*\{{")
_RE_POOL_MARKER = re.compile(rf"(?<![\w\\]){re.escape(POOL_KEYWORD)}\s*\{{")
BIND_KEYWORD = "BIND"
BIND_PREVIEW_PREFIX = "BIND<"
_RE_BIND_MARKER = re.compile(rf"(?<![\w\\]){re.escape(BIND_KEYWORD)}(?:\s*\^[^\{{]*)?\s*\{{")
ASSEMBLE_KEYWORD = "ASSEMBLE"
ASSEMBLE_PREVIEW_PREFIX = "ASSEMBLE<"
_RE_ASSEMBLE_MARKER = re.compile(rf"(?<![\w\\]){re.escape(ASSEMBLE_KEYWORD)}\s*\{{")
BLEND_KEYWORD = "BLEND"
BLEND_PREVIEW_PREFIX = "BLEND<"
BLEND_MODES = frozenset({"mean", "sum"})
BACKEND_CHANNEL_TARGETS = frozenset({"both", "cross", "pooled", "enc1", "enc2"})
SDXL_ENCODER_CHANNEL_TARGETS = frozenset({"enc1", "enc2"})
SDXL_ENCODER1_CROSS_DIM = 768
SDXL_TOTAL_CROSS_DIM = 2048
_RE_BLEND_MARKER = re.compile(rf"(?<![\w\\]){re.escape(BLEND_KEYWORD)}(?:\s*\^[^\[\{{]*)?(?:\s*\[[^\]]*\])?\s*\{{")
MORPH_KEYWORD = "MORPH"
MORPH_PREVIEW_PREFIX = "MORPH<"
MORPH_CURVES = frozenset({"linear", "bezier", "catmull"})
_RE_MORPH_MARKER = re.compile(rf"(?<![\w\\]){re.escape(MORPH_KEYWORD)}(?:\s*\^[^\[\{{@]*)?(?:\s*@[^\[\{{]*)?(?:\s*\[[^\]]*\])?\s*\{{")


@dataclass(frozen=True)
class ChunkBranchSpec:
    text: str
    weight: float = 1.0


@dataclass(frozen=True)
class ChunkPromptSpec:
    prefix: str
    suffix: str
    branches: tuple[ChunkBranchSpec, ...]
    shared_channel: str = "none"


@dataclass(frozen=True)
class PoolPromptSpec:
    prefix: str
    suffix: str
    body: str
    source: str = ""


@dataclass(frozen=True)
class BindPromptSpec:
    owner: str
    attrs: str
    weight: float = 1.0
    source: str = ""


@dataclass(frozen=True)
class AssemblePromptSpec:
    prefix: str
    suffix: str
    enc1: str
    enc2: str
    pooled: str | None = None
    source: str = ""


@dataclass(frozen=True)
class BlendBranchSpec:
    text: str
    weight: float = 1.0


@dataclass(frozen=True)
class BlendPromptSpec:
    prefix: str
    suffix: str
    branches: tuple[BlendBranchSpec, ...]
    mode: str = "mean"
    intensity: float = 1.0
    channel_target: str = "both"
    source: str = ""


@dataclass(frozen=True)
class MorphPointSpec:
    text: str
    boundary: _BoundarySpec | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class MorphPromptSpec:
    prefix: str
    suffix: str
    points: tuple[MorphPointSpec, ...]
    curve: str = "linear"
    intensity: float = 1.0
    channel_target: str = "both"
    window_start: _BoundarySpec | None = None
    window_end: _BoundarySpec | None = None
    source: str = ""


@dataclass(frozen=True)
class BackendPromptState:
    chunk_spec: ChunkPromptSpec | None = None
    pool_spec: PoolPromptSpec | None = None
    bind_specs: tuple[BindPromptSpec, ...] = ()
    bind_base_prompt: str = ""
    assemble_spec: AssemblePromptSpec | None = None
    blend_spec: BlendPromptSpec | None = None
    morph_spec: MorphPromptSpec | None = None
    allow_chunk_morph_sugar: bool = False

    @property
    def backend_specs(self) -> tuple[object, ...]:
        return tuple(spec for spec in (self.chunk_spec, self.assemble_spec, self.blend_spec, self.morph_spec) if spec is not None)

    @property
    def has_backend(self) -> bool:
        return any(spec is not None for spec in (self.chunk_spec, self.assemble_spec, self.blend_spec, self.morph_spec))

    @property
    def has_pool(self) -> bool:
        return self.pool_spec is not None

    @property
    def has_bind(self) -> bool:
        return bool(self.bind_specs)

    @property
    def has_bind_backend_conflict(self) -> bool:
        return self.has_bind and self.has_backend

    @property
    def has_mixed_backends(self) -> bool:
        return len(self.backend_specs) > 1 and not self.allow_chunk_morph_sugar

    @property
    def active_morph_spec(self) -> MorphPromptSpec | None:
        return None if self.allow_chunk_morph_sugar else self.morph_spec

    @property
    def primary_token(self) -> str:
        if self.bind_specs and self.has_backend:
            return BIND_KEYWORD
        if self.chunk_spec is not None:
            return CHUNK_KEYWORD
        if self.assemble_spec is not None:
            return ASSEMBLE_KEYWORD
        if self.blend_spec is not None:
            return BLEND_KEYWORD
        return MORPH_KEYWORD


def _match_chunk_keyword_at(text: str, index: int) -> tuple[int | None, int | None, int] | None:
    if not text.startswith(CHUNK_KEYWORD, index):
        return None
    prev = text[index - 1] if index > 0 else ""
    if prev and (prev.isalnum() or prev == "_"):
        return None
    j = index + len(CHUNK_KEYWORD)
    while j < len(text) and text[j].isspace():
        j += 1
    if j < len(text) and text[j] == "[":
        mode_open = j
        bracket_depth = 1
        j += 1
        while j < len(text):
            ch = text[j]
            if ch == "\\":
                j += 2 if j + 1 < len(text) else 1
                continue
            if ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1
                if bracket_depth == 0:
                    mode_close = j
                    j += 1
                    while j < len(text) and text[j].isspace():
                        j += 1
                    if j < len(text) and text[j] == "{":
                        return mode_open, mode_close, j
                    return None
            j += 1
        raise PromptSyntaxError(
            "Unclosed CHUNK mode: expected ']'.",
            kind="invalid_chunk_mode",
            token=f"{CHUNK_KEYWORD}[",
            full=text,
        )
    if j < len(text) and text[j] == "{":
        return None, None, j
    return None


def _contains_chunk_marker(text: str) -> bool:
    if not text or CHUNK_KEYWORD not in text:
        return False
    protected, _ = _protect_escaped_literal_spans(text)
    protected = _protect_escaped_literals(protected)
    return bool(_RE_CHUNK_MARKER.search(protected))


def _contains_pool_marker(text: str) -> bool:
    if not text or POOL_KEYWORD not in text:
        return False
    protected, _ = _protect_escaped_literal_spans(text)
    protected = _protect_escaped_literals(protected)
    return bool(_RE_POOL_MARKER.search(protected))


def _contains_bind_marker(text: str) -> bool:
    if not text or BIND_KEYWORD not in text:
        return False
    protected, _ = _protect_escaped_literal_spans(text)
    protected = _protect_escaped_literals(protected)
    return bool(_RE_BIND_MARKER.search(protected))


def _contains_assemble_marker(text: str) -> bool:
    if not text or ASSEMBLE_KEYWORD not in text:
        return False
    protected, _ = _protect_escaped_literal_spans(text)
    protected = _protect_escaped_literals(protected)
    return bool(_RE_ASSEMBLE_MARKER.search(protected))


def _match_blend_keyword_at(text: str, index: int) -> tuple[int | None, int | None, int] | None:
    if not text.startswith(BLEND_KEYWORD, index):
        return None
    prev = text[index - 1] if index > 0 else ""
    if prev and (prev.isalnum() or prev == "_"):
        return None
    j = index + len(BLEND_KEYWORD)
    while j < len(text) and text[j].isspace():
        j += 1
    if j < len(text) and text[j] == "^":
        j += 1
        while j < len(text) and text[j].isspace():
            j += 1
        while j < len(text) and text[j] not in "[{":
            j += 1
        while j < len(text) and text[j].isspace():
            j += 1
    if j < len(text) and text[j] == "[":
        mode_open = j
        bracket_depth = 1
        j += 1
        while j < len(text):
            ch = text[j]
            if ch == "\\":
                j += 2 if j + 1 < len(text) else 1
                continue
            if ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1
                if bracket_depth == 0:
                    mode_close = j
                    j += 1
                    while j < len(text) and text[j].isspace():
                        j += 1
                    if j < len(text) and text[j] == "{":
                        return mode_open, mode_close, j
                    return None
            j += 1
        raise PromptSyntaxError(
            "Unclosed BLEND mode: expected ']'.",
            kind="invalid_blend_mode",
            token=f"{BLEND_KEYWORD}[",
            full=text,
        )
    if j < len(text) and text[j] == "{":
        return None, None, j
    return None


def _contains_blend_marker(text: str) -> bool:
    if not text or BLEND_KEYWORD not in text:
        return False
    protected, _ = _protect_escaped_literal_spans(text)
    protected = _protect_escaped_literals(protected)
    return bool(_RE_BLEND_MARKER.search(protected))


def _find_top_level_blend_blocks(text: str) -> list[tuple[int, int | None, int | None, int, int]]:
    blocks: list[tuple[int, int | None, int | None, int, int]] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue

        if depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            blend_match = _match_blend_keyword_at(text, i)
            if blend_match is not None:
                mode_open, mode_close, brace_open = blend_match
                brace_depth = 1
                j = brace_open + 1
                while j < n:
                    inner = text[j]
                    if inner == "\\":
                        j += 2 if j + 1 < n else 1
                        continue
                    if inner == "{":
                        brace_depth += 1
                    elif inner == "}":
                        brace_depth -= 1
                        if brace_depth == 0:
                            blocks.append((i, mode_open, mode_close, brace_open, j))
                            i = j + 1
                            break
                    j += 1
                else:
                    raise PromptSyntaxError(
                        "Unclosed BLEND block: expected '}'",
                        kind="invalid_blend_syntax",
                        token=f"{BLEND_KEYWORD}{{",
                        full=text,
                    )
                continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        i += 1

    return blocks


def _split_top_level_blend_body(body: str, *, full_text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(body)

    while i < n:
        ch = body[i]
        if ch == "\\":
            buf.append(ch)
            if i + 1 < n:
                buf.append(body[i + 1])
                i += 2
                continue
            i += 1
            continue

        if ch == "|" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            part = "".join(buf).strip()
            if not part:
                raise PromptSyntaxError(
                    "Empty BLEND branch is not allowed.",
                    kind="empty_blend_branch",
                    token="|",
                    full=full_text,
                )
            parts.append(part)
            buf.clear()
            i += 1
            continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1

        buf.append(ch)
        i += 1

    last = "".join(buf).strip()
    if not last:
        raise PromptSyntaxError(
            "Empty BLEND branch is not allowed.",
            kind="empty_blend_branch",
            token="|",
            full=full_text,
        )
    parts.append(last)
    return parts


def _parse_blend_mode(text: str, *, full_text: str) -> str:
    mode = (text or "").strip().lower() or "mean"
    if mode not in BLEND_MODES:
        raise PromptSyntaxError(
            f"Unsupported BLEND mode {mode!r}",
            kind="invalid_blend_mode",
            token=(text or BLEND_KEYWORD).strip() or BLEND_KEYWORD,
            full=full_text,
        )
    return mode


def _parse_backend_channel_target(text: str, *, full_text: str, kind: str) -> str:
    channel_target = (text or "").strip().lower() or "both"
    if channel_target not in BACKEND_CHANNEL_TARGETS:
        raise PromptSyntaxError(
            f"Unsupported {kind} channel target {channel_target!r}",
            kind=f"invalid_{kind.lower()}_channel_target",
            token=(text or kind).strip() or kind,
            full=full_text,
        )
    return channel_target


def _parse_blend_mode_and_channel_spec(text: str, *, full_text: str) -> tuple[str, str]:
    stripped = (text or "").strip()
    if not stripped:
        return "mean", "both"

    if "@" not in stripped:
        return _parse_blend_mode(stripped, full_text=full_text), "both"

    mode_text, channel_text = stripped.split("@", 1)
    mode = _parse_blend_mode(mode_text, full_text=full_text) if mode_text.strip() else "mean"
    channel_target = _parse_backend_channel_target(channel_text, full_text=full_text, kind="BLEND")
    return mode, channel_target


def _parse_blend_intensity_spec(text: str, *, full_text: str) -> float:
    stripped = (text or "").strip()
    if not stripped:
        return 1.0

    match = re.fullmatch(rf"\^\s*({NUMERIC_RE})\s*", stripped)
    if not match:
        raise PromptSyntaxError(
            f"Invalid BLEND intensity {text!r}",
            kind="invalid_blend_intensity",
            token=stripped,
            full=full_text,
        )

    intensity = float(match.group(1))
    if not math.isfinite(intensity) or intensity <= 0.0:
        raise PromptSyntaxError(
            f"Invalid BLEND intensity {text!r}",
            kind="invalid_blend_intensity",
            token=stripped,
            full=full_text,
        )
    return intensity


def _split_blend_branch_weight(text: str, *, full_text: str) -> tuple[str, float]:
    stripped = (text or "").strip()
    if not stripped:
        raise PromptSyntaxError(
            "Empty BLEND branch is not allowed.",
            kind="empty_blend_branch",
            token="|",
            full=full_text,
        )

    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    star_pos = -1
    i = 0
    n = len(stripped)
    while i < n:
        ch = stripped[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        elif ch == "*" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            star_pos = i
        i += 1

    if star_pos == -1:
        return stripped, 1.0

    weight_text = stripped[star_pos + 1 :].strip()
    if not RE_NUMERIC_FULL.fullmatch(weight_text or ""):
        raise PromptSyntaxError(
            f"Invalid BLEND branch weight {weight_text!r}",
            kind="invalid_blend_weight",
            token=weight_text or "*",
            full=full_text,
        )

    branch_text = stripped[:star_pos].rstrip()
    if not branch_text.strip():
        raise PromptSyntaxError(
            "Empty BLEND branch is not allowed.",
            kind="empty_blend_branch",
            token="*",
            full=full_text,
        )

    weight = float(weight_text)
    if weight <= 0:
        raise PromptSyntaxError(
            f"BLEND branch weight must be positive, got {weight_text!r}",
            kind="invalid_blend_weight",
            token=weight_text,
            full=full_text,
        )
    return branch_text, weight


def _extract_blend_prompt_spec(text: str) -> BlendPromptSpec | None:
    if not text or BLEND_KEYWORD not in text:
        return None

    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    blocks = _find_top_level_blend_blocks(protected)
    if not blocks:
        return None
    if len(blocks) > 1:
        raise PromptSyntaxError(
            "Only one top-level BLEND block is supported in v1.",
            kind="multiple_blend_blocks_not_supported",
            token=BLEND_KEYWORD,
            full=text,
        )

    start, mode_open, mode_close, brace_open, brace_close = blocks[0]
    body = protected[brace_open + 1 : brace_close]
    nested = _find_top_level_blend_blocks(body)
    if nested or _contains_blend_marker(body):
        raise PromptSyntaxError(
            "Nested BLEND blocks are not supported in v1.",
            kind="nested_blend_not_supported",
            token=BLEND_KEYWORD,
            full=text,
        )
    if _contains_chunk_marker(body) or _contains_morph_marker(body) or _contains_pool_marker(body) or _contains_assemble_marker(body):
        raise PromptSyntaxError(
            "BLEND branches cannot contain CHUNK, MORPH, POOL, or ASSEMBLE blocks in v1.",
            kind="nested_backend_in_blend_not_supported",
            token=(
                CHUNK_KEYWORD
                if _contains_chunk_marker(body)
                else MORPH_KEYWORD
                if _contains_morph_marker(body)
                else POOL_KEYWORD
                if _contains_pool_marker(body)
                else ASSEMBLE_KEYWORD
            ),
            full=text,
        )

    raw_parts = _split_top_level_blend_body(body, full_text=text)
    if len(raw_parts) < 2:
        raise PromptSyntaxError(
            "BLEND requires at least two branches.",
            kind="invalid_blend_syntax",
            token=BLEND_KEYWORD,
            full=text,
        )

    mode = "mean"
    channel_target = "both"
    intensity = _parse_blend_intensity_spec(
        protected[start + len(BLEND_KEYWORD) : (mode_open if mode_open is not None else brace_open)],
        full_text=text,
    )
    if mode_open is not None and mode_close is not None:
        mode, channel_target = _parse_blend_mode_and_channel_spec(protected[mode_open + 1 : mode_close], full_text=text)

    prefix = _restore_escaped_literal_source(protected[:start], span_restore)
    suffix = _restore_escaped_literal_source(protected[brace_close + 1 :], span_restore)

    branches: list[BlendBranchSpec] = []
    for raw_part in raw_parts:
        branch_text, weight = _split_blend_branch_weight(raw_part, full_text=text)
        if _contains_top_level_multicond(branch_text):
            raise PromptSyntaxError(
                "BLEND branches cannot contain top-level AND branches in v1.",
                kind="blend_inner_multicond_not_supported",
                token=BLEND_KEYWORD,
                full=text,
            )
        branch_text = _restore_escaped_literal_source(branch_text, span_restore).strip()
        if not branch_text:
            raise PromptSyntaxError(
                "Empty BLEND branch is not allowed.",
                kind="empty_blend_branch",
                token="|",
                full=text,
            )
        branches.append(BlendBranchSpec(branch_text, weight))

    return BlendPromptSpec(
        prefix=prefix,
        suffix=suffix,
        branches=tuple(branches),
        mode=mode,
        intensity=intensity,
        channel_target=channel_target,
        source=text,
    )


def _expand_blend_branch_prompt(spec: BlendPromptSpec, branch: BlendBranchSpec | str) -> str:
    branch_text = branch.text if isinstance(branch, BlendBranchSpec) else str(branch)
    return _concat_prefix_text_suffix(spec.prefix, branch_text, spec.suffix)


def _resolve_blend_mode_weights(weights: Sequence[float], mode: str, intensity: float = 1.0) -> list[float]:
    raw = [float(weight) for weight in weights]
    k = float(intensity)
    if not math.isfinite(k) or k <= 0.0:
        raise PromptSyntaxError(
            f"Invalid BLEND intensity {intensity!r}",
            kind="invalid_blend_intensity",
            token=str(intensity),
            full=str(intensity),
        )
    if abs(k - 1.0) <= 1e-8:
        if mode == "sum":
            return raw
        total = float(sum(raw))
        if abs(total) <= 1e-8:
            return [0.0 for _ in raw]
        return [weight / total for weight in raw]

    shaped = [weight ** k for weight in raw]
    shaped_total = float(sum(shaped))
    if abs(shaped_total) <= 1e-8:
        return [0.0 for _ in raw]
    normalized = [weight / shaped_total for weight in shaped]
    if mode == "sum":
        original_total = float(sum(raw))
        return [weight * original_total for weight in normalized]
    return normalized


def _build_blend_preview_text(active_texts: Sequence[str], weights: Sequence[float]) -> str:
    nonzero = [
        (normalized_text, weight)
        for text, weight in zip(active_texts, weights)
        for normalized_text in [_normalize_preview_fragment(text)]
        if normalized_text and abs(float(weight)) > 1e-8
    ]
    if not nonzero:
        return SAFE_EMPTY
    parts = [f"{text}*{_format_interp_weight(float(weight))}" for text, weight in nonzero]
    return f"{BLEND_PREVIEW_PREFIX}{' + '.join(parts)}>"


def _blend_preview_prefix(channel_target: str) -> str:
    if channel_target == "both":
        return BLEND_PREVIEW_PREFIX
    return f"{BLEND_KEYWORD}@{channel_target}<"


def _build_blend_preview_text_with_target(active_texts: Sequence[str], weights: Sequence[float], channel_target: str) -> str:
    nonzero = [
        (normalized_text, weight)
        for text, weight in zip(active_texts, weights)
        for normalized_text in [_normalize_preview_fragment(text)]
        if normalized_text and abs(float(weight)) > 1e-8
    ]
    if not nonzero:
        return SAFE_EMPTY
    parts = [f"{text}*{_format_interp_weight(float(weight))}" for text, weight in nonzero]
    return f"{_blend_preview_prefix(channel_target)}{' + '.join(parts)}>"


def _build_blend_text_schedule_from_spec(
    spec: BlendPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
) -> list[list[int, str]]:
    branch_prompts = [_expand_blend_branch_prompt(spec, branch) for branch in spec.branches]
    branch_schedules = [
        get_schedule(branch_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
        for branch_prompt in branch_prompts
    ]
    effective_weights = _resolve_blend_mode_weights([branch.weight for branch in spec.branches], spec.mode, spec.intensity)

    if not use_scheduling:
        final_step = int(steps)
        active_texts = [_select_text_from_schedule(schedule, final_step) or SAFE_EMPTY for schedule in branch_schedules]
        return [[int(steps), _build_blend_preview_text_with_target(active_texts, effective_weights, spec.channel_target)]]

    boundaries = _collect_schedule_boundaries(branch_schedules, steps)
    out: list[list[int, str]] = []
    previous_key = None
    for end_at_step in boundaries:
        active_texts = [
            _select_text_from_schedule(schedule, end_at_step) or SAFE_EMPTY
            for schedule in branch_schedules
        ]
        preview = _build_blend_preview_text_with_target(active_texts, effective_weights, spec.channel_target)
        key = (tuple(active_texts), tuple(round(float(weight), 8) for weight in effective_weights))
        if out and previous_key == key:
            out[-1][0] = int(end_at_step)
        else:
            out.append([int(end_at_step), preview])
            previous_key = key
    return out or [[int(steps), SAFE_EMPTY]]

def _contains_morph_marker(text: str) -> bool:
    if not text or MORPH_KEYWORD not in text:
        return False
    protected, _ = _protect_escaped_literal_spans(text)
    protected = _protect_escaped_literals(protected)
    return bool(_RE_MORPH_MARKER.search(protected))


def _find_top_level_chunk_blocks(text: str) -> list[tuple[int, int | None, int | None, int, int]]:
    blocks: list[tuple[int, int | None, int | None, int, int]] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue

        if depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            chunk_match = _match_chunk_keyword_at(text, i)
            if chunk_match is not None:
                mode_open, mode_close, brace_open = chunk_match
                brace_depth = 1
                j = brace_open + 1
                while j < n:
                    inner = text[j]
                    if inner == "\\":
                        j += 2 if j + 1 < n else 1
                        continue
                    if inner == "{":
                        brace_depth += 1
                    elif inner == "}":
                        brace_depth -= 1
                        if brace_depth == 0:
                            blocks.append((i, mode_open, mode_close, brace_open, j))
                            i = j + 1
                            break
                    j += 1
                else:
                    raise PromptSyntaxError(
                        "Unclosed CHUNK block: expected '}'",
                        kind="invalid_chunk_syntax",
                        token=f"{CHUNK_KEYWORD}{{",
                        full=text,
                    )
                continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        i += 1

    return blocks


def _split_top_level_chunk_body(body: str, *, full_text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(body)

    while i < n:
        ch = body[i]
        if ch == "\\":
            buf.append(ch)
            if i + 1 < n:
                buf.append(body[i + 1])
                i += 2
                continue
            i += 1
            continue

        if ch == "|" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            part = "".join(buf).strip()
            if not part:
                raise PromptSyntaxError(
                    "Empty CHUNK branch is not allowed.",
                    kind="empty_chunk_branch",
                    token="|",
                    full=full_text,
                )
            parts.append(part)
            buf.clear()
            i += 1
            continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1

        buf.append(ch)
        i += 1

    last = "".join(buf).strip()
    if not last:
        raise PromptSyntaxError(
            "Empty CHUNK branch is not allowed.",
            kind="empty_chunk_branch",
            token="|",
            full=full_text,
        )
    parts.append(last)
    return parts


def _split_chunk_branch_weight(text: str, *, full_text: str) -> tuple[str, float]:
    stripped = (text or "").strip()
    if not stripped:
        raise PromptSyntaxError(
            "Empty CHUNK branch is not allowed.",
            kind="empty_chunk_branch",
            token="|",
            full=full_text,
        )

    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    star_pos = -1
    i = 0
    n = len(stripped)
    while i < n:
        ch = stripped[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        elif ch == "*" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            star_pos = i
        i += 1

    if star_pos == -1:
        return stripped, 1.0

    weight_text = stripped[star_pos + 1 :].strip()
    if not RE_NUMERIC_FULL.fullmatch(weight_text or ""):
        raise PromptSyntaxError(
            f"Invalid CHUNK branch weight {weight_text!r}",
            kind="invalid_chunk_weight",
            token=weight_text or "*",
            full=full_text,
        )

    branch_text = stripped[:star_pos].rstrip()
    if not branch_text.strip():
        raise PromptSyntaxError(
            "Empty CHUNK branch is not allowed.",
            kind="empty_chunk_branch",
            token="*",
            full=full_text,
        )

    weight = float(weight_text)
    if weight <= 0:
        raise PromptSyntaxError(
            f"CHUNK branch weight must be positive, got {weight_text!r}",
            kind="invalid_chunk_weight",
            token=weight_text,
            full=full_text,
        )
    return branch_text, weight


def _parse_chunk_shared_mode(text: str, *, full_text: str) -> str:
    mode = (text or "").strip().lower()
    if not mode:
        raise PromptSyntaxError(
            "Unsupported CHUNK mode ''.",
            kind="invalid_chunk_mode",
            token="[]",
            full=full_text,
        )
    if mode not in CHUNK_SHARED_MODES:
        raise PromptSyntaxError(
            f"Unsupported CHUNK mode {mode!r}",
            kind="invalid_chunk_mode",
            token=mode,
            full=full_text,
        )
    if mode == "share-pooled":
        return "pooled"
    if mode == "share-cross":
        return "cross"
    return "none"


def _find_top_level_pool_blocks(text: str) -> list[tuple[int, int, int]]:
    blocks: list[tuple[int, int, int]] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue

        if depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            if text.startswith(POOL_KEYWORD, i):
                prev = text[i - 1] if i > 0 else ""
                if not prev or (not prev.isalnum() and prev != "_"):
                    j = i + len(POOL_KEYWORD)
                    while j < n and text[j].isspace():
                        j += 1
                    if j < n and text[j] == "{":
                        brace_open = j
                        brace_depth = 1
                        k = brace_open + 1
                        while k < n:
                            inner = text[k]
                            if inner == "\\":
                                k += 2 if k + 1 < n else 1
                                continue
                            if inner == "{":
                                brace_depth += 1
                            elif inner == "}":
                                brace_depth -= 1
                                if brace_depth == 0:
                                    blocks.append((i, brace_open, k))
                                    i = k + 1
                                    break
                            k += 1
                        else:
                            raise PromptSyntaxError(
                                "Unclosed POOL block: expected '}'",
                                kind="invalid_pool_syntax",
                                token=f"{POOL_KEYWORD}{{",
                                full=text,
                            )
                        continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        i += 1

    return blocks


def _extract_pool_prompt_spec(text: str) -> PoolPromptSpec | None:
    if not text or POOL_KEYWORD not in text:
        return None

    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    blocks = _find_top_level_pool_blocks(protected)
    if not blocks:
        return None
    if len(blocks) > 1:
        raise PromptSyntaxError(
            "Only one top-level POOL block is supported in v1.",
            kind="multiple_pool_blocks_not_supported",
            token=POOL_KEYWORD,
            full=text,
        )

    start, brace_open, brace_close = blocks[0]
    body = protected[brace_open + 1 : brace_close]
    nested = _find_top_level_pool_blocks(body)
    if nested or _contains_pool_marker(body):
        raise PromptSyntaxError(
            "Nested POOL blocks are not supported in v1.",
            kind="nested_pool_not_supported",
            token=POOL_KEYWORD,
            full=text,
        )
    if (
        _contains_chunk_marker(body)
        or _contains_blend_marker(body)
        or _contains_morph_marker(body)
        or _contains_assemble_marker(body)
        or _contains_bind_marker(body)
    ):
        raise PromptSyntaxError(
            "POOL body must contain regular prompt grammar only in v1.",
            kind="nested_backend_in_pool_not_supported",
            token=POOL_KEYWORD,
            full=text,
        )

    prefix = _restore_escaped_literal_source(protected[:start], span_restore)
    suffix = _restore_escaped_literal_source(protected[brace_close + 1 :], span_restore)
    body = _restore_escaped_literal_source(body, span_restore).strip()
    if not body:
        raise PromptSyntaxError(
            "POOL body cannot be empty.",
            kind="invalid_pool_syntax",
            token=POOL_KEYWORD,
            full=text,
        )

    return PoolPromptSpec(prefix=prefix, suffix=suffix, body=body, source=text)


def _build_pool_base_prompt(spec: PoolPromptSpec) -> str:
    text = _cleanup_adjunct_base_prompt_text(_concat_prefix_text_suffix(spec.prefix, "", spec.suffix))
    return text if str(text).strip() else SAFE_EMPTY


def _build_pool_preview_text(base_text: str, pool_text: str) -> str:
    base = str(base_text or "").strip()
    pooled = str(pool_text or "").strip() or SAFE_EMPTY
    if not base:
        return f"{POOL_PREVIEW_PREFIX}{pooled}>"
    return f"{base} {POOL_PREVIEW_PREFIX}{pooled}>"


def _match_bind_keyword_at(text: str, index: int) -> tuple[str | None, int] | None:
    if not text.startswith(BIND_KEYWORD, index):
        return None
    prev = text[index - 1] if index > 0 else ""
    if prev and (prev.isalnum() or prev == "_"):
        return None

    j = index + len(BIND_KEYWORD)
    while j < len(text) and text[j].isspace():
        j += 1

    weight_raw = None
    if j < len(text) and text[j] == "^":
        j += 1
        weight_start = j
        while j < len(text) and text[j] != "{":
            j += 1
        weight_raw = text[weight_start:j].strip()

    while j < len(text) and text[j].isspace():
        j += 1
    if j < len(text) and text[j] == "{":
        return weight_raw, j
    return None


def _parse_bind_weight(raw: str | None, *, full_text: str) -> float:
    if raw is None:
        return 1.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0
    if value <= 0.0:
        raise PromptSyntaxError(
            "BIND weight must be a positive number.",
            kind="invalid_bind_weight",
            token=raw or BIND_KEYWORD,
            full=full_text,
        )
    return value


def _find_top_level_bind_blocks(text: str) -> list[tuple[int, int, int, str | None]]:
    blocks: list[tuple[int, int, int, str | None]] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue

        if depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            matched = _match_bind_keyword_at(text, i)
            if matched is not None:
                weight_raw, brace_open = matched
                brace_depth = 1
                k = brace_open + 1
                while k < n:
                    inner = text[k]
                    if inner == "\\":
                        k += 2 if k + 1 < n else 1
                        continue
                    if inner == "{":
                        brace_depth += 1
                    elif inner == "}":
                        brace_depth -= 1
                        if brace_depth == 0:
                            blocks.append((i, brace_open, k, weight_raw))
                            i = k + 1
                            break
                    k += 1
                else:
                    raise PromptSyntaxError(
                        "Unclosed BIND block: expected '}'",
                        kind="invalid_bind_syntax",
                        token=f"{BIND_KEYWORD}{{",
                        full=text,
                    )
                continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        i += 1

    return blocks


def _split_bind_owner_attrs(body: str, *, full_text: str) -> tuple[str, str]:
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        elif (
            ch == "="
            and i + 1 < n
            and body[i + 1] == ">"
            and depth_paren == 0
            and depth_brace == 0
            and depth_brack == 0
        ):
            owner = body[:i].strip()
            attrs = body[i + 2 :].strip()
            if owner and attrs:
                return owner, attrs
            break
        i += 1
    raise PromptSyntaxError(
        "BIND blocks must use 'owner => attrs' syntax.",
        kind="invalid_bind_syntax",
        token=BIND_KEYWORD,
        full=full_text,
    )


def _extract_bind_prompt_bundle(text: str) -> tuple[tuple[BindPromptSpec, ...], str]:
    if not text or BIND_KEYWORD not in text:
        return (), text

    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    blocks = _find_top_level_bind_blocks(protected)
    if not blocks:
        return (), text

    specs: list[BindPromptSpec] = []
    base_parts: list[str] = []
    last = 0
    for start, brace_open, brace_close, weight_raw in blocks:
        body = protected[brace_open + 1 : brace_close]
        nested = _find_top_level_bind_blocks(body)
        if nested or _contains_bind_marker(body):
            raise PromptSyntaxError(
                "Nested BIND blocks are not supported in v1.",
                kind="nested_bind_not_supported",
                token=BIND_KEYWORD,
                full=text,
            )
        if (
            _contains_chunk_marker(body)
            or _contains_blend_marker(body)
            or _contains_morph_marker(body)
            or _contains_pool_marker(body)
            or _contains_assemble_marker(body)
        ):
            raise PromptSyntaxError(
                "BIND fields must contain regular prompt grammar only in v1.",
                kind="nested_backend_in_bind_not_supported",
                token=BIND_KEYWORD,
                full=text,
            )

        owner_body, attrs_body = _split_bind_owner_attrs(body, full_text=text)
        if _contains_top_level_multicond(owner_body) or _contains_top_level_multicond(attrs_body):
            raise PromptSyntaxError(
                "BIND owner and attrs cannot contain top-level AND branches in v1.",
                kind="bind_inner_multicond_not_supported",
                token=BIND_KEYWORD,
                full=text,
            )
        owner = _restore_escaped_literal_source(owner_body, span_restore).strip()
        attrs = _restore_escaped_literal_source(attrs_body, span_restore).strip()
        if not owner or not attrs:
            raise PromptSyntaxError(
                "BIND requires non-empty owner and attrs sections.",
                kind="invalid_bind_syntax",
                token=BIND_KEYWORD,
                full=text,
            )
        specs.append(
            BindPromptSpec(
                owner=owner,
                attrs=attrs,
                weight=_parse_bind_weight(weight_raw, full_text=text),
                source=text,
            )
        )
        base_parts.append(protected[last:start])
        last = brace_close + 1

    base_parts.append(protected[last:])
    base_prompt = _cleanup_adjunct_base_prompt_text(
        _restore_escaped_literal_source("".join(base_parts), span_restore)
    )
    if not str(base_prompt).strip():
        raise PromptSyntaxError(
            "BIND requires a non-empty base prompt outside the BIND blocks.",
            kind="bind_requires_base_prompt",
            token=BIND_KEYWORD,
            full=text,
        )
    return tuple(specs), base_prompt


def _build_bind_branch_prompt(spec: BindPromptSpec) -> str:
    return _collapse_spaces(f"{spec.owner}, {spec.attrs}")


def _compose_bind_branch_prompt(owner_text: str, attrs_text: str) -> str:
    return _collapse_spaces(f"{owner_text}, {attrs_text}")


def _build_bind_component_text_schedule(
    text: str,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    *,
    strict: bool,
) -> list[list[int, str]]:
    if strict:
        return _strict_schedule_preview(text, steps, seed)
    return get_schedule(text, steps, use_scheduling, seed, use_visitor=use_visitor)


def _build_bind_branch_timeline_entries(
    spec: BindPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    *,
    strict: bool,
) -> list[tuple[int, str, float]]:
    owner_schedule = _build_bind_component_text_schedule(
        spec.owner,
        steps,
        use_scheduling,
        seed,
        use_visitor,
        strict=strict,
    )
    attrs_schedule = _build_bind_component_text_schedule(
        spec.attrs,
        steps,
        use_scheduling,
        seed,
        use_visitor,
        strict=strict,
    )
    if not owner_schedule or not attrs_schedule:
        raise ValueError("Empty schedule for BIND owner or attrs")

    boundaries = _collect_schedule_boundaries([owner_schedule, attrs_schedule], steps)
    out: list[tuple[int, str, float]] = []
    previous_key = None
    for end_at_step in boundaries:
        owner_text = _select_text_from_schedule(owner_schedule, end_at_step) or SAFE_EMPTY
        attrs_text = _select_text_from_schedule(attrs_schedule, end_at_step) or SAFE_EMPTY
        is_active = bool(str(owner_text).strip()) and bool(str(attrs_text).strip())
        bind_text = _compose_bind_branch_prompt(owner_text, attrs_text) if is_active else SAFE_EMPTY
        bind_weight = float(spec.weight) if is_active else 0.0
        key = (bind_text, round(bind_weight, 8))
        if out and previous_key == key:
            out[-1] = (int(end_at_step), out[-1][1], out[-1][2])
        else:
            out.append((int(end_at_step), bind_text, bind_weight))
            previous_key = key
    return out or [(int(steps), SAFE_EMPTY, 0.0)]


def _pick_bind_timeline_entry(
    timeline: Sequence[tuple[int, str, float]],
    current_step: int,
) -> tuple[int, str, float]:
    if not timeline:
        raise ValueError("Empty BIND timeline")
    for entry in timeline:
        if current_step <= int(entry[0]):
            return entry
    return timeline[-1]


def _build_bind_preview_text(active_text: str) -> str:
    return f"{BIND_PREVIEW_PREFIX}{str(active_text or '').strip() or SAFE_EMPTY}>"


def _find_top_level_assemble_blocks(text: str) -> list[tuple[int, int, int]]:
    blocks: list[tuple[int, int, int]] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue

        if depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            if text.startswith(ASSEMBLE_KEYWORD, i):
                prev = text[i - 1] if i > 0 else ""
                if not prev or (not prev.isalnum() and prev != "_"):
                    j = i + len(ASSEMBLE_KEYWORD)
                    while j < n and text[j].isspace():
                        j += 1
                    if j < n and text[j] == "{":
                        brace_open = j
                        brace_depth = 1
                        k = brace_open + 1
                        while k < n:
                            inner = text[k]
                            if inner == "\\":
                                k += 2 if k + 1 < n else 1
                                continue
                            if inner == "{":
                                brace_depth += 1
                            elif inner == "}":
                                brace_depth -= 1
                                if brace_depth == 0:
                                    blocks.append((i, brace_open, k))
                                    i = k + 1
                                    break
                            k += 1
                        else:
                            raise PromptSyntaxError(
                                "Unclosed ASSEMBLE block: expected '}'",
                                kind="invalid_assemble_syntax",
                                token=f"{ASSEMBLE_KEYWORD}{{",
                                full=text,
                            )
                        continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        i += 1

    return blocks


def _split_top_level_assemble_fields(body: str, *, full_text: str) -> list[str]:
    parts: list[str] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    last = 0
    i = 0
    n = len(body)

    while i < n:
        ch = body[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        elif ch == ";" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            parts.append(body[last:i])
            last = i + 1
        i += 1

    parts.append(body[last:])
    cleaned = [part.strip() for part in parts if part.strip()]
    if not cleaned:
        raise PromptSyntaxError(
            "ASSEMBLE must define enc1 and enc2 fields.",
            kind="invalid_assemble_syntax",
            token=ASSEMBLE_KEYWORD,
            full=full_text,
        )
    return cleaned


def _split_assemble_field(part: str, *, full_text: str) -> tuple[str, str]:
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(part)
    while i < n:
        ch = part[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        elif ch == "=" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            name = part[:i].strip().lower()
            value = part[i + 1 :].strip()
            if not name or not value:
                break
            return name, value
        i += 1
    raise PromptSyntaxError(
        "ASSEMBLE fields must use 'name=value' syntax.",
        kind="invalid_assemble_syntax",
        token=part.strip() or ASSEMBLE_KEYWORD,
        full=full_text,
    )


def _extract_assemble_prompt_spec(text: str) -> AssemblePromptSpec | None:
    if not text or ASSEMBLE_KEYWORD not in text:
        return None

    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    blocks = _find_top_level_assemble_blocks(protected)
    if not blocks:
        return None
    if len(blocks) > 1:
        raise PromptSyntaxError(
            "Only one top-level ASSEMBLE block is supported in v1.",
            kind="multiple_assemble_blocks_not_supported",
            token=ASSEMBLE_KEYWORD,
            full=text,
        )

    start, brace_open, brace_close = blocks[0]
    body = protected[brace_open + 1 : brace_close]
    nested = _find_top_level_assemble_blocks(body)
    if nested or _contains_assemble_marker(body):
        raise PromptSyntaxError(
            "Nested ASSEMBLE blocks are not supported in v1.",
            kind="nested_assemble_not_supported",
            token=ASSEMBLE_KEYWORD,
            full=text,
        )
    if _contains_chunk_marker(body) or _contains_blend_marker(body) or _contains_morph_marker(body) or _contains_pool_marker(body):
        raise PromptSyntaxError(
            "ASSEMBLE fields must contain regular prompt grammar only in v1.",
            kind="nested_backend_in_assemble_not_supported",
            token=ASSEMBLE_KEYWORD,
            full=text,
        )

    fields = {}
    for part in _split_top_level_assemble_fields(body, full_text=text):
        name, value = _split_assemble_field(part, full_text=text)
        if name not in {"enc1", "enc2", "pooled"}:
            raise PromptSyntaxError(
                f"Unsupported ASSEMBLE field {name!r}",
                kind="invalid_assemble_field",
                token=name,
                full=text,
            )
        if name in fields:
            raise PromptSyntaxError(
                f"Duplicate ASSEMBLE field {name!r}",
                kind="duplicate_assemble_field",
                token=name,
                full=text,
            )
        if _contains_top_level_multicond(value):
            raise PromptSyntaxError(
                "ASSEMBLE fields cannot contain top-level AND branches in v1.",
                kind="assemble_inner_multicond_not_supported",
                token=ASSEMBLE_KEYWORD,
                full=text,
            )
        fields[name] = _restore_escaped_literal_source(value, span_restore).strip()

    if not fields.get("enc1") or not fields.get("enc2"):
        raise PromptSyntaxError(
            "ASSEMBLE requires non-empty enc1 and enc2 fields.",
            kind="invalid_assemble_syntax",
            token=ASSEMBLE_KEYWORD,
            full=text,
        )

    prefix = _restore_escaped_literal_source(protected[:start], span_restore)
    suffix = _restore_escaped_literal_source(protected[brace_close + 1 :], span_restore)
    pooled = fields.get("pooled") or None
    return AssemblePromptSpec(
        prefix=prefix,
        suffix=suffix,
        enc1=fields["enc1"],
        enc2=fields["enc2"],
        pooled=pooled,
        source=text,
    )


def _expand_assemble_section_prompt(spec: AssemblePromptSpec, body: str) -> str:
    return _concat_prefix_text_suffix(spec.prefix, body, spec.suffix)


def _build_assemble_preview_text(enc1_text: str, enc2_text: str, pooled_text: str) -> str:
    return f"{ASSEMBLE_PREVIEW_PREFIX}enc1={enc1_text} | enc2={enc2_text} | pooled={pooled_text}>"


def _build_bind_text_schedule_from_specs(
    base_prompt: str,
    bind_specs: Sequence[BindPromptSpec],
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    *,
    strict: bool,
) -> list[list[int, str]]:
    if strict:
        base_schedule = _strict_schedule_preview(base_prompt, steps, seed)
        bind_timelines = [
            _build_bind_branch_timeline_entries(
                spec,
                steps,
                use_scheduling,
                seed,
                use_visitor,
                strict=True,
            )
            for spec in bind_specs
        ]
    else:
        base_schedule = get_schedule(base_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
        bind_timelines = [
            _build_bind_branch_timeline_entries(
                spec,
                steps,
                use_scheduling,
                seed,
                use_visitor,
                strict=False,
            )
            for spec in bind_specs
        ]

    bind_text_schedules = [
        [[int(end_at_step), text] for end_at_step, text, _weight in timeline]
        for timeline in bind_timelines
    ]

    if not base_schedule or any(not schedule for schedule in bind_text_schedules):
        raise ValueError("Empty schedule for BIND prompt")

    boundaries = _collect_schedule_boundaries([base_schedule, *bind_text_schedules], steps)
    out: list[list[int, str]] = []
    previous_key = None
    for end_at_step in boundaries:
        base_text = _select_text_from_schedule(base_schedule, end_at_step) or SAFE_EMPTY
        active_texts = [base_text]
        weights = [1.0]
        bind_keys: list[tuple[str, float]] = []
        for timeline in bind_timelines:
            _bind_end_at_step, bind_text, bind_weight = _pick_bind_timeline_entry(timeline, end_at_step)
            bind_keys.append((bind_text, bind_weight))
            if bind_weight <= 1e-8:
                continue
            active_texts.append(_build_bind_preview_text(bind_text))
            weights.append(bind_weight)
        key = (base_text, tuple((text, round(weight, 8)) for text, weight in bind_keys))
        preview = _build_multicond_preview_text(active_texts, weights)
        if out and previous_key == key:
            out[-1][0] = int(end_at_step)
        else:
            out.append([int(end_at_step), preview])
            previous_key = key
    return out or [[int(steps), SAFE_EMPTY]]


def _extract_chunk_prompt_spec(text: str) -> ChunkPromptSpec | None:
    if not text or CHUNK_KEYWORD not in text:
        return None

    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    blocks = _find_top_level_chunk_blocks(protected)
    if not blocks:
        return None
    if len(blocks) > 1:
        raise PromptSyntaxError(
            "Only one top-level CHUNK block is supported in v1.",
            kind="multiple_chunk_blocks_not_supported",
            token=CHUNK_KEYWORD,
            full=text,
        )

    start, mode_open, mode_close, brace_open, brace_close = blocks[0]
    body = protected[brace_open + 1 : brace_close]
    nested = _find_top_level_chunk_blocks(body)
    if nested or re.search(rf"(?<![\w\\]){re.escape(CHUNK_KEYWORD)}\s*\{{", body):
        raise PromptSyntaxError(
            "Nested CHUNK blocks are not supported in v1.",
            kind="nested_chunk_not_supported",
            token=CHUNK_KEYWORD,
            full=text,
        )
    if _contains_blend_marker(body) or _contains_pool_marker(body) or _contains_assemble_marker(body):
        raise PromptSyntaxError(
            "CHUNK branches cannot contain BLEND, POOL, or ASSEMBLE blocks in v1.",
            kind="nested_backend_in_chunk_not_supported",
            token=(
                BLEND_KEYWORD
                if _contains_blend_marker(body)
                else POOL_KEYWORD
                if _contains_pool_marker(body)
                else ASSEMBLE_KEYWORD
            ),
            full=text,
        )

    prefix = _restore_escaped_literal_source(protected[:start], span_restore)
    suffix = _restore_escaped_literal_source(protected[brace_close + 1 :], span_restore)
    raw_parts = _split_top_level_chunk_body(body, full_text=text)
    shared_channel = "none"
    if mode_open is not None and mode_close is not None:
        shared_channel = _parse_chunk_shared_mode(protected[mode_open + 1 : mode_close], full_text=text)

    branches: list[ChunkBranchSpec] = []
    for raw_part in raw_parts:
        branch_text, weight = _split_chunk_branch_weight(raw_part, full_text=text)
        if _contains_top_level_multicond(branch_text):
            raise PromptSyntaxError(
                "CHUNK branches cannot contain top-level AND branches in v1.",
                kind="chunk_inner_multicond_not_supported",
                token=CHUNK_KEYWORD,
                full=text,
            )
        branch_text = _restore_escaped_literal_source(branch_text, span_restore).strip()
        if not branch_text:
            raise PromptSyntaxError(
                "Empty CHUNK branch is not allowed.",
                kind="empty_chunk_branch",
                token="|",
                full=text,
            )
        branches.append(ChunkBranchSpec(branch_text, weight))

    return ChunkPromptSpec(prefix=prefix, suffix=suffix, branches=tuple(branches), shared_channel=shared_channel)


def _expand_chunk_branch_prompt(spec: ChunkPromptSpec, branch: ChunkBranchSpec | str) -> str:
    branch_text = branch.text if isinstance(branch, ChunkBranchSpec) else str(branch)
    return _concat_prefix_text_suffix(spec.prefix, branch_text, spec.suffix)


def _build_chunk_shared_anchor_prompt(spec: ChunkPromptSpec) -> str:
    anchor = _concat_prefix_text_suffix(spec.prefix, "", spec.suffix)
    return anchor if str(anchor).strip() else SAFE_EMPTY


def _select_text_from_schedule(schedule: Sequence[Sequence[int | str]], step: int) -> str:
    if not schedule:
        return ""
    for end_at_step, text in schedule:
        if step <= int(end_at_step):
            return str(text)
    return str(schedule[-1][1])


def _pick_text_schedule_index(schedule: Sequence[Sequence[int | str]], step: int) -> int:
    if not schedule:
        raise ValueError("Empty text schedule")
    for current, (end_at_step, _text) in enumerate(schedule):
        if step <= int(end_at_step):
            return current
    return len(schedule) - 1


def _collect_schedule_boundaries(schedules: Sequence[Sequence[Sequence[int | str]]], steps: int) -> list[int]:
    boundaries = sorted({int(end) for schedule in schedules for end, _ in schedule})
    if not boundaries or boundaries[-1] < int(steps):
        boundaries.append(int(steps))
    return boundaries


def _build_chunk_text_schedule_from_spec(
    spec: ChunkPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
) -> list[list[int, str]]:
    branch_schedules = [
        get_schedule(_expand_chunk_branch_prompt(spec, branch), steps, use_scheduling, seed, use_visitor=use_visitor)
        for branch in spec.branches
    ]

    boundaries = _collect_schedule_boundaries(branch_schedules, steps)
    out: list[list[int, str]] = []
    for end_at_step in boundaries:
        active_texts = [
            _normalize_preview_fragment(_select_text_from_schedule(schedule, end_at_step) or SAFE_EMPTY)
            for schedule in branch_schedules
        ]
        preview_parts = [text for text in active_texts if text]
        preview = CHUNK_PREVIEW_SEPARATOR.join(preview_parts) if preview_parts else SAFE_EMPTY
        if out and out[-1][1] == preview:
            out[-1][0] = int(end_at_step)
        else:
            out.append([int(end_at_step), preview])
    return out or [[int(steps), SAFE_EMPTY]]


def _match_morph_keyword_at(text: str, index: int) -> tuple[int | None, int | None, int] | None:
    if not text.startswith(MORPH_KEYWORD, index):
        return None
    prev = text[index - 1] if index > 0 else ""
    if prev and (prev.isalnum() or prev == "_"):
        return None
    j = index + len(MORPH_KEYWORD)
    while j < len(text) and text[j].isspace():
        j += 1
    if j < len(text) and text[j] == "^":
        j += 1
        while j < len(text) and text[j].isspace():
            j += 1
        while j < len(text) and text[j] not in "@[{":
            j += 1
        while j < len(text) and text[j].isspace():
            j += 1
    if j < len(text) and text[j] == "@":
        j += 1
        while j < len(text) and text[j].isspace():
            j += 1
        while j < len(text) and text[j] not in "[{":
            j += 1
        while j < len(text) and text[j].isspace():
            j += 1
    if j < len(text) and text[j] == "[":
        window_open = j
        bracket_depth = 1
        j += 1
        while j < len(text):
            ch = text[j]
            if ch == "\\":
                j += 2 if j + 1 < len(text) else 1
                continue
            if ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1
                if bracket_depth == 0:
                    window_close = j
                    j += 1
                    while j < len(text) and text[j].isspace():
                        j += 1
                    if j < len(text) and text[j] == "{":
                        return window_open, window_close, j
                    return None
            j += 1
        raise PromptSyntaxError(
            "Unclosed MORPH window: expected ']'.",
            kind="invalid_morph_window",
            token=f"{MORPH_KEYWORD}[",
            full=text,
        )
    if j < len(text) and text[j] == "{":
        return None, None, j
    return None


def _parse_morph_intensity_spec(text: str, *, full_text: str) -> float:
    stripped = (text or "").strip()
    if not stripped:
        return 1.0

    match = re.fullmatch(rf"\^\s*({NUMERIC_RE})\s*", stripped)
    if not match:
        raise PromptSyntaxError(
            f"Invalid MORPH intensity {text!r}",
            kind="invalid_morph_intensity",
            token=stripped,
            full=full_text,
        )

    intensity = float(match.group(1))
    if not math.isfinite(intensity) or intensity <= 0.0:
        raise PromptSyntaxError(
            f"Invalid MORPH intensity {text!r}",
            kind="invalid_morph_intensity",
            token=stripped,
            full=full_text,
        )
    return intensity


def _parse_morph_channel_target_spec(text: str, *, full_text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return "both"

    match = re.fullmatch(r"@\s*([A-Za-z0-9_-]+)\s*", stripped)
    if not match:
        raise PromptSyntaxError(
            f"Invalid MORPH channel target {text!r}",
            kind="invalid_morph_channel_target",
            token=stripped,
            full=full_text,
        )

    channel_target = match.group(1).strip().lower()
    if channel_target not in BACKEND_CHANNEL_TARGETS:
        raise PromptSyntaxError(
            f"Invalid MORPH channel target {text!r}",
            kind="invalid_morph_channel_target",
            token=stripped,
            full=full_text,
        )
    return channel_target


def _parse_morph_header_spec(text: str, *, full_text: str) -> tuple[float, str]:
    stripped = (text or "").strip()
    if not stripped:
        return 1.0, "both"

    intensity_text = stripped
    channel_text = ""
    if "@" in stripped:
        at_pos = stripped.find("@")
        intensity_text = stripped[:at_pos].rstrip()
        channel_text = stripped[at_pos:].strip()

    intensity = _parse_morph_intensity_spec(intensity_text, full_text=full_text)
    channel_target = _parse_morph_channel_target_spec(channel_text, full_text=full_text)
    return intensity, channel_target


def _find_top_level_morph_blocks(text: str) -> list[tuple[int, int | None, int | None, int, int]]:
    blocks: list[tuple[int, int | None, int | None, int, int]] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue

        if depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            morph_match = _match_morph_keyword_at(text, i)
            if morph_match is not None:
                window_open, window_close, brace_open = morph_match
                brace_depth = 1
                j = brace_open + 1
                while j < n:
                    inner = text[j]
                    if inner == "\\":
                        j += 2 if j + 1 < n else 1
                        continue
                    if inner == "{":
                        brace_depth += 1
                    elif inner == "}":
                        brace_depth -= 1
                        if brace_depth == 0:
                            blocks.append((i, window_open, window_close, brace_open, j))
                            i = j + 1
                            break
                    j += 1
                else:
                    raise PromptSyntaxError(
                        "Unclosed MORPH block: expected '}'",
                        kind="invalid_morph_syntax",
                        token=f"{MORPH_KEYWORD}{{",
                        full=text,
                    )
                continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        i += 1

    return blocks


def _split_top_level_morph_curve(body: str, *, full_text: str) -> tuple[str, str]:
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    curve_pos = -1
    i = 0
    n = len(body)

    while i < n:
        ch = body[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        elif ch == "~" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            curve_pos = i
        i += 1

    if curve_pos < 0:
        return body, "linear"

    curve = body[curve_pos + 1 :].strip().lower()
    if curve not in MORPH_CURVES:
        raise PromptSyntaxError(
            f"Unsupported MORPH curve {curve!r}",
            kind="invalid_morph_curve",
            token=curve or "~",
            full=full_text,
        )
    return body[:curve_pos].rstrip(), curve


def _split_top_level_morph_points(body: str, *, full_text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(body)

    while i < n:
        ch = body[i]
        if ch == "\\":
            buf.append(ch)
            if i + 1 < n:
                buf.append(body[i + 1])
                i += 2
                continue
            i += 1
            continue

        if (
            ch == "=" and i + 1 < n and body[i + 1] == ">"
            and depth_paren == 0 and depth_brace == 0 and depth_brack == 0
        ):
            part = "".join(buf).strip()
            if not part:
                raise PromptSyntaxError(
                    "MORPH control prompts cannot be empty.",
                    kind="invalid_morph_syntax",
                    token="=>",
                    full=full_text,
                )
            parts.append(part)
            buf.clear()
            i += 2
            continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1

        buf.append(ch)
        i += 1

    last = "".join(buf).strip()
    if not last:
        raise PromptSyntaxError(
            "MORPH control prompts cannot be empty.",
            kind="invalid_morph_syntax",
            token="=>",
            full=full_text,
        )
    parts.append(last)
    return parts


def _parse_morph_boundary_spec(text: str, *, full_text: str) -> _BoundarySpec:
    match = re.fullmatch(rf"\s*({NUMERIC_RE})(%?)\s*", text or "")
    if not match:
        raise PromptSyntaxError(
            f"Invalid MORPH boundary {text!r}",
            kind="invalid_morph_boundary",
            token=(text or "@").strip() or "@",
            full=full_text,
        )
    return _make_boundary_spec(float(match.group(1)), is_percent=bool(match.group(2)))


def _parse_morph_window_spec(text: str, *, full_text: str) -> tuple[_BoundarySpec, _BoundarySpec]:
    match = re.fullmatch(r"\s*(.+?)\s*-\s*(.+?)\s*", text or "")
    if not match:
        raise PromptSyntaxError(
            f"Invalid MORPH activation window {text!r}",
            kind="invalid_morph_window",
            token=(text or "[]").strip() or "[]",
            full=full_text,
        )
    try:
        start = _parse_morph_boundary_spec(match.group(1), full_text=full_text)
        end = _parse_morph_boundary_spec(match.group(2), full_text=full_text)
    except PromptSyntaxError as exc:
        raise PromptSyntaxError(
            f"Invalid MORPH activation window {text!r}",
            kind="invalid_morph_window",
            token=(text or "[]").strip() or "[]",
            full=full_text,
        ) from exc
    return start, end


def _split_morph_point_boundary(text: str, *, full_text: str) -> tuple[str, _BoundarySpec | None]:
    stripped = (text or "").strip()
    if not stripped:
        raise PromptSyntaxError(
            "MORPH control prompts cannot be empty.",
            kind="invalid_morph_syntax",
            token="=>",
            full=full_text,
        )

    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    at_pos = -1
    i = 0
    n = len(stripped)
    while i < n:
        ch = stripped[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        elif ch == "@" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            at_pos = i
        i += 1

    if at_pos < 0:
        return stripped, None

    point_text = stripped[:at_pos].rstrip()
    boundary_text = stripped[at_pos + 1 :].strip()
    if not point_text:
        raise PromptSyntaxError(
            "MORPH control prompts cannot be empty.",
            kind="invalid_morph_syntax",
            token="@",
            full=full_text,
        )
    return point_text, _parse_morph_boundary_spec(boundary_text, full_text=full_text)


def _split_morph_point_weight(text: str, *, full_text: str) -> tuple[str, float]:
    stripped = (text or "").strip()
    if not stripped:
        raise PromptSyntaxError(
            "MORPH control prompts cannot be empty.",
            kind="invalid_morph_syntax",
            token="=>",
            full=full_text,
        )

    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    star_pos = -1
    i = 0
    n = len(stripped)
    while i < n:
        ch = stripped[i]
        if ch == "\\":
            i += 2 if i + 1 < n else 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        elif ch == "*" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            star_pos = i
        i += 1

    if star_pos == -1:
        return stripped, 1.0

    weight_text = stripped[star_pos + 1 :].strip()
    if not RE_NUMERIC_FULL.fullmatch(weight_text or ""):
        raise PromptSyntaxError(
            f"Invalid MORPH point weight {weight_text!r}",
            kind="invalid_morph_point_weight",
            token=weight_text or "*",
            full=full_text,
        )

    point_text = stripped[:star_pos].rstrip()
    if not point_text.strip():
        raise PromptSyntaxError(
            "MORPH control prompts cannot be empty.",
            kind="invalid_morph_syntax",
            token="*",
            full=full_text,
        )

    weight = float(weight_text)
    if not math.isfinite(weight) or weight <= 0.0:
        raise PromptSyntaxError(
            f"MORPH point weight must be positive, got {weight_text!r}",
            kind="invalid_morph_point_weight",
            token=weight_text,
            full=full_text,
        )
    return point_text, weight


def _extract_morph_prompt_spec(text: str) -> MorphPromptSpec | None:
    if not text or MORPH_KEYWORD not in text:
        return None

    protected, span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    blocks = _find_top_level_morph_blocks(protected)
    if not blocks:
        return None
    if len(blocks) > 1:
        raise PromptSyntaxError(
            "Only one top-level MORPH block is supported in v1.",
            kind="multiple_morph_blocks_not_supported",
            token=MORPH_KEYWORD,
            full=text,
        )

    start, window_open, window_close, brace_open, brace_close = blocks[0]
    body = protected[brace_open + 1 : brace_close]
    nested = _find_top_level_morph_blocks(body)
    if nested or _contains_morph_marker(body):
        raise PromptSyntaxError(
            "Nested MORPH blocks are not supported in v1.",
            kind="nested_morph_not_supported",
            token=MORPH_KEYWORD,
            full=text,
        )
    if _contains_pool_marker(body) or _contains_assemble_marker(body):
        raise PromptSyntaxError(
            "MORPH control prompts cannot contain POOL or ASSEMBLE blocks in v1.",
            kind="nested_backend_in_morph_not_supported",
            token=POOL_KEYWORD if _contains_pool_marker(body) else ASSEMBLE_KEYWORD,
            full=text,
        )

    body, curve = _split_top_level_morph_curve(body, full_text=text)
    raw_points = _split_top_level_morph_points(body, full_text=text)
    if len(raw_points) < 2:
        raise PromptSyntaxError(
            "MORPH requires at least two control prompts.",
            kind="invalid_morph_syntax",
            token=MORPH_KEYWORD,
            full=text,
        )

    prefix = _restore_escaped_literal_source(protected[:start], span_restore)
    suffix = _restore_escaped_literal_source(protected[brace_close + 1 :], span_restore)

    intensity, channel_target = _parse_morph_header_spec(
        protected[start + len(MORPH_KEYWORD) : (window_open if window_open is not None else brace_open)],
        full_text=text,
    )

    window_start = None
    window_end = None
    if window_open is not None and window_close is not None:
        window_start, window_end = _parse_morph_window_spec(
            protected[window_open + 1 : window_close],
            full_text=text,
        )

    points: list[MorphPointSpec] = []
    for i, raw_point in enumerate(raw_points):
        point_text, boundary = _split_morph_point_boundary(raw_point, full_text=text)
        point_text, point_weight = _split_morph_point_weight(point_text, full_text=text)
        if _contains_top_level_multicond(point_text):
            raise PromptSyntaxError(
                "MORPH control prompts cannot contain top-level AND branches in v1.",
                kind="morph_inner_multicond_not_supported",
                token=MORPH_KEYWORD,
                full=text,
            )
        point_text = _restore_escaped_literal_source(point_text, span_restore).strip()
        if not point_text:
            raise PromptSyntaxError(
                "MORPH control prompts cannot be empty.",
                kind="invalid_morph_syntax",
                token="=>",
                full=text,
            )
        if window_start is not None or window_end is not None:
            if boundary is not None:
                raise PromptSyntaxError(
                    "MORPH window syntax cannot be combined with per-point '@boundary' markers in v1.",
                    kind="morph_window_with_point_boundaries_not_supported",
                    token="@",
                    full=text,
                )
        else:
            if i == 0 and boundary is not None:
                raise PromptSyntaxError(
                    "The first MORPH control prompt cannot have an explicit boundary in v1.",
                    kind="invalid_morph_boundary",
                    token="@",
                    full=text,
                )
            if i not in (0, len(raw_points) - 1) and boundary is None:
                raise PromptSyntaxError(
                    "Intermediate MORPH control prompts must declare a boundary with '@'.",
                    kind="invalid_morph_boundary",
                    token=point_text,
                    full=text,
                )
        points.append(MorphPointSpec(point_text, boundary, point_weight))

    return MorphPromptSpec(
        prefix=prefix,
        suffix=suffix,
        points=tuple(points),
        curve=curve,
        intensity=intensity,
        channel_target=channel_target,
        window_start=window_start,
        window_end=window_end,
        source=text,
    )


def _is_chunk_morph_sugar(text: str) -> bool:
    if not text or CHUNK_KEYWORD not in text or MORPH_KEYWORD not in text:
        return False

    protected, _span_restore = _protect_escaped_literal_spans_for_source(text)
    protected = _protect_escaped_literals(protected)
    chunk_blocks = _find_top_level_chunk_blocks(protected)
    morph_blocks = _find_top_level_morph_blocks(protected)
    blend_blocks = _find_top_level_blend_blocks(protected)

    if len(chunk_blocks) != 1 or len(morph_blocks) != 1 or blend_blocks:
        return False

    chunk_start, _chunk_mode_open, _chunk_mode_close, chunk_brace_open, chunk_brace_close = chunk_blocks[0]
    morph_start, _window_open, _window_close, _morph_brace_open, morph_brace_close = morph_blocks[0]

    non_overlapping = (
        chunk_start < morph_start and chunk_brace_close < morph_start
    ) or (
        morph_start < chunk_start and morph_brace_close < chunk_start
    )
    if not non_overlapping:
        return False

    body = protected[chunk_brace_open + 1 : chunk_brace_close]
    if _contains_morph_marker(body) or _contains_blend_marker(body):
        return False

    return True


def _extract_backend_prompt_state(text: str) -> BackendPromptState:
    bind_marker = _contains_bind_marker(text)
    bind_specs, bind_base_prompt = _extract_bind_prompt_bundle(text)
    if bind_specs and len(_split_top_level_multicond(_protect_escaped_literals(_protect_escaped_literal_spans(bind_base_prompt)[0]))) > 1:
        raise PromptSyntaxError(
            "BIND cannot be combined with top-level AND in v1.",
            kind="bind_with_and_not_supported",
            token=BIND_KEYWORD,
            full=text,
        )
    post_bind_text = bind_base_prompt if bind_specs else text
    pool_marker = _contains_pool_marker(post_bind_text)
    pool_spec = _extract_pool_prompt_spec(post_bind_text)
    primary_text = _build_pool_base_prompt(pool_spec) if pool_spec is not None else post_bind_text
    chunk_spec = _extract_chunk_prompt_spec(primary_text)
    assemble_spec = _extract_assemble_prompt_spec(primary_text)
    blend_spec = _extract_blend_prompt_spec(primary_text)
    morph_spec = _extract_morph_prompt_spec(primary_text)
    if pool_spec is None and pool_marker:
        raise PromptSyntaxError(
            "POOL blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_pool_context",
            token=POOL_KEYWORD,
            full=text,
        )
    if not bind_specs and bind_marker:
        raise PromptSyntaxError(
            "BIND blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_bind_context",
            token=BIND_KEYWORD,
            full=text,
        )
    allow_chunk_morph_sugar = (
        chunk_spec is not None
        and assemble_spec is None
        and blend_spec is None
        and morph_spec is not None
        and _is_chunk_morph_sugar(primary_text)
    )
    return BackendPromptState(
        chunk_spec=chunk_spec,
        pool_spec=pool_spec,
        bind_specs=bind_specs,
        bind_base_prompt=bind_base_prompt,
        assemble_spec=assemble_spec,
        blend_spec=blend_spec,
        morph_spec=morph_spec,
        allow_chunk_morph_sugar=allow_chunk_morph_sugar,
    )


def _raise_mixed_backend_prompt_error(text: str, state: BackendPromptState) -> None:
    raise PromptSyntaxError(
        "CHUNK, ASSEMBLE, BLEND, and MORPH cannot be combined in the same prompt branch in v1.",
        kind="mixed_backend_blocks_not_supported",
        token=state.primary_token,
        full=text,
    )


def _raise_bind_backend_prompt_error(text: str) -> None:
    raise PromptSyntaxError(
        "BIND can only be combined with plain prompt text or POOL in v1.",
        kind="bind_with_backend_not_supported",
        token=BIND_KEYWORD,
        full=text,
    )


def _raise_unsupported_backend_context_error(text: str) -> None:
    if _contains_chunk_marker(text):
        raise PromptSyntaxError(
            "CHUNK blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_chunk_context",
            token=CHUNK_KEYWORD,
            full=text,
        )
    if _contains_blend_marker(text):
        raise PromptSyntaxError(
            "BLEND blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_blend_context",
            token=BLEND_KEYWORD,
            full=text,
        )
    if _contains_assemble_marker(text):
        raise PromptSyntaxError(
            "ASSEMBLE blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_assemble_context",
            token=ASSEMBLE_KEYWORD,
            full=text,
        )
    if _contains_bind_marker(text):
        raise PromptSyntaxError(
            "BIND blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_bind_context",
            token=BIND_KEYWORD,
            full=text,
        )
    if _contains_morph_marker(text):
        raise PromptSyntaxError(
            "MORPH blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_morph_context",
            token=MORPH_KEYWORD,
            full=text,
        )
    if _contains_pool_marker(text):
        raise PromptSyntaxError(
            "POOL blocks must appear at the top level of a prompt branch in v1.",
            kind="unsupported_pool_context",
            token=POOL_KEYWORD,
            full=text,
        )


def _expand_morph_point_prompt(spec: MorphPromptSpec, point: MorphPointSpec | str) -> str:
    point_text = point.text if isinstance(point, MorphPointSpec) else str(point)
    return _concat_prefix_text_suffix(spec.prefix, point_text, spec.suffix)


def _build_morph_inactive_text(spec: MorphPromptSpec) -> str:
    text = _concat_prefix_text_suffix(spec.prefix, "", spec.suffix)
    return text if str(text).strip() else SAFE_EMPTY


def _resolve_morph_window_steps(spec: MorphPromptSpec, steps: int) -> tuple[int, int] | None:
    if spec.window_start is None and spec.window_end is None:
        return None
    if spec.window_start is None or spec.window_end is None:
        raise PromptSyntaxError(
            "MORPH activation window must define both start and end boundaries.",
            kind="invalid_morph_window",
            token=MORPH_KEYWORD,
            full=spec.source or None,
        )
    start = _to_end_step(spec.window_start, steps)
    end = _to_end_step(spec.window_end, steps)
    if end <= start:
        raise PromptSyntaxError(
            "MORPH activation window end must be greater than the start.",
            kind="invalid_morph_window",
            token=MORPH_KEYWORD,
            full=spec.source or None,
        )
    return int(start), int(end)


def _resolve_morph_positions(spec: MorphPromptSpec, steps: int) -> list[int]:
    window_steps = _resolve_morph_window_steps(spec, steps)
    if window_steps is not None:
        start, end = window_steps
        span = int(end) - int(start)
        count = len(spec.points)
        if span < count - 1:
            raise PromptSyntaxError(
                "MORPH activation window is too narrow for the number of control prompts.",
                kind="invalid_morph_window",
                token=MORPH_KEYWORD,
                full=spec.source or None,
            )
        positions = [int(start)]
        for i in range(1, count - 1):
            remaining_slots = (count - 1) - i
            raw = int(start) + (span * i) / (count - 1)
            pos = int(start) + int(math.floor(raw - int(start) + 0.5))
            pos = max(positions[-1] + 1, pos)
            pos = min(pos, int(end) - remaining_slots)
            if pos <= positions[-1]:
                raise PromptSyntaxError(
                    "MORPH activation window is too narrow for the number of control prompts.",
                    kind="invalid_morph_window",
                    token=MORPH_KEYWORD,
                    full=spec.source or None,
                )
            positions.append(int(pos))
        positions.append(int(end))
        return positions

    positions: list[int] = [1]
    for i, point in enumerate(spec.points[1:], start=1):
        if point.boundary is None:
            pos = int(steps)
        else:
            pos = _to_end_step(point.boundary, steps)
        pos = max(1, int(pos))
        if pos <= positions[-1]:
            raise PromptSyntaxError(
                "MORPH boundaries must be strictly increasing.",
                kind="invalid_morph_boundary",
                token="@",
                full=_expand_morph_point_prompt(spec, point),
            )
        positions.append(pos)
    return positions


def _compute_morph_scaled_t(positions: Sequence[int], step: int) -> float:
    if not positions or len(positions) == 1:
        return 0.0
    if step <= positions[0]:
        return 0.0
    if step >= positions[-1]:
        return 1.0

    for idx in range(len(positions) - 1):
        left = positions[idx]
        right = positions[idx + 1]
        if step <= right:
            local = 0.0 if right == left else (step - left) / max(1, right - left)
            return (idx + local) / (len(positions) - 1)
    return 1.0


def _shape_morph_progress_t(scaled_t: float, intensity: float) -> float:
    t = max(0.0, min(1.0, float(scaled_t)))
    k = float(intensity)
    if not math.isfinite(k) or k <= 0.0:
        raise PromptSyntaxError(
            f"Invalid MORPH intensity {intensity!r}",
            kind="invalid_morph_intensity",
            token=str(intensity),
            full=str(intensity),
        )
    if t <= 0.0 or t >= 1.0 or abs(k - 1.0) <= 1e-8:
        return t

    left = t ** k
    right = (1.0 - t) ** k
    denom = left + right
    if denom <= 0.0:
        return t
    return left / denom


def _compute_morph_curve_weights(
    count: int,
    positions: Sequence[int],
    step: int,
    curve: str,
    intensity: float = 1.0,
) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [1.0]

    scaled_t = _compute_morph_scaled_t(positions, step)
    scaled_t = _shape_morph_progress_t(scaled_t, intensity)
    if curve == "linear":
        if scaled_t <= 0.0:
            return [1.0] + [0.0] * (count - 1)
        if scaled_t >= 1.0:
            return [0.0] * (count - 1) + [1.0]
        segment_pos = scaled_t * (count - 1)
        left = min(int(segment_pos), count - 2)
        alpha = segment_pos - left
        weights = [0.0] * count
        weights[left] = 1.0 - alpha
        weights[left + 1] = alpha
        return weights

    if curve == "bezier":
        weights = []
        degree = count - 1
        for i in range(count):
            weights.append(math.comb(degree, i) * ((1.0 - scaled_t) ** (degree - i)) * (scaled_t ** i))
        return weights

    if curve == "catmull":
        if scaled_t <= 0.0:
            return [1.0] + [0.0] * (count - 1)
        if scaled_t >= 1.0:
            return [0.0] * (count - 1) + [1.0]
        segment_pos = scaled_t * (count - 1)
        i = min(int(segment_pos), count - 2)
        u = segment_pos - i
        idx0 = max(i - 1, 0)
        idx1 = i
        idx2 = i + 1
        idx3 = min(i + 2, count - 1)
        w0 = -0.5 * u * u * u + u * u - 0.5 * u
        w1 = 1.5 * u * u * u - 2.5 * u * u + 1.0
        w2 = -1.5 * u * u * u + 2.0 * u * u + 0.5 * u
        w3 = 0.5 * u * u * u - 0.5 * u * u
        weights = [0.0] * count
        weights[idx0] += w0
        weights[idx1] += w1
        weights[idx2] += w2
        weights[idx3] += w3
        return weights

    raise PromptSyntaxError(
        f"Unsupported MORPH curve {curve!r}",
        kind="invalid_morph_curve",
        token=curve,
        full=curve,
    )


def _resolve_morph_point_weights(points: Sequence[MorphPointSpec], weights: Sequence[float]) -> list[float]:
    base_weights = [float(weight) for weight in weights]
    if not points:
        return base_weights

    point_weights = [float(getattr(point, "weight", 1.0)) for point in points]
    if all(abs(weight - 1.0) <= 1e-8 for weight in point_weights):
        return base_weights

    effective_weights = [base * point_weight for base, point_weight in zip(base_weights, point_weights)]
    total = float(sum(effective_weights)) if effective_weights else 0.0
    if math.isfinite(total) and abs(total) > 1e-8:
        return [weight / total for weight in effective_weights]
    return effective_weights


def _apply_condition_channel_target(base_cond, target_cond, channel_target: str):
    if channel_target == "both" or not isinstance(base_cond, dict) or not isinstance(target_cond, dict):
        return target_cond

    merged = {}
    all_keys = set(base_cond.keys()) | set(target_cond.keys())
    for key in all_keys:
        if channel_target in SDXL_ENCODER_CHANNEL_TARGETS:
            if key in SDXL_SPLITTABLE_CROSS_KEYS:
                if key in target_cond and key in base_cond:
                    merged[key] = _apply_sdxl_encoder_channel_target_to_value(
                        base_cond[key],
                        target_cond[key],
                        channel_target,
                    )
                elif key in target_cond:
                    merged[key] = target_cond[key]
                elif key in base_cond:
                    merged[key] = base_cond[key]
                continue

            if key in base_cond:
                merged[key] = base_cond[key]
            elif key in target_cond:
                merged[key] = target_cond[key]
            continue

        use_target = (
            key in CHUNK_CROSSATTN_KEYS
            if channel_target == "cross"
            else key not in CHUNK_CROSSATTN_KEYS
        )
        if use_target:
            if key in target_cond:
                merged[key] = target_cond[key]
            elif key in base_cond:
                merged[key] = base_cond[key]
        else:
            if key in base_cond:
                merged[key] = base_cond[key]
            elif key in target_cond:
                merged[key] = target_cond[key]
    return merged


def _apply_pool_prompt_conditioning(base_cond, pool_cond):
    if not isinstance(base_cond, dict) or not isinstance(pool_cond, dict):
        return base_cond
    return _apply_condition_channel_target(base_cond, pool_cond, "pooled")


def _morph_preview_prefix(channel_target: str) -> str:
    if channel_target == "both":
        return MORPH_PREVIEW_PREFIX
    return f"{MORPH_KEYWORD}@{channel_target}<"


def _build_morph_preview_text(active_texts: Sequence[str], weights: Sequence[float], channel_target: str = "both") -> str:
    nonzero = [
        (normalized_text, weight)
        for text, weight in zip(active_texts, weights)
        for normalized_text in [_normalize_preview_fragment(text)]
        if normalized_text and abs(float(weight)) > 1e-8
    ]
    if not nonzero:
        return SAFE_EMPTY
    if len(nonzero) == 1 and abs(float(nonzero[0][1]) - 1.0) <= 1e-8:
        return str(nonzero[0][0])
    parts = [f"{text}*{_format_interp_weight(float(weight))}" for text, weight in nonzero]
    return f"{_morph_preview_prefix(channel_target)}{' + '.join(parts)}>"


def _build_morph_text_schedule_from_spec(
    spec: MorphPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
) -> list[list[int, str]]:
    point_schedules = [
        get_schedule(_expand_morph_point_prompt(spec, point), steps, use_scheduling, seed, use_visitor=use_visitor)
        for point in spec.points
    ]
    positions = _resolve_morph_positions(spec, steps)
    window_steps = _resolve_morph_window_steps(spec, steps)
    inactive_text = _build_morph_inactive_text(spec)

    if not use_scheduling:
        final_step = int(steps)
        if window_steps is not None and not (window_steps[0] <= final_step <= window_steps[1]):
            return [[int(steps), inactive_text]]
        active_texts = [_select_text_from_schedule(schedule, final_step) or SAFE_EMPTY for schedule in point_schedules]
        weights = _resolve_morph_point_weights(
            spec.points,
            _compute_morph_curve_weights(len(spec.points), positions, final_step, spec.curve, spec.intensity),
        )
        return [[int(steps), _build_morph_preview_text(active_texts, weights, spec.channel_target)]]

    out: list[list[int, str]] = []
    previous_key = None
    for step in range(1, int(steps) + 1):
        if window_steps is not None and not (window_steps[0] <= step <= window_steps[1]):
            preview = inactive_text
            key = ("inactive", preview)
            if out and previous_key == key:
                out[-1][0] = int(step)
            else:
                out.append([int(step), preview])
                previous_key = key
            continue
        active_texts = [_select_text_from_schedule(schedule, step) or SAFE_EMPTY for schedule in point_schedules]
        weights = _resolve_morph_point_weights(
            spec.points,
            _compute_morph_curve_weights(len(spec.points), positions, step, spec.curve, spec.intensity),
        )
        preview = _build_morph_preview_text(active_texts, weights, spec.channel_target)
        key = (tuple(active_texts), tuple(round(float(weight), 8) for weight in weights))
        if out and previous_key == key:
            out[-1][0] = int(step)
        else:
            out.append([int(step), preview])
            previous_key = key
    return out or [[int(steps), SAFE_EMPTY]]


def _build_pool_text_schedule_from_spec(
    spec: PoolPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    *,
    strict: bool,
) -> list[list[int, str]]:
    base_prompt = _build_pool_base_prompt(spec)
    if strict:
        base_schedule = _strict_schedule_preview(base_prompt, steps, seed)
        pool_schedule = _strict_schedule_preview(spec.body, steps, seed)
    else:
        base_schedule = get_schedule(base_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
        pool_schedule = get_schedule(spec.body, steps, use_scheduling, seed, use_visitor=use_visitor)

    boundaries = _collect_schedule_boundaries([base_schedule, pool_schedule], steps)
    out: list[list[int, str]] = []
    previous_key = None
    for end_at_step in boundaries:
        base_text = _select_text_from_schedule(base_schedule, end_at_step) or SAFE_EMPTY
        pool_text = _select_text_from_schedule(pool_schedule, end_at_step) or SAFE_EMPTY
        preview = _build_pool_preview_text(base_text, pool_text)
        key = (base_text, pool_text)
        if out and previous_key == key:
            out[-1][0] = int(end_at_step)
        else:
            out.append([int(end_at_step), preview])
            previous_key = key
    return out or [[int(steps), SAFE_EMPTY]]


def _build_assemble_text_schedule_from_spec(
    spec: AssemblePromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    *,
    strict: bool,
) -> list[list[int, str]]:
    enc1_prompt = _expand_assemble_section_prompt(spec, spec.enc1)
    enc2_prompt = _expand_assemble_section_prompt(spec, spec.enc2)
    pooled_prompt = _expand_assemble_section_prompt(spec, spec.pooled) if spec.pooled is not None else enc2_prompt

    if strict:
        enc1_schedule = _strict_schedule_preview(enc1_prompt, steps, seed)
        enc2_schedule = _strict_schedule_preview(enc2_prompt, steps, seed)
        pooled_schedule = _strict_schedule_preview(pooled_prompt, steps, seed)
    else:
        enc1_schedule = get_schedule(enc1_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
        enc2_schedule = get_schedule(enc2_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
        pooled_schedule = get_schedule(pooled_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)

    boundaries = _collect_schedule_boundaries([enc1_schedule, enc2_schedule, pooled_schedule], steps)
    out: list[list[int, str]] = []
    previous_key = None
    for end_at_step in boundaries:
        enc1_text = _select_text_from_schedule(enc1_schedule, end_at_step) or SAFE_EMPTY
        enc2_text = _select_text_from_schedule(enc2_schedule, end_at_step) or SAFE_EMPTY
        pooled_text = _select_text_from_schedule(pooled_schedule, end_at_step) or SAFE_EMPTY
        preview = _build_assemble_preview_text(enc1_text, enc2_text, pooled_text)
        key = (enc1_text, enc2_text, pooled_text)
        if out and previous_key == key:
            out[-1][0] = int(end_at_step)
        else:
            out.append([int(end_at_step), preview])
            previous_key = key
    return out or [[int(steps), SAFE_EMPTY]]


def _tensor_ndim(x) -> int:
    shape = getattr(x, "shape", None)
    if shape is None:
        return 0
    try:
        return len(shape)
    except TypeError:
        return 0


def _sdxl_cross_split_axis(x):
    shape = getattr(x, "shape", None)
    if shape is None:
        return None
    try:
        if int(shape[-1]) != SDXL_TOTAL_CROSS_DIM:
            return None
    except Exception:
        return None
    ndim = _tensor_ndim(x)
    if ndim <= 0:
        return None
    return ndim - 1


def _apply_sdxl_encoder_channel_target_to_value(base_value, target_value, channel_target: str):
    axis = _sdxl_cross_split_axis(base_value)
    target_axis = _sdxl_cross_split_axis(target_value)
    if axis is None or target_axis is None or axis != target_axis:
        return target_value

    ndim = _tensor_ndim(base_value)
    if ndim != _tensor_ndim(target_value):
        return target_value

    head = [slice(None)] * ndim
    tail = [slice(None)] * ndim
    head[axis] = slice(None, SDXL_ENCODER1_CROSS_DIM)
    tail[axis] = slice(SDXL_ENCODER1_CROSS_DIM, None)

    try:
        base_head = base_value[tuple(head)]
        base_tail = base_value[tuple(tail)]
        target_head = target_value[tuple(head)]
        target_tail = target_value[tuple(tail)]
        parts = [target_head, base_tail] if channel_target == "enc1" else [base_head, target_tail]
        return _ensure_torch().cat(parts, dim=axis)
    except Exception:
        # Fall back to whole-cross routing on non-SDXL or non-sliceable tensor types.
        return target_value


def _weighted_average_condition_values(values, weights):
    total = float(sum(weights)) if weights else 0.0
    if total == 0.0:
        total = 1.0

    acc = None
    for value, weight in zip(values, weights):
        weighted = value * float(weight)
        acc = weighted if acc is None else acc + weighted
    return acc / total


def _merge_chunk_condition_values(cond_list, weights):
    if not cond_list:
        raise ValueError("Empty CHUNK conditioning branch list")

    first = cond_list[0]
    if isinstance(first, dict):
        _torch = _ensure_torch()
        merged = {}
        for key in first.keys():
            values = [cond[key] for cond in cond_list if key in cond]
            if not values:
                continue

            if key in CHUNK_CROSSATTN_KEYS:
                weighted = [value * float(weight) if float(weight) != 1.0 else value for value, weight in zip(values, weights)]
                dim = 1 if _tensor_ndim(weighted[0]) >= 3 else 0
                merged[key] = _torch.cat(weighted, dim=dim)
                continue

            ndim = _tensor_ndim(values[0])
            if ndim <= 0:
                merged[key] = values[0]
            else:
                merged[key] = _weighted_average_condition_values(values, weights)
        return merged

    _torch = _ensure_torch()
    weighted = [value * float(weight) if float(weight) != 1.0 else value for value, weight in zip(cond_list, weights)]
    dim = 1 if _tensor_ndim(weighted[0]) >= 3 else 0
    return _torch.cat(weighted, dim=dim)


def _align_condition_values_for_blend(values):
    if not values:
        return []
    if any(getattr(value, "shape", None) is None for value in values):
        return list(values)

    counts = []
    for value in values:
        axis = 1 if _tensor_ndim(value) >= 3 else 0
        try:
            counts.append(int(value.shape[axis]))
        except Exception:
            return list(values)

    target = max(counts)
    if all(count == target for count in counts):
        return list(values)

    _torch = _ensure_torch()
    aligned = []
    for value, count in zip(values, counts):
        axis = 1 if _tensor_ndim(value) >= 3 else 0
        if count >= target:
            aligned.append(value)
            continue
        try:
            pad_count = target - count
            if axis == 0:
                tail = value[-1:]
                repeats = [pad_count] + [1] * max(0, len(value.shape) - 1)
                tail_repeated = tail.repeat(repeats)
                aligned.append(_torch.cat([value, tail_repeated], dim=0))
            else:
                tail = value[:, -1:, ...]
                repeats = [1, pad_count] + [1] * max(0, len(value.shape) - 2)
                tail_repeated = tail.repeat(repeats)
                aligned.append(_torch.cat([value, tail_repeated], dim=1))
        except Exception:
            return list(values)
    return aligned


def _weighted_sum_condition_values(values, weights):
    acc = None
    for value, weight in zip(values, weights):
        weight_f = float(weight)
        if abs(weight_f) <= 1e-8:
            continue
        weighted = value if abs(weight_f - 1.0) <= 1e-8 else value * weight_f
        acc = weighted if acc is None else acc + weighted
    return acc if acc is not None else values[0] * 0.0


def _blend_morph_condition_values(cond_list, weights):
    if not cond_list:
        raise ValueError("Empty MORPH conditioning point list")

    first = cond_list[0]
    if isinstance(first, dict):
        merged = {}
        for key in first.keys():
            keyed = [(cond[key], weight) for cond, weight in zip(cond_list, weights) if key in cond]
            if not keyed:
                continue
            values = [value for value, _weight in keyed]
            value_weights = [weight for _value, weight in keyed]
            if key in CHUNK_CROSSATTN_KEYS:
                values = _align_condition_values_for_blend(values)
            merged[key] = _weighted_sum_condition_values(values, value_weights)
        return merged

    aligned = _align_condition_values_for_blend(cond_list)
    return _weighted_sum_condition_values(aligned, weights)


def _conditioning_shape_from_cond(cond):
    if isinstance(cond, dict):
        any_val = cond.get("crossattn")
        if any_val is None and cond:
            any_val = next(iter(cond.values()))
        return getattr(any_val, "shape", None) or (0,)
    return getattr(cond, "shape", None) or (0,)


def _build_plain_prompt_conditioning_schedule(
    model,
    prompt: str,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    prompt_schedule = get_schedule(prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
    if not prompt_schedule:
        raise ValueError(f"Empty schedule for prompt '{prompt}'")

    texts = SdConditioning([x[1] for x in prompt_schedule], copy_from=copy_from)
    conds = model.get_learned_conditioning(texts)
    cond_schedule = []
    for i, (end_at_step, _) in enumerate(prompt_schedule):
        if isinstance(conds, dict):
            cond = {k: v[i] for k, v in conds.items()}
        else:
            cond = conds[i]
        cond_schedule.append(ScheduledPromptConditioning(int(end_at_step), cond))
    return cond_schedule


def _build_pool_prompt_conditioning_schedule(
    model,
    spec: PoolPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    base_schedule = _build_prompt_conditioning_schedule(
        model,
        _build_pool_base_prompt(spec),
        steps,
        use_scheduling,
        seed,
        use_visitor,
        copy_from,
    )
    pool_schedule = _build_plain_prompt_conditioning_schedule(
        model,
        spec.body,
        steps,
        use_scheduling,
        seed,
        use_visitor,
        copy_from,
    )
    if not base_schedule or not pool_schedule:
        raise ValueError("Empty schedule for POOL prompt")

    boundaries = _collect_schedule_boundaries([base_schedule, pool_schedule], steps)
    out: list[ScheduledPromptConditioning] = []
    previous_key = None
    for end_at_step in boundaries:
        base_index = _pick_schedule_entry_index(base_schedule, end_at_step)
        pool_index = _pick_schedule_entry_index(pool_schedule, end_at_step)
        base_cond = base_schedule[base_index].cond
        pool_cond = pool_schedule[pool_index].cond
        key = (base_index, pool_index)
        if out and previous_key == key:
            out[-1] = ScheduledPromptConditioning(int(end_at_step), out[-1].cond)
            continue
        merged = _apply_pool_prompt_conditioning(base_cond, pool_cond)
        out.append(ScheduledPromptConditioning(int(end_at_step), merged))
        previous_key = key

    return out


def _assemble_condition_values(enc1_cond, enc2_cond, pooled_cond):
    if not isinstance(enc1_cond, dict) or not isinstance(enc2_cond, dict) or not isinstance(pooled_cond, dict):
        raise ValueError("ASSEMBLE requires SDXL-style dict conditioning values.")

    has_split_cross = any(
        key in enc1_cond
        and key in enc2_cond
        and _sdxl_cross_split_axis(enc1_cond[key]) is not None
        and _sdxl_cross_split_axis(enc2_cond[key]) is not None
        for key in SDXL_SPLITTABLE_CROSS_KEYS
    )
    has_pooled = any(key not in CHUNK_CROSSATTN_KEYS for key in pooled_cond.keys())
    if not has_split_cross or not has_pooled:
        raise ValueError(
            "ASSEMBLE requires SDXL-style dict conditioning with split cross-attention and pooled/global keys."
        )

    merged = _apply_condition_channel_target(enc1_cond, enc2_cond, "enc2")
    merged = _apply_condition_channel_target(merged, pooled_cond, "pooled")
    return merged


def _build_assemble_prompt_conditioning_schedule(
    model,
    spec: AssemblePromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    enc1_prompt = _expand_assemble_section_prompt(spec, spec.enc1)
    enc2_prompt = _expand_assemble_section_prompt(spec, spec.enc2)
    pooled_prompt = _expand_assemble_section_prompt(spec, spec.pooled) if spec.pooled is not None else enc2_prompt

    enc1_schedule = get_schedule(enc1_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
    enc2_schedule = get_schedule(enc2_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
    pooled_schedule = get_schedule(pooled_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
    if not enc1_schedule or not enc2_schedule or not pooled_schedule:
        raise ValueError("Empty schedule for ASSEMBLE section prompt")

    text_to_index: dict[str, int] = {}
    unique_texts: list[str] = []
    section_indices: list[list[int]] = []
    for schedule in (enc1_schedule, enc2_schedule, pooled_schedule):
        indices: list[int] = []
        for _end_at_step, text in schedule:
            if text not in text_to_index:
                text_to_index[text] = len(unique_texts)
                unique_texts.append(text)
            indices.append(text_to_index[text])
        section_indices.append(indices)

    texts_conditioning = SdConditioning(unique_texts, copy_from=copy_from)
    model_conds = model.get_learned_conditioning(texts_conditioning)

    def get_i_cond(i: int):
        if isinstance(model_conds, dict):
            return {k: v[i] for k, v in model_conds.items()}
        return model_conds[i]

    boundaries = [int(steps)] if not use_scheduling else _collect_schedule_boundaries(
        [enc1_schedule, enc2_schedule, pooled_schedule], steps
    )
    out: list[ScheduledPromptConditioning] = []
    previous_key = None
    for end_at_step in boundaries:
        enc1_local = _pick_text_schedule_index(enc1_schedule, end_at_step)
        enc2_local = _pick_text_schedule_index(enc2_schedule, end_at_step)
        pooled_local = _pick_text_schedule_index(pooled_schedule, end_at_step)
        key = (
            enc1_schedule[enc1_local][1],
            enc2_schedule[enc2_local][1],
            pooled_schedule[pooled_local][1],
        )
        if out and previous_key == key:
            out[-1] = ScheduledPromptConditioning(int(end_at_step), out[-1].cond)
            continue
        enc1_cond = get_i_cond(section_indices[0][enc1_local])
        enc2_cond = get_i_cond(section_indices[1][enc2_local])
        pooled_cond = get_i_cond(section_indices[2][pooled_local])
        merged = _assemble_condition_values(enc1_cond, enc2_cond, pooled_cond)
        out.append(ScheduledPromptConditioning(int(end_at_step), merged))
        previous_key = key

    return out


def _build_bind_branch_prompt_conditioning_schedule(
    model,
    spec: BindPromptSpec,
    base_schedule,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    bind_timeline = _build_bind_branch_timeline_entries(
        spec,
        steps,
        use_scheduling,
        seed,
        use_visitor,
        strict=False,
    )
    bind_text_schedule = [[int(end_at_step), text] for end_at_step, text, _weight in bind_timeline]
    if not base_schedule or not bind_text_schedule:
        raise ValueError("Empty schedule for BIND branch")

    text_to_index: dict[str, int] = {}
    unique_texts: list[str] = []
    for _end_at_step, bind_text, _bind_weight in bind_timeline:
        if bind_text not in text_to_index:
            text_to_index[bind_text] = len(unique_texts)
            unique_texts.append(bind_text)

    texts_conditioning = SdConditioning(unique_texts, copy_from=copy_from)
    model_conds = model.get_learned_conditioning(texts_conditioning)

    def get_i_cond(i: int):
        if isinstance(model_conds, dict):
            return {k: v[i] for k, v in model_conds.items()}
        return model_conds[i]

    boundaries = _collect_schedule_boundaries([base_schedule, bind_text_schedule], steps)
    out: list[ScheduledPromptConditioning] = []
    weight_schedule: list[tuple[int, float]] = []
    previous_key = None
    previous_weight = None
    for end_at_step in boundaries:
        base_index = _pick_schedule_entry_index(base_schedule, end_at_step)
        _bind_end_at_step, bind_text, bind_weight = _pick_bind_timeline_entry(bind_timeline, end_at_step)
        key = (base_index, bind_text)
        if out and previous_key == key:
            out[-1] = ScheduledPromptConditioning(int(end_at_step), out[-1].cond)
        else:
            base_cond = base_schedule[base_index].cond
            bind_cond = get_i_cond(text_to_index[bind_text])
            merged = _apply_condition_channel_target(base_cond, bind_cond, "cross")
            out.append(ScheduledPromptConditioning(int(end_at_step), merged))
            previous_key = key
        if weight_schedule and previous_weight is not None and abs(previous_weight - bind_weight) <= 1e-8:
            weight_schedule[-1] = (int(end_at_step), weight_schedule[-1][1])
        else:
            weight_schedule.append((int(end_at_step), float(bind_weight)))
            previous_weight = float(bind_weight)
    return out, weight_schedule


def _build_chunk_branch_prompt_conditioning_schedule(
    model,
    prompt: str,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    state = _extract_backend_prompt_state(prompt)

    if state.pool_spec is not None or _contains_pool_marker(prompt):
        raise PromptSyntaxError(
            "CHUNK branches cannot contain POOL blocks in v1.",
            kind="nested_backend_in_chunk_not_supported",
            token=POOL_KEYWORD,
            full=prompt,
        )
    if state.assemble_spec is not None or _contains_assemble_marker(prompt):
        raise PromptSyntaxError(
            "CHUNK branches cannot contain ASSEMBLE blocks in v1.",
            kind="nested_backend_in_chunk_not_supported",
            token=ASSEMBLE_KEYWORD,
            full=prompt,
        )
    if state.chunk_spec is not None:
        raise PromptSyntaxError(
            "Nested CHUNK blocks are not supported in v1.",
            kind="nested_chunk_not_supported",
            token=CHUNK_KEYWORD,
            full=prompt,
        )
    if state.blend_spec is not None or _contains_blend_marker(prompt):
        raise PromptSyntaxError(
            "CHUNK branches cannot contain BLEND blocks in v1.",
            kind="nested_backend_in_chunk_not_supported",
            token=BLEND_KEYWORD,
            full=prompt,
        )
    if state.active_morph_spec is not None:
        return _build_morph_prompt_conditioning_schedule(
            model,
            state.active_morph_spec,
            steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )
    if _contains_morph_marker(prompt):
        raise PromptSyntaxError(
            "MORPH blocks inside CHUNK branches must appear as one top-level branch block in v1.",
            kind="unsupported_morph_context",
            token=MORPH_KEYWORD,
            full=prompt,
        )
    return _build_plain_prompt_conditioning_schedule(
        model,
        prompt,
        steps,
        use_scheduling,
        seed,
        use_visitor,
        copy_from,
    )


def _build_chunk_prompt_conditioning_schedule(
    model,
    spec: ChunkPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    branch_prompts = [_expand_chunk_branch_prompt(spec, branch) for branch in spec.branches]
    branch_text_schedules = [
        get_schedule(branch_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
        for branch_prompt in branch_prompts
    ]
    branch_cond_schedules = [
        _build_chunk_branch_prompt_conditioning_schedule(
            model,
            branch_prompt,
            steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )
        for branch_prompt in branch_prompts
    ]
    anchor_cond_schedule = None
    target_channel = None
    if spec.shared_channel in {"pooled", "cross"}:
        anchor_cond_schedule = _build_plain_prompt_conditioning_schedule(
            model,
            _build_chunk_shared_anchor_prompt(spec),
            steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )
        target_channel = "cross" if spec.shared_channel == "pooled" else "pooled"
    if (
        not branch_text_schedules
        or any(not schedule for schedule in branch_text_schedules)
        or any(not schedule for schedule in branch_cond_schedules)
        or (anchor_cond_schedule is not None and not anchor_cond_schedule)
    ):
        raise ValueError("Empty schedule for at least one CHUNK branch")

    boundaries = _collect_schedule_boundaries(branch_text_schedules, steps)
    out: list[ScheduledPromptConditioning] = []
    previous_key = None
    weights = [branch.weight for branch in spec.branches]

    for end_at_step in boundaries:
        active_texts: list[str] = []
        active_conds = []

        for text_schedule, cond_schedule in zip(branch_text_schedules, branch_cond_schedules):
            text_index = _pick_text_schedule_index(text_schedule, end_at_step)
            active_texts.append(str(text_schedule[text_index][1]))
            cond_index = _pick_schedule_entry_index(cond_schedule, end_at_step)
            active_conds.append(cond_schedule[cond_index].cond)

        active_key = tuple(active_texts)
        if out and previous_key == active_key:
            out[-1] = ScheduledPromptConditioning(int(end_at_step), out[-1].cond)
            continue

        merged = _merge_chunk_condition_values(active_conds, weights)
        if anchor_cond_schedule is not None and target_channel is not None:
            anchor_index = _pick_schedule_entry_index(anchor_cond_schedule, end_at_step)
            anchor_cond = anchor_cond_schedule[anchor_index].cond
            merged = _apply_condition_channel_target(anchor_cond, merged, target_channel)
        out.append(ScheduledPromptConditioning(int(end_at_step), merged))
        previous_key = active_key

    return out


def _build_blend_prompt_conditioning_schedule(
    model,
    spec: BlendPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    branch_prompts = [_expand_blend_branch_prompt(spec, branch) for branch in spec.branches]
    branch_schedules = [
        get_schedule(branch_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
        for branch_prompt in branch_prompts
    ]
    if not branch_schedules or any(not schedule for schedule in branch_schedules):
        raise ValueError("Empty schedule for at least one BLEND branch")

    text_to_index: dict[str, int] = {}
    unique_texts: list[str] = []
    branch_text_indices: list[list[int]] = []

    for schedule in branch_schedules:
        indices: list[int] = []
        for _end_at_step, text in schedule:
            if text not in text_to_index:
                text_to_index[text] = len(unique_texts)
                unique_texts.append(text)
            indices.append(text_to_index[text])
        branch_text_indices.append(indices)

    texts_conditioning = SdConditioning(unique_texts, copy_from=copy_from)
    model_conds = model.get_learned_conditioning(texts_conditioning)

    def get_i_cond(i: int):
        if isinstance(model_conds, dict):
            return {k: v[i] for k, v in model_conds.items()}
        return model_conds[i]

    boundaries = [int(steps)] if not use_scheduling else _collect_schedule_boundaries(branch_schedules, steps)
    out: list[ScheduledPromptConditioning] = []
    previous_key = None
    effective_weights = _resolve_blend_mode_weights([branch.weight for branch in spec.branches], spec.mode, spec.intensity)

    for end_at_step in boundaries:
        active_texts: list[str] = []
        active_conds = []

        for schedule, indices in zip(branch_schedules, branch_text_indices):
            local_index = _pick_text_schedule_index(schedule, end_at_step)
            active_texts.append(str(schedule[local_index][1]))
            active_conds.append(get_i_cond(indices[local_index]))

        key = (tuple(active_texts), tuple(round(float(weight), 8) for weight in effective_weights))
        if out and previous_key == key:
            out[-1] = ScheduledPromptConditioning(int(end_at_step), out[-1].cond)
            continue

        merged = _blend_morph_condition_values(active_conds, effective_weights)
        merged = _apply_condition_channel_target(active_conds[0], merged, spec.channel_target)
        out.append(ScheduledPromptConditioning(int(end_at_step), merged))
        previous_key = key

    return out


def _build_morph_prompt_conditioning_schedule(
    model,
    spec: MorphPromptSpec,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    point_prompts = [_expand_morph_point_prompt(spec, point) for point in spec.points]
    point_schedules = [
        get_schedule(point_prompt, steps, use_scheduling, seed, use_visitor=use_visitor)
        for point_prompt in point_prompts
    ]
    if not point_schedules or any(not schedule for schedule in point_schedules):
        raise ValueError("Empty schedule for at least one MORPH control prompt")

    positions = _resolve_morph_positions(spec, steps)
    window_steps = _resolve_morph_window_steps(spec, steps)
    inactive_text = _build_morph_inactive_text(spec) if window_steps is not None else None

    text_to_index: dict[str, int] = {}
    unique_texts: list[str] = []
    point_text_indices: list[list[int]] = []

    for schedule in point_schedules:
        indices: list[int] = []
        for _end_at_step, text in schedule:
            if text not in text_to_index:
                text_to_index[text] = len(unique_texts)
                unique_texts.append(text)
            indices.append(text_to_index[text])
        point_text_indices.append(indices)

    if inactive_text is not None and inactive_text not in text_to_index:
        text_to_index[inactive_text] = len(unique_texts)
        unique_texts.append(inactive_text)

    texts_conditioning = SdConditioning(unique_texts, copy_from=copy_from)
    model_conds = model.get_learned_conditioning(texts_conditioning)

    def get_i_cond(i: int):
        if isinstance(model_conds, dict):
            return {k: v[i] for k, v in model_conds.items()}
        return model_conds[i]

    loop_steps = [int(steps)] if not use_scheduling else list(range(1, int(steps) + 1))

    out: list[ScheduledPromptConditioning] = []
    previous_key = None
    for step in loop_steps:
        if window_steps is not None and not (window_steps[0] <= step <= window_steps[1]):
            key = ("inactive", inactive_text)
            inactive_cond = get_i_cond(text_to_index[inactive_text])
            if out and previous_key == key:
                out[-1] = ScheduledPromptConditioning(int(step), out[-1].cond)
            else:
                out.append(ScheduledPromptConditioning(int(step), inactive_cond))
                previous_key = key
            continue
        active_texts: list[str] = []
        active_conds = []
        for schedule, indices in zip(point_schedules, point_text_indices):
            local_index = _pick_text_schedule_index(schedule, step)
            active_texts.append(str(schedule[local_index][1]))
            active_conds.append(get_i_cond(indices[local_index]))

        weights = _resolve_morph_point_weights(
            spec.points,
            _compute_morph_curve_weights(len(spec.points), positions, step, spec.curve, spec.intensity),
        )
        key = (tuple(active_texts), tuple(round(float(weight), 8) for weight in weights))
        if out and previous_key == key:
            out[-1] = ScheduledPromptConditioning(int(step), out[-1].cond)
            continue

        merged = _blend_morph_condition_values(active_conds, weights)
        merged = _apply_condition_channel_target(active_conds[0], merged, spec.channel_target)
        out.append(ScheduledPromptConditioning(int(step), merged))
        previous_key = key

    return out


def _build_prompt_conditioning_schedule(
    model,
    prompt: str,
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
):
    state = _extract_backend_prompt_state(prompt)
    if state.has_mixed_backends:
        _raise_mixed_backend_prompt_error(prompt, state)
    if state.has_bind_backend_conflict:
        _raise_bind_backend_prompt_error(prompt)
    if state.has_bind:
        raise ValueError("BIND requires the composable conditioning path via get_multicond_learned_conditioning().")
    if state.pool_spec is not None:
        return _build_pool_prompt_conditioning_schedule(
            model,
            state.pool_spec,
            steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )
    if state.assemble_spec is not None:
        return _build_assemble_prompt_conditioning_schedule(
            model,
            state.assemble_spec,
            steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )

    if state.chunk_spec is not None:
        return _build_chunk_prompt_conditioning_schedule(
            model,
            state.chunk_spec,
            steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )
    if state.blend_spec is not None:
        return _build_blend_prompt_conditioning_schedule(
            model,
            state.blend_spec,
            steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )
    if state.active_morph_spec is not None:
        return _build_morph_prompt_conditioning_schedule(
            model,
            state.active_morph_spec,
            steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )
    _raise_unsupported_backend_context_error(prompt)
    return _build_plain_prompt_conditioning_schedule(
        model,
        prompt,
        steps,
        use_scheduling,
        seed,
        use_visitor,
        copy_from,
    )


def _normalize_scheduler_surface_syntax(text: str) -> str:
    """Normalize scheduler spelling without changing semantic intent."""
    if not text:
        return text

    normalized = text

    # Tolerate legacy inner reverse syntax by moving the terminal token outside
    # the closing bracket. This keeps the canonical grammar ("[...] reverse")
    # without rewriting inner syntax to postfix.
    if "[" in normalized and "reverse" in normalized:
        out: list[str] = []
        stack: list[int] = []
        last_emit = 0
        i = 0
        while i < len(normalized):
            ch = normalized[i]
            if ch == "\\":
                i += 2 if i + 1 < len(normalized) else 1
                continue
            if ch == "[":
                stack.append(i)
                i += 1
                continue
            if ch == "]" and stack:
                start = stack.pop()
                if not stack:
                    inner = normalized[start + 1 : i]
                    last_colon = _find_last_top_level_colon_index(inner)
                    if last_colon >= 0:
                        tail = inner[last_colon + 1 :]
                        mrev = RE_REVERSE_SUFFIX.search(tail)
                        if mrev:
                            base_tail = tail[:mrev.start()].rstrip()
                            if _parse_inner_sched_tail(base_tail, 1) is not None:
                                out.append(normalized[last_emit:start])
                                out.append(f"[{inner[:last_colon + 1]}{base_tail}] reverse")
                                last_emit = i + 1
            i += 1
        if out:
            out.append(normalized[last_emit:])
            normalized = "".join(out)

    normalized = _RE_POSTFIX_STEP_COLON_SPACING.sub(
        lambda m: f"]:{m.group('num')}",
        normalized,
    )

    # If there is extra word-like text after "reverse", treat it as literal text,
    # not as reverse_flag for ranges.
    normalized = _RE_LITERAL_REVERSE_AFTER_RANGES.sub(
        lambda m: f"{m.group('head')} {LITERAL_REVERSE_TOKEN}",
        normalized,
    )
    return normalized


# ──────────────────────────────────────────────────────────────────────────────
# Pre-parse protection for postfix scheduled blocks: "[...]:N ..."
#
# Lark grammar intentionally disallows raw commas inside `plain` tokens.
# That is fine for most top-level prompts (commas are handled by `start`),
# but it breaks complex parses when a postfix scheduled block contains
# comma-separated tags (a very common SD prompt style), e.g.:
#   "[(a:1.2), b]:10"
#
# To make this reliable, we replace ONLY those commas that are inside the
# BRACKET CONTENT of postfix scheduled blocks, and only at the top level of
# that bracket content (not inside nested (), {}, or nested [] blocks).
#
# IMPORTANT: we replace commas with a 1-char placeholder to keep indices stable
# for linting and UI highlighting.
# ──────────────────────────────────────────────────────────────────────────────

# Tail pattern for inner scheduled block detection:
# matches "N", "N%", "N reverse", "N 1-4,5-8", "N% reverse", etc.
# Structural regex for inner scheduled tail: NUMBER[%] [step_ranges]
# NOTE: 'reverse' is NOT included here — it lives OUTSIDE ] per grammar rule:
#   scheduled: "[" [...] "]" (WHITESPACE* reverse_flag)?
# So [cat:8 reverse] is ambiguous (unlikely in practice), but
# [cat:8] reverse is the canonical form.
_RE_INNER_SCHED_TAIL = re.compile(
    rf"^\s*({NUMERIC_RE})(%?)"                                        # num + optional %
    rf"((?:\s+\d+%?\s*-\s*\d+%?(?:\s*,\s*\d+%?\s*-\s*\d+%?)*)?)?"   # optional step ranges
    rf"\s*$"
)

# Split pattern for step ranges in inner tail
_RE_INNER_RANGE_PART = re.compile(r'\d+%?\s*-\s*\d+%?')


def _extract_inner_sched_parts(inner: str, steps: int):
    """Return parsed inner scheduler parts or None."""
    last_colon = _find_last_top_level_colon_index(inner)
    if last_colon < 0:
        return None
    tail = (inner[last_colon + 1 :] or "").strip()
    tail_result = _parse_inner_sched_tail(tail, steps)
    if tail_result is None:
        return None
    boundary_spec, ranges_txt = tail_result
    return inner[:last_colon], boundary_spec, ranges_txt, last_colon


def _parse_inner_sched_tail(tail: str, steps: int):
    """Structurally parse an inner-scheduler tail: NUMBER[%] [ranges].

    Returns (boundary_spec, ranges_txt) or None if not a valid tail.
    Does NOT handle 'reverse' — that belongs outside ] in the grammar.

    Examples:
      "8"           → (absolute 8,  "")
      "75%"         → (percent 75, "")
      "8 1-4,5-8"   → (absolute 8,  "1-4,5-8")
      "8 reverse"   → None         # reverse is outside [], not in tail
    """
    _ = steps  # kept for backward-compatible signature
    m = _RE_INNER_SCHED_TAIL.fullmatch(tail.strip())
    if not m:
        return None
    num_str  = m.group(1)
    is_pct   = bool(m.group(2))
    rest     = (m.group(3) or "").strip()

    try:
        num_f = float(num_str)
    except (ValueError, TypeError):
        return None

    boundary_spec = _make_boundary_spec(num_f, is_percent=is_pct)
    return boundary_spec, rest


def _is_inner_scheduled_block(inner: str) -> bool:
    """Return True if inner content looks like a scheduler block [content:N].

    Uses _parse_inner_sched_tail for structural validation — correctly handles
    NUMBER, NUMBER%, NUMBER ranges. Does NOT match [cat|dog] (alternate),
    [text] (attention), or tails with 'reverse' (which lives outside brackets).
    """
    return _extract_inner_sched_parts(inner, 1) is not None


def _placeholderize_scheduled_block_commas(text: str) -> str:
    """Replace top-level commas inside ALL scheduled bracket blocks with
    SCHEDULE_COMMA_PLACEHOLDER so Lark can parse comma-separated tags.

    Handles both forms:
      postfix:  [(a:1.2), b]:10   — number AFTER closing bracket
      inner:    [a, b, c:10]      — number INSIDE bracket as last colon-segment

    Only commas at the top nesting level of the bracket content are replaced
    (commas inside nested (), {}, [] are left untouched).

    This is called in _parse_lark_cached BEFORE Lark parsing, so Lark never
    sees raw commas in scheduled content regardless of block count.
    """
    if not text or "[" not in text or "," not in text:
        return text

    chars = list(text)
    stack: list[int] = []
    i = 0

    # O(N) total: inner k-loop runs only over non-overlapping bracket spans.
    while i < len(chars):
        ch = chars[i]

        if ch == "\\":
            i += 2 if i + 1 < len(chars) else 1
            continue

        if ch == "[":
            stack.append(i)
            i += 1
            continue

        if ch == "]" and stack:
            start = stack.pop()
            # Only act on closed top-level bracket blocks (stack now empty)
            if not stack:
                inner_text = "".join(chars[start + 1 : i])

                # --- Determine if this is a scheduled block ---
                is_scheduled = False

                # Check 1: postfix form — ]:N after the closing bracket
                j = i + 1
                while j < len(chars) and str(chars[j]).isspace():
                    j += 1
                if j < len(chars) and chars[j] == ":":
                    j += 1
                    while j < len(chars) and str(chars[j]).isspace():
                        j += 1
                    m = RE_NUMERIC.match("".join(chars[j:]))
                    if m:
                        is_scheduled = True

                # Check 2: inner form — last top-level colon-segment is numeric
                replace_limit = i

                if not is_scheduled:
                    inner_sched = _extract_inner_sched_parts(inner_text, 1)
                    if inner_sched is not None:
                        is_scheduled = True
                        replace_limit = start + 1 + inner_sched[3]

                if is_scheduled:
                    # Replace commas inside prompt content at all nesting levels,
                    # except inside nested [] blocks. For inner syntax, stop before
                    # the final ":tail" so range separators like "1-4,5-8" survive.
                    #
                    # WHY: Lark grammar forbids raw comma inside `prompt` rule — it only
                    # exists at `start` level. So (cat, dog) inside [(...):N] fails Lark
                    # even though the comma is "inside ()". We must replace all commas
                    # inside scheduled prompt content so Lark never sees them as syntax.
                    # Commas inside nested [] are skipped (they're sub-bracket structure).
                    depth_brack = 0
                    k = start + 1
                    while k < replace_limit:
                        c = chars[k]
                        if c == "\\":
                            k += 2
                            continue
                        if c == "[":
                            depth_brack += 1
                        elif c == "]" and depth_brack > 0:
                            depth_brack -= 1
                        elif c == "," and depth_brack == 0:
                            chars[k] = SCHEDULE_COMMA_PLACEHOLDER
                        k += 1

        i += 1

    return "".join(chars)


# Keep old name as alias for backward compatibility
_placeholderize_postfix_scheduled_blocks = _placeholderize_scheduled_block_commas



# ──────────────────────────────────────────────────────────────────────────────
# Узкие хелперы для извлечения компонентов fast-path'ов (без сборки расписаний)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_after_with_ranges(full: str, steps: int):
    """Извлечь компоненты для вида "[a:b]:N 1-4,6-8 [reverse]".

    Возвращает tuple(prompts, ranges, rev_flag) либо None.
    ВАЖНО:
      - Порядок ranges сохраняется как в исходной строке (нужно для last-wins).
      - Обратные диапазоны (start > end) считаются ошибкой (строгий режим).
      - Одиночные шаги (start == end) разрешены.
    """
    m = RE_BRACKET_AFTER_WITH_RANGES.match(full or "")
    if not m:
        return None
    inner = m.group("inner")
    ranges_txt = m.group("ranges") or ""
    rev = m.group("rev")

    prompts = _parse_inner_prompts(inner)
    ranges: list[tuple[int, int]] = []

    def _raw_value(txt: str, is_pct: bool) -> float:
        val = float(txt[:-1] if is_pct else txt)
        if is_pct:
            val = val / 100.0 * steps
        return val

    for part in (ranges_txt or "").split(','):
        part = part.strip()
        if not part:
            continue
        if '-' not in part:
            continue
        start_raw, end_raw = [x.strip() for x in part.split('-', 1)]
        start_is_pct = start_raw.endswith('%')
        end_is_pct = end_raw.endswith('%')

        try:
            raw_start = _raw_value(start_raw, start_is_pct)
            raw_end = _raw_value(end_raw, end_is_pct)
        except Exception as exc:
            raise PromptSyntaxError(
                f"Invalid range token {part!r}",
                kind="invalid_range_token",
                token=part,
                full=full,
            ) from exc

        # Строго: обратный диапазон — ошибка (чтобы не скрывать опечатки)
        if raw_start > raw_end:
            raise PromptSyntaxError(
                f"Reverse range {part!r} (start > end)",
                kind="reverse_range",
                token=part,
                full=full,
            )

        start_val = _clamp(round(raw_start), steps)
        end_val = _clamp(round(raw_end), steps)
        ranges.append((start_val, end_val))

    rev_flag = bool(rev)
    return (prompts, ranges, rev_flag)




def _build_ranges_schedules_from_components(
    prompts: Sequence[str],
    ranges: Sequence[tuple[int, int]],
    steps: int,
    *,
    prefix: str = "",
    suffix: str = "",
    rev_flag: bool = False,
    cycle_prompts: bool = True,
    empty_center: str = "",
) -> list[list[int, str]]:
    """Собрать расписание для синтаксиса "[...]:N a-b,c-d [reverse]".

    Политики (зафиксированы под твой выбор):
      - Overlap: last-wins (последний диапазон перезаписывает пересечения).
      - Reverse ranges (start > end): не допускаются (должны ловиться на этапе извлечения).
      - Single-step (start == end): разрешён (один шаг).
      - Порядок ranges важен: он определяет, какой prompt будет применён к диапазону.

    Реализация:
      1) строим mapping step -> center_text (по умолчанию empty_center)
      2) проходим ranges по порядку и записываем (перезаписывая) center_text
      3) сжимаем mapping обратно в список [end_step, text]
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    # База промптов
    prompt_list = list(prompts) if prompts else [empty_center]
    if rev_flag:
        prompt_list = list(reversed(prompt_list))

    # mapping: шаг -> центральный текст
    slot = [empty_center] * (int(steps) + 1)  # slot[0] не используется

    for idx, (start, end) in enumerate(ranges or ()):
        start_i = int(start)
        end_i = int(end)

        # безопасность (обычно уже нормализовано)
        if start_i > end_i:
            raise ValueError(f"Invalid range ({start_i}-{end_i}) after normalization for: {ranges!r}")

        if cycle_prompts:
            center = prompt_list[idx % len(prompt_list)]
        else:
            if idx >= len(prompt_list):
                break
            center = prompt_list[idx]

        # clamp и заполнение inclusive
        start_i = max(1, min(start_i, int(steps)))
        end_i = max(1, min(end_i, int(steps)))

        for s in range(start_i, end_i + 1):
            slot[s] = center  # last-wins: всегда перезаписываем

    # Сжатие mapping -> schedule segments
    schedules: list[list[int, str]] = []

    def _append(end_step: int, center_text: str) -> None:
        end_step = min(int(end_step), int(steps))
        text = _concat_prefix_text_suffix(prefix, center_text, suffix)
        if schedules and schedules[-1][1] == text:
            schedules[-1][0] = end_step
        else:
            schedules.append([end_step, text])

    cur = slot[1]
    for s in range(2, int(steps) + 1):
        if slot[s] != cur:
            _append(s - 1, cur)
            cur = slot[s]
    _append(int(steps), cur)

    return schedules


# ──────────────────────────────────────────────────────────────────────────────
# Безопасный сплит по ':' только на верхнем уровне
# ──────────────────────────────────────────────────────────────────────────────

def _split_top_level_colon_all(s: str, keep_empty: bool) -> list[str]:
    """
    Разбить строку по ':' только на ВЕРХНЕМ уровне:
    игнорировать ':' внутри круглых (), фигурных {}, а ТАКЖЕ квадратных [].
    Учитывать экранирование '\\'.
    """
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = 0  # (...)
    depth_brace = 0  # {...}
    depth_brack = 0  # [...]
    i = 0
    while i < len(s):
        ch = s[i]
        # экранирование
        if ch == '\\':
            if i + 1 < len(s):
                buf.append(ch); buf.append(s[i + 1])
                i += 2
                continue

        # уровни скобок
        if ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren = max(0, depth_paren - 1)
        elif ch == '{':
            depth_brace += 1
        elif ch == '}':
            depth_brace = max(0, depth_brace - 1)
        elif ch == '[':
            depth_brack += 1
        elif ch == ']':
            depth_brack = max(0, depth_brack - 1)

        # разрыв только на самом верхнем уровне
        if ch == ':' and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            seg = "".join(buf)
            seg_trim = seg.strip()
            if keep_empty:
                parts.append(seg_trim if seg_trim != "" else "")
            else:
                if seg_trim != "":
                    parts.append(seg_trim)
            buf = []
        else:
            buf.append(ch)
        i += 1

    seg = "".join(buf)
    seg_trim = seg.strip()
    if keep_empty:
        parts.append(seg_trim if seg_trim != "" else "")
    else:
        if seg_trim != "":
            parts.append(seg_trim)
    return parts


def _split_top_level_colon(s: str) -> list[str]:
    return _split_top_level_colon_all(s, keep_empty=False)

def _split_top_level_colon_keep_empty(s: str) -> list[str]:
    return _split_top_level_colon_all(s, keep_empty=True)



# ──────────────────────────────────────────────────────────────────────────────
# Хелперы, устраняющие дублирование reverse-логики и prompts-разбора
# ──────────────────────────────────────────────────────────────────────────────

def _strip_reverse_from_post(post: str, extra_rev: bool = False) -> tuple[str, bool]:
    """Вытащить 'reverse' из начала или конца строки post.

    Возвращает (cleaned_post, rev_found: bool).
    extra_rev=True означает что reverse уже обнаружен (из токена/regex-группы);
    в этом случае post не изменяем содержательно, но зачищаем ведущую
    пунктуацию (запятая/пробел) которая могла остаться после извлечения
    токена 'reverse' в RE_BRACKET_AFTER.
    Порядок поиска: PREFIX ('reverse hd') → SUFFIX ('hd reverse').

    Fix: '[a:b]:10 reverse, shot' ранее давал post=', shot' (ведущая запятая),
    теперь возвращает 'shot' — одинаково с '[a:b]:10 reverse shot'.
    """
    if extra_rev:
        # Strip leading comma/space left by the regex group split
        return post.lstrip(", "), True
    if not post:
        return post, False
    mrev = RE_REVERSE_PREFIX.match(post)
    if mrev:
        return post[mrev.end():].lstrip(", "), True
    mrev_s = RE_REVERSE_SUFFIX.search(post)
    if mrev_s:
        return post[:mrev_s.start()].rstrip().rstrip(",").rstrip(), True
    return post, False


def _parse_inner_prompts(inner: str) -> list[str]:
    """Разбить inner на список промптов: split по ':' → unescape → strip.
    Устраняет дублирование одинакового list-comp в 4+ местах.
    """
    return [_unescape_literals(p.strip()) for p in _split_top_level_colon_keep_empty(inner)]

def _finalize_schedules(schedules) -> list:
    """Применить _apply_and + _collapse_spaces к каждому тексту расписания.
    Устраняет дублирование идентичных return-строк вида:
      return [[e, _apply_and(_collapse_spaces(t))] for e, t in schedules]
    После нормализации схлопывает соседние сегменты с одинаковым текстом
    (два сегмента могут стать идентичными после collapse_spaces).
    """
    out = []
    for e, t in schedules:
        normalized = _apply_and(_collapse_spaces(t)).replace(LITERAL_REVERSE_TOKEN, "reverse")
        if out and out[-1][1] == normalized:
            out[-1][0] = e   # extend last segment
        else:
            out.append([e, normalized])
    return out



def _has_multiple_bracket_blocks(s: str) -> bool:
    """Проверить, есть ли более одного независимого блока ``[...]``.

    Игнорирует экранированные скобки и вложенные уровни, чтобы не считать
    ``[[nested]]`` как два блока. Возвращает ``True`` при втором входе в
    глубину 1.
    """
    depth = 0
    blocks = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "[":
            depth += 1
            if depth == 1:
                blocks += 1
                if blocks >= 2:
                    return True
        elif ch == "]" and depth > 0:
            depth -= 1
        i += 1
    return False

# ──────────────────────────────────────────────────────────────────────────────
# Грамматика Lark
# ──────────────────────────────────────────────────────────────────────────────

# динамическое правило для alternate
_alt_rule = r' "[" prompt ("|" prompt)* "]" '                 # без пустых опций
if ALLOW_EMPTY_ALTERNATE:
    # '!' передаёт '|' как токены в alternate(args) — необходимо для пустых веток [fe|]
    _alt_rule = r' "[" prompt ("|" [prompt])+ "]" '

# NB: расширили класс одиночных знаков в start, чтобы '!' не валил парсер для plain'ов/compound'ов
_grammar = r"""
!start: (prompt | /[():,!|&]/+)*

prompt: (scheduled | emphasized | grouped
        | alternate | alternate_distinct
        | alternate1 | alternate2
        | top_level_sequence3 | top_level_sequence | sequence
        | weighted | numbered | and_rule
        | plain | WHITESPACE)*

!emphasized: "(" prompt ")"
           | "(" prompt ":" prompt ")"
           | "(" prompt ":" NUMBER ")"


!weighted: plain ":" NUMBER


scheduled: "[" [prompt (":" prompt)* ":" boundary_value (WHITESPACE* step_range_list)?] "]" (WHITESPACE* reverse_flag)?
        | "[" [prompt (":" prompt)*] "]" ":" boundary_value (WHITESPACE* step_range_list)? (WHITESPACE* reverse_flag)?
boundary_value: NUMBER           -> boundary_number
              | NUMBER PERCENT   -> boundary_percent
reverse_flag: "reverse"
step_range_list: step_range (WHITESPACE* "," WHITESPACE* step_range)*
step_range: NUMBER "-" NUMBER        -> range_abs
          | NUMBER "%" "-" NUMBER "%" -> range_pct



!alternate: """ + _alt_rule + r"""
!alternate_distinct: "[" prompt ("|" prompt)* "]!"
alternate1: (prompt) "|" (prompt)+
alternate2: plain ("|" plain)+

grouped: "{" ((NUMBER_Q | prompt | sequence | grouped) ("," | "|")?)+ "}"

top_level_sequence: prompt ("::" sequence)+ "!!" ("," plain)?
top_level_sequence3: prompt ":::" (sequence | plain) (WHITESPACE* "," WHITESPACE* (sequence | plain))* "!!!" (WHITESPACE* "," WHITESPACE* (plain | sequence))*
sequence: prompt "::" prompt ("," | WHITESPACE)* nested_sequence* ("!" | ";")?
nested_sequence: "::" prompt ("," | WHITESPACE)* ("!" | ";" | "~")?

compound: /[a-zA-Z0-9]+(_[a-zA-Z0-9]+)+/
numbered: NUMBER_Q ("!" | "_")? (grouped | sequence | alternate | alternate_distinct | alternate2 | alternate1)

and_rule: (plain | weighted | emphasized | numbered | grouped | alternate | alternate_distinct | alternate2 | alternate1 | scheduled) ("&" (plain | weighted | emphasized | numbered | grouped | alternate | alternate_distinct | alternate1 | scheduled))+
 WHITESPACE: /\s+/
plain: /([^\\[\]\{\}\(\),&:!|]|\\.)+/

%import common.SIGNED_NUMBER -> NUMBER
%import common.INT -> NUMBER_Q
PERCENT: "%"
"""

# Варианты планирования: число может быть до или после закрывающей скобки

schedule_parser = lark.Lark(_grammar, start="start")

# --- ДОБАВИТЬ ЭТУ ФУНКЦИЮ ---
@lru_cache(maxsize=CACHE_SIZE)
def _parse_lark_cached(prompt: str):
    """Кэшируем только построение дерева, так как оно не зависит от steps/seed.
    Нормализация уже выполнена в _get_schedule_impl до вызова этой функции.
    """
    return schedule_parser.parse(_placeholderize_postfix_scheduled_blocks(prompt))
# ----------------------------
# ──────────────────────────────────────────────────────────────────────────────
# Общие вспомогалки для fast-path "[...]:N" (и CollectSteps, и get_schedule)
# ──────────────────────────────────────────────────────────────────────────────

def _clamp(x: int, steps: int) -> int:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    # ИСПРАВЛЕНИЕ: Защита от деления на ноль и некорректных значений
    if not isinstance(x, (int, float)) or not isinstance(steps, (int, float)):
        raise TypeError(f"x and steps must be numeric, got {type(x)}, {type(steps)}")
    return max(1, min(int(x), int(steps)))

def _to_end_step(num: float | _BoundarySpec, steps: int) -> int:
    """Convert a typed boundary (or legacy numeric fallback) into an end step."""
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if isinstance(num, _BoundarySpec):
        if num.kind == "percent":
            raw = num.value / 100.0 * steps
        elif num.kind == "fraction":
            raw = num.value * steps
        else:
            raw = num.value
        return _clamp(round(raw), steps)

    num_f = float(num)
    if 0.0 < num_f < 1.0:
        return _clamp(round(num_f * steps), steps)
    return _clamp(round(num_f), steps)


def _extract_scheduled_boundary_value(children) -> _BoundarySpec | None:
    boundary_node = next(
        (
            c
            for c in children
            if isinstance(c, lark.Tree)
            and getattr(c, "data", None) in ("boundary_number", "boundary_percent")
        ),
        None,
    )
    if boundary_node is not None:
        num_tok = next(
            (
                c
                for c in boundary_node.children
                if isinstance(c, lark.Token) and c.type == "NUMBER"
            ),
            None,
        )
        if num_tok is None:
            return None
        try:
            value = float(num_tok.value)
        except (TypeError, ValueError):
            return None
        return _make_boundary_spec(
            value,
            is_percent=(getattr(boundary_node, "data", None) == "boundary_percent"),
        )

    num_tok = next(
        (c for c in children if isinstance(c, lark.Token) and c.type == "NUMBER"),
        None,
    )
    if num_tok is None:
        return None
    try:
        return _make_boundary_spec(float(num_tok.value))
    except (TypeError, ValueError):
        return None

def _build_bracket_after_schedules(
    pre: str, prompts: list[str], boundary_end: int, post: str, steps: int
) -> list[list[int, str]]:
    """
    Семантика:
      • если НЕТ префикса/суффикса → как для [p1:p2:...:N] (интервалы ВНУТРИ [1..N]);
      • если ЕСТЬ префикс/суффикс → «после N»: до N – пролог (pre+post), далее хвост.
    """
    # ИСПРАВЛЕНИЕ: Проверка на пустой prompts В НАЧАЛЕ
    if not prompts:
        return [[steps, _concat_prefix_text_suffix(pre, "", post)]]
    
    if not (pre.strip() or post.strip()):
        return _build_bracket_inner_schedules(pre, prompts, boundary_end, post, steps)

    schedules: list[list[int, str]] = []
    boundary_end = int(boundary_end)
    steps = int(steps)

    tail = max(0, steps - boundary_end)

    # Пролог до N (если есть хвост)
    if tail > 0:
        schedules.append([boundary_end, _concat_prefix_text_suffix(pre, "", post)])

    # Теперь безопасно использовать prompts
    if tail <= 0:
        # boundary >= steps: after-сегмент никогда не активируется.
        # Показываем первый промпт на всех шагах (A1111 совместимость).
        # Пустой центр здесь неверен — пользователь ожидает увидеть промпт.
        schedules.append([steps, _concat_prefix_text_suffix(pre, prompts[0], post)])
    else:
        # Строго монотонные границы по floor, без round-слипания.
        # Обрезаем prompts до tail доступных шагов — лишние промпты
        # всё равно не получат уникальный шаг и только создадут дубли.
        usable = prompts[:tail]
        if not usable:
            usable = prompts[:1]
        k = max(1, len(usable))
        prev_end = boundary_end
        for i, p in enumerate(usable, start=1):
            end = boundary_end + (tail * i) // k  # floor
            if end <= prev_end:
                end = min(steps, prev_end + 1)
            if end > steps:
                break
            schedules.append([end, _concat_prefix_text_suffix(pre, p, post)])
            prev_end = end

    # Схлопнуть соседние сегменты с одинаковым текстом
    out: list[list[int, str]] = []
    for e, t in schedules:
        if out and out[-1][1] == t:
            out[-1][0] = int(e)   # extend
        else:
            out.append([int(e), t])
    return out


def _build_bracket_inner_schedules(
    pre: str, prompts: list[str], boundary_end: int, post: str, steps: int
) -> list[list[int, str]]:
    """
    A [p1:p2:...:N] Z — интервалы внутри [1..N] + хвост.
    Особый случай: при ОДНОМ prompt до границы показываем только pre+post,
    а после границы — pre+prompt+post (ожидание тестов).
    Для >=2 prompt'ов интервалы совпадают с логикой CollectSteps.visit_scheduled.
    """
    schedules: list[list[int, str]] = []

    if not prompts:
        return [[steps, _concat_prefix_text_suffix(pre, "", post)]]

    # 1 prompt — SD/A1111 semantics:
    #   boundary < steps  → empty 1..boundary, then prompt boundary+1..steps
    #   boundary >= steps → prompt active on all steps (no "after" window exists)
    if len(prompts) == 1:
        if boundary_end >= steps:
            # [cat:10] at steps=10: cat is shown on all steps
            schedules.append([int(steps), _concat_prefix_text_suffix(pre, prompts[0], post)])
        else:
            schedules.append([int(boundary_end), _concat_prefix_text_suffix(pre, "", post)])
            schedules.append([int(steps), _concat_prefix_text_suffix(pre, prompts[0], post)])
        return schedules

    # >=2 prompts: standard SD/A1111 semantics.
    # The LAST prompt is the "after boundary" tail (active from boundary+1 to steps).
    # The FIRST (n-1) prompts divide [1..boundary_end] evenly.
    #
    # Examples:
    #   [cat:dog:5] steps=10  → cat 1-5, dog 6-10
    #   [a:b:c:6]   steps=10  → a 1-3, b 4-6, c 7-10
    #
    # Old code divided ALL n prompts within [1..boundary], causing:
    #   [cat:dog:5] steps=10 → cat 1-2, dog 3-5, dog 6-10  ← wrong (cat too short)
    num_before = len(prompts) - 1   # prompts active within [1..boundary_end]

    if num_before == 0:
        # Edge case: num_before can't be 0 when len(prompts)>=2, guard only
        schedules.append([int(boundary_end), _concat_prefix_text_suffix(pre, prompts[0], post)])
    else:
        step_size = float(boundary_end) / num_before
        for i in range(num_before):
            end = _clamp(int(round((i + 1) * step_size)), steps)
            schedules.append([end, _concat_prefix_text_suffix(pre, prompts[i], post)])

    # Last prompt as tail after boundary
    if boundary_end < steps:
        schedules.append([int(steps), _concat_prefix_text_suffix(pre, prompts[-1], post)])
    elif not schedules:
        # boundary_end >= steps: tail has no room; still need at least one segment
        schedules.append([int(steps), _concat_prefix_text_suffix(pre, prompts[-1], post)])

    # Fallback if rounding produced nothing
    if not schedules:
        schedules.append([int(steps), _concat_prefix_text_suffix(pre, prompts[-1], post)])



    # Схлопнем соседние сегменты с одинаковым текстом (разные end_step — тоже)
    out: list[list[int, str]] = []
    for e, t in schedules:
        if out and out[-1][1] == t:
            out[-1][0] = int(e)   # extend same-text segment
        else:
            out.append([int(e), t])

    return out



# ──────────────────────────────────────────────────────────────────────────────
# Унификация resolve_tree
# ──────────────────────────────────────────────────────────────────────────────

def resolve_tree(tree: lark.Tree | lark.Token, keep_spacing: bool = True) -> str:
    if isinstance(tree, lark.Tree):
        parts = []
        for child in tree.children:
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                parts.append(" ")
                continue
            parts.append(resolve_tree(child, keep_spacing))
        result = "".join(str(c) for c in parts if c is not None)
        if keep_spacing:
            # схлопываем все пробельные последовательности до одиночного пробела,
            # но сохраняем крайние пробелы, если они были
            leading = len(result) - len(result.lstrip())
            trailing = len(result) - len(result.rstrip())
            core = _re_ws_collapse.sub(" ", result.strip())
            return ((" " * leading) + core + (" " * trailing)).replace(SCHEDULE_COMMA_PLACEHOLDER, ",")
        else:
            return _re_ws_collapse.sub("", result).replace(SCHEDULE_COMMA_PLACEHOLDER, ",")
    return str(tree).replace(SCHEDULE_COMMA_PLACEHOLDER, ",")


# ──────────────────────────────────────────────────────────────────────────────
# Комплексность/перевод узлов → текст
# ──────────────────────────────────────────────────────────────────────────────
_RE_UNESCAPED_ALT_OR_BANG = re.compile(r'(?<!\\)[|!]')
_RE_POST_STEP_RANGES = re.compile(r'\b\d+%?\s*-\s*\d+%?')

def _needs_complex_parse(inner: str, full: str) -> bool:
    """
    True, если fast-path лучше отключить и отдать парсеру/Visitor.
    FIX #4: Усиленная проверка для предотвращения багов с вложенностью.
    """
    # Если внутри inner есть квадратные скобки (вложенность) — это работа для Lark, а не Regex.
    if "[" in inner or "]" in inner:
        return True
        
    # Если есть экранированные скобки, Regex может ошибиться с позициями
    if "\\[" in inner or "\\]" in inner:
        return True

    # НЕэкранированные '|' или '!' в полной строке требуют парсера
    if _RE_UNESCAPED_ALT_OR_BANG.search(full or ""):
        return True

    # '::' считаем «сложностью» только ВНЕ квадратных скобок fast-path'а
    if "::" in (full or "") and "::" not in (inner or ""):
        return True
        
    return False


def _try_fast_after_scheduler(
    full: str,
    steps: int,
    *,
    prefix: str = "",
    suffix: str = "",
    allow_ranges: bool = True,
    allow_bracket_after: bool = True,
    disallow_alt_bang: bool = True,
    boundary_fallback: float | _BoundarySpec | None = None,
) -> list[list[int | str]] | None:
    """Shared fast-path for "[...]:N" and optional step ranges."""
    if _has_multiple_bracket_blocks(full):
        return None

    if allow_ranges and "|" not in full:
        m_res = _extract_after_with_ranges(full, steps)
        if m_res:
            prompts, ranges, rev_flag = m_res
            if ranges and prompts:
                schedules = _build_ranges_schedules_from_components(
                    prompts,
                    ranges,
                    steps,
                    prefix=prefix,
                    suffix=suffix,
                    rev_flag=rev_flag,
                    cycle_prompts=True,
                    empty_center="",
                )
                return _finalize_schedules(schedules)

    if not allow_bracket_after:
        return None

    if full.count("[") != 1 or full.count("]") != 1:
        return None

    if disallow_alt_bang and _RE_UNESCAPED_ALT_OR_BANG.search(full):
        return None

    m = RE_BRACKET_AFTER.match(full)
    if not m:
        return None

    pre, inner, boundary_txt, rev_token, post = m.groups()
    if _needs_complex_parse(inner or "", full):
        return None
    if _RE_POST_STEP_RANGES.search(post or ""):
        return None

    try:
        if boundary_txt and boundary_txt.endswith("%"):
            boundary_spec = _make_boundary_spec(float(boundary_txt[:-1]), is_percent=True)
        else:
            boundary_spec = _make_boundary_spec(float(boundary_txt))
    except (ValueError, TypeError):
        if boundary_fallback is None:
            return None
        boundary_spec = _make_boundary_spec(boundary_fallback, assume_absolute=True)

    prompts = _parse_inner_prompts(inner)
    post, rev_flag = _strip_reverse_from_post(post or "", bool(rev_token))
    if rev_flag:
        prompts = list(reversed(prompts))

    boundary = _to_end_step(boundary_spec, steps)
    schedules = _build_bracket_after_schedules(
        prefix + (pre or ""),
        prompts,
        boundary,
        (post or "") + suffix,
        steps,
    )
    return _finalize_schedules(schedules)

def _to_text(x) -> str:
    """Единый путь узел/строку → финальный текст, как у Visitor."""
    s = resolve_tree(x, keep_spacing=True) if not isinstance(x, str) else x
    return _unescape_literals(s)

# ──────────────────────────────────────────────────────────────────────────────
# Transformer: преобразование дерева в текст (для TL последовательностей и т.п.)
# ──────────────────────────────────────────────────────────────────────────────

class ScheduleTransformer(lark.Transformer):
    def __init__(self, total_steps: int, current_step: int = 1, seed: int | None = 42):
        super().__init__()
        self.total_steps = total_steps
        self.current_step = current_step
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random

    def start(self, args):
        s = "".join(str(arg) for arg in args if arg)
        # owner::a::b!!, extra -> owner -> owner: a, b, extra
        if "::" in s and "!!" in s and all(ch not in s for ch in "[]()"):
            left, trailing = s.split("!!", 1)
            owner, rest = left.split("::", 1)
            descriptors = [x.strip(" ,~!;") for x in rest.split("::") if x.strip(" ,~!;")]
            seq_text = f"{owner.strip()}: {', '.join(descriptors)}" if descriptors else owner.strip()
            trailing_text = [t.strip(" ,") for t in trailing.split(",") if t.strip(" ,")]
            out = f"{owner.strip()} -> {seq_text}"
            if trailing_text:
                out += f", {', '.join(trailing_text)}"
            return out
        # owner::a::b!  ->  owner: a, b
        if "::" in s and (s.endswith("!") or s.endswith(";")) and all(ch not in s for ch in "[]()"):
            owner, rest = s.split("::", 1)
            rest = rest[:-1]
            descriptors = [x.strip(" ,~!;") for x in rest.split("::") if x.strip(" ,~!;")]
            return f"{owner.strip()}: {', '.join(descriptors)}"
        return s

    def prompt(self, args):
        return "".join(str(arg) for arg in args if arg)

    def plain(self, args):
        return args[0].value

    def compound(self, args):
        return "_".join(str(arg) for arg in args)

    def and_rule(self, args):
        parts = []
        for child in args:
            if isinstance(child, lark.Tree) and getattr(child, "data", None) in ("weighted", "emphasized"):
                parts.append(self.transform(child))  # получим '(token:1.2)'
            else:
                s = resolve_tree(child, keep_spacing=True)
                if s:
                    parts.append(s)
        return f" {ATTENTION_AND_OPERATOR} ".join(p for p in parts if p)

    def weighted(self, args):
        """
        Преобразуем 'token:NUMBER' -> '(token:NUMBER)' с тем же форматированием, как в emphasized.
        """
        # левая часть (plain|compound)
        left = resolve_tree(args[0], keep_spacing=True).strip()

        # правая часть — число; достаём как текст с подстраховкой
        tail_txt = "".join(
            (a.value if isinstance(a, lark.Token) else resolve_tree(a, keep_spacing=True))
            for a in args[1:]
        )
        mnum = RE_NUMERIC.search(tail_txt or "")
        if mnum:
            num_txt = mnum.group(0)
            if "." not in num_txt and "e" not in num_txt.lower():
                try:
                    weight_str = f"{float(num_txt):.1f}"
                except (ValueError, TypeError):
                    weight_str = _RB_WEIGHT_STR
            else:
                weight_str = num_txt
        else:
            weight_str = _RB_WEIGHT_STR

        return f"({left}:{weight_str})"

    def grouped(self, args):
        return ", ".join(resolve_tree(arg, keep_spacing=True) for arg in args if resolve_tree(arg).strip(" ,|"))

    def alternate(self, args):
        # После '!alternate' Lark передаёт '[', '|', ']' как токены наряду с prompt-деревьями.
        # '[' и ']' — структурные скобки, их нужно игнорировать.
        # Только '|' является разделителем опций.
        #
        # Алгоритм: строим список опций, считая '|' разделителями.
        # Каждая пара смежных '|' (или начало/конец) без prompt между ними — пустая опция.
        #
        # Пример '[fe|]male':  args = ['[', Tree('fe'), '|', ']']
        #   -> игнорируем '[' и ']'
        #   -> остаток: [Tree('fe'), '|']
        #   -> опции: ['fe', '']  (пустая после последнего '|')

        vals: list[str] = []
        prev_was_pipe = True   # True в начале: если первый токен '|', это ведущая пустая опция

        for arg in args:
            s = resolve_tree(arg, keep_spacing=True)
            tok = (s or "").strip()

            if tok in ("[", "]"):
                # Структурные скобки — пропускаем, не меняем состояние
                continue

            if tok == "|":
                if prev_was_pipe:
                    vals.append("")   # два '|' подряд → пустая опция между ними
                prev_was_pipe = True
                continue

            # Реальный контент
            vals.append(s)
            prev_was_pipe = False

        if prev_was_pipe:
            vals.append("")   # trailing '|' → пустая завершающая опция

        return vals[(self.current_step - 1) % len(vals)] if vals else "empty_prompt"


    def alternate_distinct(self, args):
        options = []
        for arg in args:
            s = resolve_tree(arg, keep_spacing=True)
            # фильтруем разделители
            if s.strip() in ("|", ","):
                continue
            if s:
                options.append(s)
            else:
                options.append("")  # пустая опция допустима при ALLOW_EMPTY_ALTERNATE
        return self.rng.choice(options) if options else "empty_prompt"

    def alternate1(self, args):
        # "a | b" — поддержка простых разделителей с возможными пустыми вариантами
        options = []
        last_was_sep = True

        for arg in args:
            s = resolve_tree(arg, keep_spacing=True)
            tok = (s or "").strip()
            if tok in ("|", ","):
                if last_was_sep:
                    options.append("")  # пустая альтернатива
                last_was_sep = True
                continue
            options.append(s)
            last_was_sep = False

        if last_was_sep:
            options.append("")

        return self.rng.choice(options) if options else "empty_prompt"


    def alternate2(self, args):
        options = [resolve_tree(a, keep_spacing=True).strip() for a in args if resolve_tree(a, keep_spacing=True).strip()]
        suffix = options[0].split("_", 1)[1] if options and "_" in options[0] else ""
        combined = [(o if "_" in o or not suffix else f"{o}_{suffix}") for o in options]
        return "|".join(combined) if combined else "empty_prompt"

    def numbered(self, args):
        quantity = int(args[0])
        distinct = False
        if len(args) > 1:
            mark = str(args[1])
            distinct = mark in ("!", "_")
        # В Transformer дети уже преобразованы → приводим к строке.
        target = args[-1]
        target_str = "" if target is None else str(target)
        # Допускаем, что источником был alternate; если пришла единичная строка — берём её как единственный вариант.
        options = [s for s in (p.strip() for p in target_str.split("|")) if s] or ([target_str] if target_str != "" else [])

        if not options:
            return "empty_prompt"

        if distinct:
            seen = []
            for opt in options:
                if opt not in seen:
                    seen.append(opt)
                if len(seen) >= quantity:
                    break
            selected = seen if len(seen) >= quantity else seen + options[:max(0, quantity - len(seen))]
        else:
            selected = self.rng.choices(options, k=quantity)

        return ", ".join(selected)

    def sequence(self, args, parent=None):
        owner = resolve_tree(args[0], keep_spacing=True) if parent is None else parent
        descriptors = [resolve_tree(arg, keep_spacing=True).strip(" ,~!;") for arg in args[1:] if resolve_tree(arg, keep_spacing=True).strip(" ,~!;")]
        return f"{owner}: {', '.join(descriptors)}"

    def top_level_sequence(self, args):
        owner = resolve_tree(args[0], keep_spacing=True).strip()
        sequences = []
        trailing_text = []
        for child in args[1:]:
            s = child if isinstance(child, str) else resolve_tree(child, keep_spacing=True)
            if not s:
                continue
            s = s.strip()
            if ':' in s:
                head, rhs = s.split(':', 1)
                parts = [head.strip()] + [p.strip() for p in rhs.split(',') if p.strip()]
                for part in parts:
                    if part and part != owner:
                        sequences.append(f"{owner}: {part}")
            else:
                t = s.strip(' ,')
                if t:
                    trailing_text.append(t)
        text = f"{owner} -> {', '.join(sequences)}"
        if trailing_text:
            text += f", {', '.join(trailing_text)}"
        return text

    def top_level_sequence3(self, args):
        owner = resolve_tree(args[0], keep_spacing=True).strip()
        before = []
        after = []
        after_bang = False

        for child in args[1:]:
            # Токен-граница "!!!"
            if isinstance(child, lark.Token):
                if child.type == "WHITESPACE":
                    continue
                if str(child) == "!!!":
                    after_bang = True
                    continue
                if str(child) == ",":
                    continue

            if not after_bang:
                # Левая часть: "owner::: (sequence|plain) (, ...)*"
                if isinstance(child, lark.Tree) and child.data == "sequence":
                    # Принудительно используем owner в левой части
                    txt = self.sequence(child.children, parent=owner)
                    if txt:
                        before.append(txt)
                else:
                    s = resolve_tree(child, keep_spacing=True).strip(" ,")
                    if s:
                        before.append(f"{owner}: {s}")
            else:
                # Правая часть после "!!!": сохраняем собственных владельцев
                if isinstance(child, lark.Tree) and child.data == "sequence":
                    txt = self.sequence(child.children)  # owner берём из самого sequence
                    if txt:
                        after.append(txt)
                else:
                    s = resolve_tree(child, keep_spacing=True).strip(" ,")
                    if s:
                        after.append(s)

        text = f"{owner} -> {', '.join(before)}"
        if after:
            text += f", {', '.join(after)}"
        return text

        
    def nested_sequence(self, args):
        def _is_term(x): return isinstance(x, str) and x in ('!', ';', '~')
        has_term = bool(args and _is_term(args[-1]))
        payload = args[:-1] if has_term else args
        elements = [resolve_tree(arg, keep_spacing=True).strip(" ,~!;")
                    for arg in payload
                    if resolve_tree(arg, keep_spacing=True).strip(" ,~!;")]
        terminator = args[-1] if has_term else None
        if terminator == "~":
            return self.rng.choice(elements) if elements else "empty_prompt"
        return ", ".join(elements)

    def emphasized(self, args):
        """
        (cat) -> (cat:1.1)
        (cat:2) -> (cat:2.0)
        (cat: dog) -> (cat:1.1)
        ((bird:2):3) -> содержит "(bird:2.0)"
        (  fox  :  1.25 ) -> (fox:1.25)
        (wolf : (2)) -> (wolf:2.0)

        Доп. правило: если внутри уже "(...:w)" и рассчитанный внешний вес == 1.1,
        не добавляем второй слой — возвращаем как есть.
        """
        prompt_text = ""
        weight_str: str | None = None

        # Собираем сырой текст аргументов
        raw_parts: list[str] = []
        for a in args:
            if isinstance(a, lark.Token):
                raw_parts.append(a.value)
            elif isinstance(a, lark.Tree):
                raw_parts.append(resolve_tree(a, keep_spacing=True))
            else:
                raw_parts.append(str(a))

        # Уберём служебные токены '(', ')', ':' — оставим полезные куски
        parts = [p.strip() for p in raw_parts if p not in (":", "(", ")", "", None)]
        if parts:
            prompt_text = parts[0].strip()

        if len(parts) >= 2:
            mnum = RE_NUMERIC.search(parts[1])
            if mnum:
                num_txt = mnum.group(0)
                if "." not in num_txt and "e" not in num_txt.lower():
                    try:
                        weight_str = f"{float(num_txt):.1f}"
                    except (ValueError, TypeError):
                        weight_str = _RB_WEIGHT_STR
                else:
                    weight_str = num_txt
            else:
                weight_str = _RB_WEIGHT_STR
        else:
            weight_str = _RB_WEIGHT_STR

        # ★ Если уже имеем "(...:w)" как текст и внешний вес дефолтный — не оборачиваем повторно
        #   Это устраняет артефакт вида "((cat:1.2):1.1)" в Visitor.
        # Внутри ScheduleTransformer.emphasized, перед return: 
        pt = prompt_text.strip()
        # (существующее правило «не оборачивать второй раз при 1.1» — оставить)
        if (
            weight_str in _RB_WEIGHT_STRS
            and len(pt) >= 5 and pt[0] == "(" and pt[-1] == ")" and ":" in pt
        ):
            return pt

        # ★ добавка: единообразно разэкранируем литералы в prompt_text
        prompt_text = _unescape_literals(prompt_text)

        return f"({prompt_text}:{weight_str})"




def _compact_schedule(raw: list[list], steps: int) -> list[list]:
    """Схлопнуть смежные шаги с одинаковым текстом → список [end_step, text].

    raw — список [step, text] по одной записи на шаг (выход пошагового цикла).
    Последний сегмент всегда растягивается до steps.
    """
    result: list[list] = []
    for step, text in raw:
        if result and result[-1][1] == text:
            result[-1][0] = step
        else:
            result.append([step, text])
    if result and result[-1][0] < steps:
        result[-1][0] = steps
    return result


def _merge_child_schedules_for_scheduled(
    child_scheds: list[list],   # расписание каждого prompt-слота внутри [...]
    boundary: int,               # конец периода переключений [1..boundary]
    steps: int,
    prefix: str = "",
    suffix: str = "",
    is_reverse: bool = False,
) -> list[list]:
    """Аналог _build_bracket_inner_schedules, но принимает расписания дочерних
    узлов вместо статичных строк.

    Используется только из CollectSteps.visit_scheduled когда inner-prompt
    содержит dynamic-узлы (alternate, nested scheduled и т.п.).
    Fast-path ветки и _build_bracket_inner_schedules(list[str]) не трогаем.
    """
    if not child_scheds:
        return [[steps, _concat_prefix_text_suffix(prefix, "", suffix)]]

    if is_reverse:
        child_scheds = list(reversed(child_scheds))

    def pick(sched: list, step: int) -> str:
        for e, t in sched:
            if step <= int(e):
                return t
        return sched[-1][1]

    n = len(child_scheds)

    raw: list[list] = []

    if n == 1:
        if boundary >= steps:
            # boundary >= steps: "after" window never activates.
            # Show content on all steps (mirrors _build_bracket_inner_schedules).
            for step in range(1, steps + 1):
                raw.append([step, _concat_prefix_text_suffix(prefix, pick(child_scheds[0], step), suffix)])
        else:
            # 1 слот: шаги 1..boundary — пустой центр, boundary+1..steps — контент
            for step in range(1, boundary + 1):
                raw.append([step, _concat_prefix_text_suffix(prefix, "", suffix)])
            for step in range(boundary + 1, steps + 1):
                raw.append([step, _concat_prefix_text_suffix(prefix, pick(child_scheds[0], step), suffix)])
    else:
        # n>=2: равномерно делим [1..boundary] между слотами
        step_size = float(boundary) / n
        for i, sched in enumerate(child_scheds):
            seg_start = _clamp(round(i * step_size) + 1, steps)
            seg_end   = _clamp(round((i + 1) * step_size), steps)
            for step in range(seg_start, seg_end + 1):
                raw.append([step, _concat_prefix_text_suffix(prefix, pick(sched, step), suffix)])
        # Хвост после boundary
        for step in range(boundary + 1, steps + 1):
            raw.append([step, _concat_prefix_text_suffix(prefix, pick(child_scheds[-1], step), suffix)])

    return _compact_schedule(raw, steps)


# ──────────────────────────────────────────────────────────────────────────────
# Visitor: сборка расписаний
# ──────────────────────────────────────────────────────────────────────────────

class CollectSteps(lark.Visitor):
    def __init__(self, steps, prefix="", suffix="", use_scheduling=True, seed=None, _rng=None):
        super().__init__()
        self.steps = steps
        self.prefix = prefix
        self.suffix = suffix
        self.use_scheduling = use_scheduling
        self.seed = seed
        # Если передан уже инициализированный rng — используем его (разделяем состояние
        # с родительским CollectSteps, чтобы [a|b] car [c|d] давал независимые выборы).
        # Иначе создаём свежий генератор из seed.
        if _rng is not None:
            self.rng = _rng
        else:
            self.rng = random.Random(seed) if seed is not None else random
        self.schedules = []

    # — fast-path для "[...]:N [reverse]" на уровне prompt
    def visit_prompt(self, tree):
        full = resolve_tree(tree, keep_spacing=True)
        if not self.use_scheduling:
            return self._default_visit(tree)
        schedules = _try_fast_after_scheduler(
            full,
            self.steps,
            prefix=self.prefix,
            suffix=self.suffix,
            allow_ranges=True,
            allow_bracket_after=True,
            disallow_alt_bang=True,
            boundary_fallback=None,
        )
        if schedules is not None:
            return schedules
        return self._default_visit(tree)


    def visit(self, tree):
        if isinstance(tree, lark.Tree):
            method_name = f"visit_{tree.data}"
            method = getattr(self, method_name, self._default_visit)
            return method(tree)
        elif isinstance(tree, lark.Token):
            return self._visit_token(tree)
        return []

    def visit_start(self, tree):
        full = resolve_tree(tree, keep_spacing=True).strip()
        if not self.use_scheduling:
            return self._default_visit(tree)
        schedules = _try_fast_after_scheduler(
            full,
            self.steps,
            prefix=self.prefix,
            suffix=self.suffix,
            allow_ranges=True,
            allow_bracket_after=True,
            disallow_alt_bang=True,
            boundary_fallback=_make_boundary_spec(1.0, assume_absolute=True),
        )
        if schedules is not None:
            return schedules

        # 0) owner::a::b!!, trailing
        if "::" in full and "!!" in full and all(ch not in full for ch in '[]()'):
            left, trailing = full.split("!!", 1)
            owner, rest = left.split("::", 1)
            descriptors = [x.strip(' ,~!;') for x in rest.split('::') if x.strip(' ,~!;')]
            sequences = [f"{owner.strip()}: {d}" for d in descriptors]
            trailing_text = [t.strip(' ,') for t in trailing.split(',') if t.strip(' ,')]
            out = f"{owner.strip()} -> {', '.join(sequences)}"
            if trailing_text:
                out += f", {', '.join(trailing_text)}"
            return [[self.steps, _collapse_spaces(self.prefix + out + self.suffix)]]

        # 1) owner::a::b!
        if '::' in full and (full.endswith('!') or full.endswith(';')) and all(ch not in full for ch in '[]()'):
            owner, rest = full.split('::', 1)
            rest = rest[:-1]
            descriptors = [x.strip(' ,~!;') for x in rest.split('::') if x.strip(' ,~!;')]
            text = f"{owner.strip()}: {', '.join(descriptors)}"
            return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]

        # 2) [a:b:...:N] — допускаем скобки внутри и безопасно сплитим,
        # запускаем fast-path только если в строке ровно одна пара []
        if full.count('[') == 1 and full.count(']') == 1 and '|' not in full:
            try:
                lb, rb = full.index('['), full.rindex(']')
                pre, inner, post = full[:lb], full[lb+1:rb], full[rb+1:]
                parts = _split_top_level_colon_keep_empty(inner)
                if len(parts) >= 2 and re.fullmatch(r'[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?', parts[-1]):
                    boundary_spec = _make_boundary_spec(float(parts[-1]))
                    prompts = [_unescape_literals(p.strip()) for p in parts[:-1]]
                    if prompts:
                        post, rev_flag = _strip_reverse_from_post(post or "")
                        if rev_flag:
                            prompts = list(reversed(prompts))
                        boundary = _to_end_step(boundary_spec, self.steps)
                        schedules = _build_bracket_inner_schedules(self.prefix + pre, prompts, boundary, post + self.suffix, self.steps)
                        return _finalize_schedules(schedules)
            except ValueError:
                pass

        return self._default_visit(tree)

    def _default_visit(self, tree):
        # 1) Собираем расписание для каждого ребёнка без внешних аффиксов
        child_scheds = []
        for child in tree.children:
            if isinstance(child, lark.Token):
                if child.type == "WHITESPACE":
                    # пробел — константное расписание
                    child_scheds.append([[self.steps, " "]])
                else:
                    child_scheds.append([[self.steps, str(child)]])
            else:
                sub = CollectSteps(
                    self.steps,
                    prefix="",   # важно: без префикса/суффикса на уровне детей
                    suffix="",
                    use_scheduling=self.use_scheduling,
                    seed=self.seed,
                    _rng=self.rng,  # ФИЙ14: передаём общий rng, чтобы разные [a|b] блоки
                                    # не сбрасывали генератор к тому же начальному состоянию
                ).visit(child)
                if not sub:
                    # на всякий случай — literal fallback
                    sub = [[self.steps, resolve_tree(child, keep_spacing=True)]]
                child_scheds.append(sub)

        if not child_scheds:
            return [[self.steps, _collapse_spaces(self.prefix + self.suffix)]]

        # 2) Объединённые границы всех детей
        boundaries = sorted({int(e) for sched in child_scheds for (e, _) in sched})
        if boundaries and boundaries[-1] < self.steps:
            boundaries.append(self.steps)
        elif not boundaries:
            boundaries = [self.steps]

        def pick_text(sched, step):
            for e, t in sched:
                if step <= int(e):
                    return t
            return sched[-1][1]

        # 3) Склейка «картинки шага» = конкатенация текущих текстов детей
        out = []
        for end in boundaries:
            parts = [pick_text(s, end) for s in child_scheds]

            if parts:
                combined = parts[0]
                for part in parts[1:]:
                    combined = _concat_prefix_text_suffix(combined, "", part)
            else:
                combined = ""

            text = _concat_prefix_text_suffix(self.prefix, combined, self.suffix)
            text = _apply_and(_collapse_spaces(text))
            if not out or out[-1][1] != text:
                out.append([end, text])

        return out


    def _visit_token(self, token):
        if token.type == "WHITESPACE":
            return []
        return [[self.steps, _collapse_spaces(self.prefix + str(token) + self.suffix)]]

    def visit_plain(self, tree):
        text = resolve_tree(tree, keep_spacing=True)
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]

    def visit_top_level_sequence3(self, tree):
        transformer = ScheduleTransformer(self.steps, 1, self.seed)
        text = transformer.transform(tree)
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]

    def visit_top_level_sequence(self, tree):
        # Аналогично visit_top_level_sequence3, но для 'owner::a::b!!, tail'
        transformer = ScheduleTransformer(self.steps, 1, self.seed)
        text = transformer.transform(tree)
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]
    def visit_scheduled(self, tree):
            if not tree.children:
                return [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix)]]

            # IMPORTANT:
            # Do NOT flatten the whole subtree to find NUMBER / prompt nodes.
            # Otherwise nested constructs like "(red eyes:1.4)" will leak their NUMBER (1.4)
            # and inner prompt nodes into the scheduling boundary/alternates.
            #
            # We only consider DIRECT children of the `scheduled` node.
            direct_prompts = [
                c for c in tree.children
                if isinstance(c, lark.Tree) and getattr(c, "data", None) == "prompt"
            ]
            boundary_value = _extract_scheduled_boundary_value(tree.children)
            step_range_list = next(
                (c for c in tree.children if isinstance(c, lark.Tree) and getattr(c, "data", None) == "step_range_list"),
                None,
            )
            is_reverse = any(
                isinstance(c, lark.Tree) and getattr(c, "data", None) == "reverse_flag"
                for c in tree.children
            )

            # Проверяем, есть ли dynamic-узлы внутри prompt-детей
            # (alternate, nested scheduled и т.п.)
            def _has_dynamic_children(prompt_node) -> bool:
                DYNAMIC = {"scheduled", "alternate", "alternate1", "alternate2",
                           "alternate_distinct", "grouped", "numbered"}
                return any(
                    isinstance(c, lark.Tree) and getattr(c, "data", None) in DYNAMIC
                    for c in prompt_node.children
                )

            has_dynamic = any(_has_dynamic_children(p) for p in direct_prompts)

            if has_dynamic:
                # Новый путь: строим расписание для каждого prompt-слота через CollectSteps
                child_scheds = []
                for p in direct_prompts:
                    sub = CollectSteps(
                        self.steps, prefix="", suffix="",
                        use_scheduling=self.use_scheduling,
                        seed=self.seed, _rng=self.rng,
                    ).visit(p)
                    if not sub:
                        sub = [[self.steps, _unescape_literals(resolve_tree(p, keep_spacing=True))]]
                    child_scheds.append(sub)

                if is_reverse:
                    child_scheds = list(reversed(child_scheds))

                if boundary_value is None:
                    return [[self.steps, _collapse_spaces(self.prefix + resolve_tree(tree, keep_spacing=True) + self.suffix)]]

                if step_range_list is not None:
                    # ranges-mode с dynamic-детьми:
                    # строим step-mapping slot[step] = text из child_scheds
                    # аналогично _build_ranges_schedules_from_components, но
                    # для каждого range берём текст из child-расписания на данном шаге.

                    def _pick_child(sched, step):
                        for e, t in sched:
                            if step <= int(e):
                                return t
                        return sched[-1][1]

                    ranges_dyn: list[tuple[int, int]] = []

                    def _to_steps_local_dyn(val_txt: str, is_pct: bool) -> int:
                        v = float(val_txt)
                        if is_pct:
                            v = v / 100.0 * float(self.steps)
                        return _clamp(int(round(v)), self.steps)

                    for r in step_range_list.children:
                        if not isinstance(r, lark.Tree):
                            continue
                        if getattr(r, "data", None) not in ("range_abs", "range_pct"):
                            continue
                        is_pct = (r.data == "range_pct")
                        try:
                            a = str(r.children[0].value)
                            b = str(r.children[1].value)
                        except (AttributeError, TypeError):
                            continue
                        sv = _to_steps_local_dyn(a, is_pct)
                        ev = _to_steps_local_dyn(b, is_pct)
                        if sv > ev:
                            continue
                        ranges_dyn.append((sv, ev))

                    if not ranges_dyn:
                        return [[self.steps, _concat_prefix_text_suffix(self.prefix, "", self.suffix)]]

                    # slot[step] = center text (last-wins)
                    slot = [""] * (self.steps + 1)
                    for idx, (sv, ev) in enumerate(ranges_dyn):
                        child_idx = idx % len(child_scheds)
                        for step in range(sv, ev + 1):
                            slot[step] = _pick_child(child_scheds[child_idx], step)

                    # Compress slot → schedule segments
                    raw_dyn = []
                    for step in range(1, self.steps + 1):
                        text = _concat_prefix_text_suffix(self.prefix, slot[step], self.suffix)
                        raw_dyn.append([step, text])
                    return _finalize_schedules(_compact_schedule(raw_dyn, self.steps))

                boundary = _to_end_step(boundary_value, self.steps)
                return _finalize_schedules(
                    _merge_child_schedules_for_scheduled(
                        child_scheds, boundary,
                        self.steps, self.prefix, self.suffix,
                        is_reverse=False,  # already reversed above
                    )
                )

            # Статичный путь (нет dynamic-детей) — без изменений
            # Тексты промптов (только верхнего уровня)
            prompt_texts = [_unescape_literals(resolve_tree(p, keep_spacing=True)) for p in direct_prompts]
            if is_reverse:
                prompt_texts = list(reversed(prompt_texts))

            if boundary_value is None:
                # В норме такого быть не должно, но оставим безопасный fallback
                return [[self.steps, _collapse_spaces(self.prefix + resolve_tree(tree, keep_spacing=True) + self.suffix)]]

            # ── RANGES MODE: "[...]: N 2-3,4-9 [reverse]" ─────────────────────────
            if step_range_list is not None:
                ranges: list[tuple[int, int]] = []

                def _to_steps_local(val_txt: str, is_pct: bool) -> int:
                    v = float(val_txt)
                    if is_pct:
                        v = v / 100.0 * float(self.steps)
                    return _clamp(int(round(v)), self.steps)

                # Preserve original order (needed for last-wins overlap policy)
                for r in step_range_list.children:
                    if not isinstance(r, lark.Tree):
                        continue
                    if getattr(r, "data", None) not in ("range_abs", "range_pct"):
                        continue

                    is_pct = (r.data == "range_pct")
                    try:
                        a = str(r.children[0].value)
                        b = str(r.children[1].value)
                    except (AttributeError, TypeError):
                        continue

                    start_val = _to_steps_local(a, is_pct)
                    end_val = _to_steps_local(b, is_pct)

                    if start_val > end_val:
                        tok = f"{a}{'%' if is_pct else ''}-{b}{'%' if is_pct else ''}"
                        raise PromptSyntaxError(
                            f"Reverse range detected: {tok}",
                            kind="reverse_range",
                            token=tok,
                            full=resolve_tree(tree, keep_spacing=True),
                        )

                    # Single-step allowed (start == end)
                    ranges.append((start_val, end_val))

                if not ranges:
                    return [[self.steps, _concat_prefix_text_suffix(self.prefix, "", self.suffix)]]

                # Build with unified policy: last-wins overlaps, strict reverse, single-step ok
                # NOTE: prompt_texts already reversed above (if is_reverse), so pass rev_flag=False
                # to avoid double-reverse inside _build_ranges_schedules_from_components.
                schedules = _build_ranges_schedules_from_components(
                    prompt_texts,
                    ranges,
                    self.steps,
                    prefix=self.prefix,
                    suffix=self.suffix,
                    rev_flag=False,
                    cycle_prompts=True,
                    empty_center="",
                )
                return _finalize_schedules(schedules)

            # ── Boundary mode: "[...:N]" / "[...]:N" ─────────────────────────────
            boundary = _to_end_step(boundary_value, self.steps)
            schedules = _build_bracket_after_schedules(self.prefix, prompt_texts, boundary, self.suffix, self.steps)
            return _finalize_schedules(schedules)

    def visit_alternate(self, tree):
        vals = []
        prev_was_pipe = True   # ведущий '|' → пустая опция в начале
        for child in tree.children:
            if child is None:
                vals.append("")
                prev_was_pipe = False
                continue
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                continue
            tok = str(child).strip()
            # '[' и ']' — структурные скобки от !alternate, игнорируем
            if tok in ("[", "]", "!"):
                continue
            if tok in ("|", ","):
                if prev_was_pipe:
                    vals.append("")   # два '|' подряд → пустая опция
                prev_was_pipe = True
                continue
            child_schedules = self.visit(child)
            if child_schedules:
                # Если дочерний узел сам даёт несколько вариантов (например alternate1),
                # учитываем их все, сохраняя порядок.
                dedup_vals: list[str] = []
                for _, t in child_schedules:
                    if not dedup_vals or dedup_vals[-1] != t:
                        dedup_vals.append(t)
                child_vals = []
                for t in dedup_vals:
                    if t not in child_vals:
                        child_vals.append(t)
            else:
                child_vals = [resolve_tree(child, keep_spacing=True)]
            if not child_vals:
                child_vals = [""]
            vals.extend(child_vals)
            prev_was_pipe = False

        if prev_was_pipe:
            vals.append("")   # trailing '|' → пустая завершающая опция

        # ИСПРАВЛЕНИЕ: Более точная фильтрация
        # Удаляем избыточные пустые, но сохраняем хотя бы один если все пустые
        if ALLOW_EMPTY_ALTERNATE:
            # Разрешаем пустые, не фильтруем
            filtered_vals = vals if vals else [""]
        else:
            # Убираем все пустые
            filtered_vals = [v for v in vals if v.strip()]
            if not filtered_vals:
                filtered_vals = [""]

        if not filtered_vals:
            return [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix)]]

        if EXPAND_ALTERNATE_PER_STEP:
            schedules = []
            for step in range(1, self.steps + 1):
                choice = filtered_vals[(step - 1) % len(filtered_vals)]
                schedules.append([step, _collapse_spaces(self.prefix + choice + self.suffix)])
            return [[e, _apply_and(t)] for e, t in schedules]

        choice = self.rng.choice(filtered_vals)
        return [[self.steps, _collapse_spaces(self.prefix + choice + self.suffix)]]
        
    def visit_alternate_distinct(self, tree):
        options = []
        for child in tree.children:
            if isinstance(child, lark.Token):
                tok = str(child).strip()
                if tok in ("|", ",", "[", "]", "!"):
                    continue
                if tok:
                    options.append([tok])
                else:
                    options.append([""])
                continue
            child_schedules = self.visit(child)
            child_options = [sched[1] for sched in child_schedules if sched[1] != ""]
            if child_options:
                options.append(child_options)
            else:
                txt = resolve_tree(child, keep_spacing=True)
                options.append([txt])
        flat = [opt for group in options for opt in group]
        if not flat:
            return [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix)]]
        selected = self.rng.choice(flat)
        return [[self.steps, _collapse_spaces(self.prefix + selected + self.suffix)]]

    def visit_alternate1(self, tree):
        # Собираем варианты, включая потенциально пустые
        options = []
        last_was_sep = True
        for child in tree.children:
            if isinstance(child, lark.Token):
                tok = str(child).strip()
                if tok in ("|", ","):
                    if last_was_sep:
                        options.append("")
                    last_was_sep = True
                    continue
                if child.type == "WHITESPACE":
                    continue
                s = str(child)
            else:
                s = resolve_tree(child, keep_spacing=True)
            if s is None:
                continue
            tok = s.strip()
            if tok in ("|", ","):
                if last_was_sep:
                    options.append("")
                last_was_sep = True
                continue
            options.append(s)
            last_was_sep = False
        if last_was_sep:
            options.append("")

        if not options:
            return [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix)]]

        if EXPAND_ALTERNATE_PER_STEP:
            schedules = []
            for step in range(1, self.steps + 1):
                choice = options[(step - 1) % len(options)]
                schedules.append([step, _collapse_spaces(self.prefix + choice + self.suffix)])
            return [[e, _apply_and(t)] for e, t in schedules]
        else:
            choice = self.rng.choice(options)
            return [[self.steps, _collapse_spaces(self.prefix + choice + self.suffix)]]


    def visit_alternate2(self, tree):
        opts = []
        for c in tree.children:
            s = resolve_tree(c, keep_spacing=True).strip()
            if not s or s in ("|", ","):
                continue
            opts.append(s)
        options = opts
        suffix = options[0].split("_", 1)[1] if options and "_" in options[0] else ""
        combined = []
        for opt in options:
            combined.append(opt if "_" in opt or not suffix else f"{opt}_{suffix}")
        text = "|".join(combined) if combined else "empty_prompt"
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]

    def visit_weighted(self, tree):
        tr = ScheduleTransformer(self.steps, 1, self.seed)
        text = tr.transform(tree)  # '(token:1.2)'
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]

    def visit_grouped(self, tree):
        # Собираем варианты по дочерним узлам (сохраняем пустые элементы)
        all_options = []
        for child in tree.children:
            if isinstance(child, lark.Token) and child.type == "WHITESPACE":
                continue
            child_schedules = self.visit(child)
            # берём тексты без обрезки пустых, чтобы пустые варианты учитывались
            child_opts = [sched[1] for sched in child_schedules]
            if not child_opts:
                child_opts = [resolve_tree(child, keep_spacing=True)]
            all_options.append(child_opts)

        # Оценка числа комбинаций:
        total_combos = 1
        for opts in all_options:
            total_combos *= max(1, len(opts))

        if total_combos > GROUP_COMBO_LIMIT:
            mode = GROUP_COMBO_FALLBACK
            if mode == "literal":
                # Возвращаем псевдолитерал, начинающийся с "{[" чтобы сохранить сигнал исходной формы
                inner = ", ".join(resolve_tree(c, keep_spacing=True) for c in tree.children if not (isinstance(c, lark.Token) and c.type == "WHITESPACE"))
                original = "{[" + inner + "]}"
                return [[self.steps, _collapse_spaces(self.prefix + original + self.suffix)]]
            elif mode == "sample":
                k = GROUP_COMBO_LIMIT
                lens = [max(1, len(opts)) for opts in all_options]
                seen: set[tuple[int, ...]] = set()
                out = []
                max_tries = k * 10
                tries = 0
                while len(seen) < min(k, total_combos) and tries < max_tries:
                    idx = tuple(self.rng.randrange(n) for n in lens)
                    tries += 1
                    if idx in seen:
                        continue
                    seen.add(idx)
                    combo = [all_options[d][i] if all_options[d] else "" for d, i in enumerate(idx)]
                    text = ", ".join(combo).strip()
                    if text:
                        out.append([self.steps, _collapse_spaces(self.prefix + text + self.suffix)])
                if out:
                    return out
                # иначе сваливаемся на усечение (truncate)

        out = []
        for i, combo in enumerate(product(*all_options)):
            if i >= GROUP_COMBO_LIMIT:
                break
            text = ", ".join(combo).strip()
            if text:
                out.append([self.steps, _collapse_spaces(self.prefix + text + self.suffix)])
        return out or [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix)]]

    def visit_sequence(self, tree):
        transformer = ScheduleTransformer(self.steps, 1, self.seed)
        text = transformer.transform(tree)
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]

    def visit_nested_sequence(self, tree):
        # Терминатор ('!', ';', '~') — ОПЦИОНАЛЬНЫЙ. Проверяем до среза,
        # иначе [:-1] отрежет последний prompt когда терминатора нет.
        last = tree.children[-1] if tree.children else None
        has_term = isinstance(last, lark.Token) and last.value in ('!', ';', '~')
        payload = tree.children[:-1] if has_term else tree.children
        elements = [resolve_tree(c, keep_spacing=True).strip(" ,~!;")
                    for c in payload
                    if resolve_tree(c, keep_spacing=True).strip(" ,~!;")]
        terminator = last if has_term else None
        if terminator and terminator.value == "~":
            text = self.rng.choice(elements) if elements else "empty_prompt"
        else:
            # Согласовано с ScheduleTransformer.nested_sequence: возвращаем обычный текст
            text = ", ".join(elements) if elements else "empty_prompt"
        return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]

    def visit_numbered(self, tree):
        quantity = int(tree.children[0])
        distinct = False
        if len(tree.children) > 1:
            mark = tree.children[1]
            try:
                if str(mark) in ('!', '_'):
                    distinct = True
            except (AttributeError, TypeError):
                pass
        target = tree.children[-1]

        options = []
        def add_opts(node):
            if isinstance(node, lark.Tree):
                if getattr(node, "data", None) in ("alternate", "alternate1", "alternate2", "alternate_distinct", "prompt"):
                    for ch in node.children:
                        add_opts(ch)
                else:
                    txt = resolve_tree(node, keep_spacing=True)
                    if txt is not None:
                        options.append(txt)
            elif isinstance(node, lark.Token):
                if node.type == "WHITESPACE":
                    return
                txt = str(node)
                if txt in ("[", "]", "|", ",", "!"):
                    return
                options.append(txt)

        add_opts(target)
        options = [
            opt.strip() if isinstance(opt, str) else opt
            for opt in options
            if opt is not None
        ]
        if not options:
            child_schedules = self.visit(target)
            options = [s[1] for s in child_schedules]
        if not options:
            return [[self.steps, _collapse_spaces(self.prefix + "empty_prompt" + self.suffix)]]

        if distinct:
            seen = []
            for opt in options:
                if opt not in seen:
                    seen.append(opt)
                if len(seen) == quantity:
                    break
            if len(seen) < quantity:
                seen += self.rng.choices(options, k=quantity - len(seen))
            selected = seen
        else:
            selected = self.rng.choices(options, k=quantity)

        return [[self.steps, _collapse_spaces(self.prefix + ", ".join(selected) + self.suffix)]]

    def visit_and_rule(self, tree):
        # Static nodes: transform directly to text (no schedule needed)
        STATIC_NODES = ("weighted", "emphasized")

        branch_scheds: list[list] = []
        tr = ScheduleTransformer(self.steps, 1, self.seed)

        for child in tree.children:
            # Skip structural tokens (& operator, whitespace)
            if isinstance(child, lark.Token):
                continue

            node_type = getattr(child, "data", None)

            if node_type in STATIC_NODES:
                # Fast static path — transform to "(text:weight)" string
                text = tr.transform(child)
                branch_scheds.append([[self.steps, text]])
            else:
                # General dynamic path: alternate, grouped, scheduled,
                # numbered, alternate1/2/distinct — build sub-schedule
                sub = CollectSteps(
                    self.steps, prefix="", suffix="",
                    use_scheduling=self.use_scheduling,
                    seed=self.seed, _rng=self.rng,
                ).visit(child)
                if not sub:
                    sub = [[self.steps, resolve_tree(child, keep_spacing=True)]]
                branch_scheds.append(sub)

        if not branch_scheds:
            return [[self.steps, _collapse_spaces(self.prefix + self.suffix)]]

        # Merge all branches over their combined boundary points
        boundaries = sorted({int(e) for s in branch_scheds for (e, _) in s})
        if boundaries and boundaries[-1] < self.steps:
            boundaries.append(self.steps)

        def _pick(sched, step):
            for e, t in sched:
                if step <= int(e):
                    return t
            return sched[-1][1]

        out = []
        for end in boundaries:
            parts = [_pick(s, end) for s in branch_scheds]
            text = f" {ATTENTION_AND_OPERATOR} ".join(p for p in parts if p)
            text = _collapse_spaces(self.prefix + text + self.suffix)
            if not out or out[-1][1] != text:
                out.append([end, text])
        return out

    def visit_emphasized(self, tree):
        # Check if the inner prompt contains dynamic nodes that produce schedules
        inner_prompt = next(
            (c for c in tree.children
             if isinstance(c, lark.Tree) and c.data == "prompt"),
            None,
        )
        DYNAMIC_NODES = {
            "scheduled", "alternate", "alternate1", "alternate2",
            "alternate_distinct", "grouped", "numbered",
        }
        has_dynamic = inner_prompt and any(
            isinstance(c, lark.Tree) and getattr(c, "data", None) in DYNAMIC_NODES
            for c in inner_prompt.children
        )

        if not has_dynamic:
            # Fast static path — no change from original
            tr = ScheduleTransformer(self.steps, 1, self.seed)
            text = tr.transform(tree)
            return [[self.steps, _collapse_spaces(self.prefix + text + self.suffix)]]

        # Dynamic path: build inner schedule, wrap each segment in (text:weight)
        weight_tok = next(
            (c for c in tree.children
             if isinstance(c, lark.Token) and c.type == "NUMBER"),
            None,
        )
        if weight_tok is not None:
            w_raw = weight_tok.value
            try:
                w_f = float(w_raw)
                w_str = f"{w_f:.1f}" if ("." not in w_raw and "e" not in w_raw.lower()) else w_raw
            except (ValueError, TypeError):
                w_str = _RB_WEIGHT_STR
        else:
            w_str = _RB_WEIGHT_STR

        inner_sched = CollectSteps(
            self.steps, prefix="", suffix="",
            use_scheduling=self.use_scheduling,
            seed=self.seed, _rng=self.rng,
        ).visit(inner_prompt)

        out = []
        for end, text in (inner_sched or [[self.steps, ""]]):
            t = text.strip()
            wrapped = f"({t}:{w_str})" if t else ""
            full = _collapse_spaces(self.prefix + wrapped + self.suffix)
            if not out or out[-1][1] != full:
                out.append([end, full])
        return out

    def __call__(self, tree):
        self.schedules = self.visit(tree)
        # dedup по паре (step,text)
        uniq = []
        seen = set()
        for end_step, text in self.schedules or []:
            key = (int(end_step), text)
            if key not in seen:
                uniq.append([int(end_step), text])
                seen.add(key)
        # Паритет с get_schedule: последний сегмент эксплицитно тянется до self.steps
        if uniq:
            try:
                last_end = int(uniq[-1][0])
            except (ValueError, TypeError):
                last_end = None
            if isinstance(last_end, int) and last_end < int(self.steps):
                uniq.append([int(self.steps), uniq[-1][1]])
        result = uniq or [[self.steps, _collapse_spaces(self.prefix + resolve_tree(tree, keep_spacing=True) + self.suffix)]]
        if not self.use_scheduling:
            return _apply_scheduling_mode(result, self.steps, False)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Внешние утилиты для расписаний
# ──────────────────────────────────────────────────────────────────────────────

def at_step_from_schedule(step: int, schedule: Sequence[Sequence[int | str]]) -> str:
    if not schedule:
        return ""
    for end_step, text in schedule:
        try:
            if step <= int(end_step):
                return text
        except (ValueError, TypeError):
            continue
    return schedule[-1][1]

def at_step(step: int, prompt_or_schedule, *, steps: int | None = None,
            seed: int | None = 42, use_visitor: bool = True, use_scheduling: bool = True) -> str:
    if isinstance(prompt_or_schedule, list) and prompt_or_schedule and isinstance(prompt_or_schedule[0], list):
        return at_step_from_schedule(step, prompt_or_schedule)
    prompt = str(prompt_or_schedule)
    if steps is None:
        raise ValueError("steps is required when passing a prompt string to at_step(...)")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if step <= 0 or step > steps:
        raise ValueError(f"step must be between 1 and {steps}, got {step}")
    sched = get_schedule(prompt, steps, use_scheduling=use_scheduling, seed=seed, use_visitor=use_visitor)
    return at_step_from_schedule(step, sched)

def _apply_and(text: str) -> str:
    # 1) Нормализуем пробелы вокруг двоеточия в шаблоне 'token : number' -> 'token:number'
    text = re.sub(
        rf'(?<=[^\W\d_])\s*[:：]\s*(?=(?:{NUMERIC_RE})\b)',
        ':',
        text,
    )

    # 2) Приводим операторы к единому виду, игнорируя экранированный '&'
    text = _normalize_and_operators_for_parse(text)
    text = re.sub(
        rf'(?<!\\)(\S)\s*{re.escape(ATTENTION_AND_OPERATOR)}\s*(\S)',
        lambda m: f"{m.group(1)} {ATTENTION_AND_OPERATOR} {m.group(2)}",
        text,
    )  # Нормализуем пробелы вокруг &

    # 3) Схлопнуть множественные пробелы
    text = _re_ws_collapse.sub(" ", text)

    # 4) Разэкранируем литералы перед финальной отдачей
    text = _unescape_literals(text)
    text = text.replace(ESCAPED_AMP_PLACEHOLDER, "&")

    # 5) Удаляем лишние/висячие '&' из начала и конца выражения.
    tokens = [tok for tok in text.strip().split(" ") if tok]
    if not tokens:
        return ""
    if len(tokens) == 1 and tokens[0] == ATTENTION_AND_OPERATOR:
        return ATTENTION_AND_OPERATOR

    cleaned_tokens: list[str] = []
    for i, tok in enumerate(tokens):
        if tok != ATTENTION_AND_OPERATOR:
            cleaned_tokens.append(tok)
            continue

        has_prev_term = bool(cleaned_tokens) and cleaned_tokens[-1] != ATTENTION_AND_OPERATOR
        has_next_term = any(t != ATTENTION_AND_OPERATOR for t in tokens[i + 1 :])
        if has_prev_term and has_next_term:
            cleaned_tokens.append(tok)

    return " ".join(cleaned_tokens)


def _validate_inputs(prompt: str, steps: int) -> None:
    """Единая валидация входных параметров"""
    if not isinstance(steps, int):
        raise TypeError(f"steps must be int, got {type(steps)}")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if steps > 10000:
        raise ValueError(f"steps too large (max 10000), got {steps}")
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be str, got {type(prompt)}")


def _copy_schedule(schedule: Sequence[Sequence[int | str]]) -> list[list[int | str]]:
    """Defensive copy so callers cannot mutate cached schedule objects."""
    return [[row[0], row[1]] for row in schedule]


def _apply_scheduling_mode(
    schedule: Sequence[Sequence[int | str]],
    steps: int,
    use_scheduling: bool,
) -> list[list[int | str]]:
    """Finalize schedules and optionally collapse them to a single static segment."""
    finalized = _finalize_schedules(schedule)
    if use_scheduling:
        return finalized
    final_text = at_step_from_schedule(int(steps), finalized)
    return [[int(steps), final_text]]


def _strict_check_delimiter_balance(text: str) -> None:
    """Best-effort bracket/paren balance check for strict lint mode."""
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {v: k for k, v in opening.items()}
    stack: list[str] = []
    escaped = False

    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch in opening:
            stack.append(ch)
            continue
        if ch in closing:
            if not stack or stack[-1] != closing[ch]:
                raise PromptSyntaxError(
                    f"Unmatched closing delimiter: {ch}",
                    kind="syntax_error",
                    token=ch,
                    full=text,
                )
            stack.pop()

    if stack:
        opener = stack[-1]
        raise PromptSyntaxError(
            f"Unclosed delimiter: {opener}",
            kind="syntax_error",
            token=opener,
            full=text,
        )


_RE_STRICT_RANGE_PREFIX = re.compile(r'^\s*(?:-|%|\d+\s*%?\s*-)')
_RE_STRICT_RANGE_CHARS = re.compile(r'^[\s\d%,-]+')
_RE_STRICT_RANGE_TOKEN = re.compile(r'^\d+%?\s*-\s*\d+%?$')
_RE_DUPLICATE_REVERSE_PREFIX = re.compile(r'^\s*reverse(?:\s*,)?\s+reverse\b')
_RE_INCOMPLETE_SEQUENCE_SUFFIX = re.compile(r'(?<!\\)::\s*$')
_RE_INCOMPLETE_SEQUENCE_BEFORE_OPERATOR = re.compile(r'(?<!\\)::(?=\s*(?:AND\b|&|,|$|\)|\]|\}))')
_RE_BOUNDARY_WITH_OPTIONAL_PERCENT = re.compile(rf'{NUMERIC_RE}%?')


def _iter_parenthesized_spans(text: str):
    """Yield balanced parenthesized spans as (start, end, inner_text)."""
    stack: list[int] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2 if i + 1 < len(text) else 1
            continue
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            start = stack.pop()
            yield start, i + 1, text[start + 1 : i]
        i += 1


def _iter_postfix_scheduler_tail_starts(text: str):
    """Yield indices right after top-level postfix scheduler boundaries like ``]:10``."""
    stack: list[int] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2 if i + 1 < len(text) else 1
            continue
        if ch == "[":
            stack.append(i)
            i += 1
            continue
        if ch == "]" and stack:
            stack.pop()
            if not stack:
                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1
                if j < len(text) and text[j] == ":":
                    j += 1
                    while j < len(text) and text[j].isspace():
                        j += 1
                    m = _RE_BOUNDARY_WITH_OPTIONAL_PERCENT.match(text[j:])
                    if m is not None:
                        yield j + m.end()
        i += 1


def _raise_invalid_range_token(full: str, token: str) -> None:
    bad = token.strip() or token
    raise PromptSyntaxError(
        f"Invalid range token {bad!r}",
        kind="invalid_range_token",
        token=bad or token,
        full=full,
    )


def _strict_check_postfix_scheduler_surfaces(text: str) -> None:
    """Reject obviously broken postfix scheduler tails before lenient parsing hides them."""
    for tail_start in _iter_postfix_scheduler_tail_starts(text):
        tail = text[tail_start:]
        if not tail.strip():
            continue

        if _RE_DUPLICATE_REVERSE_PREFIX.match(tail):
            raise PromptSyntaxError(
                "Duplicate 'reverse' flag after scheduler block.",
                kind="duplicate_reverse_flag",
                token="reverse",
                full=text,
            )

        stripped = tail.lstrip()
        if stripped.startswith("reverse"):
            continue

        if not _RE_STRICT_RANGE_PREFIX.match(stripped):
            continue

        m = _RE_STRICT_RANGE_CHARS.match(stripped)
        candidate = (m.group(0) if m else stripped).rstrip()
        parts = [part.strip() for part in candidate.split(",")]
        valid_parts: list[str] = []

        for part in parts:
            if not part:
                _raise_invalid_range_token(text, candidate)
            if not _RE_STRICT_RANGE_TOKEN.fullmatch(part):
                _raise_invalid_range_token(text, part)
            valid_parts.append(part)

        if not valid_parts:
            _raise_invalid_range_token(text, candidate)

        rest = stripped[len(candidate):]
        if _RE_DUPLICATE_REVERSE_PREFIX.match(rest):
            raise PromptSyntaxError(
                "Duplicate 'reverse' flag after scheduler block.",
                kind="duplicate_reverse_flag",
                token="reverse",
                full=text,
            )


def _strict_check_sequence_surfaces(text: str) -> None:
    """Reject obviously incomplete top-level sequence markers."""
    m_incomplete = _RE_INCOMPLETE_SEQUENCE_SUFFIX.search(text) or _RE_INCOMPLETE_SEQUENCE_BEFORE_OPERATOR.search(text)
    if m_incomplete:
        raise PromptSyntaxError(
            "Incomplete sequence: expected text after '::'.",
            kind="incomplete_sequence",
            token="::",
            full=text,
        )

    idx = 0
    while True:
        pos = text.find(":::", idx)
        if pos == -1:
            break
        if pos > 0 and text[pos - 1] == "\\":
            idx = pos + 3
            continue
        if "!!!" not in text[pos + 3 :]:
            raise PromptSyntaxError(
                "Incomplete top-level sequence: expected '!!!' terminator after ':::'.",
                kind="incomplete_sequence",
                token=":::",
                full=text,
            )
        idx = pos + 3


def _strict_check_interpolation_surfaces(text: str) -> None:
    """Reject broken attention interpolation shells like ``(cat:1.0->)``."""
    exc = _detect_invalid_interpolation_surface(text)
    if exc is not None:
        raise exc


def _detect_invalid_interpolation_surface(text: str) -> PromptSyntaxError | None:
    """Return a syntax error for malformed interpolation shells, if any."""
    for start, end, inner in _iter_parenthesized_spans(text):
        if "->" not in inner or ":" not in inner:
            continue

        last_colon = _find_last_top_level_colon_index(inner)
        if last_colon < 0:
            continue

        body = inner[:last_colon].strip()
        tail = inner[last_colon + 1 :].strip()
        if "->" not in tail:
            continue
        if body and _RE_ATT_INTERP_TAIL.fullmatch(tail):
            continue

        return PromptSyntaxError(
            "Invalid attention interpolation. Use '(text:w0->w1)'.",
            kind="invalid_interpolation",
            token=text[start:end],
            full=text,
        )
    return None


def _strict_check_surface_forms(text: str) -> None:
    """Validate malformed surface forms that the lenient runtime path would otherwise swallow."""
    state = _extract_backend_prompt_state(text)
    if state.has_mixed_backends:
        _raise_mixed_backend_prompt_error(text, state)
    if state.has_bind_backend_conflict:
        _raise_bind_backend_prompt_error(text)
    if state.has_backend or state.has_pool or state.has_bind:
        return
    _raise_unsupported_backend_context_error(text)
    _strict_check_interpolation_surfaces(text)
    probe, _ = _protect_escaped_literal_spans(text)
    probe = _protect_escaped_literals(probe)
    _strict_check_postfix_scheduler_surfaces(probe)
    _strict_check_sequence_surfaces(probe)


def _strict_schedule_preview(text: str, steps: int, seed: int | None):
    """Strict dry-run for lint/explain: parse must succeed (no lenient fallback)."""
    _validate_inputs(text, steps)
    _strict_check_delimiter_balance(text)
    multicond_branches = _extract_multicond_preview_branches(text)
    if multicond_branches is not None:
        return _build_multicond_text_schedule_from_branches(
            multicond_branches,
            int(steps),
            True,
            seed,
            True,
            strict=True,
        )
    state = _extract_backend_prompt_state(text)
    if state.has_mixed_backends:
        _raise_mixed_backend_prompt_error(text, state)
    if state.has_bind_backend_conflict:
        _raise_bind_backend_prompt_error(text)
    if state.has_bind:
        return _build_bind_text_schedule_from_specs(
            state.bind_base_prompt,
            state.bind_specs,
            int(steps),
            True,
            seed,
            True,
            strict=True,
        )
    if state.pool_spec is not None:
        return _build_pool_text_schedule_from_spec(
            state.pool_spec,
            int(steps),
            True,
            seed,
            True,
            strict=True,
        )
    if state.assemble_spec is not None:
        return _build_assemble_text_schedule_from_spec(
            state.assemble_spec,
            int(steps),
            True,
            seed,
            True,
            strict=True,
        )
    if state.chunk_spec is not None:
        branch_schedules = []
        for branch in state.chunk_spec.branches:
            branch_prompt = _expand_chunk_branch_prompt(state.chunk_spec, branch)
            branch_schedules.append(_strict_schedule_preview(branch_prompt, steps, seed))

        boundaries = _collect_schedule_boundaries(branch_schedules, steps)
        out: list[list[int, str]] = []
        for end_at_step in boundaries:
            active_texts = [
                _normalize_preview_fragment(_select_text_from_schedule(schedule, end_at_step) or SAFE_EMPTY)
                for schedule in branch_schedules
            ]
            preview_parts = [text for text in active_texts if text]
            preview = CHUNK_PREVIEW_SEPARATOR.join(preview_parts) if preview_parts else SAFE_EMPTY
            if out and out[-1][1] == preview:
                out[-1][0] = int(end_at_step)
            else:
                out.append([int(end_at_step), preview])
        return out or [[int(steps), SAFE_EMPTY]]
    if state.blend_spec is not None:
        branch_schedules = []
        for branch in state.blend_spec.branches:
            branch_prompt = _expand_blend_branch_prompt(state.blend_spec, branch)
            branch_schedules.append(_strict_schedule_preview(branch_prompt, steps, seed))

        boundaries = _collect_schedule_boundaries(branch_schedules, steps)
        effective_weights = _resolve_blend_mode_weights(
            [branch.weight for branch in state.blend_spec.branches],
            state.blend_spec.mode,
            state.blend_spec.intensity,
        )
        out: list[list[int, str]] = []
        previous_key = None
        for end_at_step in boundaries:
            active_texts = [
                _select_text_from_schedule(schedule, end_at_step) or SAFE_EMPTY
                for schedule in branch_schedules
            ]
            preview = _build_blend_preview_text_with_target(active_texts, effective_weights, state.blend_spec.channel_target)
            key = (tuple(active_texts), tuple(round(float(weight), 8) for weight in effective_weights))
            if out and previous_key == key:
                out[-1][0] = int(end_at_step)
            else:
                out.append([int(end_at_step), preview])
                previous_key = key
        return out or [[int(steps), SAFE_EMPTY]]
    if state.active_morph_spec is not None:
        point_schedules = []
        for point in state.active_morph_spec.points:
            point_prompt = _expand_morph_point_prompt(state.active_morph_spec, point)
            point_schedules.append(_strict_schedule_preview(point_prompt, steps, seed))

        positions = _resolve_morph_positions(state.active_morph_spec, steps)
        window_steps = _resolve_morph_window_steps(state.active_morph_spec, steps)
        inactive_text = _build_morph_inactive_text(state.active_morph_spec)
        out: list[list[int, str]] = []
        previous_key = None
        for step in range(1, int(steps) + 1):
            if window_steps is not None and not (window_steps[0] <= step <= window_steps[1]):
                preview = inactive_text
                key = ("inactive", preview)
                if out and previous_key == key:
                    out[-1][0] = int(step)
                else:
                    out.append([int(step), preview])
                    previous_key = key
                continue
            active_texts = [
                _select_text_from_schedule(schedule, step) or SAFE_EMPTY
                for schedule in point_schedules
            ]
            weights = _resolve_morph_point_weights(
                state.active_morph_spec.points,
                _compute_morph_curve_weights(
                    len(state.active_morph_spec.points),
                    positions,
                    step,
                    state.active_morph_spec.curve,
                    state.active_morph_spec.intensity,
                ),
            )
            preview = _build_morph_preview_text(active_texts, weights, state.active_morph_spec.channel_target)
            key = (tuple(active_texts), tuple(round(float(weight), 8) for weight in weights))
            if out and previous_key == key:
                out[-1][0] = int(step)
            else:
                out.append([int(step), preview])
                previous_key = key
        return out or [[int(steps), SAFE_EMPTY]]
    _strict_check_surface_forms(text)
    probe = text
    if "\\n" in probe or "\\t" in probe:
        probe = probe.replace("\\n", "\n").replace("\\t", "\t")
    probe, _ = _protect_escaped_literal_spans(probe)
    probe = _protect_escaped_literals(probe)
    probe = _normalize_and_operators_for_parse(probe)
    probe = _normalize_scheduler_surface_syntax(probe)
    _parse_lark_cached(probe)
    return get_schedule(text, steps=steps, use_scheduling=True, seed=seed, use_visitor=True)


def _schedule_runtime_cache_key() -> tuple:
    """Capture runtime flags that affect schedule construction results."""
    return (
        bool(ALLOW_EMPTY_ALTERNATE),
        bool(EXPAND_ALTERNATE_PER_STEP),
        int(GROUP_COMBO_LIMIT),
        str(GROUP_COMBO_FALLBACK),
        bool(DEDUP_SCHEDULE_STEPS),
        float(ROUND_BRACKET_MULTIPLIER),  # affects emphasized/weighted default weight
        # SUPPRESS_STANDALONE_COLON intentionally excluded: belongs to parse_prompt_attention only
    )
        
# ──────────────────────────────────────────────────────────────────────────────
# Главный API построения расписания из строки
# ──────────────────────────────────────────────────────────────────────────────

def get_schedule(prompt: str, steps: int, use_scheduling: bool, seed: int | None, use_visitor: bool = True):
    """
    Args:
        seed: Если None, генерируется стабильный seed на основе хеша промпта.
    """
    prompt_text = str(prompt or "")
    multicond_branches = _extract_multicond_preview_branches(prompt_text)
    if multicond_branches is not None:
        return _build_multicond_text_schedule_from_branches(
            multicond_branches,
            int(steps),
            bool(use_scheduling),
            seed,
            use_visitor,
            strict=False,
        )
    if CHUNK_KEYWORD in prompt_text or ASSEMBLE_KEYWORD in prompt_text or BLEND_KEYWORD in prompt_text or MORPH_KEYWORD in prompt_text or POOL_KEYWORD in prompt_text or BIND_KEYWORD in prompt_text:
        try:
            state = _extract_backend_prompt_state(prompt_text)
        except PromptSyntaxError:
            return [[int(steps), _collapse_spaces(prompt_text)]]
        if state.has_mixed_backends or state.has_bind_backend_conflict:
            return [[int(steps), _collapse_spaces(prompt_text)]]
        if state.has_bind:
            return _build_bind_text_schedule_from_specs(
                state.bind_base_prompt,
                state.bind_specs,
                int(steps),
                bool(use_scheduling),
                seed,
                use_visitor,
                strict=False,
            )
        if state.pool_spec is not None:
            return _build_pool_text_schedule_from_spec(
                state.pool_spec,
                int(steps),
                bool(use_scheduling),
                seed,
                use_visitor,
                strict=False,
            )
        if state.assemble_spec is not None:
            return _build_assemble_text_schedule_from_spec(
                state.assemble_spec,
                int(steps),
                bool(use_scheduling),
                seed,
                use_visitor,
                strict=False,
            )
        if state.chunk_spec is not None:
            return _build_chunk_text_schedule_from_spec(
                state.chunk_spec,
                int(steps),
                bool(use_scheduling),
                seed,
                use_visitor,
            )
        if state.blend_spec is not None:
            return _build_blend_text_schedule_from_spec(
                state.blend_spec,
                int(steps),
                bool(use_scheduling),
                seed,
                use_visitor,
            )
        if state.active_morph_spec is not None:
            return _build_morph_text_schedule_from_spec(
                state.active_morph_spec,
                int(steps),
                bool(use_scheduling),
                seed,
                use_visitor,
            )
        if _contains_chunk_marker(prompt_text) or _contains_assemble_marker(prompt_text) or _contains_blend_marker(prompt_text) or _contains_morph_marker(prompt_text) or _contains_pool_marker(prompt_text):
            return [[int(steps), _collapse_spaces(prompt_text)]]

    if _detect_invalid_interpolation_surface(prompt_text) is not None:
        return [[int(steps), _collapse_spaces(prompt_text)]]

    protected_prompt, span_restore = _protect_escaped_literal_spans(prompt)
    protected_prompt = _protect_escaped_literals(protected_prompt)

    if seed is not None:
        runtime_key = _schedule_runtime_cache_key()
        cached = _get_schedule_cached(protected_prompt, steps, use_scheduling, seed, use_visitor, runtime_key)
        return [[row[0], _restore_escaped_literals(str(row[1]), span_restore)] for row in cached]
    else:
        # FIX #3: Детерминированный хеш вместо random hash()
        # hash() в Python меняется при каждом перезапуске процесса.
        # Используем sha256 для стабильности генерации при seed=None.
        prompt_bytes = protected_prompt.encode("utf-8")
        hash_digest = hashlib.sha256(prompt_bytes).digest()
        # Берем первые 4 байта и превращаем в int (до 2^31)
        temp_seed = int.from_bytes(hash_digest[:4], 'big') & 0x7fffffff
        schedule = _get_schedule_impl(protected_prompt, steps, use_scheduling, temp_seed, use_visitor)
        return [[row[0], _restore_escaped_literals(str(row[1]), span_restore)] for row in schedule]
 

@lru_cache(maxsize=CACHE_SIZE)
def _get_schedule_cached(
    prompt: str,
    steps: int,
    use_scheduling: bool,
    seed: int,
    use_visitor: bool,
    _runtime_key: tuple,
):
    """Кешируемая версия для детерминированных вызовов"""
    schedule = _get_schedule_impl(prompt, steps, use_scheduling, seed, use_visitor)
    return tuple((row[0], row[1]) for row in schedule)

def _placeholderize_attention_interpolations(text: str) -> str:
    """Pre-pass: replace (body:w0->w1) patterns with private-use marker strings.

    v2: bracket-balanced scanner — supports nested parentheses in body, e.g.:
        (red (glowing eyes):0.8->1.4)
        ((cat:1.2):1.0->2.0)
        (\\(escaped\\) face:1.0->2.0)

    Algorithm:
      1. Walk the string char-by-char, tracking paren depth and escape state.
      2. For each outer '(' … ')' span, extract the inner text.
      3. Find the LAST top-level ':' inside the span (depth=0 relative to inner).
      4. Check whether the tail after that ':' matches 'w0->w1' strictly.
      5. If yes → replace the whole span with a marker; if no → keep as-is.
      6. Replacements are collected and applied right-to-left so earlier
         indices stay stable.

    Non-interpolation '(...)' spans are never modified.
    Called in _get_schedule_impl BEFORE Lark parsing.
    """
    if "->" not in text:
        return text

    spans: list[tuple[int, int, str]] = []  # (start, end_exclusive, replacement)
    i = 0
    n = len(text)

    while i < n:
        # Handle escape sequences — skip next char
        if text[i] == "\\" and i + 1 < n:
            i += 2
            continue

        if text[i] != "(":
            i += 1
            continue

        # Found an unescaped '(' — scan for matching ')'
        start = i
        depth = 0
        j = i
        while j < n:
            if text[j] == "\\" and j + 1 < n:
                j += 2
                continue
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1

        if depth != 0:
            # Unmatched '(' — skip
            i += 1
            continue

        end = j  # index of closing ')'
        inner = text[start + 1 : end]  # content between the outer parens

        # Quick bail-out: if '->' not in inner, this can't be interpolation
        if "->" not in inner:
            i = end + 1
            continue

        # Find the LAST top-level ':' in inner (depth=0 inside inner)
        last_colon = -1
        d = 0
        escaped = False
        for k, ch in enumerate(inner):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == "(":
                d += 1
            elif ch == ")":
                d -= 1
            elif ch == ":" and d == 0:
                last_colon = k

        if last_colon == -1:
            i = end + 1
            continue

        body = inner[:last_colon].strip()
        tail = inner[last_colon + 1 :]

        # Validate tail: must be 'w0 -> w1' or 'w0 -> w1 ~ mode'
        m = _RE_ATT_INTERP_TAIL.match(tail)
        if not m:
            i = end + 1
            continue

        # Body must be non-empty
        if not body:
            i = end + 1
            continue

        w0   = float(m.group(1))
        w1   = float(m.group(2))
        mode = (m.group(3) or _EASING_DEFAULT).strip().lower()
        if mode not in _EASING_MODES:
            mode = _EASING_DEFAULT
        replacement = _serialize_att_interp(body, w0, w1, mode)
        spans.append((start, end + 1, replacement))

        # Skip past this span (don't recurse into already-matched regions)
        i = end + 1

    if not spans:
        return text

    # Apply replacements right-to-left to keep earlier indices valid
    result = list(text)
    for start, end, replacement in reversed(spans):
        result[start:end] = list(replacement)

    return "".join(result)


def _expand_att_interp_text_for_step(
    text: str,
    step: int,
    interp_start: int,
    interp_end: int,
) -> str:
    """Replace all interpolation markers in *text* with their weight at *step*.

    interp_start / interp_end — the GLOBAL active range for each marker
    (computed by _expand_attention_interpolations Phase 1).

    t=0 at interp_start, t=1 at interp_end.  When span==0 uses t=0.
    Easing is applied after computing t_lin.
    """
    span = max(0, int(interp_end) - int(interp_start))

    def repl(m: re.Match) -> str:
        body = m.group(1).strip()
        w0   = float(m.group(2))
        w1   = float(m.group(3))
        mode = (m.group(4) or _EASING_DEFAULT).strip().lower()
        if mode not in _EASING_MODES:
            mode = _EASING_DEFAULT
        t_lin = 0.0 if span == 0 else (int(step) - int(interp_start)) / span
        t_lin = max(0.0, min(1.0, t_lin))          # clamp for safety
        t     = _apply_easing(t_lin, mode)
        w     = w0 + (w1 - w0) * t
        return f"({body}:{_format_interp_weight(w)})"

    return RE_ATTN_INTERP_LITERAL.sub(repl, text)


def _expand_attention_interpolations(schedule: list[list], steps: int) -> list[list]:
    """Post-pass: expand (body:w0->w1) markers with correct global-range semantics.

    Two-phase algorithm that eliminates the sawtooth/reset bug while preserving
    contextual-aware behaviour for markers that live inside a single scheduler
    window:

    Phase 1 — global range discovery
        Walk the schedule and, for each unique marker identity
        (body, w0, w1, mode), record the FIRST segment's start_step and the
        LAST segment's end_step where that marker appears.

        Key insight:
        * Top-level marker alongside [a:b:N]: appears in MULTIPLE segments
          → global range spans all of them → smooth monotone interpolation.
        * Marker inside [(cat:0→1):N]: appears in ONE segment only
          → global range == that segment → original contextual behaviour
          preserved.

    Phase 2 — expansion
        Each marker is expanded using its discovered global range instead of
        the local segment boundary, so weight never resets at foreign
        scheduler boundaries.

    Segments without markers pass through in O(1) (no marker scan).
    Adjacent identical entries are merged to keep the output compact.
    """
    if not schedule:
        return [[steps, SAFE_EMPTY]]

    # ── Phase 1: discover global active range per unique marker identity ──────
    # Identity key: (body_stripped, raw_w0, raw_w1, mode)
    # Value: [first_start_step, last_end_step]
    MarkerRange = list  # [int, int]
    marker_global: dict[tuple, MarkerRange] = {}

    prev_end = 0
    for end_step, text in schedule:
        end_step   = int(end_step)
        start_step = prev_end + 1
        if ATTN_INTERP_OPEN in str(text):
            for m in RE_ATTN_INTERP_LITERAL.finditer(text):
                key = (
                    m.group(1).strip(),
                    m.group(2),
                    m.group(3),
                    (m.group(4) or _EASING_DEFAULT).strip().lower(),
                )
                if key not in marker_global:
                    marker_global[key] = [start_step, end_step]
                else:
                    marker_global[key][1] = end_step   # extend to latest end
        prev_end = end_step

    if not marker_global:
        # No markers anywhere — return as-is (shouldn't reach here normally)
        return schedule  # type: ignore[return-value]

    # ── Phase 2: expand each segment using per-marker global range ────────────
    out: list[list] = []
    prev_end = 0

    for end_step, text in schedule:
        end_step   = int(end_step)
        start_step = prev_end + 1

        if ATTN_INTERP_OPEN not in str(text):
            # Fast path: no marker — merge identical neighbours
            if out and out[-1][1] == text:
                out[-1][0] = end_step
            else:
                out.append([end_step, text])
            prev_end = end_step
            continue

        # Expand step-by-step; each marker uses its own global range
        for step in range(start_step, end_step + 1):

            def _make_repl(step: int, marker_global: dict) -> re.Match:
                def repl(m: re.Match) -> str:
                    key = (
                        m.group(1).strip(),
                        m.group(2),
                        m.group(3),
                        (m.group(4) or _EASING_DEFAULT).strip().lower(),
                    )
                    body = m.group(1).strip()
                    w0   = float(m.group(2))
                    w1   = float(m.group(3))
                    mode = key[3]
                    g_start, g_end = marker_global.get(key, (step, step))
                    span = max(0, g_end - g_start)
                    t_lin = 0.0 if span == 0 else (step - g_start) / span
                    t_lin = max(0.0, min(1.0, t_lin))
                    t     = _apply_easing(t_lin, mode)
                    w     = w0 + (w1 - w0) * t
                    return f"({body}:{_format_interp_weight(w)})"
                return repl

            expanded = RE_ATTN_INTERP_LITERAL.sub(_make_repl(step, marker_global), text)
            if out and out[-1][1] == expanded:
                out[-1][0] = step
            else:
                out.append([step, expanded])

        prev_end = end_step

    return out


def _get_schedule_impl(prompt: str, steps: int, use_scheduling: bool, seed: int | None, use_visitor: bool):
    """Основная реализация без кеширования"""

    _validate_inputs(prompt, steps)
    result: list[list[int | str]] | None = None

    if not str(prompt).strip():
        result = [[steps, SAFE_EMPTY]]
    else:
        if "\\n" in prompt or "\\t" in prompt:
            prompt = prompt.replace("\\n", "\n").replace("\\t", "\t")

        prompt = _normalize_and_operators_for_parse(prompt)
        prompt = _normalize_scheduler_surface_syntax(prompt)
        # ── Attention interpolation pre-pass ──────────────────────────────────
        # Must run AFTER and-operator normalization (so '&' is already canonical)
        # and BEFORE Lark parsing. Replaces (body:w0->w1) with private-use markers
        # so Lark never encounters '->' and no grammar ambiguity arises.
        prompt = _placeholderize_attention_interpolations(prompt)
        fast_path_allowed = not _has_multiple_bracket_blocks(prompt)

        m_tl3 = re.match(r'(?s)^\s*([^:\[\]\{\}\(\)]+?):::(.+?)!!!(?:,\s*(.*))?\s*$', prompt)
        if m_tl3:
            owner, rest, trailing = m_tl3.groups()
            parts = [p.strip() for p in rest.split(',') if p.strip()]
            seq_texts = []
            for seg in parts:
                seg = seg.rstrip('!;').strip()
                toks = [t.strip() for t in seg.split('::') if t.strip()]
                if toks:
                    label, descs = toks[0], toks[1:]
                    s = f"{owner.strip()}: {label}"
                    if descs:
                        s += f", {', '.join(descs)}"
                    seq_texts.append(s)
            trailing_texts = []
            if trailing:
                for t in trailing.split(','):
                    t = t.strip()
                    if not t:
                        continue
                    mseq = re.match(r'^\s*([^:\[\]\{\}\(\)]+?)::(.+?)([!;])\s*$', t)
                    if mseq:
                        seq_owner, rest2, _term = mseq.groups()
                        descs = [x.strip(' ,~!;') for x in rest2.split('::') if x.strip(' ,~!;')]
                        trailing_texts.append(f"{seq_owner.strip()}: {', '.join(descs)}")
                    else:
                        trailing_texts.append(t)
            text_out = f"{owner.strip()} -> {', '.join(seq_texts)}"
            if trailing_texts:
                text_out += f", {', '.join(trailing_texts)}"
            result = [[steps, _apply_and(_collapse_spaces(text_out))]]

        if result is None:
            m_num_alt = re.match(r'^\s*(\d+)\s*([!_])?\s*\[([^\]]+)\]\s*$', str(prompt))
            if m_num_alt:
                qty_txt, mark, inner = m_num_alt.groups()
                quantity = int(qty_txt)
                options = [_unescape_literals(x.strip()) for x in re.split(r'(?<!\\)\|', inner)]
                options = [opt if opt != "" else SAFE_EMPTY for opt in options]
                options_unique = list(dict.fromkeys(options)) or [SAFE_EMPTY]
                if mark:
                    if quantity <= len(options_unique):
                        chosen = options_unique[:quantity]
                    else:
                        need = quantity - len(options_unique)
                        pad = (options_unique * ((need + len(options_unique) - 1)//len(options_unique)))[:need]
                        chosen = options_unique + pad
                else:
                    rng = random.Random(seed) if seed is not None else random
                    chosen = rng.choices(options_unique, k=quantity)
                result = [[steps, _apply_and(_collapse_spaces(', '.join(chosen)))]]

        if result is None and fast_path_allowed:
            schedules = _try_fast_after_scheduler(
                prompt,
                steps,
                prefix="",
                suffix="",
                allow_ranges=True,
                allow_bracket_after=False,
                disallow_alt_bang=False,
                boundary_fallback=None,
            )
            if schedules is not None:
                result = schedules

        if result is None and fast_path_allowed:
            m_inner = re.match(r'(?s)^([^\[\]]*)\[([^\[\]]+)\]([^\[\]]*)$', prompt)
            if m_inner and prompt.count('[') == 1 and prompt.count(']') == 1:
                pre, inner, post = m_inner.groups()
                # If post starts with :N this is bracket-AFTER syntax "[...]:N"
                # handled by _try_fast_after_scheduler below — don't steal it here.
                _post_is_boundary = bool(re.match(
                    rf'^\s*:\s*{re.escape(RE_NUMERIC.pattern[:-1]).replace(re.escape(RE_NUMERIC.pattern[:-1]), RE_NUMERIC.pattern)}',
                    post or ""
                ))
                # Simpler: just check if post stripped starts with ":"
                _post_stripped = (post or "").lstrip()
                _skip = _post_stripped.startswith(":") and bool(
                    RE_NUMERIC.match(_post_stripped[1:].lstrip())
                )
                if not _skip and not _needs_complex_parse(inner or "", inner or ""):
                    parts = _split_top_level_colon_keep_empty(inner)
                    if len(parts) >= 2:
                        tail_result = _parse_inner_sched_tail((parts[-1] or "").strip(), steps)
                        if tail_result is not None:
                            boundary_spec, ranges_txt = tail_result
                            prompts = [_unescape_literals(p.strip()) for p in parts[:-1]]
                            post, rev_flag = _strip_reverse_from_post(post or "")
                            if rev_flag:
                                prompts = list(reversed(prompts))
                            boundary = _to_end_step(boundary_spec, steps)
                            if ranges_txt:
                                # Inner form with ranges: build via ranges builder
                                ranges: list[tuple[int, int]] = []
                                for rng in ranges_txt.split(','):
                                    rng = rng.strip()
                                    if not rng or '-' not in rng:
                                        continue
                                    a_raw, b_raw = [x.strip() for x in rng.split('-', 1)]
                                    a_pct = a_raw.endswith('%')
                                    b_pct = b_raw.endswith('%')
                                    try:
                                        av = float(a_raw[:-1] if a_pct else a_raw)
                                        bv = float(b_raw[:-1] if b_pct else b_raw)
                                    except (ValueError, TypeError):
                                        continue
                                    sv = _clamp(round((av / 100.0 * steps) if a_pct else av), steps)
                                    ev = _clamp(round((bv / 100.0 * steps) if b_pct else bv), steps)
                                    if sv <= ev:
                                        ranges.append((sv, ev))
                                if ranges:
                                    result = _finalize_schedules(
                                        _build_ranges_schedules_from_components(
                                            prompts, ranges, steps,
                                            prefix=pre, suffix=post,
                                            rev_flag=False, cycle_prompts=True,
                                        )
                                    )
                            else:
                                result = _build_bracket_inner_schedules(
                                    pre, prompts, boundary, post, steps
                                )

        if result is None and fast_path_allowed:
            schedules = _try_fast_after_scheduler(
                prompt,
                steps,
                prefix="",
                suffix="",
                allow_ranges=False,
                allow_bracket_after=True,
                disallow_alt_bang=False,
                boundary_fallback=None,
            )
            if schedules is not None:
                result = schedules

        if result is None:
            try:
                tree = _parse_lark_cached(prompt)
            except (lark.exceptions.LarkError, RecursionError) as e:
                logger.warning("Prompt parse failed: '%s' — %s", prompt, e)
                result = [[steps, _apply_and(_collapse_spaces(prompt))]]
            else:
                collector = CollectSteps(steps, use_scheduling=use_scheduling, seed=seed)
                schedules = collector(tree)
                try:
                    schedules.sort(key=lambda x: int(x[0]))
                except (ValueError, TypeError):
                    pass

                if DEDUP_SCHEDULE_STEPS:
                    try:
                        schedules = _dedup_schedules(schedules)
                    except (ValueError, TypeError, AttributeError):
                        pass

                if not schedules:
                    result = [[steps, _collapse_spaces(prompt)]]
                else:
                    if not use_visitor:
                        logger.debug("use_visitor=False falls back to visitor schedules for consistency.")
                    result = schedules

    if result is None:
        result = [[steps, SAFE_EMPTY]]

    # ── Attention interpolation post-pass ─────────────────────────────────────
    # Expand (body:w0->w1) markers per local active segment BEFORE
    # _apply_scheduling_mode collapses use_scheduling=False to a single entry.
    # The check is O(n) over segments and skips entirely when no marker present.
    if any(ATTN_INTERP_OPEN in str(t) for _, t in result):
        result = _expand_attention_interpolations(result, steps)

    return _apply_scheduling_mode(result, steps, use_scheduling)


# ──────────────────────────────────────────────────────────────────────────────
# Сервиска
# ──────────────────────────────────────────────────────────────────────────────

def _dedup_schedules(schedules: list[list[int, str]], joiner: str = ", ") -> list[list[int, str]]:
    if not schedules:
        return schedules
    last_by_step: dict[int, str] = {}
    for end, text in schedules:
        try:
            key = int(end)
        except (ValueError, TypeError):
            key = end
        last_by_step[key] = text
    # Устойчивая сортировка: сначала числовые ключи по возрастанию, потом строковые.
    return [[k, v] for k, v in sorted(
        last_by_step.items(),
        key=lambda kv: (0, int(kv[0])) if isinstance(kv[0], int) else (1, str(kv[0]))
    )]

# ──────────────────────────────────────────────────────────────────────────────
# parse_prompt_attention — внимательность/веса
# ──────────────────────────────────────────────────────────────────────────────

re_attention = re.compile(rf"""
\\\(|
\\\)|
\\\[|
\\\]|
\\\\|
\\|
\(|
\[|
:\s*(x?{NUMERIC_RE})\s*\)|
\)|
]|
[^\\()\[\]:\s]+|
\s+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

# ──────────────────────────────────────────────────────────────────────────────
# Настройки и предкомпилированные regex для parse_prompt_attention
# ──────────────────────────────────────────────────────────────────────────────
# Опциональная АДДИТИВНАЯ семантика дельт (word +0.2 → 1.2 вместо 0.2)
# По умолчанию ВЫКЛ (0), чтобы поведение не менялось.
ATTENTION_DELTA_ADDITIVE = _env_bool("ATTENTION_DELTA_ADDITIVE", "0")


def _attention_runtime_cache_key() -> tuple:
    """Capture runtime knobs that change parse_prompt_attention output."""
    return (
        bool(ATTENTION_DELTA_ADDITIVE),
        bool(SUPPRESS_STANDALONE_COLON),
        float(ROUND_BRACKET_MULTIPLIER),
        float(SQUARE_BRACKET_MULTIPLIER),
    )

RX_FIX_ATT_WEIGHT = re.compile(
    rf'(\b[^\s:()\[\]{{}}]+)\s*:\s*({NUMERIC_RE})'
)
RX_DELTA_TOKEN = re.compile(
    rf'^\s*([-+]{NUMERIC_NOSIGN_RE})\s*$'
)
RX_INLINE_ATT = re.compile(
    r'(?:'
    r'(\b[^\s:(){{}}\[\]]+)\s*:\s*'         # g1: слово для абсолютного веса
    rf'({NUMERIC_RE})'                      # g2: абсолютный вес
    r')|(?:'
    r'(\b[^\s:(){{}}\[\]]+)\s*'             # g3: слово для дельты
    rf'([-+](?:{NUMERIC_NOSIGN_RE}))'       # g4: дельта со знаком
    r')',
    re.X
)

# ──────────────────────────────────────────────────────────────────────────────
# Объект состояния для pending_colon-автомата в _parse_prompt_attention_impl.
# Заменяет три разрозненные переменные (pending_colon, pending_colon_had_space,
# pending_colon_position) единым объектом: сброс/применение атомарны и не могут
# рассинхронизироваться при будущих правках кода.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _ColonState:
    """Состояние ожидания числового веса после двоеточия.

    active           — ждём число после последнего ':'
    had_space        — между ':' и следующим токеном был пробел
    res_position     — индекс в res, куда был (мог бы быть) вставлен ':'
    last_token_index — индекс последнего «настоящего» слова в res
    """
    active: bool = False
    had_space: bool = False
    res_position: int = -1
    last_token_index: int = -1

    def arm(self, res_len: int, last_tok_idx: int) -> None:
        """Зафиксировать двоеточие: начать ожидание числа."""
        self.active = True
        self.had_space = False
        self.res_position = res_len       # позиция В МОМЕНТ встречи ':'
        self.last_token_index = last_tok_idx

    def reset(self) -> None:
        """Сбросить состояние."""
        self.active = False
        self.had_space = False
        self.res_position = -1

    def apply_weight(self, res: list, weight: float, suppress: bool) -> None:
        """Применить вес к предыдущему токену и убрать ':' из res (если он был добавлен)."""
        j = self.last_token_index
        # Пропускаем: ':' / 'BREAK' / пробельные токены / уже взвешенные токены.
        # Без проверки на пробел вес мог применяться к ' ' вместо слова
        # при паттерне "word : 1.5" (пробелы вокруг двоеточия).
        while j >= 0 and (
            res[j][0] in (':', 'BREAK')
            or res[j][0].strip() == ''
            or res[j][1] != 1.0
        ):
            j -= 1
        if j >= 0:
            res[j][1] = weight
        # Если ':' был добавлен явно (SUPPRESS=False), удаляем по сохранённой позиции.
        # res между arm() и сюда растёт только append'ами → позиция стабильна.
        if not suppress and 0 <= self.res_position < len(res):
            if res[self.res_position][0] == ':':
                res.pop(self.res_position)
        self.reset()

    def restore_as_text(self, res: list, suppress: bool) -> None:
        """':' оказалось текстом — приклеить к предыдущему слову."""
        if suppress:
            j = self.last_token_index
            # Пропускаем пробельные токены и BREAK — ищем реальное слово.
            while j >= 0 and (res[j][0] == 'BREAK' or res[j][0].strip() == ''):
                j -= 1
            if j >= 0:
                res[j][0] = res[j][0] + ":" + (" " if self.had_space else "")
            else:
                res.append([':', 1.0])
                if self.had_space:
                    res.append([' ', 1.0])
        self.reset()


@lru_cache(maxsize=CACHE_SIZE)
def _parse_prompt_attention_impl(text, _runtime_key: tuple | bool | None = None):
    """Внутренняя реализация с кешем.
    _additive_delta — копия ATTENTION_DELTA_ADDITIVE на момент вызова;
    включена в ключ кеша, чтобы смена флага не возвращала устаревший результат.
    """
    if _runtime_key is None:
        _runtime_key = _attention_runtime_cache_key()
    elif isinstance(_runtime_key, bool):
        # Backward compatibility with older internal calls that passed only additive flag.
        _runtime_key = (
            bool(_runtime_key),
            bool(SUPPRESS_STANDALONE_COLON),
            float(ROUND_BRACKET_MULTIPLIER),
            float(SQUARE_BRACKET_MULTIPLIER),
        )
    try:
        additive_delta, suppress_standalone_colon, round_bracket_multiplier, square_bracket_multiplier = _runtime_key
    except Exception:
        additive_delta = bool(ATTENTION_DELTA_ADDITIVE)
        suppress_standalone_colon = bool(SUPPRESS_STANDALONE_COLON)
        round_bracket_multiplier = float(ROUND_BRACKET_MULTIPLIER)
        square_bracket_multiplier = float(SQUARE_BRACKET_MULTIPLIER)

    if not text or not str(text).strip():
        return [[SAFE_EMPTY, 1.0]]

    text = str(text).replace('\r\n', '\n').replace('\r', '\n')
    # FIX: нормализуем и реальный таб, и литерал таба до пробелов ДО токенизации
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = text.replace('\\n', ' ').replace('\\t', ' ')

    # Нормализуем AND → & и пробелы перед токенизацией
    text = _apply_and(text)

    res = []
    round_brackets = []
    square_brackets = []

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    # Единый объект состояния вместо трёх разрозненных переменных.
    colon = _ColonState()
    last_token_index = -1

    for m in re_attention.finditer(text):
        text_match = m.group(0)
        wgrp = m.group(1)

        if text_match.startswith('\\'):
            res.append([text_match[1:], 1.0])
            last_token_index = len(res) - 1
            colon.reset()

        elif text_match == '(':
            round_brackets.append(len(res))
            colon.reset()

        elif text_match == '[':
            square_brackets.append(len(res))
            colon.reset()

        elif wgrp is not None and round_brackets:
            if wgrp.lower().startswith('x'):
                # ──────────────────────────────────────────────────────────
                # Синтаксис :xN.M — «расширение токена» (token repetition).
                # (текст:x3.2) → три копии текста, последняя с весом 1.2.
                # Математика: val=3.2 → count=3 (целая), tail=0.2+1=1.2.
                # Применение: усиление токена через повторение в cross-attention.
                # Отрицательное значение: val<0 → инвертировать вес (* -1).
                # ──────────────────────────────────────────────────────────
                try:
                    val = float(wgrp[1:])
                except ValueError:
                    val = 1.0
                start = round_brackets.pop()
                segment = res[start:]
                if segment:
                    abs_val = abs(val)
                    count = max(1, int(abs_val))
                    tail = (abs_val % 1) + (abs_val >= 1)
                    res[start:] = [[t, w] for _ in range(count) for t, w in segment]
                    if tail > 1e-4:
                        for seg in res[-len(segment):]:
                            seg[1] *= tail
                    if val < 0:
                        multiply_range(start, -1.0)
                else:
                    round_brackets.append(start)  # пустой сегмент — не трогаем
            else:
                try:
                    multiply_range(round_brackets.pop(), float(wgrp))
                except ValueError:
                    multiply_range(round_brackets.pop(), 1.0)
            colon.reset()

        elif text_match == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
            colon.reset()

        elif text_match == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
            colon.reset()

        elif text_match == ':' and not round_brackets and not square_brackets:
            # Встретили ':' — начинаем ожидание числового веса.
            colon.arm(res_len=len(res), last_tok_idx=last_token_index)
            if not suppress_standalone_colon:
                res.append([':', 1.0])
                last_token_index = len(res) - 1

        else:
            chunk = text_match

            # Если ждём число после ':' и пришли только пробелы — продолжаем ждать
            if colon.active and not (round_brackets or square_brackets):
                if chunk.strip() == "":
                    colon.had_space = True
                    continue

            # Проверяем, пришло ли чистое число (вес)
            if colon.active and not (round_brackets or square_brackets):
                mnum = re.match(r"^\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$", chunk)
                if mnum:
                    # Это вес: применяем через метод объекта (атомарно).
                    try:
                        weight = float(mnum.group(1))
                        colon.apply_weight(res, weight, suppress_standalone_colon)
                        last_token_index = len(res) - 1
                        chunk = chunk[mnum.end():]
                        if not chunk.strip():
                            continue
                    except ValueError:
                        pass

                # Если это НЕ число — ':' оказалось текстом: восстанавливаем через метод.
                if colon.active:
                    colon.restore_as_text(res, suppress_standalone_colon)
                    last_token_index = len(res) - 1

            # Обработка обычного текста (chunk)
            parts = re.split(re_break, chunk)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                if part == "":
                    continue
                res.append([part, 1.0])
                last_token_index = len(res) - 1

    # --- Висящее двоеточие в конце строки --- #
    # Число так и не встретилось → ':' трактуем как литерал.
    if colon.active and not (round_brackets or square_brackets):
        colon.restore_as_text(res, suppress_standalone_colon)

    if round_brackets or square_brackets:
        pass

    if not res:
        return [[SAFE_EMPTY, 1.0]]

    # --- Ремонтный проход для шаблона "word : number" (с пробелами) -----------
    rx_fix = RX_FIX_ATT_WEIGHT

    # Создаем временный текст из res для поиска
    temp_text = "".join(t for t, w in res if t != "BREAK")
    used = {}
    
    for mm in rx_fix.finditer(temp_text):
        word = mm.group(1)
        try:
            wt = float(mm.group(2))
        except ValueError:
            continue
        k = used.get(word, 0)
        # найдём k-ую по счёту неподправленную запись этого слова
        # ФИЙ16: сравниваем t.strip() == word — токен может содержать ведущий
        # пробел (например [' cat', 1.0]) и тогда t == word никогда не сработает.
        cnt = 0
        for i, (t, w) in enumerate(res):
            if t.strip() == word and w == 1.0:
                if cnt == k:
                    res[i][1] = wt
                    used[word] = k + 1
                    break
                cnt += 1


    # --- Сшивка "word  +0.2" / "word  -0.1" через ПРОБЕЛ ---
    # use precompiled RX_DELTA_TOKEN
    # Цель: интерпретировать отдельный токен "+0.2" (или "-0.1") как изменение веса
    # предыдущего НЕпустого токена (слова/фразы), даже если у него уже есть вес.
    #
    # Исправления:
    #  - не применяем дельту к пробельным токенам (t.strip()==""), чтобы "+0.2" не "взвешивал" пробел
    #  - разрешаем повторные дельты: "word +0.2 +0.3"
    #  - разрешаем дельту после уже-взвешенного слова: "word:1.2 +0.3"
    # ФИЙ15: O(N) однопроходный сбор в new_delta вместо O(N²) del/insert в цикле.
    # Алгоритм: для каждого «слова» жадно поглощаем все следующие дельта-токены подряд.
    new_delta: list = []
    i = 0
    while i < len(res):
        t, w = res[i]

        # Разделители/пустоту оставляем как есть
        if t in (':', 'BREAK') or (isinstance(t, str) and t.strip() == ""):
            new_delta.append([t, w])
            i += 1
            continue

        # Жадно поглощаем все последовательные дельты после этого слова
        cur_w = float(w)
        j = i + 1
        last_had_space = False
        consumed_to = i  # последний индекс, уже вошедший в результат

        while j <= len(res):  # j — следующий кандидат на дельту
            # Пропускаем пробельные токены между словом и дельтой
            k = j
            sp = False
            while k < len(res) and res[k][1] == 1.0 and isinstance(res[k][0], str) and res[k][0].strip() == "":
                sp = True
                k += 1
            if k >= len(res) or res[k][1] != 1.0:
                break
            m = RX_DELTA_TOKEN.match(res[k][0])
            if not m:
                break
            try:
                delta = float(m.group(1))
            except ValueError:
                break
            if additive_delta:
                cur_w = cur_w + delta
            else:
                cur_w = delta
            last_had_space = sp
            consumed_to = k
            j = k + 1

        new_delta.append([t, cur_w])
        # Если последняя поглощённая дельта была через пробел — восстанавливаем его
        if last_had_space and consumed_to > i:
            new_delta.append([" ", 1.0])
        i = consumed_to + 1

    res = new_delta
    # второй проход: inline-веса word:weight ИЛИ word(+/-)delta

    new_res = []
    for txt, w in res:
        if txt == "BREAK" or w != 1.0:
            new_res.append([txt, w])
            continue
        pos = 0; changed = False
        for mm in RX_INLINE_ATT.finditer(txt):
            pre = txt[pos:mm.start()]
            if pre:
                new_res.append([pre, 1.0])
            if mm.group(1) is not None:
                word, wt = mm.group(1), float(mm.group(2))
            else:
                word, delta = mm.group(3), float(mm.group(4))
                wt = (1.0 + delta) if additive_delta else delta
            new_res.append([word, wt]); changed = True
            pos = mm.end()
        if changed:
            tail = txt[pos:]
            if tail:
                new_res.append([tail, 1.0])
        else:
            new_res.append([txt, w])
    res = new_res

    # убрать ведущий "AND " (но сохранить пробелы после)
    norm = []
    for t, w in res:
        if t == "BREAK":
            norm.append([t, w]); continue
        m = re.match(r'^(\s*)AND\s+(.*)$', t)
        if m:
            t = m.group(1) + m.group(2)
        norm.append([t, w])
    res = norm

    # финальное схлопывание соседей с одинаковым весом — ВСТАВЛЯЕМ ПРОБЕЛ при склейке
    def _smart_concat(a: str, b: str):
        if not a: return b
        if not b: return a
        
        ar = a.rstrip()
        # Если в конце первого слова запятая/точка - пробел обычно желателен для токенизатора
        if ar and ar[-1] in ",.;?!":
            return ar + " " + b.lstrip()

        if a.endswith(" ") or b.startswith(" "):
            return a + b

        # Старая проверка была: if a[-1].isalnum() and b[0].isalnum():
        # Новая проверка: ставим пробел, если это не спецсимволы скобок/двоеточий
        # (грубая эвристика, но лучше чем терять пробелы после запятых)
        if a[-1] not in "(:[" and b[0] not in "):]":
            return a + " " + b
            
        return a + b


    # Слияние соседних токенов с одинаковым весом — O(N) через новый список,
    # вместо O(N²) res.pop(i+1) внутри while.
    merged: list[list] = []
    for tok, w in res:
        if (merged
                and merged[-1][1] == w
                and merged[-1][0] not in (':', 'BREAK')
                and tok not in (':', 'BREAK')):
            merged[-1][0] = _smart_concat(merged[-1][0], tok)
        else:
            merged.append([tok, w])
    res = merged

    # Сжать повторные пробелы внутри каждого токена
    res = [[_collapse_spaces(t), w] if t != "BREAK" else [t, w] for t, w in res]
    # Удалить пустые, если вдруг остались
    res = [[t, w] for t, w in res if t or t == "BREAK"]

    # Если остались только BREAK — вернём безопасный токен
    if all(t == "BREAK" for t, _ in res):
        return [[SAFE_EMPTY, 1.0]]

    return res

# Обёртка публичного API: сохраняем прежнее имя и тип возврата
def parse_prompt_attention(text):
    protected = _protect_escaped_literals(str(text or ""))
    out = _parse_prompt_attention_impl(protected, _attention_runtime_cache_key())
    # возвращаем копию list-of-[text, weight]
    return [[_restore_escaped_literals(str(t)), w] for t, w in out]

# ──────────────────────────────────────────────────────────────────────────────
# Мультиконд — как в старых версиях, но с дедупом текстов расписаний
# ──────────────────────────────────────────────────────────────────────────────

ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])

class SdConditioning(list):
    def __init__(self, prompts, is_negative_prompt=False, width=None, height=None, copy_from=None, distilled_cfg_scale=None):
        super().__init__()
        self.extend(prompts)
        if copy_from is None:
            copy_from = prompts
        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, 'is_negative_prompt', False)
        self.width = width if width is not None else getattr(copy_from, 'width', None)
        self.height = height if height is not None else getattr(copy_from, 'height', None)
        self.distilled_cfg_scale = distilled_cfg_scale if distilled_cfg_scale is not None else getattr(copy_from, 'distilled_cfg_scale', None)

def get_learned_conditioning_prompt_schedules(
    prompts: list[str],
    base_steps: int,
    hires_steps: int | None = None,
    use_old_scheduling: bool = False,
    seed: int | None = 42,
    use_visitor: bool = True,
):
    steps = hires_steps if (hires_steps is not None and not use_old_scheduling) else base_steps
    use_scheduling = (hires_steps is None) or use_old_scheduling
    prompt_schedules = [get_schedule(p, steps, use_scheduling, seed, use_visitor=use_visitor) for p in prompts]
    return prompt_schedules

def get_learned_conditioning(
    model,
    prompts: SdConditioning | list[str],
    steps,
    hires_steps=None,
    use_old_scheduling=False,
    seed: int | None = 42,
    use_visitor: bool = True,
):
    effective_steps = hires_steps if (hires_steps is not None and not use_old_scheduling) else steps
    use_scheduling = (hires_steps is None) or use_old_scheduling

    if any(
        _contains_chunk_marker(prompt)
        or _contains_assemble_marker(prompt)
        or _contains_blend_marker(prompt)
        or _contains_morph_marker(prompt)
        or _contains_pool_marker(prompt)
        or _contains_bind_marker(prompt)
        for prompt in prompts
    ):
        res = []
        cache = {}
        for prompt in prompts:
            cached = cache.get(prompt, None)
            if cached is not None:
                res.append(cached)
                continue
            cond_schedule = _build_prompt_conditioning_schedule(
                model,
                prompt,
                effective_steps,
                use_scheduling,
                seed,
                use_visitor,
                prompts,
            )
            cache[prompt] = cond_schedule
            res.append(cond_schedule)
        return res

    res = []
    prompt_schedules = get_learned_conditioning_prompt_schedules(
        prompts,
        steps,
        hires_steps,
        use_old_scheduling,
        seed,
        use_visitor,
    )
    cache = {}
    for prompt, prompt_schedule in zip(prompts, prompt_schedules):
        if not prompt_schedule:
            raise ValueError(f"Empty schedule for prompt '{prompt}'")
        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached); continue
        texts = SdConditioning([x[1] for x in prompt_schedule], copy_from=prompts)
        conds = model.get_learned_conditioning(texts)
        cond_schedule = []
        for i, (end_at_step, _) in enumerate(prompt_schedule):
            if isinstance(conds, dict):
                cond = {k: v[i] for k, v in conds.items()}
            else:
                cond = conds[i]
            cond_schedule.append(ScheduledPromptConditioning(end_at_step, cond))
        cache[prompt] = cond_schedule
        res.append(cond_schedule)
    return res

# === DROP-IN REPLACEMENT ===
class ComposableScheduledPromptConditioning:
    def __init__(self, schedules, weight=1.0, weight_schedule=None):
        self.schedules: list[ScheduledPromptConditioning] = schedules
        self.weight: float = weight
        self.weight_schedule: list[tuple[int, float]] | None = weight_schedule

    def weight_at_step(self, current_step: int) -> float:
        if not self.weight_schedule:
            return float(self.weight)
        for end_at_step, weight in self.weight_schedule:
            if current_step <= int(end_at_step):
                return float(weight)
        return float(self.weight_schedule[-1][1])

class MulticondLearnedConditioning:
    def __init__(self, shape, batch):
        self.shape: tuple = shape
        self.batch: list[list[ComposableScheduledPromptConditioning]] = batch

def _build_single_prompt_multicond_parts(
    model,
    prompt: str,
    effective_steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    copy_from,
) -> list[ComposableScheduledPromptConditioning]:
    state = _extract_backend_prompt_state(prompt)
    if state.has_mixed_backends:
        _raise_mixed_backend_prompt_error(prompt, state)
    if state.has_bind_backend_conflict:
        _raise_bind_backend_prompt_error(prompt)

    if state.has_bind:
        base_schedule = _build_prompt_conditioning_schedule(
            model,
            state.bind_base_prompt,
            effective_steps,
            use_scheduling,
            seed,
            use_visitor,
            copy_from,
        )
        if not base_schedule:
            raise ValueError("Empty schedule for BIND base prompt")
        composable_parts = [ComposableScheduledPromptConditioning(base_schedule, 1.0)]
        for spec in state.bind_specs:
            bind_schedule, bind_weight_schedule = _build_bind_branch_prompt_conditioning_schedule(
                model,
                spec,
                base_schedule,
                effective_steps,
                use_scheduling,
                seed,
                use_visitor,
                copy_from,
            )
            if not bind_schedule:
                raise ValueError("Empty schedule for BIND branch")
            composable_parts.append(
                ComposableScheduledPromptConditioning(
                    bind_schedule,
                    spec.weight,
                    bind_weight_schedule,
                )
            )
        return composable_parts

    conds_list, prompt_flat_list, _prompt_indexes = get_multicond_prompt_list([prompt])
    if not conds_list:
        raise ValueError("Empty multicond prompt decomposition")
    cond_parts = conds_list[0]
    if not cond_parts:
        raise ValueError("Prompt did not produce any composable branches")

    if any(
        _contains_chunk_marker(subprompt)
        or _contains_assemble_marker(subprompt)
        or _contains_blend_marker(subprompt)
        or _contains_morph_marker(subprompt)
        or _contains_pool_marker(subprompt)
        or _contains_bind_marker(subprompt)
        for subprompt in prompt_flat_list
    ):
        flat_cond_schedules = [
            _build_prompt_conditioning_schedule(
                model,
                subprompt,
                effective_steps,
                use_scheduling,
                seed,
                use_visitor,
                copy_from,
            )
            for subprompt in prompt_flat_list
        ]
        if not flat_cond_schedules or any(not schedule for schedule in flat_cond_schedules):
            raise ValueError("Empty schedule for at least one sub-prompt")
        return [
            ComposableScheduledPromptConditioning(flat_cond_schedules[flat_index], weight)
            for flat_index, weight in cond_parts
        ]

    flat_schedules = [
        get_schedule(subprompt, effective_steps, use_scheduling, seed, use_visitor=use_visitor)
        for subprompt in prompt_flat_list
    ]
    if not flat_schedules or any(not schedule for schedule in flat_schedules):
        raise ValueError("Empty schedule for at least one sub-prompt")

    text_to_index: dict[str, int] = {}
    unique_texts: list[str] = []
    flat_schedule_text_indices: list[list[int]] = []

    for schedule in flat_schedules:
        indices: list[int] = []
        for _end_at_step, text in schedule:
            if text not in text_to_index:
                text_to_index[text] = len(unique_texts)
                unique_texts.append(text)
            indices.append(text_to_index[text])
        flat_schedule_text_indices.append(indices)

    texts_conditioning = SdConditioning(unique_texts, copy_from=copy_from)
    model_conds = model.get_learned_conditioning(texts_conditioning)

    def get_i_cond(i: int):
        if isinstance(model_conds, dict):
            return {k: v[i] for k, v in model_conds.items()}
        return model_conds[i]

    composable_parts: list[ComposableScheduledPromptConditioning] = []
    for flat_index, weight in cond_parts:
        schedule = flat_schedules[flat_index]
        conds_for_steps = []
        for local_step_idx, (end_at_step, _text) in enumerate(schedule):
            i_global = flat_schedule_text_indices[flat_index][local_step_idx]
            conds_for_steps.append(ScheduledPromptConditioning(int(end_at_step), get_i_cond(i_global)))
        composable_parts.append(ComposableScheduledPromptConditioning(conds_for_steps, weight))
    return composable_parts

def get_multicond_learned_conditioning(
    model,
    prompts,
    steps,
    hires_steps=None,
    use_old_scheduling=False,
    seed: int | None = 42,
    use_visitor: bool = True,
):
    effective_steps = hires_steps if hires_steps is not None and not use_old_scheduling else steps
    use_scheduling = (hires_steps is None) or use_old_scheduling

    if any(_contains_bind_marker(prompt) for prompt in prompts):
        res_batch: list[list[ComposableScheduledPromptConditioning]] = []
        first_cond = None
        cache: dict[str, list[ComposableScheduledPromptConditioning]] = {}
        for prompt in prompts:
            cached = cache.get(prompt)
            if cached is None:
                cached = _build_single_prompt_multicond_parts(
                    model,
                    prompt,
                    effective_steps,
                    use_scheduling,
                    seed,
                    use_visitor,
                    prompts,
                )
                cache[prompt] = cached
            if not cached or any(not part.schedules for part in cached):
                raise ValueError("Empty schedule for at least one composable prompt part")
            if first_cond is None:
                first_cond = cached[0].schedules[0].cond
            res_batch.append(cached)
        if first_cond is None:
            raise ValueError("Empty multicond batch")
        shape = _conditioning_shape_from_cond(first_cond)
        return MulticondLearnedConditioning(shape, res_batch)

    # ФИЙ17: сначала разбиваем на подпромпты по AND, затем строим расписание
    # для каждого подпромпта отдельно — иначе все части AND получали бы одинаковый
    # тензор (весь "cat AND dog" целиком) вместо независимых тензоров "cat" и "dog".
    conds_list, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)

    if any(
        _contains_chunk_marker(prompt)
        or _contains_assemble_marker(prompt)
        or _contains_blend_marker(prompt)
        or _contains_morph_marker(prompt)
        or _contains_pool_marker(prompt)
        for prompt in prompt_flat_list
    ):
        flat_cond_schedules = [
            _build_prompt_conditioning_schedule(
                model,
                prompt,
                effective_steps,
                (hires_steps is None) or use_old_scheduling,
                seed,
                use_visitor,
                prompts,
            )
            for prompt in prompt_flat_list
        ]
        if not flat_cond_schedules or any(not schedule for schedule in flat_cond_schedules):
            raise ValueError("Empty schedule for at least one sub-prompt")

        res_batch: list[list[ComposableScheduledPromptConditioning]] = []
        for cond_parts in conds_list:
            composable_parts: list[ComposableScheduledPromptConditioning] = []
            for flat_index, weight in cond_parts:
                composable_parts.append(
                    ComposableScheduledPromptConditioning(flat_cond_schedules[flat_index], weight)
                )
            res_batch.append(composable_parts)

        first_cond = flat_cond_schedules[0][0].cond
        shape = _conditioning_shape_from_cond(first_cond)
        return MulticondLearnedConditioning(shape, res_batch)

    # Строим расписание для каждого уникального подпромпта из prompt_flat_list
    flat_schedules = get_learned_conditioning_prompt_schedules(
        prompt_flat_list,
        effective_steps,
        hires_steps,
        use_old_scheduling,
        seed,
        use_visitor,
    )
    if not flat_schedules or any(not sch for sch in flat_schedules):
        raise ValueError("Empty schedule for at least one sub-prompt")

    # Дедупликация текстов для батча model.get_learned_conditioning
    text_to_index: dict[str, int] = {}
    unique_texts: list[str] = []
    flat_schedule_text_indices: list[list[int]] = []

    for schedule in flat_schedules:
        indices: list[int] = []
        for end_at_step, text in schedule:
            if text not in text_to_index:
                text_to_index[text] = len(unique_texts)
                unique_texts.append(text)
            indices.append(text_to_index[text])
        flat_schedule_text_indices.append(indices)

    texts_conditioning = SdConditioning(unique_texts, copy_from=prompts)
    model_conds = model.get_learned_conditioning(texts_conditioning)

    def get_i_cond(i: int):
        if isinstance(model_conds, dict):
            return {k: v[i] for k, v in model_conds.items()}
        return model_conds[i]

    res_batch: list[list[ComposableScheduledPromptConditioning]] = []
    for cond_parts in conds_list:
        composable_parts: list[ComposableScheduledPromptConditioning] = []
        for flat_index, weight in cond_parts:
            # Каждый flat_index → своё расписание и свой набор тензоров
            schedule = flat_schedules[flat_index]
            conds_for_steps = []
            for local_step_idx, (end_at_step, _text) in enumerate(schedule):
                i_global = flat_schedule_text_indices[flat_index][local_step_idx]
                cond_at_step = get_i_cond(i_global)
                conds_for_steps.append(ScheduledPromptConditioning(int(end_at_step), cond_at_step))
            composable_parts.append(ComposableScheduledPromptConditioning(conds_for_steps, weight))
        res_batch.append(composable_parts)

    if isinstance(model_conds, dict):
        ca = model_conds.get('crossattn')
        if isinstance(ca, list) and ca:
            shape = getattr(ca[0], 'shape', None) or (0,)
        else:
            shape = getattr(ca, 'shape', None) or (0,)
    else:
        shape = getattr(model_conds, 'shape', None) or (0,)

    return MulticondLearnedConditioning(shape, res_batch)


# Делим по слову AND и по одиночному '&' как самостоятельному токену.
# Не трогаем AND_PERP / AND_SALT / AND_TOPK.
re_AND = re.compile(r"(?:\bAND\b|(?<!\S)&(?!\S))(?!_PERP|_SALT|_TOPK)")
# Вес в конце подпрампта: "text : 1.2" (якорь на конец строки, поддержка китайского двоеточия)
RE_END_WEIGHT = re.compile(
    r"^(?P<text>.*?)"
    rf"\s*[:：]\s*(?P<w>{NUMERIC_RE})\s*$"  # Убрать необязательность группы веса
)

def _split_top_level_multicond(prompt: str) -> list[str]:
    if not prompt:
        return [prompt]

    parts: list[str] = []
    buf: list[str] = []
    depth_paren = 0
    depth_brace = 0
    depth_brack = 0
    i = 0
    n = len(prompt)
    escaped_backslash_placeholder = _ESCAPED_LITERAL_SINGLE_PLACEHOLDERS["\\"]

    def flush_part():
        part = "".join(buf)
        buf.clear()
        parts.append(part)

    def is_word_char(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    while i < n:
        ch = prompt[i]

        if depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
            prev_ch = prompt[i - 1] if i > 0 else ""
            next_ch = prompt[i + 1] if i + 1 < n else ""

            if ch == "&":
                if (not prev_ch or prev_ch.isspace()) and (not next_ch or next_ch.isspace()):
                    flush_part()
                    i += 1
                    continue

            if prompt.startswith("AND", i):
                prev_blocks_split = prev_ch == "\\" or prev_ch == escaped_backslash_placeholder or is_word_char(prev_ch)
                next_boundary_ch = prompt[i + 3] if i + 3 < n else ""
                next_blocks_split = is_word_char(next_boundary_ch)
                if not prev_blocks_split and not next_blocks_split:
                    flush_part()
                    i += 3
                    continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1

        buf.append(ch)
        i += 1

    flush_part()
    return parts


def get_multicond_prompt_list(prompts: SdConditioning | list[str]):
    res_indexes = []
    prompt_indexes = {}
    prompt_flat_list = SdConditioning(prompts)
    prompt_flat_list.clear()

    for prompt in prompts:
        protected_prompt, span_restore = _protect_escaped_literal_spans(prompt)
        protected_prompt = _protect_escaped_literals(protected_prompt)
        subprompts = _split_top_level_multicond(protected_prompt)
        indexes = []
        for subprompt in subprompts:
            s = subprompt if isinstance(subprompt, str) else str(subprompt)

            # 1) Нормализация скобочных весов: "(cat:2.0)" или просто "(cat)"
            m_emph = re.match(
                r'^\s*\(\s*(.*?)\s*(?::\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)\s*)?\)\s*$',
                s
            )
            if m_emph:
                text = m_emph.group(1)
                weight = m_emph.group(2) if m_emph.group(2) is not None else 1.1
            else:
                # 2) Надёжный разбор "word:weight"
                s_clean = (s or "").strip().rstrip(",;")
                m_end = RE_END_WEIGHT.match(s_clean)
                if m_end and m_end.group("w"):  # Проверяем что вес найден
                    text, weight = m_end.group("text"), m_end.group("w")
                else:
                    text, weight = (s_clean, None)

            text = (text or "").strip()
            # Нормализуем артефакты вроде "dog:" -> "dog"
            if text.endswith(":") or text.endswith("："):
                text = text[:-1].rstrip()
            text = _restore_escaped_literals(text, span_restore)
            if not text.strip():
                text = SAFE_EMPTY
            try:
                weight = float(weight) if weight is not None else 1.0
            except (ValueError, TypeError):
                weight = 1.0
            index = prompt_indexes.get(text, None)
            if index is None:
                index = len(prompt_flat_list)
                prompt_flat_list.append(text)
                prompt_indexes[text] = index
            indexes.append((index, weight))
        res_indexes.append(indexes)

    return res_indexes, prompt_flat_list, prompt_indexes


def _extract_multicond_preview_branches(text: str) -> list[tuple[str, float]] | None:
    conds_list, prompt_flat_list, _prompt_indexes = get_multicond_prompt_list([text])
    if not conds_list or len(conds_list[0]) <= 1:
        return None

    branches: list[tuple[str, float]] = []
    has_backend = False
    for prompt_index, weight in conds_list[0]:
        branch_text = str(prompt_flat_list[prompt_index])
        branch_weight = float(weight)
        branches.append((branch_text, branch_weight))
        if (
            _contains_chunk_marker(branch_text)
            or _contains_assemble_marker(branch_text)
            or _contains_blend_marker(branch_text)
            or _contains_morph_marker(branch_text)
            or _contains_pool_marker(branch_text)
        ):
            has_backend = True

    return branches if has_backend else None


def _build_multicond_preview_text(active_texts: Sequence[str], weights: Sequence[float]) -> str:
    parts: list[str] = []
    for text, weight in zip(active_texts, weights):
        normalized_text = _normalize_preview_fragment(text)
        if not normalized_text:
            continue
        if abs(float(weight) - 1.0) <= 1e-8:
            parts.append(normalized_text)
        else:
            parts.append(f"({normalized_text})*{_format_interp_weight(float(weight))}")
    return " AND ".join(parts) if parts else SAFE_EMPTY


def _build_multicond_text_schedule_from_branches(
    branches: Sequence[tuple[str, float]],
    steps: int,
    use_scheduling: bool,
    seed: int | None,
    use_visitor: bool,
    *,
    strict: bool,
) -> list[list[int, str]]:
    if not branches:
        return [[int(steps), SAFE_EMPTY]]

    if strict:
        branch_schedules = [
            _strict_schedule_preview(branch_text, steps, seed)
            for branch_text, _branch_weight in branches
        ]
    else:
        branch_schedules = [
            get_schedule(branch_text, steps, use_scheduling, seed, use_visitor=use_visitor)
            for branch_text, _branch_weight in branches
        ]

    boundaries = _collect_schedule_boundaries(branch_schedules, steps)
    out: list[list[int, str]] = []
    previous_key = None
    weights = [float(weight) for _branch_text, weight in branches]

    for end_at_step in boundaries:
        active_texts = [
            _select_text_from_schedule(schedule, end_at_step) or SAFE_EMPTY
            for schedule in branch_schedules
        ]
        preview = _build_multicond_preview_text(active_texts, weights)
        key = (tuple(active_texts), tuple(round(weight, 8) for weight in weights))
        if out and previous_key == key:
            out[-1][0] = int(end_at_step)
        else:
            out.append([int(end_at_step), preview])
            previous_key = key

    return out or [[int(steps), SAFE_EMPTY]]


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогалки для реконструкции батчей
# ──────────────────────────────────────────────────────────────────────────────

class DictWithShape(dict):
    def __init__(self, x, shape=None):
        super().__init__()
        self.update(x)
        self._shape = shape
    @property
    def shape(self):
        if self._shape is not None:
            return self._shape
        any_val = self.get("crossattn")
        if any_val is None and self:
            any_val = next(iter(self.values()))
        return getattr(any_val, "shape", None)


def _pick_schedule_entry_index(schedule, current_step):
    if not schedule:
        raise ValueError("Empty conditioning schedule")

    for current, entry in enumerate(schedule):
        if current_step <= int(entry.end_at_step):
            return current

    return len(schedule) - 1

def reconstruct_cond_batch(c: list[list[ScheduledPromptConditioning]], current_step):
    _torch = _ensure_torch()
        
    if not c or not c[0]:
        raise ValueError("Empty conditioning schedule")
    param = c[0][0].cond
    if param is None:
        raise ValueError("Invalid conditioning parameter")
    is_dict = isinstance(param, dict)
    if is_dict:
        dict_cond = param
        res = {k: _torch.zeros((len(c),) + v.shape, device=getattr(v, "device", "cpu"), dtype=getattr(v, "dtype", _torch.float32)) for k, v in dict_cond.items()}
        res = DictWithShape(res, (len(c),) + dict_cond.get('crossattn', next(iter(dict_cond.values()))).shape)
    else:
        res = _torch.zeros((len(c),) + param.shape, device=getattr(param, "device", "cpu"), dtype=getattr(param, "dtype", _torch.float32))

    for i, cond_schedule in enumerate(c):
        target_index = _pick_schedule_entry_index(cond_schedule, current_step)
        if is_dict:
            for k, v in cond_schedule[target_index].cond.items():
                res[k][i] = v
        else:
            res[i] = cond_schedule[target_index].cond
    return res

def stack_conds(tensors):
    _torch = _ensure_torch()
    token_count = max([x.shape[0] for x in tensors])
    for i in range(len(tensors)):
        if tensors[i].shape[0] != token_count:
            last_vector = tensors[i][-1: ]
            last_vector_repeated = last_vector.repeat([token_count - tensors[i].shape[0], 1])
            tensors[i] = _torch.vstack([tensors[i], last_vector_repeated])
    return _torch.stack(tensors)

def reconstruct_multicond_batch(c: MulticondLearnedConditioning, current_step):
    _torch = _ensure_torch()

    if not c.batch or not c.batch[0]:
        raise ValueError("Empty multicond batch")
    param = c.batch[0][0].schedules[0].cond
    if param is None:
        raise ValueError("Invalid conditioning parameter")
    tensors = []
    conds_list = []
    for composable_prompts in c.batch:
        conds_for_batch = []
        for composable_prompt in composable_prompts:
            weight = composable_prompt.weight_at_step(current_step)
            if abs(float(weight)) <= 1e-8:
                continue
            target_index = _pick_schedule_entry_index(composable_prompt.schedules, current_step)
            conds_for_batch.append((len(tensors), weight))
            tensors.append(composable_prompt.schedules[target_index].cond)
        conds_list.append(conds_for_batch)

    if isinstance(tensors[0], dict):
        keys = list(tensors[0].keys())
        stacked = {k: stack_conds([x[k] for x in tensors]) for k in keys}
        stacked = DictWithShape(stacked, stacked.get('crossattn', next(iter(stacked.values()))).shape)
    else:
        stacked = stack_conds(tensors).to(device=getattr(param, "device", "cpu"), dtype=getattr(param, "dtype", _torch.float32))
    return conds_list, stacked


# ──────────────────────────────────────────────────────────────────────────────
# Тестовые самопроверки
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные утилиты (не влияют на основной API/поведение)
# ──────────────────────────────────────────────────────────────────────────────

def _syntax_error_details(exc: Exception, source_text: str) -> dict:
    """Return structured syntax error details suitable for UIs.

    The returned dict is intentionally language-neutral: UI can map `kind` to
    localized messages and implement smart highlighting via `pos`/`end`.
    """
    # Default (unknown error)
    details: dict = {"kind": "error", "message": str(exc), "fix_suggestion": None}

    # Our explicit parser errors (e.g., strict reverse ranges)
    if isinstance(exc, PromptSyntaxError):
        src = exc.full or source_text
        token = exc.token
        pos = None
        end = None
        if token and src:
            p = src.find(token)
            if p != -1:
                pos = p
                end = p + len(token)
        details = {
            "kind": exc.kind or "prompt_syntax_error",
            "message": str(exc),
            "token": token,
            "pos": pos,
            "end": end,
            "fix_suggestion": None,
        }

        # Optional automatic fix for UIs (e.g., reverse range "8-4" -> "4-8")
        if details["kind"] == "reverse_range" and isinstance(token, str):
            m = re.match(r'^(\s*\d+%?\s*)-(\s*\d+%?\s*)$', token)
            if m:
                details["fix_suggestion"] = f"{m.group(2)}-{m.group(1)}"

        return details

    # Lark syntax errors
    if isinstance(exc, lark.exceptions.UnexpectedInput):
        pos = getattr(exc, "pos_in_stream", None)
        details = {"kind": "syntax_error", "message": str(exc), "pos": pos, "fix_suggestion": None}
        try:
            details["context"] = exc.get_context(source_text)
        except AttributeError:
            pass
        return details

    return details

def _format_syntax_error(exc: Exception, source_text: str) -> str:
    """Format a human-friendly prompt syntax error message.

    For programmatic handling (UI highlighting / localization), prefer `lint_prompt()`,
    which returns structured fields like `kind`, `pos`, `end`, and `token`.
    """
    details = _syntax_error_details(exc, source_text)
    kind = details.get("kind", "error")
    token = details.get("token")
    pos = details.get("pos")
    msg = details.get("message") or str(exc)

    # Short, language-neutral hints (UI can localize by `kind`).
    hint = ""
    if kind == "reverse_range":
        hint = "Range start is greater than end. Fix it (e.g. '4-8')."
    elif kind == "invalid_range_token":
        hint = "Use 'a-b' or 'a%-b%' (e.g. '1-4' or '10%-50%')."
    elif kind == "invalid_chunk_syntax":
        hint = "Use exactly one CHUNK{branch|branch} block with a closing '}'."
    elif kind == "empty_chunk_branch":
        hint = "CHUNK branches cannot be empty. Remove duplicate '|' or add text."
    elif kind == "invalid_chunk_weight":
        hint = "Use a positive numeric chunk weight at the end of a branch, e.g. 'wolf*1.5'."
    elif kind == "invalid_chunk_mode":
        hint = "Use CHUNK{...}, CHUNK[share-pooled]{...}, or CHUNK[share-cross]{...}."
    elif kind == "nested_chunk_not_supported":
        hint = "Nested CHUNK blocks are not supported in v1."
    elif kind == "nested_backend_in_chunk_not_supported":
        hint = "In v1, CHUNK branches may contain regular prompt grammar or MORPH blocks, but not BLEND, POOL, or ASSEMBLE blocks."
    elif kind == "chunk_inner_multicond_not_supported":
        hint = "In v1, CHUNK branches must stay single-branch prompt grammar; top-level AND is not supported inside CHUNK branches."
    elif kind == "multiple_chunk_blocks_not_supported":
        hint = "Use only one CHUNK block per prompt branch in v1."
    elif kind == "unsupported_chunk_context":
        hint = "In v1, CHUNK must appear as one top-level prompt block, not inside scheduler/group wrappers."
    elif kind == "invalid_pool_syntax":
        hint = "Use one POOL{prompt} block with regular prompt grammar inside and a closing '}'."
    elif kind == "nested_pool_not_supported":
        hint = "Nested POOL blocks are not supported in v1."
    elif kind == "nested_backend_in_pool_not_supported":
        hint = "In v1, POOL may contain regular prompt grammar only, not nested CHUNK, ASSEMBLE, BLEND, MORPH, BIND, or POOL blocks."
    elif kind == "invalid_assemble_syntax":
        hint = "Use ASSEMBLE{enc1=...; enc2=...; pooled=...}. Fields use '=' and enc1/enc2 are required."
    elif kind == "invalid_assemble_field":
        hint = "Supported ASSEMBLE fields are enc1, enc2, and pooled."
    elif kind == "duplicate_assemble_field":
        hint = "Each ASSEMBLE field may appear only once in v1."
    elif kind == "nested_assemble_not_supported":
        hint = "Nested ASSEMBLE blocks are not supported in v1."
    elif kind == "nested_backend_in_assemble_not_supported":
        hint = "In v1, ASSEMBLE fields must contain regular prompt grammar only, not nested CHUNK, BLEND, MORPH, POOL, or ASSEMBLE blocks."
    elif kind == "assemble_inner_multicond_not_supported":
        hint = "In v1, ASSEMBLE fields must stay single-branch prompt grammar; top-level AND is not supported inside enc1/enc2/pooled."
    elif kind == "multiple_assemble_blocks_not_supported":
        hint = "Use only one ASSEMBLE block per prompt branch in v1."
    elif kind == "unsupported_assemble_context":
        hint = "In v1, ASSEMBLE must appear as one top-level prompt block, not inside scheduler/group wrappers."
    elif kind == "multiple_pool_blocks_not_supported":
        hint = "Use only one POOL block per prompt branch in v1."
    elif kind == "unsupported_pool_context":
        hint = "In v1, POOL must appear as one top-level prompt adjunct, not inside scheduler/group wrappers."
    elif kind == "invalid_bind_syntax":
        hint = "Use BIND{owner => attrs} with non-empty owner and attrs sections and a closing '}'."
    elif kind == "invalid_bind_weight":
        hint = "Use a positive numeric BIND weight like BIND^1.2{owner => attrs}."
    elif kind == "nested_bind_not_supported":
        hint = "Nested BIND blocks are not supported in v1."
    elif kind == "nested_backend_in_bind_not_supported":
        hint = "In v1, BIND fields may contain regular prompt grammar only, not nested CHUNK, ASSEMBLE, BLEND, MORPH, POOL, or BIND blocks."
    elif kind == "bind_inner_multicond_not_supported":
        hint = "In v1, BIND owner and attrs must stay single-branch prompt grammar; top-level AND is not supported inside BIND fields."
    elif kind == "bind_requires_base_prompt":
        hint = "BIND needs a non-empty base prompt outside the BIND blocks in v1."
    elif kind == "unsupported_bind_context":
        hint = "In v1, BIND must appear as a top-level prompt adjunct, not inside scheduler/group wrappers."
    elif kind == "bind_with_backend_not_supported":
        hint = "In v1, BIND can be combined only with plain prompt text and optional POOL."
    elif kind == "bind_with_and_not_supported":
        hint = "In v1, BIND cannot be combined with top-level AND. Use separate prompts or keep BIND on a single branch."
    elif kind == "invalid_blend_syntax":
        hint = "Use BLEND{branch|branch} with at least two branches and a closing '}'."
    elif kind == "empty_blend_branch":
        hint = "BLEND branches cannot be empty. Remove duplicate '|' or add text."
    elif kind == "invalid_blend_weight":
        hint = "Use a positive numeric BLEND branch weight at the end of a branch, e.g. 'wolf*1.5'."
    elif kind == "invalid_blend_mode":
        hint = "Use a supported BLEND mode: mean or sum."
    elif kind == "invalid_blend_channel_target":
        hint = "Use BLEND[mean@both]{...}, BLEND[mean@cross]{...}, BLEND[mean@pooled]{...}, BLEND[mean@enc1]{...}, BLEND[mean@enc2]{...}, or omit @channel for old behavior."
    elif kind == "invalid_blend_intensity":
        hint = "Use BLEND^number{...} or BLEND^number[mode]{...} with a positive numeric intensity."
    elif kind == "nested_blend_not_supported":
        hint = "Nested BLEND blocks are not supported in v1."
    elif kind == "nested_backend_in_blend_not_supported":
        hint = "In v1, BLEND branches must contain regular prompt grammar only, not nested CHUNK, MORPH, POOL, or ASSEMBLE blocks."
    elif kind == "blend_inner_multicond_not_supported":
        hint = "In v1, BLEND branches must stay single-branch prompt grammar; top-level AND is not supported inside BLEND branches."
    elif kind == "multiple_blend_blocks_not_supported":
        hint = "Use only one BLEND block per prompt branch in v1."
    elif kind == "unsupported_blend_context":
        hint = "In v1, BLEND must appear as one top-level prompt block, not inside scheduler/group wrappers."
    elif kind == "invalid_morph_syntax":
        hint = "Use MORPH{prompt => prompt@step} with at least two control prompts and a closing '}'."
    elif kind == "invalid_morph_curve":
        hint = "Use a supported MORPH curve: linear, bezier, or catmull."
    elif kind == "invalid_morph_intensity":
        hint = "Use MORPH^number{...} or MORPH^number[start-end]{...} with a positive numeric intensity."
    elif kind == "invalid_morph_channel_target":
        hint = "Use MORPH@both{...}, MORPH@cross{...}, MORPH@pooled{...}, MORPH@enc1{...}, or MORPH@enc2{...}; intensity may come first, e.g. MORPH^1.3@enc1{...}."
    elif kind == "invalid_morph_point_weight":
        hint = "Use a positive numeric MORPH point weight like 'human*0.8' or 'cyborg*1.3@0.6'."
    elif kind == "invalid_morph_boundary":
        hint = "Use increasing MORPH boundaries like @0.6, @75%, or @12."
    elif kind == "invalid_morph_window":
        hint = "Use MORPH[start-end]{...} with increasing boundaries like 5-12 or 25%-50%."
    elif kind == "morph_window_with_point_boundaries_not_supported":
        hint = "In v1, use either MORPH[start-end]{a => b} or MORPH{a => b@0.6}, but not both together."
    elif kind == "nested_morph_not_supported":
        hint = "Nested MORPH blocks are not supported in v1."
    elif kind == "nested_backend_in_morph_not_supported":
        hint = "In v1, MORPH control prompts must contain regular prompt grammar only, not nested POOL or ASSEMBLE blocks."
    elif kind == "morph_inner_multicond_not_supported":
        hint = "In v1, MORPH control prompts must stay single-branch prompt grammar; top-level AND is not supported inside MORPH points."
    elif kind == "multiple_morph_blocks_not_supported":
        hint = "Use only one MORPH block per prompt branch in v1."
    elif kind == "unsupported_morph_context":
        hint = "In v1, MORPH must appear as one top-level prompt block, not inside scheduler/group wrappers."
    elif kind == "mixed_backend_blocks_not_supported":
        hint = "Use only one backend block per prompt branch in v1: CHUNK, ASSEMBLE, BLEND, or MORPH."
    elif kind == "duplicate_reverse_flag":
        hint = "Use the 'reverse' flag only once after a scheduler block."
    elif kind == "incomplete_sequence":
        hint = "Complete the sequence syntax or remove the dangling '::' / ':::'."
    elif kind == "invalid_interpolation":
        hint = "Use '(text:w0->w1)' with text and both weights present."
    elif kind == "syntax_error":
        msg = f"Syntax error at position {pos if pos is not None else '?'}."
        hint = ""

    # Pointer to the offending token (best-effort)
    pointer = ""
    if token and source_text:
        p = source_text.find(token)
        if p != -1:
            pointer = "\n" + source_text + "\n" + (" " * p) + ("^" * max(1, len(token)))

    # Lark often provides useful context lines
    context = details.get("context")
    if isinstance(context, str) and context.strip():
        msg = msg + "\n" + context.strip()

    if hint:
        msg = msg + "\n" + hint

    return (msg + pointer).strip()


def explain_syntax_error(text: str) -> str:
    """Вернуть человекочитаемое описание ошибки синтаксиса.
    При успехе вернуть пустую строку.
    """
    try:
        # Небольшой dry-parse через основной путь
        _ = _strict_schedule_preview(text, steps=20, seed=None)
        return ""
    except Exception as e:
        return _format_syntax_error(e, text)

def lint_prompt(text: str, steps: int = 20, seed: int | None = None) -> dict:
    """Dry-run validation and schedule preview.

    On success returns:
      - ok: True
      - kind: None
      - spans: number of schedule segments
      - preview: short preview string

    On failure returns:
      - ok: False
      - kind: machine-readable error code (e.g. 'reverse_range', 'syntax_error')
      - token/pos/end: best-effort location info for UI highlighting
      - fix_suggestion: optional replacement text for [pos:end] (e.g. swapped range)
      - error: human-friendly message (optional for UIs; can be localized externally)
      - details: structured payload (language-neutral)
    """
    try:
        sched = _strict_schedule_preview(text, steps=steps, seed=seed)
        spans = len(sched)
        preview = '; '.join([f"to {end}: {t}" for end, t in sched[:3]])
        return {"ok": True, "kind": None, "spans": spans, "preview": preview, "fix_suggestion": None}
    except Exception as e:
        details = _syntax_error_details(e, text)
        return {
            "ok": False,
            "kind": details.get("kind"),
            "token": details.get("token"),
            "pos": details.get("pos"),
            "end": details.get("end"),
            "error": _format_syntax_error(e, text),
            "fix_suggestion": details.get("fix_suggestion"),
            "details": details,
        }


def visualize_schedule(text: str, steps: int = 20, seed: int | None = None) -> str:

    """Сформатировать человекочитаемое представление расписания."""
    sched = get_schedule(text, steps=steps, use_scheduling=True, seed=seed, use_visitor=True)
    out_lines = []
    prev_end = 0
    for end, t in sched:
        start = prev_end + 1
        out_lines.append(f"Шаги {start}-{end}: {t}")
        prev_end = end
    return "\n".join(out_lines)


if __name__ == "__main__":
    import doctest, unittest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
    random.seed(42)

    g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]

    assert g("test") == [[10, 'test']]
    assert g("a [b:3]") == [[3, 'a'], [10, 'a b']]
    assert g("[(a:2):3]") == [[3, ''], [10, '(a:2)']]

    big = "{[a|b|c|d|e|f|g],[h|i|j|k|l|m|n],[o|p|q|r|s|t|u]}"
    res = g(big)
    assert len(res) <= GROUP_COMBO_LIMIT

    g2 = lambda p: get_learned_conditioning_prompt_schedules([p], 6)[0]
    one = g2("[cat|dog|fox]!")
    assert len(set([txt for _, txt in one])) == 1
    
    # reverse для «чистой» формы со ступенчатым делением внутри скобок
    g3 = lambda p: get_learned_conditioning_prompt_schedules([p], 6)[0]
    assert g3("[a:b:4] reverse") == [[4, 'b'], [6, 'a']]

    # Нумерованные альтернативы с нормализацией пробелов/AND
    g4 = lambda p: get_learned_conditioning_prompt_schedules([p], 5)[0]
    txt = g4("3[_a|b| |c]")[0][1]
    assert isinstance(txt, str) and len(txt) > 0

    # Пустые сегменты внутри [...:N] (чистая форма и reverse)
    g5 = lambda p: get_learned_conditioning_prompt_schedules([p], 6)[0]
    assert g5("[a::4]") == [[4, 'a'], [6, '']]
    assert g5("[a::4] reverse") == [[4, ''], [6, 'a']]

    # Legacy inner reverse should be tolerated without rewriting to postfix.
    g5r = lambda p, s=20: get_schedule(p, s, True, seed=123)
    assert g5r("[cat, dog:8 reverse]") == [[8, ''], [20, 'cat, dog']]
    assert g5r("[cat:dog:8 reverse]") == [[8, 'dog'], [20, 'cat']]
    assert g5r("[cat, dog:8 reverse] [night, moon:15 reverse]") == [
        [8, ''],
        [15, 'cat, dog'],
        [20, 'cat, dog night, moon'],
    ]
    assert g5r("[cat, dog:8 1-4,5-8] [night, moon:15 1-7,8-15]") == [
        [8, 'cat, dog night, moon'],
        [15, 'night, moon'],
        [20, ''],
    ]

    # Префикс/суффикс + пустой сегмент: добавляется пролог до первого интервала
    g6 = lambda p: get_learned_conditioning_prompt_schedules([p], 6)[0]
    assert g6("X [a::4] Y") == [[4, 'X a Y'], [6, 'X Y']]

    # --- новые проверки табов/переводов строки ---
    pa1 = parse_prompt_attention("dog\\tcat")
    joined1 = "".join(t for t, w in pa1 if t != "BREAK").strip()
    assert joined1 == "dog cat", f"Expected 'dog cat', got {joined1!r}"

    pa2 = parse_prompt_attention("dog\tcat")
    joined2 = "".join(t for t, w in pa2 if t != "BREAK").strip()
    assert joined2 == "dog cat", f"Expected 'dog cat', got {joined2!r}"

    # --- AND operator normalization ---
    g_and = lambda p: get_schedule(p, 4, True, seed=123)
    assert g_and("cat and dog") == [[4, "cat and dog"]]
    assert g_and("cat AND dog") == [[4, f"cat {ATTENTION_AND_OPERATOR} dog"]]
    assert g_and("cat & dog") == [[4, f"cat {ATTENTION_AND_OPERATOR} dog"]]
    assert g_and(r"R\&D") == [[4, "R&D"]]
    assert g_and(r"\&") == [[4, "&"]]

    _join_pa = lambda txt: "".join(t for t, w in parse_prompt_attention(txt) if t != "BREAK").strip()
    assert _join_pa(r"R\&D") == "R&D"
    assert _join_pa(r"\&") == "&"
    assert _join_pa("cat AND dog") == f"cat {ATTENTION_AND_OPERATOR} dog"

    py_compile.compile(__file__, doraise=True)
    # ФИЙ19: убрана загрузка __file__ через importlib — она инициализировала весь модуль
    # второй раз в памяти (регулярки, lark.Lark и т.д.) без какой-либо пользы.
    # Smoke-тесты выше уже работают через текущий модуль напрямую.

    # ── Attention interpolation tests (v21) ─────────────────────────────────
    import re as _re

    def _w(text, token):
        """Extract weight from '(token:W)' in text."""
        m = _re.search(rf'\({_re.escape(token)}:([\d.]+)\)', text)
        assert m, f"Token '({token}:...)' not found in: {text!r}"
        return float(m.group(1))

    g5 = lambda p, s=5: get_schedule(p, s, True,  seed=123)
    g5f= lambda p, s=5: get_schedule(p, s, False, seed=123)

    # 1. Basic linear interpolation: first step = w0, last step = w1
    sc = g5("(cat:1.0->2.0)", 5)
    assert sc[0][1]  == "(cat:1.0)", f"w0 wrong: {sc[0][1]}"
    assert sc[-1][1] == "(cat:2.0)", f"w1 wrong: {sc[-1][1]}"
    assert len(sc) == 5, f"Expected 5 steps, got {len(sc)}: {sc}"
    ws = [_w(t, "cat") for _, t in sc]
    assert all(ws[i] <= ws[i+1] for i in range(len(ws)-1)), f"Not monotone: {ws}"

    # 2. Prefix text preserved
    sc2 = g5("dog, (red eyes:0.8->1.4)", 5)
    assert sc2[0][1].startswith("dog,"),   f"Prefix lost: {sc2[0][1]}"
    assert abs(_w(sc2[0][1],  "red eyes") - 0.8) < 1e-4
    assert abs(_w(sc2[-1][1], "red eyes") - 1.4) < 1e-4

    # 3. Two interpolations in one prompt
    sc3 = g5("(cat:1.0->2.0), (dog:0.5->1.5)", 5)
    assert "(cat:2.0)" in sc3[-1][1], f"cat at last step: {sc3[-1][1]}"
    assert "(dog:1.5)" in sc3[-1][1], f"dog at last step: {sc3[-1][1]}"
    # First block must not be mangled (v20 bug: parsed as weighted + plain)
    assert "(cat:1.0)" in sc3[0][1],  f"cat at first step: {sc3[0][1]}"

    # 4. AND operator: both parts survive
    sc4 = g5("(cat:1.0->2.0) & dog", 5)
    assert "dog" in sc4[-1][1],   f"AND part lost: {sc4[-1][1]}"
    assert "(cat:2.0)" in sc4[-1][1], f"Interp at last step: {sc4[-1][1]}"

    # 5. use_scheduling=False -> collapses to w1 (last step)
    sc5 = g5f("(cat:1.0->2.0)", 5)
    assert len(sc5) == 1,              f"Expected 1 segment, got {len(sc5)}: {sc5}"
    assert sc5[0][1] == "(cat:2.0)",   f"Expected cat:2.0, got {sc5[0][1]}"

    # 6. Contextual-aware: inside scheduled window
    # [(cat:0.5->1.0):2] -> before boundary: empty, after: cat interpolating
    sc6 = g5("[(cat:0.5->1.0):2]", 5)
    # Key assertions: no raw '->' and no unexpanded marker leaks
    for end, txt in sc6:
        assert "->"            not in txt, f"Raw '->' leaked at step {end}: {txt!r}"
        assert ATTN_INTERP_OPEN not in txt, f"Unexpanded marker at step {end}: {txt!r}"
    # After boundary: cat weight present and in [0.5, 1.0]
    active = [(end, txt) for end, txt in sc6 if "(cat:" in txt]
    assert active, f"No active cat segments in: {sc6}"
    for end, txt in active:
        w = _w(txt, "cat")
        assert 0.5 - 1e-4 <= w <= 1.0 + 1e-4, f"Weight out of range [{end}]: {w}"

    # 7. Regression: static (cat:1.2) unchanged
    sr = g5("(cat:1.2)", 5)
    assert all("(cat:1.2)" in t for _, t in sr), f"Regression static emphasis: {sr}"

    # 8. Regression: scheduler [a:b:3] unchanged
    sr2 = g5("[a:b:3]", 5)
    assert "->" not in str(sr2), f"Regression scheduled: {sr2}"

    # 9. Regression: reverse unchanged
    sr3 = g5("[a:b:4] reverse", 5)
    assert "->" not in str(sr3), f"Regression reverse: {sr3}"

    # 10. No false positive: 'word -> something' plain text stays plain
    sc10 = g5("go -> here", 5)
    assert sc10[0][1] == "go -> here", f"False positive interp: {sc10[0][1]}"

    # ── lint_prompt regression tests ─────────────────────────────────────────
    # Lark parses (body:w0->w1) as emphasized(weighted(body:w0) + plain("->w1"))
    # — a valid tree, no exception. So lint_prompt must return ok=True,
    # not a syntax error. These tests lock that behaviour so a future grammar
    # change can't silently break it.

    # 11. lint_prompt: basic interpolation passes
    lr = lint_prompt("(cat:1.0->2.0)")
    assert lr["ok"], f"lint_prompt basic interp failed: {lr}"
    assert lr["kind"] is None, f"Unexpected error kind: {lr['kind']}"

    # 12. lint_prompt: multi-token body passes
    lr2 = lint_prompt("(dog, red hair, hair, blue eyes, sword:0.5->1.5)")
    assert lr2["ok"], f"lint_prompt multi-token body failed: {lr2}"

    # 13. lint_prompt: interpolation inside scheduled window passes
    lr3 = lint_prompt("[(cat:1.0->2.0):3]")
    assert lr3["ok"], f"lint_prompt scheduled interp failed: {lr3}"

    # 14. lint_prompt: mixed static + interpolation passes
    lr4 = lint_prompt("(cat:1.2), (dog:0.8->1.5)")
    assert lr4["ok"], f"lint_prompt mixed failed: {lr4}"

    # 15. lint_prompt: plain '->' text passes (not a false-positive interp error)
    lr5 = lint_prompt("go -> here")
    assert lr5["ok"], f"lint_prompt plain arrow failed: {lr5}"

    # 16. lint_prompt: spans count is reasonable (not zero)
    lr6 = lint_prompt("(cat:1.0->2.0)", steps=5)
    assert lr6["ok"],       f"lint_prompt interp steps failed: {lr6}"
    assert lr6["spans"] > 0, f"lint_prompt spans=0: {lr6}"

    # ── visualize_schedule regression tests ─────────────────────────────────
    # visualize_schedule must not crash and must not leak raw '->' or markers
    # in its output for any interpolation input.

    # 17. visualize_schedule: basic interpolation — no raw '->' in output
    vs = visualize_schedule("(cat:1.0->2.0)", steps=5)
    assert "->"            not in vs, f"Raw '->' in visualize output:\n{vs}"
    assert ATTN_INTERP_OPEN not in vs, f"Unexpanded marker in visualize output:\n{vs}"
    assert "(cat:"          in vs,     f"'(cat:...)' missing in visualize output:\n{vs}"

    # 18. visualize_schedule: multi-token body
    vs2 = visualize_schedule("(dog, red hair, sword:0.5->1.5)", steps=5)
    assert "->"            not in vs2, f"Raw '->' in visualize multi:\n{vs2}"
    assert ATTN_INTERP_OPEN not in vs2, f"Marker leaked in visualize multi:\n{vs2}"

    # 19. visualize_schedule: inside scheduled window
    vs3 = visualize_schedule("[(cat:1.0->2.0):3]", steps=5)
    assert "->"            not in vs3, f"Raw '->' in visualize scheduled:\n{vs3}"
    assert ATTN_INTERP_OPEN not in vs3, f"Marker leaked in visualize scheduled:\n{vs3}"

    # 20. visualize_schedule: plain '->' text is preserved as-is
    vs4 = visualize_schedule("go -> here", steps=5)
    assert "go -> here" in vs4, f"Plain arrow mangled in visualize:\n{vs4}"

    # ── 5 дополнительных тестов (по рекомендации GPT) ─────────────────────────

    # 21. lint_prompt.preview содержит раскрытые веса, а не сырой синтаксис
    # Фиксируем, что preview строится из развёрнутого расписания, а не из исходной строки.
    lr7 = lint_prompt("(cat:1.0->2.0)", steps=5)
    assert lr7["ok"], f"lint_prompt preview test failed: {lr7}"
    preview = lr7.get("preview", "")
    assert "->"            not in preview, f"Raw '->' in preview: {preview!r}"
    assert ATTN_INTERP_OPEN not in preview, f"Marker leaked in preview: {preview!r}"
    assert "(cat:"          in preview,    f"'(cat:...)' missing in preview: {preview!r}"

    # 22. use_visitor=True vs use_visitor=False — parity для базовой интерполяции
    # Оба маршрута должны давать одинаковый результат, так как pre-pass работает до Lark.
    sv1 = get_schedule("(cat:1.0->2.0)", 5, True, seed=123, use_visitor=True)
    sv2 = get_schedule("(cat:1.0->2.0)", 5, True, seed=123, use_visitor=False)
    assert sv1 == sv2, f"use_visitor parity failed:\n  visitor=True:  {sv1}\n  visitor=False: {sv2}"

    # 23. use_visitor parity для interpolation внутри scheduled-окна
    sv3 = get_schedule("[(cat:0.5->1.0):2]", 5, True, seed=123, use_visitor=True)
    sv4 = get_schedule("[(cat:0.5->1.0):2]", 5, True, seed=123, use_visitor=False)
    assert sv3 == sv4, f"use_visitor scheduled parity failed:\n  True:  {sv3}\n  False: {sv4}"

    # 24. Single-step семантика: steps=1, span==0 → вес w0, а не w1
    # Важно: при span==0 в _expand_att_interp_text_for_step берётся t=0.0, то есть w0.
    ss1 = get_schedule("(cat:1.0->2.0)", 1, True,  seed=123)
    assert ss1 == [[1, "(cat:1.0)"]], f"Single-step use_scheduling=True: {ss1}"
    ss2 = get_schedule("(cat:1.0->2.0)", 1, False, seed=123)
    assert ss2 == [[1, "(cat:1.0)"]], f"Single-step use_scheduling=False: {ss2}"

    # 25. Точный финальный текст для двух interpolation-блоков в одной строке
    # Проверяет, что оба блока раскрываются корректно — не только "что-то там есть".
    sm = get_schedule("(cat:1.0->2.0), (dog:0.5->1.5)", 5, True, seed=123)
    assert sm[0][1]  == "(cat:1.0), (dog:0.5)", f"Mixed first step: {sm[0][1]!r}"
    assert sm[-1][1] == "(cat:2.0), (dog:1.5)", f"Mixed last step:  {sm[-1][1]!r}"

    print("All quick integration tests passed!")

    # ── GPT P2/P3 regression tests ───────────────────────────────────────────

    # P1 (already fixed): ESCAPED_AMP_PLACEHOLDER uses private-use char \uE004
    # Verify user text containing the old string placeholder is NOT mangled
    r_amp = get_schedule("__PROMPT_PARSER_ESCAPED_AMP__", 4, True, seed=123)
    assert r_amp[0][1] == "__PROMPT_PARSER_ESCAPED_AMP__", \
        f"P1 regression: amp placeholder leaked into output: {r_amp}"

    # P2 (already fixed): LITERAL_REVERSE_TOKEN uses private-use char \uE005
    r_rev = get_schedule("__PP_LITERAL_REVERSE__", 4, True, seed=123)
    assert r_rev[0][1] == "__PP_LITERAL_REVERSE__", \
        f"P2 regression: reverse token leaked into output: {r_rev}"

    # P3: 'reverse, shot' and 'reverse shot' must give same suffix (no leading comma)
    g_p3 = lambda p: get_schedule(p, 4, True, seed=123)
    r_a = g_p3("[a:b]:10 reverse shot")
    r_b = g_p3("[a:b]:10 reverse, shot")
    # Both should have 'shot' in the suffix text, no leading ', '
    for end, txt in r_a:
        assert not txt.startswith(", "), f"P3a: leading comma in text: {txt!r}"
    for end, txt in r_b:
        assert not txt.startswith(", "), f"P3b: leading comma in text: {txt!r}"
    # And the non-reverse content should match between the two
    texts_a = [t for _, t in r_a]
    texts_b = [t for _, t in r_b]
    assert texts_a == texts_b, \
        f"P3: 'reverse shot' vs 'reverse, shot' gave different results:\n  A={texts_a}\n  B={texts_b}"

    # P4 (already fixed): SdConditioning preserves falsy numeric values (0)
    sd = SdConditioning([], width=0, height=0, distilled_cfg_scale=0)
    assert sd.width == 0,               f"P4: width=0 lost, got {sd.width!r}"
    assert sd.height == 0,              f"P4: height=0 lost, got {sd.height!r}"
    assert sd.distilled_cfg_scale == 0, f"P4: distilled_cfg_scale=0 lost, got {sd.distilled_cfg_scale!r}"

    print("GPT P2/P3 regression tests passed!")

    # ── P4 regression: empty alternate compatibility with A1111 ──────────────
    # [fe|]male: step1 -> 'female', step2 -> 'male' (cycles like A1111)
    g_alt = lambda p, s=10: get_learned_conditioning_prompt_schedules([p], s)[0]
    r_p4 = g_alt("[fe|]male", 10)
    step1 = r_p4[0][1]  # first entry
    # Find step-1 and step-2 texts by simulating get_schedule per step
    sc_p4 = get_schedule("[fe|]male", 10, True, seed=None)
    texts = {end: txt for end, txt in sc_p4}
    # At minimum: should contain 'female' and 'male' entries (alternating)
    all_texts = [txt for _, txt in sc_p4]
    assert "female" in all_texts, f"P4: 'female' missing from schedule: {sc_p4}"
    assert "male"   in all_texts, f"P4: 'male' missing from schedule: {sc_p4}"
    # Should NOT have 'fe male' (space between fe and male)
    assert "fe male" not in all_texts, f"P4: got 'fe male' instead of 'female': {sc_p4}"
    print("P4 empty alternate regression passed!")

    # ── Edge case tests ──────────────────────────────────────────────────────

    import re as _re2

    def _w2(text, token):
        m = _re2.search(rf'\({_re2.escape(token)}:([-\d.]+)\)', text)
        assert m, f"'({token}:...)' not found in: {text!r}"
        return float(m.group(1))

    g_ec = lambda p, s=5: get_schedule(p, s, True, seed=123)

    # EC-1: w0 == w1 — constant interpolation, all steps same weight
    sc_eq = g_ec("(cat:1.2->1.2)", 5)
    for end, txt in sc_eq:
        assert "(cat:1.2)" in txt, f"EC-1 constant interp, step {end}: {txt!r}"

    # EC-2: negative start weight (-0.5 -> 0.5)
    sc_neg = g_ec("(cat:-0.5->0.5)", 5)
    w_neg_first = _w2(sc_neg[0][1],  "cat")
    w_neg_last  = _w2(sc_neg[-1][1], "cat")
    assert abs(w_neg_first - (-0.5)) < 1e-4, f"EC-2 neg start: {w_neg_first}"
    assert abs(w_neg_last  -   0.5)  < 1e-4, f"EC-2 neg end:   {w_neg_last}"
    assert w_neg_first < w_neg_last,          f"EC-2 not increasing: {w_neg_first} -> {w_neg_last}"

    # EC-3: negative end weight (1.0 -> -0.5)
    sc_neg2 = g_ec("(cat:1.0->-0.5)", 5)
    w_n2_first = _w2(sc_neg2[0][1],  "cat")
    w_n2_last  = _w2(sc_neg2[-1][1], "cat")
    assert abs(w_n2_first - 1.0)  < 1e-4, f"EC-3 start: {w_n2_first}"
    assert abs(w_n2_last  - (-0.5)) < 1e-4, f"EC-3 end: {w_n2_last}"
    assert w_n2_first > w_n2_last,           f"EC-3 not decreasing: {w_n2_first} -> {w_n2_last}"

    # EC-4: alternate [a|b] combined with interpolation — no crash, no marker leak
    sc_alt = g_ec("[cat|dog], (red eyes:0.8->1.4)", 5)
    for end, txt in sc_alt:
        assert ATTN_INTERP_OPEN not in txt, f"EC-4 marker leak at step {end}: {txt!r}"
        assert "->" not in txt,             f"EC-4 raw arrow at step {end}: {txt!r}"
        assert "(red eyes:" in txt,         f"EC-4 interp missing at step {end}: {txt!r}"

    # EC-5: reverse + interpolation in same prompt — no crash, no marker leak
    sc_rev = g_ec("[a:b:3] reverse, (cat:1.0->2.0)", 5)
    for end, txt in sc_rev:
        assert ATTN_INTERP_OPEN not in txt, f"EC-5 marker leak at step {end}: {txt!r}"
        assert "->" not in txt,             f"EC-5 raw arrow at step {end}: {txt!r}"
    # Interpolation ramps across the whole schedule
    last_txt = sc_rev[-1][1]
    assert "(cat:2.0)" in last_txt, f"EC-5 cat not at w1 on last step: {last_txt!r}"

    # EC-6: AND with both branches having weights / interpolation
    sc_and2 = g_ec("(cat:1.5) & (dog:0.5->1.5)", 5)
    for end, txt in sc_and2:
        assert ATTN_INTERP_OPEN not in txt, f"EC-6 marker leak at step {end}: {txt!r}"
    assert "(cat:1.5)" in sc_and2[0][1],  f"EC-6 static branch first step:  {sc_and2[0][1]!r}"
    assert "(dog:1.5)" in sc_and2[-1][1], f"EC-6 interp branch last step:   {sc_and2[-1][1]!r}"

    # EC-7: AND with two interpolation branches
    sc_and3 = g_ec("(cat:1.0->2.0) & (dog:0.5->1.5)", 5)
    assert "(cat:2.0)" in sc_and3[-1][1], f"EC-7 cat last step:  {sc_and3[-1][1]!r}"
    assert "(dog:1.5)" in sc_and3[-1][1], f"EC-7 dog last step:  {sc_and3[-1][1]!r}"
    assert "(cat:1.0)" in sc_and3[0][1],  f"EC-7 cat first step: {sc_and3[0][1]!r}"
    assert "(dog:0.5)" in sc_and3[0][1],  f"EC-7 dog first step: {sc_and3[0][1]!r}"

    # EC-8: multi-token body with negative weight
    sc_mt_neg = g_ec("(red hair, blue eyes:-0.3->0.8)", 5)
    w_mt_first = _w2(sc_mt_neg[0][1],  "red hair, blue eyes")
    w_mt_last  = _w2(sc_mt_neg[-1][1], "red hair, blue eyes")
    assert abs(w_mt_first - (-0.3)) < 1e-4, f"EC-8 start: {w_mt_first}"
    assert abs(w_mt_last  -   0.8)  < 1e-4, f"EC-8 end:   {w_mt_last}"

    print("Edge case tests passed!")

    # ── v2 nested body tests ─────────────────────────────────────────────────

    g_v2 = lambda p, s=5: get_schedule(p, s, True, seed=123)

    # V2-1: nested body '(cat)' — emphasized transformer converts (cat)->(cat:1.1),
    # then outer interpolation weight wraps it: ((cat:1.1):w)
    sc_v2_1 = g_v2("((cat):1.0->2.0)", 5)
    assert "((cat:1.1):1.0)" in sc_v2_1[0][1],  f"V2-1 first: {sc_v2_1[0][1]!r}"
    assert "((cat:1.1):2.0)" in sc_v2_1[-1][1], f"V2-1 last:  {sc_v2_1[-1][1]!r}"

    # V2-2: inner '(glowing eyes)' → emphasized → '(glowing eyes:1.1)',
    # outer interpolation weight wraps the whole body.
    sc_v2_2 = g_v2("(red (glowing eyes):0.8->1.4)", 5)
    assert "->" not in sc_v2_2[0][1],              f"V2-2 raw arrow: {sc_v2_2[0][1]!r}"
    assert ATTN_INTERP_OPEN not in sc_v2_2[-1][1], f"V2-2 marker leak: {sc_v2_2[-1][1]!r}"
    assert "(red (glowing eyes:1.1):0.8)" in sc_v2_2[0][1],  f"V2-2 first: {sc_v2_2[0][1]!r}"
    assert "(red (glowing eyes:1.1):1.4)" in sc_v2_2[-1][1], f"V2-2 last:  {sc_v2_2[-1][1]!r}"

    # V2-3: '(a:b)' where b is non-numeric → emphasized gives (a:1.1),
    # outer interpolation wraps: ((a:1.1):w)
    sc_v2_3 = g_v2("((a:b):1.0->2.0)", 5)
    assert "((a:1.1):1.0)" in sc_v2_3[0][1],  f"V2-3 first: {sc_v2_3[0][1]!r}"
    assert "((a:1.1):2.0)" in sc_v2_3[-1][1], f"V2-3 last:  {sc_v2_3[-1][1]!r}"

    # V2-4: plain '->' text is NOT treated as interpolation (no outer parens)
    sc_v2_4 = g_v2("go -> here", 5)
    assert sc_v2_4[0][1] == "go -> here", f"V2-4 false positive: {sc_v2_4[0][1]!r}"

    # V2-5: unmatched paren — must not crash, left as-is
    sc_v2_5 = g_v2("(unclosed:1.0->2.0", 5)
    assert "->" in sc_v2_5[0][1] or sc_v2_5[0][1], f"V2-5 crash or empty"

    # V2-6: nested body + scheduled window (contextual-aware still works)
    sc_v2_6 = g_v2("[(red (glowing eyes):0.5->1.0):2]", 5)
    for end, txt in sc_v2_6:
        assert ATTN_INTERP_OPEN not in txt, f"V2-6 marker leak step {end}: {txt!r}"
        assert "->" not in txt,             f"V2-6 raw arrow step {end}: {txt!r}"

    # V2-7: v1 multi-token body still works (no regression)
    sc_v2_7 = g_v2("(cat, dog, sword:1.0->2.0)", 5)
    assert "(cat, dog, sword:1.0)" in sc_v2_7[0][1],  f"V2-7 first: {sc_v2_7[0][1]!r}"
    assert "(cat, dog, sword:2.0)" in sc_v2_7[-1][1], f"V2-7 last:  {sc_v2_7[-1][1]!r}"

    print("v2 nested body tests passed!")

    # ── Easing tests ─────────────────────────────────────────────────────────

    import math as _math

    g_ez = lambda p, s=5: get_schedule(p, s, True, seed=123)

    def _w_at(schedule, step_idx):
        """Extract float weight from '(cat:W)' in schedule entry."""
        import re as _re3
        txt = schedule[step_idx][1]
        m = _re3.search(r'\(cat:([-\d.]+)\)', txt)
        assert m, f"no (cat:w) in {txt!r}"
        return float(m.group(1))

    # EZ-0: no suffix → linear (baseline)
    sc_lin = g_ez("(cat:0.0->1.0)", 5)
    w_lin  = [_w_at(sc_lin, i) for i in range(len(sc_lin))]
    # linear: differences between consecutive steps should be equal
    diffs = [round(w_lin[i+1] - w_lin[i], 4) for i in range(len(w_lin)-1)]
    assert len(set(diffs)) == 1, f"EZ-0 linear not uniform: {diffs}"

    # EZ-1: ease-in — starts slow, ends fast → w[midpoint] < linear midpoint
    sc_ei = g_ez("(cat:0.0->1.0~ease-in)", 5)
    w_ei  = [_w_at(sc_ei, i) for i in range(len(sc_ei))]
    mid_lin = 0.5
    mid_ei  = w_ei[len(w_ei)//2]
    assert mid_ei < mid_lin + 0.01, f"EZ-1 ease-in midpoint {mid_ei} not below linear {mid_lin}"
    assert w_ei[0]  <= w_lin[0]  + 1e-3, f"EZ-1 ease-in start mismatch: {w_ei[0]}"
    assert w_ei[-1] >= w_lin[-1] - 1e-3, f"EZ-1 ease-in end mismatch:   {w_ei[-1]}"

    # EZ-2: ease-out — starts fast, ends slow → w[midpoint] > linear midpoint
    sc_eo = g_ez("(cat:0.0->1.0~ease-out)", 5)
    w_eo  = [_w_at(sc_eo, i) for i in range(len(sc_eo))]
    mid_eo = w_eo[len(w_eo)//2]
    assert mid_eo > mid_lin - 0.01, f"EZ-2 ease-out midpoint {mid_eo} not above linear {mid_lin}"

    # EZ-3: ease-in-out — symmetric S-curve; midpoint ≈ 0.5 (exact for smooth step)
    sc_eio = g_ez("(cat:0.0->1.0~ease-in-out)", 5)
    w_eio  = [_w_at(sc_eio, i) for i in range(len(sc_eio))]
    mid_eio = w_eio[len(w_eio)//2]
    assert abs(mid_eio - 0.5) < 0.15, f"EZ-3 ease-in-out midpoint {mid_eio} far from 0.5"

    # EZ-4: unknown mode falls back to linear
    sc_unk = g_ez("(cat:0.0->1.0~bogus-mode)", 5)
    w_unk  = [_w_at(sc_unk, i) for i in range(len(sc_unk))]
    assert w_unk == w_lin, f"EZ-4 unknown mode not linear: {w_unk}"

    # EZ-5: easing with non-zero w0 (0.5->1.5~ease-in) — endpoints exact
    sc_e5 = g_ez("(cat:0.5->1.5~ease-in)", 5)
    assert abs(_w_at(sc_e5, 0) - 0.5) < 1e-3,  f"EZ-5 start: {_w_at(sc_e5, 0)}"
    assert abs(_w_at(sc_e5, -1) - 1.5) < 1e-3, f"EZ-5 end:   {_w_at(sc_e5, -1)}"

    # EZ-6: easing inside scheduled window — no marker leak, no raw ~
    sc_e6 = g_ez("[(cat:0.0->1.0~ease-out):2]", 5)
    for end, txt in sc_e6:
        assert ATTN_INTERP_OPEN not in txt, f"EZ-6 marker leak: {txt!r}"
        assert "~" not in txt,              f"EZ-6 raw tilde: {txt!r}"
        assert "->" not in txt,             f"EZ-6 raw arrow: {txt!r}"

    # EZ-7: backward compat — old-style no easing still works
    sc_e7 = g_ez("(cat:1.0->2.0)", 5)
    assert abs(_w_at(sc_e7, 0)  - 1.0) < 1e-3, f"EZ-7 compat start: {_w_at(sc_e7, 0)}"
    assert abs(_w_at(sc_e7, -1) - 2.0) < 1e-3, f"EZ-7 compat end:   {_w_at(sc_e7, -1)}"

    # EZ-8: case-insensitive mode — EASE-IN / Ease_In → same as ease-in
    sc_upper = g_ez("(cat:0.0->1.0~EASE-IN)", 5)
    w_upper  = [_w_at(sc_upper, i) for i in range(len(sc_upper))]
    assert w_upper == w_ei, f"EZ-8 EASE-IN != ease-in: {w_upper} vs {w_ei}"

    print("Easing tests passed!")