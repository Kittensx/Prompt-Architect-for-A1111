from __future__ import annotations

import re

BACKEND_WORDS = ("BLEND", "CHUNK", "MORPH", "POOL", "ASSEMBLE")


def normalize_backend_prompt(prompt: str) -> str:
    """
    Lift simple wrapped backend blocks into top-level backend blocks.

    Example:
        {portrait BLEND{photo realism*0.9 | oil painting*1.5}}

    Becomes:
        BLEND{portrait photo realism*0.9 | portrait oil painting*1.5}
    """
    text = str(prompt or "")

    text = _normalize_wrapped_blend(text)
    text = _normalize_wrapped_chunk(text)

    return text


def _normalize_wrapped_blend(text: str) -> str:
    return _normalize_wrapped_branch_backend(text, "BLEND")


def _normalize_wrapped_chunk(text: str) -> str:
    return _normalize_wrapped_branch_backend(text, "CHUNK")


def _normalize_wrapped_branch_backend(text: str, keyword: str) -> str:
    stripped = text.strip()

    if not (stripped.startswith("{") and stripped.endswith("}")):
        return text

    inner = stripped[1:-1].strip()

    block = _find_top_level_backend_block(inner, keyword)
    if block is None:
        return text

    start, open_brace, close_brace = block

    prefix = inner[:start].strip()
    suffix = inner[close_brace + 1:].strip()
    body = inner[open_brace + 1:close_brace].strip()

    if not prefix and not suffix:
        return text

    branches = _split_top_level(body, "|")
    lifted = []

    for branch in branches:
        branch = branch.strip()
        parts = [part for part in (prefix, branch, suffix) if part]
        lifted.append(" ".join(parts))

    return f"{keyword}{{{' | '.join(lifted)}}}"


def _find_top_level_backend_block(text: str, keyword: str):
    pattern = re.compile(rf"(?<![\w\\]){re.escape(keyword)}(?:\s*\^[^\[\{{]*)?(?:\s*\[[^\]]*\])?\s*\{{")

    for match in pattern.finditer(text):
        start = match.start()
        open_brace = text.find("{", match.start())

        if _depth_at(text, start) != (0, 0, 0):
            continue

        close_brace = _find_matching(text, open_brace, "{", "}")
        if close_brace is None:
            return None

        return start, open_brace, close_brace

    return None


def _split_top_level(text: str, sep: str) -> list[str]:
    parts = []
    buf = []
    round_depth = square_depth = curly_depth = 0
    i = 0

    while i < len(text):
        ch = text[i]

        if ch == "\\":
            buf.append(ch)
            if i + 1 < len(text):
                buf.append(text[i + 1])
                i += 2
                continue

        if ch == sep and round_depth == square_depth == curly_depth == 0:
            parts.append("".join(buf))
            buf.clear()
            i += 1
            continue

        if ch == "(":
            round_depth += 1
        elif ch == ")" and round_depth:
            round_depth -= 1
        elif ch == "[":
            square_depth += 1
        elif ch == "]" and square_depth:
            square_depth -= 1
        elif ch == "{":
            curly_depth += 1
        elif ch == "}" and curly_depth:
            curly_depth -= 1

        buf.append(ch)
        i += 1

    parts.append("".join(buf))
    return parts


def _find_matching(text: str, open_index: int, open_char: str, close_char: str) -> int | None:
    depth = 0
    i = open_index

    while i < len(text):
        ch = text[i]

        if ch == "\\":
            i += 2
            continue

        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return i

        i += 1

    return None


def _depth_at(text: str, index: int) -> tuple[int, int, int]:
    round_depth = square_depth = curly_depth = 0
    i = 0

    while i < index:
        ch = text[i]

        if ch == "\\":
            i += 2
            continue

        if ch == "(":
            round_depth += 1
        elif ch == ")" and round_depth:
            round_depth -= 1
        elif ch == "[":
            square_depth += 1
        elif ch == "]" and square_depth:
            square_depth -= 1
        elif ch == "{":
            curly_depth += 1
        elif ch == "}" and curly_depth:
            curly_depth -= 1

        i += 1

    return round_depth, square_depth, curly_depth