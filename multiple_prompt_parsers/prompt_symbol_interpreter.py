# prompt_symbol_interpreter.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_SYMBOL_CONFIG = {
    "reserved_symbols": {
        "semantic_prompt": "%%",
    },
    "backend_symbols": {
        "chunk": "&&",
        "blend": "<+>",
        "morph": ">>",
        "assemble": "@@",
        "bind": "=>",
        "pool": "$$",
    },
    "sequence_symbols": {
        "group_open": "{",
        "group_close": "}",
        "sequence": "::",
        "deep_sequence": ":::",
        "close": "!",
        "top_close": "!!",
    },
    "backend_wrappers": {
        "open": "(",
        "close": ")",
    },
}


CANONICAL_BACKEND = {
    "chunk": "CHUNK",
    "blend": "BLEND",
    "morph": "MORPH",
    "assemble": "ASSEMBLE",
    "bind": "BIND",
    "pool": "POOL",
}


@dataclass
class PromptSymbolConfig:
    reserved_symbols: dict[str, str] = field(default_factory=dict)
    backend_symbols: dict[str, str] = field(default_factory=dict)
    sequence_symbols: dict[str, str] = field(default_factory=dict)
    backend_wrappers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "PromptSymbolConfig":
        return cls.from_dict(DEFAULT_SYMBOL_CONFIG)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PromptSymbolConfig":
        merged = _deep_merge(DEFAULT_SYMBOL_CONFIG, data or {})
        return cls(
            reserved_symbols=dict(merged.get("reserved_symbols", {})),
            backend_symbols=dict(merged.get("backend_symbols", {})),
            sequence_symbols=dict(merged.get("sequence_symbols", {})),
            backend_wrappers=dict(merged.get("backend_wrappers", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromptSymbolConfig":
        if yaml is None:
            raise RuntimeError("PyYAML is required to load prompt symbol YAML files.")

        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    def validate(self) -> None:
        used: dict[str, str] = {}

        for section_name, section in {
            "reserved_symbols": self.reserved_symbols,
            "backend_symbols": self.backend_symbols,
            "sequence_symbols": self.sequence_symbols,
        }.items():
            for name, symbol in section.items():
                if not isinstance(symbol, str) or symbol == "":
                    raise ValueError(f"{section_name}.{name} must be a non-empty string.")

                if symbol in used:
                    raise ValueError(
                        f"Symbol collision: {section_name}.{name} uses {symbol!r}, "
                        f"already used by {used[symbol]}."
                    )

                used[symbol] = f"{section_name}.{name}"

        semantic_symbol = self.reserved_symbols.get("semantic_prompt")
        if semantic_symbol:
            for name, symbol in self.backend_symbols.items():
                if symbol == semantic_symbol:
                    raise ValueError(
                        f"Backend symbol {name!r} cannot use reserved semantic_prompt symbol "
                        f"{semantic_symbol!r}."
                    )


class PromptSymbolInterpreter:
    """
    Converts user-facing prompt symbols into canonical parser syntax.

    User-facing example:
        forest &&(wolf*1.5 | hunter*0.8) fog

    Canonical output:
        forest CHUNK{wolf*1.5 | hunter*0.8} fog

    This file does not modify prompt_parser_21.py.
    It translates before parser/runtime calls.
    """

    def __init__(self, config: PromptSymbolConfig | None = None):
        self.config = config or PromptSymbolConfig.default()
        self.config.validate()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromptSymbolInterpreter":
        return cls(PromptSymbolConfig.from_yaml(path))

    def to_canonical(self, prompt: str) -> str:
        if not prompt:
            return prompt

        text = prompt

        text = self._translate_backend_blocks(text)

        # Optional future hook:
        # text = self._translate_sequence_symbols(text)

        return text

    def _translate_backend_blocks(self, text: str) -> str:
        symbols = self.config.backend_symbols
        open_wrap = self.config.backend_wrappers.get("open", "(")
        close_wrap = self.config.backend_wrappers.get("close", ")")

        for key in ("assemble", "chunk", "blend", "morph", "pool"):
            symbol = symbols.get(key)
            if not symbol:
                continue

            canonical = CANONICAL_BACKEND[key]
            text = self._replace_wrapped_operator(
                text=text,
                symbol=symbol,
                canonical=canonical,
                open_wrap=open_wrap,
                close_wrap=close_wrap,
                canonical_open="{",
                canonical_close="}",
            )

        text = self._translate_bind(text)

        return text

    def _translate_bind(self, text: str) -> str:
        """
        Converts:
            =>(owner: attrs)
            =>^1.2(owner: attrs)

        Into:
            BIND{owner => attrs}
            BIND^1.2{owner => attrs}

        This avoids ambiguity with the internal BIND arrow.
        """
        symbol = self.config.backend_symbols.get("bind", "=>")
        if not symbol:
            return text

        open_wrap = self.config.backend_wrappers.get("open", "(")
        close_wrap = self.config.backend_wrappers.get("close", ")")

        i = 0
        out: list[str] = []

        while i < len(text):
            if not self._matches_unescaped(text, i, symbol):
                out.append(text[i])
                i += 1
                continue

            j = i + len(symbol)
            weight = ""

            if j < len(text) and text[j] == "^":
                weight_start = j
                j += 1
                while j < len(text) and text[j] != open_wrap:
                    j += 1
                weight = text[weight_start:j].strip()

            while j < len(text) and text[j].isspace():
                j += 1

            if j >= len(text) or text[j] != open_wrap:
                out.append(text[i])
                i += 1
                continue

            end = self._find_matching(text, j, open_wrap, close_wrap)
            if end is None:
                out.append(text[i])
                i += 1
                continue

            body = text[j + 1:end].strip()
            owner, attrs = self._split_bind_body(body)

            if not owner or not attrs:
                out.append(text[i:end + 1])
                i = end + 1
                continue

            out.append(f"BIND{weight}{{{owner} => {attrs}}}")
            i = end + 1

        return "".join(out)

    def _replace_wrapped_operator(
        self,
        text: str,
        symbol: str,
        canonical: str,
        open_wrap: str,
        close_wrap: str,
        canonical_open: str,
        canonical_close: str,
    ) -> str:
        """
        Converts:
            SYMBOL(...)
            SYMBOL^1.2(...)
            SYMBOL[mode](...)

        Into:
            CANONICAL{...}
            CANONICAL^1.2{...}
            CANONICAL[mode]{...}
        """
        i = 0
        out: list[str] = []

        while i < len(text):
            if not self._matches_unescaped(text, i, symbol):
                out.append(text[i])
                i += 1
                continue

            j = i + len(symbol)
            modifiers = ""

            while j < len(text) and text[j].isspace():
                j += 1

            # Preserve modifiers like ^1.4, [mean@pooled], @cross, [5-12]
            while j < len(text) and text[j] != open_wrap:
                if text[j].isspace():
                    j += 1
                    continue

                if text[j] == "[":
                    end_mod = self._find_matching(text, j, "[", "]")
                    if end_mod is None:
                        break
                    modifiers += text[j:end_mod + 1]
                    j = end_mod + 1
                    continue

                if text[j] in "{(":
                    break

                modifiers += text[j]
                j += 1

            while j < len(text) and text[j].isspace():
                j += 1

            if j >= len(text) or text[j] != open_wrap:
                out.append(text[i])
                i += 1
                continue

            end = self._find_matching(text, j, open_wrap, close_wrap)
            if end is None:
                out.append(text[i])
                i += 1
                continue

            body = text[j + 1:end]
            out.append(f"{canonical}{modifiers}{canonical_open}{body}{canonical_close}")
            i = end + 1

        return "".join(out)

    @staticmethod
    def _split_bind_body(body: str) -> tuple[str, str]:
        """
        Supports:
            owner: attrs
            owner -> attrs
            owner => attrs

        Returns:
            owner, attrs
        """
        for sep in ("=>", "->", ":"):
            idx = _find_top_level_separator(body, sep)
            if idx != -1:
                return body[:idx].strip(), body[idx + len(sep):].strip()

        return "", ""

    @staticmethod
    def _matches_unescaped(text: str, index: int, symbol: str) -> bool:
        if not text.startswith(symbol, index):
            return False

        slash_count = 0
        j = index - 1
        while j >= 0 and text[j] == "\\":
            slash_count += 1
            j -= 1

        return slash_count % 2 == 0

    @staticmethod
    def _find_matching(
        text: str,
        open_index: int,
        open_char: str,
        close_char: str,
    ) -> int | None:
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


def to_canonical_prompt(
    prompt: str,
    config: PromptSymbolConfig | dict[str, Any] | None = None,
) -> str:
    if isinstance(config, dict):
        config = PromptSymbolConfig.from_dict(config)

    return PromptSymbolInterpreter(config).to_canonical(prompt)


def load_interpreter_from_yaml(path: str | Path) -> PromptSymbolInterpreter:
    return PromptSymbolInterpreter.from_yaml(path)


def _find_top_level_separator(text: str, sep: str) -> int:
    depth_round = 0
    depth_square = 0
    depth_curly = 0
    i = 0

    while i < len(text):
        ch = text[i]

        if ch == "\\":
            i += 2
            continue

        if depth_round == 0 and depth_square == 0 and depth_curly == 0:
            if text.startswith(sep, i):
                return i

        if ch == "(":
            depth_round += 1
        elif ch == ")" and depth_round > 0:
            depth_round -= 1
        elif ch == "[":
            depth_square += 1
        elif ch == "]" and depth_square > 0:
            depth_square -= 1
        elif ch == "{":
            depth_curly += 1
        elif ch == "}" and depth_curly > 0:
            depth_curly -= 1

        i += 1

    return -1


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


'''
if __name__ == "__main__":
    interpreter = PromptSymbolInterpreter()

    examples = [
        "forest &&(wolf*1.5 | hunter*0.8) fog",
        "portrait <>(photo realism*0.7 | oil painting*0.3)",
        "portrait >>^1.4@cross(human*0.8 => cyborg*1.3@0.6 ~ bezier)",
        "@@(enc1=wolf; enc2=moonlit forest; pooled=cold atmosphere)",
        "1girl, street scene =>(1boy: red eyes, pink scarf)",
        "$$(cold ominous atmosphere)",
    ]

    for example in examples:
        print("USER:     ", example)
        print("CANONICAL:", interpreter.to_canonical(example))
        print()
'''