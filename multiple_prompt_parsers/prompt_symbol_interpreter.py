# prompt_symbol_interpreter.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from copy import deepcopy

from modules.prompt_symbol_defaults import DEFAULT_SYMBOL_CONFIG

try:
    import yaml
except ImportError:
    yaml = None


CANONICAL_BACKEND = {
    "chunk": "CHUNK",
    "blend": "BLEND",
    "morph": "MORPH",
    "assemble": "ASSEMBLE",
    "bind": "BIND",
    "pool": "POOL",
}


@dataclass(slots=True)
class OperatorSpec:
    canonical: str
    default_symbol: str
    aliases: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "OperatorSpec":
        canonical = str(data.get("canonical", "")).strip()
        default_symbol = str(data.get("default_symbol", "")).strip()
        aliases = [str(item) for item in data.get("aliases", []) if str(item)]

        if not canonical:
            raise ValueError(f"{name}.canonical must be set.")

        if not default_symbol:
            raise ValueError(f"{name}.default_symbol must be set.")

        if default_symbol not in aliases:
            aliases.insert(0, default_symbol)

        if canonical not in aliases:
            aliases.append(canonical)

        return cls(
            canonical=canonical,
            default_symbol=default_symbol,
            aliases=aliases,
        )


@dataclass(slots=True)
class PromptSymbolConfig:
    version: int = 2
    reserved_symbols: dict[str, OperatorSpec] = field(default_factory=dict)
    backend_operators: dict[str, OperatorSpec] = field(default_factory=dict)
    sequence_operators: dict[str, OperatorSpec] = field(default_factory=dict)
    grouping_operators: dict[str, OperatorSpec] = field(default_factory=dict)
    branch_operators: dict[str, OperatorSpec] = field(default_factory=dict)
    weight_operators: dict[str, OperatorSpec] = field(default_factory=dict)
    transition_operators: dict[str, OperatorSpec] = field(default_factory=dict)
    escaping: dict[str, Any] = field(default_factory=dict)
    serialization: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromptSymbolConfig":
        if yaml is None:
            raise RuntimeError("PyYAML is required to load prompt symbol YAML files.")

        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        merged = _deep_merge_dicts(DEFAULT_SYMBOL_CONFIG, data)
        merged = _apply_user_alias_priority(merged, data)
        return cls.from_dict(merged)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PromptSymbolConfig":
        if not isinstance(data, dict):
            raise ValueError("Prompt symbol config must be a dictionary.")

        version = int(data.get("version", 2))
        if version != 2:
            raise ValueError("Only prompt_symbols.yaml version 2 is supported.")

        config = cls(
            version=version,
            reserved_symbols=_parse_operator_section(data, "reserved_symbols"),
            backend_operators=_parse_operator_section(data, "backend_operators"),
            sequence_operators=_parse_operator_section(data, "sequence_operators"),
            grouping_operators=_parse_operator_section(data, "grouping_operators"),
            branch_operators=_parse_operator_section(data, "branch_operators"),
            weight_operators=_parse_operator_section(data, "weight_operators"),
            transition_operators=_parse_operator_section(data, "transition_operators"),
            escaping=dict(data.get("escaping", {})),
            serialization=dict(data.get("serialization", {})),
            validation=dict(data.get("validation", {})),
        )

        config.validate()
        return config

    @classmethod
    def default(cls) -> "PromptSymbolConfig":
        return cls.from_dict(DEFAULT_SYMBOL_CONFIG)

    def validate(self) -> None:
        allow_alias_collisions = bool(
            self.validation.get("allow_alias_collisions", False)
        )
        allow_reserved_override = bool(
            self.validation.get("allow_reserved_symbol_override", False)
        )

        reserved_aliases: dict[str, str] = {}
        used_aliases: dict[str, str] = {}

        for group_name, section in self._all_sections().items():
            for op_name, spec in section.items():
                if not spec.aliases:
                    raise ValueError(f"{group_name}.{op_name}.aliases cannot be empty.")

                for alias in spec.aliases:
                    if not alias:
                        raise ValueError(f"{group_name}.{op_name} has an empty alias.")

                    full_name = f"{group_name}.{op_name}"

                    if group_name == "reserved_symbols":
                        reserved_aliases[alias] = full_name
                        continue

                    if alias in reserved_aliases and not allow_reserved_override:
                        raise ValueError(
                            f"{full_name} alias {alias!r} conflicts with reserved "
                            f"symbol {reserved_aliases[alias]}."
                        )

                    if alias in used_aliases and not allow_alias_collisions:
                        raise ValueError(
                            f"Alias collision: {full_name} uses {alias!r}, "
                            f"already used by {used_aliases[alias]}."
                        )

                    used_aliases[alias] = full_name

    def _all_sections(self) -> dict[str, dict[str, OperatorSpec]]:
        return {
            "reserved_symbols": self.reserved_symbols,
            "backend_operators": self.backend_operators,
            "sequence_operators": self.sequence_operators,
            "grouping_operators": self.grouping_operators,
            "branch_operators": self.branch_operators,
            "weight_operators": self.weight_operators,
            "transition_operators": self.transition_operators,
        }

    def backend_aliases(self, key: str) -> list[str]:
        spec = self.backend_operators.get(key)
        return list(spec.aliases) if spec else []

    def backend_canonical(self, key: str) -> str:
        spec = self.backend_operators.get(key)
        if spec:
            return spec.canonical
        return CANONICAL_BACKEND[key]

    def wrapper_open(self) -> str:
        return self.grouping_operators["open_attention"].default_symbol

    def wrapper_close(self) -> str:
        return self.grouping_operators["close_attention"].default_symbol


class PromptSymbolInterpreter:
    """
    Converts user-facing aliases into canonical parser syntax.

    Examples:
        <+>(a | b)      -> BLEND{a | b}
        BLEND(a | b)   -> BLEND{a | b}
        BLEND{a | b}   -> unchanged
    """

    def __init__(self, config: PromptSymbolConfig):
        self.config = config
        self.config.validate()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromptSymbolInterpreter":
        return cls(PromptSymbolConfig.from_yaml(path))

    def to_canonical(self, prompt: str) -> str:
        if not prompt:
            return prompt

        text = str(prompt)
        text = self._translate_backend_blocks(text)
        return text

    def _translate_backend_blocks(self, text: str) -> str:
        open_wrap = self.config.wrapper_open()
        close_wrap = self.config.wrapper_close()

        # Longest aliases first prevents "::" before ":::"-style issues
        # and prevents shorter symbolic aliases from partially matching longer ones.
        for key in ("assemble", "chunk", "blend", "morph", "pool"):
            canonical = self.config.backend_canonical(key)
            aliases = sorted(self.config.backend_aliases(key), key=len, reverse=True)

            for alias in aliases:
                text = self._replace_wrapped_operator(
                    text=text,
                    alias=alias,
                    canonical=canonical,
                    open_wrap=open_wrap,
                    close_wrap=close_wrap,
                    canonical_open="{",
                    canonical_close="}",
                )

        text = self._translate_bind(text)
        return text

    def _translate_bind(self, text: str) -> str:
        aliases = sorted(self.config.backend_aliases("bind"), key=len, reverse=True)
        if not aliases:
            return text

        open_wrap = self.config.wrapper_open()
        close_wrap = self.config.wrapper_close()
        canonical = self.config.backend_canonical("bind")

        for alias in aliases:
            text = self._replace_bind_alias(
                text=text,
                alias=alias,
                canonical=canonical,
                open_wrap=open_wrap,
                close_wrap=close_wrap,
            )

        return text

    def _replace_bind_alias(
        self,
        *,
        text: str,
        alias: str,
        canonical: str,
        open_wrap: str,
        close_wrap: str,
    ) -> str:
        i = 0
        out: list[str] = []

        while i < len(text):
            if not self._matches_alias_at(text, i, alias):
                out.append(text[i])
                i += 1
                continue

            j = i + len(alias)

            # Do not rewrite existing canonical BIND{...}
            k = j
            while k < len(text) and text[k].isspace():
                k += 1
            if k < len(text) and text[k] == "{":
                out.append(text[i])
                i += 1
                continue

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

            out.append(f"{canonical}{weight}{{{owner} => {attrs}}}")
            i = end + 1

        return "".join(out)

    def _replace_wrapped_operator(
        self,
        *,
        text: str,
        alias: str,
        canonical: str,
        open_wrap: str,
        close_wrap: str,
        canonical_open: str,
        canonical_close: str,
    ) -> str:
        i = 0
        out: list[str] = []

        while i < len(text):
            if not self._matches_alias_at(text, i, alias):
                out.append(text[i])
                i += 1
                continue

            j = i + len(alias)

            # Already canonical block form, e.g. BLEND{a | b}
            k = j
            while k < len(text) and text[k].isspace():
                k += 1
            if k < len(text) and text[k] == canonical_open:
                out.append(text[i])
                i += 1
                continue

            modifiers = ""

            while j < len(text) and text[j].isspace():
                j += 1

            # Preserve modifiers like ^1.4, [mean@pooled], @cross, [5-12].
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
        for sep in ("=>", "->", ":"):
            idx = _find_top_level_separator(body, sep)
            if idx != -1:
                return body[:idx].strip(), body[idx + len(sep):].strip()

        return "", ""

    @staticmethod
    def _matches_alias_at(text: str, index: int, alias: str) -> bool:
        if not text.startswith(alias, index):
            return False

        slash_count = 0
        j = index - 1
        while j >= 0 and text[j] == "\\":
            slash_count += 1
            j -= 1

        if slash_count % 2 != 0:
            return False

        # Word aliases like BLEND should not match inside MYBLEND or BLENDER.
        if alias and (alias[0].isalnum() or alias[0] == "_"):
            prev_ch = text[index - 1] if index > 0 else ""
            if prev_ch and (prev_ch.isalnum() or prev_ch == "_"):
                return False

        end = index + len(alias)
        if alias and (alias[-1].isalnum() or alias[-1] == "_"):
            next_ch = text[end] if end < len(text) else ""
            if next_ch and (next_ch.isalnum() or next_ch == "_"):
                return False

        return True

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
    config: PromptSymbolConfig | dict[str, Any],
) -> str:
    if isinstance(config, dict):
        config = PromptSymbolConfig.from_dict(config)

    if config is None:
        raise ValueError("A prompt symbol config is required.")

    return PromptSymbolInterpreter(config).to_canonical(prompt)


def load_interpreter_from_yaml(path: str | Path) -> PromptSymbolInterpreter:
    return PromptSymbolInterpreter.from_yaml(path)


def _parse_operator_section(
    data: dict[str, Any],
    section_name: str,
) -> dict[str, OperatorSpec]:
    section = data.get(section_name, {})
    if not isinstance(section, dict):
        raise ValueError(f"{section_name} must be a dictionary.")

    parsed: dict[str, OperatorSpec] = {}

    for name, value in section.items():
        if not isinstance(value, dict):
            raise ValueError(f"{section_name}.{name} must be a dictionary.")

        parsed[str(name)] = OperatorSpec.from_dict(
            f"{section_name}.{name}",
            value,
        )

    return parsed


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

_OPERATOR_SECTIONS = (
    "reserved_symbols",
    "backend_operators",
    "sequence_operators",
    "grouping_operators",
    "branch_operators",
    "weight_operators",
    "transition_operators",
)


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def _collect_user_alias_claims(user_data: dict[str, Any]) -> dict[str, str]:
    claims: dict[str, str] = {}

    for section_name in _OPERATOR_SECTIONS:
        section = user_data.get(section_name, {})
        if not isinstance(section, dict):
            continue

        for op_name, spec in section.items():
            if not isinstance(spec, dict):
                continue

            full_name = f"{section_name}.{op_name}"
            aliases = spec.get("aliases", [])

            for alias in aliases:
                alias = str(alias).strip()
                if alias:
                    claims[alias] = full_name

            default_symbol = str(spec.get("default_symbol", "")).strip()
            if default_symbol:
                claims[default_symbol] = full_name

    return claims


def _apply_user_alias_priority(
    merged: dict[str, Any],
    user_data: dict[str, Any],
) -> dict[str, Any]:
    claims = _collect_user_alias_claims(user_data)

    reserved = merged.get("reserved_symbols", {})
    reserved_aliases = set()

    for spec in reserved.values():
        if isinstance(spec, dict):
            reserved_aliases.update(str(a) for a in spec.get("aliases", []) if str(a))
            if spec.get("default_symbol"):
                reserved_aliases.add(str(spec["default_symbol"]))

    for section_name in _OPERATOR_SECTIONS:
        section = merged.get(section_name, {})
        if not isinstance(section, dict):
            continue

        for op_name, spec in section.items():
            if not isinstance(spec, dict):
                continue

            full_name = f"{section_name}.{op_name}"

            if section_name == "reserved_symbols":
                continue

            aliases = [str(a) for a in spec.get("aliases", []) if str(a)]
            aliases = [
                alias
                for alias in aliases
                if alias not in claims
                or claims[alias] == full_name
                or alias in reserved_aliases
            ]

            spec["aliases"] = aliases

            default_symbol = str(spec.get("default_symbol", "")).strip()
            if default_symbol and default_symbol not in aliases:
                if aliases:
                    spec["default_symbol"] = aliases[0]
                else:
                    spec["default_symbol"] = str(spec.get("canonical", op_name)).strip()

    return merged