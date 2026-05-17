from __future__ import annotations

from dataclasses import dataclass

try:
    from modules.prompt_graph import (
        AlternateNode,
        AndBranch,
        AndNode,
        AssembleNode,
        BackendBranch,
        BindNode,
        BlendNode,
        ChunkNode,
        GroupNode,
        MorphNode,
        MorphPoint,
        PoolNode,
        PromptGraph,
        PromptNode,
        ScheduleBoundary,
        ScheduleNode,
        ScheduleSegment,
        SequenceNode,
        TextNode,
        WeightNode,
    )
except ImportError:
    from prompt_graph import (
        AlternateNode,
        AndBranch,
        AndNode,
        AssembleNode,
        BackendBranch,
        BindNode,
        BlendNode,
        ChunkNode,
        GroupNode,
        MorphNode,
        MorphPoint,
        PoolNode,
        PromptGraph,
        PromptNode,
        ScheduleBoundary,
        ScheduleNode,
        ScheduleSegment,
        SequenceNode,
        TextNode,
        WeightNode,
    )


class PromptGraphParseError(Exception):
    pass


@dataclass(slots=True)
class Token:
    kind: str
    value: str
    start: int
    end: int


class PromptTokenizer:
    """
    Lightweight structural tokenizer.

    This does not normalize aliases. Run prompt_symbol_interpreter first so
    shorthand like <+>(a | b) becomes canonical BLEND{a | b}.
    """

    SYMBOLS = {
        "{",
        "}",
        "(",
        ")",
        "[",
        "]",
        "|",
        ":",
        "*",
        "@",
        "^",
        ";",
        "=",
        "~",
    }

    DOUBLE_SYMBOLS = {
        "=>",
    }

    KEYWORDS = {
        "BLEND",
        "CHUNK",
        "MORPH",
        "POOL",
        "BIND",
        "ASSEMBLE",
        "AND",
    }

    def tokenize(self, text: str) -> list[Token]:
        tokens: list[Token] = []
        i = 0
        length = len(text)

        while i < length:
            ch = text[i]

            if ch.isspace():
                i += 1
                continue

            if ch == "\\":
                if i + 1 >= length:
                    raise PromptGraphParseError("Trailing escape character")

                tokens.append(
                    Token(
                        kind="TEXT",
                        value=text[i + 1],
                        start=i,
                        end=i + 2,
                    )
                )
                i += 2
                continue

            matched_double = False
            for symbol in sorted(self.DOUBLE_SYMBOLS, key=len, reverse=True):
                if text.startswith(symbol, i):
                    tokens.append(
                        Token(
                            kind=symbol,
                            value=symbol,
                            start=i,
                            end=i + len(symbol),
                        )
                    )
                    i += len(symbol)
                    matched_double = True
                    break

            if matched_double:
                continue

            if ch in self.SYMBOLS:
                tokens.append(
                    Token(
                        kind=ch,
                        value=ch,
                        start=i,
                        end=i + 1,
                    )
                )
                i += 1
                continue

            start = i

            while i < length:
                current = text[i]

                if current.isspace():
                    break

                if current in self.SYMBOLS:
                    break

                if any(text.startswith(sym, i) for sym in self.DOUBLE_SYMBOLS):
                    break

                i += 1

            value = text[start:i]
            kind = value if value in self.KEYWORDS else "TEXT"

            tokens.append(
                Token(
                    kind=kind,
                    value=value,
                    start=start,
                    end=i,
                )
            )

        return tokens


class PromptGraphParser:
    VALID_BLEND_MODES = {"mean", "sum"}
    VALID_CHANNEL_TARGETS = {"both", "cross", "pooled", "enc1", "enc2"}
    VALID_CHUNK_MODES = {"share-pooled", "share-cross"}
    VALID_MORPH_CURVES = {"linear", "bezier", "catmull"}

    def __init__(self, text: str):
        self.text = str(text or "")
        self.tokens = PromptTokenizer().tokenize(self.text)
        self.index = 0

    def parse(self) -> PromptGraph:
        root = self.parse_sequence(stop_tokens=None)

        if not self.at_end():
            token = self.peek()
            raise PromptGraphParseError(f"Unexpected token: {token.value!r}")

        return PromptGraph(root=root)

    def parse_sequence(self, stop_tokens: set[str] | None) -> PromptNode:
        parts: list[PromptNode] = []

        while not self.at_end():
            token = self.peek()

            if stop_tokens and token.kind in stop_tokens:
                break

            parts.append(self.parse_expression())

        if not parts:
            return TextNode("")

        if len(parts) == 1:
            return parts[0]

        return SequenceNode(parts=parts)

    def parse_expression(self) -> PromptNode:
        return self.parse_and_expression()

    def parse_and_expression(self) -> PromptNode:
        first = self.parse_primary()

        branches = [
            AndBranch(node=first)
        ]

        while not self.at_end() and self.peek_kind() == "AND":
            self.consume("AND")
            branches.append(
                AndBranch(
                    node=self.parse_primary()
                )
            )

        if len(branches) == 1:
            return first

        return AndNode(branches=branches)

    def parse_primary(self) -> PromptNode:
        token = self.peek()

        if token.kind == "BLEND":
            return self.parse_blend()

        if token.kind == "CHUNK":
            return self.parse_chunk()

        if token.kind == "MORPH":
            return self.parse_morph()

        if token.kind == "POOL":
            return self.parse_pool()

        if token.kind == "BIND":
            return self.parse_bind()

        if token.kind == "ASSEMBLE":
            return self.parse_assemble()

        if token.kind == "{":
            return self.parse_group("brace")

        if token.kind == "(":
            return self.parse_weight_or_group()

        if token.kind == "[":
            return self.parse_alternate_or_schedule()

        if token.kind == "TEXT":
            return self.parse_text()

        raise PromptGraphParseError(
            f"Unexpected token {token.value!r}"
        )

    def parse_text(self) -> PromptNode:
        token = self.consume("TEXT")
        return TextNode(text=token.value)

    def parse_group(self, delimiter: str) -> PromptNode:
        if delimiter == "brace":
            open_tok = "{"
            close_tok = "}"
        elif delimiter == "paren":
            open_tok = "("
            close_tok = ")"
        elif delimiter == "bracket":
            open_tok = "["
            close_tok = "]"
        else:
            raise RuntimeError(f"Unknown delimiter: {delimiter!r}")

        self.consume(open_tok)

        node = self.parse_sequence(stop_tokens={close_tok})

        self.consume(close_tok)

        if isinstance(node, SequenceNode):
            return GroupNode(parts=node.parts, delimiter=delimiter)

        return GroupNode(parts=[node], delimiter=delimiter)

    def parse_weight_or_group(self) -> PromptNode:
        start = self.index

        self.consume("(")

        inner = self.parse_sequence(stop_tokens={":", ")"})

        if self.peek_kind() == ":":
            self.consume(":")
            weight = self.parse_number()
            self.consume(")")

            return WeightNode(
                node=inner,
                weight=weight,
            )

        self.index = start
        return self.parse_group("paren")

    def parse_alternate_or_schedule(self) -> PromptNode:
        self.consume("[")

        items: list[PromptNode] = []
        separators: list[str] = []

        while not self.at_end():
            if self.peek_kind() == "]":
                break

            node = self.parse_sequence(stop_tokens={"|", ":", "]"})
            items.append(node)

            if self.peek_kind() in {"|", ":"}:
                separators.append(self.consume().kind)
                continue

            break

        self.consume("]")

        if "|" in separators:
            return AlternateNode(options=items)

        if ":" in separators:
            return ScheduleNode(
                segments=[
                    ScheduleSegment(node=item)
                    for item in items
                ]
            )

        return GroupNode(parts=items, delimiter="bracket")

    def parse_blend(self) -> PromptNode:
        self.consume("BLEND")

        intensity = self.parse_optional_intensity()
        blend_mode, channel_target = self.parse_optional_blend_mode()

        self.consume("{")

        branches = self.parse_backend_branches(stop_tokens={"}"})

        self.consume("}")

        return BlendNode(
            branches=branches,
            blend_mode=blend_mode,
            channel_target=channel_target,
            intensity=intensity,
        )

    def parse_chunk(self) -> PromptNode:
        self.consume("CHUNK")

        shared_channel = "none"

        if self.peek_kind() == "[":
            mode_text = self.collect_bracket_text().strip().lower()

            if mode_text not in self.VALID_CHUNK_MODES:
                raise PromptGraphParseError(
                    f"Unsupported CHUNK mode: {mode_text!r}"
                )

            if mode_text == "share-pooled":
                shared_channel = "pooled"
            elif mode_text == "share-cross":
                shared_channel = "cross"

        self.consume("{")

        branches = self.parse_backend_branches(stop_tokens={"}"})

        self.consume("}")

        return ChunkNode(
            branches=branches,
            shared_channel=shared_channel,
        )

    def parse_morph(self) -> PromptNode:
        self.consume("MORPH")

        intensity = self.parse_optional_intensity()
        channel_target = self.parse_optional_channel_target()
        window_start, window_end = self.parse_optional_window()

        self.consume("{")

        points: list[MorphPoint] = []

        while not self.at_end():
            if self.peek_kind() == "}":
                break

            node = self.parse_sequence(
                stop_tokens={"*", "@", "=>", "~", "}"}
            )

            weight = 1.0
            boundary = None

            if self.peek_kind() == "*":
                self.consume("*")
                weight = self.parse_number()

            if self.peek_kind() == "@":
                self.consume("@")
                boundary = self.parse_boundary()

            points.append(
                MorphPoint(
                    node=node,
                    boundary=boundary,
                    weight=weight,
                )
            )

            if self.peek_kind() == "=>":
                self.consume("=>")
                continue

            break

        curve = "linear"

        if self.peek_kind() == "~":
            self.consume("~")
            curve = self.consume("TEXT").value.strip().lower()

            if curve not in self.VALID_MORPH_CURVES:
                raise PromptGraphParseError(
                    f"Unsupported MORPH curve: {curve!r}"
                )

        self.consume("}")

        return MorphNode(
            points=points,
            curve=curve,
            channel_target=channel_target,
            intensity=intensity,
            window_start=window_start,
            window_end=window_end,
        )

    def parse_pool(self) -> PromptNode:
        self.consume("POOL")
        self.consume("{")

        node = self.parse_sequence(stop_tokens={"}"})

        self.consume("}")

        return PoolNode(node=node)

    def parse_bind(self) -> PromptNode:
        self.consume("BIND")

        weight = self.parse_optional_intensity()

        self.consume("{")

        owner = self.parse_sequence(stop_tokens={"=>", "}"})

        self.consume("=>")

        attrs = self.parse_sequence(stop_tokens={"}"})

        self.consume("}")

        return BindNode(
            owner=owner,
            attrs=attrs,
            weight=weight,
        )

    def parse_assemble(self) -> PromptNode:
        self.consume("ASSEMBLE")
        self.consume("{")

        fields: dict[str, PromptNode] = {}

        while not self.at_end():
            if self.peek_kind() == "}":
                break

            name = self.consume("TEXT").value.strip().lower()

            if name not in {"enc1", "enc2", "pooled"}:
                raise PromptGraphParseError(
                    f"Unsupported ASSEMBLE field: {name!r}"
                )

            self.consume("=")

            value = self.parse_sequence(stop_tokens={";", "}"})
            fields[name] = value

            if self.peek_kind() == ";":
                self.consume(";")
                continue

            break

        self.consume("}")

        if "enc1" not in fields or "enc2" not in fields:
            raise PromptGraphParseError(
                "ASSEMBLE requires enc1 and enc2 fields."
            )

        return AssembleNode(
            enc1=fields["enc1"],
            enc2=fields["enc2"],
            pooled=fields.get("pooled"),
        )

    def parse_backend_branches(
        self,
        stop_tokens: set[str],
    ) -> list[BackendBranch]:
        branches: list[BackendBranch] = []

        while not self.at_end():
            if self.peek_kind() in stop_tokens:
                break

            branch = self.parse_sequence(
                stop_tokens={"|", "*", *stop_tokens}
            )

            weight = 1.0

            if self.peek_kind() == "*":
                self.consume("*")
                weight = self.parse_number()

            branches.append(
                BackendBranch(
                    node=branch,
                    weight=weight,
                )
            )

            if self.peek_kind() == "|":
                self.consume("|")
                continue

            break

        return branches

    def parse_optional_intensity(self) -> float:
        if self.peek_kind() != "^":
            return 1.0

        self.consume("^")
        return self.parse_number()

    def parse_optional_channel_target(self) -> str:
        if self.peek_kind() != "@":
            return "both"

        self.consume("@")
        channel = self.consume("TEXT").value.strip().lower()

        if channel not in self.VALID_CHANNEL_TARGETS:
            raise PromptGraphParseError(
                f"Unsupported channel target: {channel!r}"
            )

        return channel

    def parse_optional_blend_mode(self) -> tuple[str, str]:
        if self.peek_kind() != "[":
            return "mean", "both"

        mode_text = self.collect_bracket_text().strip().lower()

        if not mode_text:
            return "mean", "both"

        if "@" in mode_text:
            blend_mode, channel_target = mode_text.split("@", 1)
            blend_mode = blend_mode.strip() or "mean"
            channel_target = channel_target.strip() or "both"
        else:
            blend_mode = mode_text
            channel_target = "both"

        if blend_mode not in self.VALID_BLEND_MODES:
            raise PromptGraphParseError(
                f"Unsupported BLEND mode: {blend_mode!r}"
            )

        if channel_target not in self.VALID_CHANNEL_TARGETS:
            raise PromptGraphParseError(
                f"Unsupported BLEND channel target: {channel_target!r}"
            )

        return blend_mode, channel_target

    def parse_optional_window(
        self,
    ) -> tuple[ScheduleBoundary | None, ScheduleBoundary | None]:
        if self.peek_kind() != "[":
            return None, None

        window_text = self.collect_bracket_text().strip()

        if "-" not in window_text:
            raise PromptGraphParseError(
                f"Expected MORPH window as start-end, got {window_text!r}"
            )

        start_text, end_text = window_text.split("-", 1)

        return (
            self.boundary_from_text(start_text.strip()),
            self.boundary_from_text(end_text.strip()),
        )

    def parse_boundary(self) -> ScheduleBoundary:
        token = self.consume("TEXT")
        return self.boundary_from_text(token.value)

    def boundary_from_text(self, text: str) -> ScheduleBoundary:
        raw = str(text).strip()

        if not raw:
            raise PromptGraphParseError("Empty boundary")

        if raw.endswith("%"):
            return ScheduleBoundary(
                value=float(raw[:-1]),
                kind="percent",
            )

        value = float(raw)

        if 0.0 < value < 1.0:
            return ScheduleBoundary(
                value=value,
                kind="fraction",
            )

        return ScheduleBoundary(
            value=value,
            kind="step",
        )

    def collect_bracket_text(self) -> str:
        self.consume("[")

        parts: list[str] = []
        depth = 1

        while not self.at_end() and depth > 0:
            token = self.consume()

            if token.kind == "[":
                depth += 1
                parts.append(token.value)
                continue

            if token.kind == "]":
                depth -= 1
                if depth == 0:
                    break
                parts.append(token.value)
                continue

            parts.append(token.value)

        if depth != 0:
            raise PromptGraphParseError("Unclosed bracket block")

        return "".join(parts)

    def parse_number(self) -> float:
        token = self.consume("TEXT")

        try:
            return float(token.value)
        except ValueError as exc:
            raise PromptGraphParseError(
                f"Expected number, got {token.value!r}"
            ) from exc

    def at_end(self) -> bool:
        return self.index >= len(self.tokens)

    def peek(self) -> Token:
        if self.at_end():
            raise PromptGraphParseError("Unexpected end of input")

        return self.tokens[self.index]

    def peek_kind(self) -> str | None:
        if self.at_end():
            return None

        return self.tokens[self.index].kind

    def consume(self, expected_kind: str | None = None) -> Token:
        token = self.peek()

        if expected_kind is not None and token.kind != expected_kind:
            raise PromptGraphParseError(
                f"Expected {expected_kind!r}, got {token.kind!r}"
            )

        self.index += 1
        return token


def parse_prompt_graph(text: str) -> PromptGraph:
    return PromptGraphParser(text).parse()