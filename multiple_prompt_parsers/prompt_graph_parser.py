# prompt_graph_parser.py
from __future__ import annotations

from dataclasses import dataclass

try:
    from modules.prompt_graph import (
        AlternateNode,
        AndBranch,
        AndNode,
        BackendBranch,
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
        BackendBranch,
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


# ============================================================================
# ERRORS
# ============================================================================


class PromptGraphParseError(Exception):
    pass


# ============================================================================
# TOKEN
# ============================================================================


@dataclass(slots=True)
class Token:
    kind: str
    value: str
    start: int
    end: int


# ============================================================================
# TOKENIZER
# ============================================================================


class PromptTokenizer:
    """
    Lightweight tokenizer.

    Important:
    We do NOT fully interpret semantics here.
    We only create structural tokens.
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
    }

    DOUBLE_SYMBOLS = {
        "=>",
    }

    KEYWORDS = {
        "BLEND",
        "CHUNK",
        "MORPH",
        "POOL",
        "AND",
    }

    def tokenize(self, text: str) -> list[Token]:

        tokens: list[Token] = []

        i = 0
        length = len(text)

        while i < length:

            ch = text[i]

            # -------------------------------------------------------------
            # whitespace
            # -------------------------------------------------------------

            if ch.isspace():
                i += 1
                continue

            # -------------------------------------------------------------
            # escaped chars
            # -------------------------------------------------------------

            if ch == "\\":

                if i + 1 >= length:
                    raise PromptGraphParseError(
                        "Trailing escape character"
                    )

                start = i

                escaped = text[i + 1]

                tokens.append(
                    Token(
                        kind="TEXT",
                        value=escaped,
                        start=start,
                        end=i + 2,
                    )
                )

                i += 2
                continue

            # -------------------------------------------------------------
            # double symbols
            # -------------------------------------------------------------

            matched_double = False

            for symbol in self.DOUBLE_SYMBOLS:

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

            # -------------------------------------------------------------
            # single symbols
            # -------------------------------------------------------------

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

            # -------------------------------------------------------------
            # words
            # -------------------------------------------------------------

            start = i

            while i < length:

                current = text[i]

                if current.isspace():
                    break

                if current in self.SYMBOLS:
                    break

                if any(
                    text.startswith(sym, i)
                    for sym in self.DOUBLE_SYMBOLS
                ):
                    break

                i += 1

            value = text[start:i]

            kind = (
                value
                if value in self.KEYWORDS
                else "TEXT"
            )

            tokens.append(
                Token(
                    kind=kind,
                    value=value,
                    start=start,
                    end=i,
                )
            )

        return tokens


# ============================================================================
# PARSER
# ============================================================================


class PromptGraphParser:

    def __init__(self, text: str):

        self.text = text

        tokenizer = PromptTokenizer()

        self.tokens = tokenizer.tokenize(text)

        self.index = 0

    # ---------------------------------------------------------------------
    # PUBLIC
    # ---------------------------------------------------------------------

    def parse(self) -> PromptGraph:

        root = self.parse_sequence(
            stop_tokens=None
        )

        if not self.at_end():
            token = self.peek()

            raise PromptGraphParseError(
                f"Unexpected token: {token.value!r}"
            )

        return PromptGraph(
            root=root,
        )

    # ---------------------------------------------------------------------
    # CORE
    # ---------------------------------------------------------------------

    def parse_sequence(
        self,
        stop_tokens: set[str] | None,
    ) -> PromptNode:

        parts: list[PromptNode] = []

        while not self.at_end():

            token = self.peek()

            if stop_tokens and token.kind in stop_tokens:
                break

            node = self.parse_expression()

            parts.append(node)

        if not parts:
            return TextNode("")

        if len(parts) == 1:
            return parts[0]

        return SequenceNode(parts=parts)

    def parse_expression(self) -> PromptNode:

        token = self.peek()

        # -------------------------------------------------------------
        # backend nodes
        # -------------------------------------------------------------

        if token.kind == "BLEND":
            return self.parse_blend()

        if token.kind == "CHUNK":
            return self.parse_chunk()

        if token.kind == "MORPH":
            return self.parse_morph()

        if token.kind == "POOL":
            return self.parse_pool()

        # -------------------------------------------------------------
        # groups
        # -------------------------------------------------------------

        if token.kind == "{":
            return self.parse_group("brace")

        if token.kind == "(":
            return self.parse_weight_or_group()

        if token.kind == "[":
            return self.parse_alternate_or_schedule()

        # -------------------------------------------------------------
        # AND
        # -------------------------------------------------------------

        return self.parse_text_or_and()

    # =========================================================================
    # TEXT
    # =========================================================================

    def parse_text_or_and(self) -> PromptNode:

        first = self.parse_text()

        branches = [
            AndBranch(node=first)
        ]

        while not self.at_end():

            token = self.peek()

            if token.kind != "AND":
                break

            self.consume("AND")

            branch = self.parse_expression()

            branches.append(
                AndBranch(node=branch)
            )

        if len(branches) == 1:
            return first

        return AndNode(
            branches=branches
        )

    def parse_text(self) -> PromptNode:

        token = self.consume("TEXT")

        return TextNode(
            text=token.value
        )

    # =========================================================================
    # GROUPS
    # =========================================================================

    def parse_group(
        self,
        delimiter: str,
    ) -> PromptNode:

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
            raise RuntimeError(
                f"Unknown delimiter: {delimiter!r}"
            )

        self.consume(open_tok)

        node = self.parse_sequence(
            stop_tokens={close_tok}
        )

        self.consume(close_tok)

        if isinstance(node, SequenceNode):
            return GroupNode(
                parts=node.parts,
                delimiter=delimiter,
            )

        return GroupNode(
            parts=[node],
            delimiter=delimiter,
        )

    # =========================================================================
    # WEIGHT
    # =========================================================================

    def parse_weight_or_group(self) -> PromptNode:

        start = self.index

        self.consume("(")

        inner = self.parse_sequence(
            stop_tokens={":", ")"}
        )

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

    # =========================================================================
    # ALTERNATE / SCHEDULE
    # =========================================================================

    def parse_alternate_or_schedule(self) -> PromptNode:

        self.consume("[")

        items: list[PromptNode] = []

        separators: list[str] = []

        while not self.at_end():

            if self.peek_kind() == "]":
                break

            node = self.parse_sequence(
                stop_tokens={"|", ":", "]"}
            )

            items.append(node)

            if self.peek_kind() in {"|", ":"}:
                separators.append(
                    self.consume().kind
                )

            else:
                break

        self.consume("]")

        # -------------------------------------------------------------
        # alternate
        # -------------------------------------------------------------

        if "|" in separators:

            return AlternateNode(
                options=items
            )

        # -------------------------------------------------------------
        # schedule
        # -------------------------------------------------------------

        if ":" in separators:

            segments = []

            for item in items:
                segments.append(
                    ScheduleSegment(
                        node=item,
                    )
                )

            return ScheduleNode(
                segments=segments
            )

        return GroupNode(
            parts=items,
            delimiter="bracket",
        )

    # =========================================================================
    # BLEND
    # =========================================================================

    def parse_blend(self) -> PromptNode:

        self.consume("BLEND")

        self.consume("{")

        branches: list[BackendBranch] = []

        while not self.at_end():

            branch = self.parse_sequence(
                stop_tokens={"|", "}"}
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

        self.consume("}")

        return BlendNode(
            branches=branches
        )

    # =========================================================================
    # CHUNK
    # =========================================================================

    def parse_chunk(self) -> PromptNode:

        self.consume("CHUNK")

        self.consume("{")

        branches: list[BackendBranch] = []

        while not self.at_end():

            branch = self.parse_sequence(
                stop_tokens={"|", "}"}
            )

            branches.append(
                BackendBranch(
                    node=branch,
                )
            )

            if self.peek_kind() == "|":
                self.consume("|")
                continue

            break

        self.consume("}")

        return ChunkNode(
            branches=branches
        )

    # =========================================================================
    # MORPH
    # =========================================================================

    def parse_morph(self) -> PromptNode:

        self.consume("MORPH")

        self.consume("{")

        points: list[MorphPoint] = []

        while not self.at_end():

            node = self.parse_sequence(
                stop_tokens={"@", "=>", "}"}
            )

            boundary = None

            if self.peek_kind() == "@":

                self.consume("@")

                boundary_value = self.parse_number()

                boundary = ScheduleBoundary(
                    value=boundary_value,
                    kind="fraction",
                )

            points.append(
                MorphPoint(
                    node=node,
                    boundary=boundary,
                )
            )

            if self.peek_kind() == "=>":
                self.consume("=>")
                continue

            break

        self.consume("}")

        return MorphNode(
            points=points
        )

    # =========================================================================
    # POOL
    # =========================================================================

    def parse_pool(self) -> PromptNode:

        self.consume("POOL")

        self.consume("{")

        node = self.parse_sequence(
            stop_tokens={"}"}
        )

        self.consume("}")

        return PoolNode(
            node=node
        )

    # =========================================================================
    # NUMBERS
    # =========================================================================

    def parse_number(self) -> float:

        token = self.consume("TEXT")

        try:
            return float(token.value)

        except ValueError:

            raise PromptGraphParseError(
                f"Expected number, got {token.value!r}"
            )

    # =========================================================================
    # TOKEN HELPERS
    # =========================================================================

    def at_end(self) -> bool:
        return self.index >= len(self.tokens)

    def peek(self) -> Token:

        if self.at_end():
            raise PromptGraphParseError(
                "Unexpected end of input"
            )

        return self.tokens[self.index]

    def peek_kind(self) -> str | None:

        if self.at_end():
            return None

        return self.tokens[self.index].kind

    def consume(
        self,
        expected_kind: str | None = None,
    ) -> Token:

        token = self.peek()

        if (
            expected_kind is not None
            and token.kind != expected_kind
        ):
            raise PromptGraphParseError(
                f"Expected {expected_kind!r}, "
                f"got {token.kind!r}"
            )

        self.index += 1

        return token


# ============================================================================
# CONVENIENCE
# =========================================================================


def parse_prompt_graph(
    text: str,
) -> PromptGraph:

    parser = PromptGraphParser(text)

    return parser.parse()