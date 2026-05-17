# prompt_graph.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal
from uuid import uuid4


class PromptNodeKind(str, Enum):
    TEXT = "text"
    SEQUENCE = "sequence"
    GROUP = "group"
    ALTERNATE = "alternate"
    SCHEDULE = "schedule"
    WEIGHT = "weight"
    BLEND = "blend"
    CHUNK = "chunk"
    MORPH = "morph"
    POOL = "pool"
    BIND = "bind"
    ASSEMBLE = "assemble"
    AND = "and"


@dataclass(slots=True)
class SourceSpan:
    start: int | None = None
    end: int | None = None
    source: str | None = None


@dataclass(slots=True)
class GraphMeta:
    node_id: str = field(default_factory=lambda: uuid4().hex)
    span: SourceSpan | None = None
    raw: str | None = None
    notes: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptNode:
    kind: PromptNodeKind
    meta: GraphMeta = field(default_factory=GraphMeta)

    def children(self) -> list[PromptNode]:
        return []

    def clone_meta(self) -> GraphMeta:
        return GraphMeta(
            span=self.meta.span,
            raw=self.meta.raw,
            notes=list(self.meta.notes),
            attrs=dict(self.meta.attrs),
        )


@dataclass(slots=True, init=False)
class TextNode(PromptNode):
    text: str = ""

    def __init__(self, text: str, meta: GraphMeta | None = None):
        PromptNode.__init__(self, PromptNodeKind.TEXT, meta or GraphMeta())
        self.text = str(text)


@dataclass(slots=True, init=False)
class SequenceNode(PromptNode):
    parts: list[PromptNode] = field(default_factory=list)
    separator: str = " "

    def __init__(self, parts: list[PromptNode] | None = None, separator: str = " ", meta: GraphMeta | None = None):
        PromptNode.__init__(self, PromptNodeKind.SEQUENCE, meta or GraphMeta())
        self.parts = parts or []
        self.separator = separator

    def children(self) -> list[PromptNode]:
        return list(self.parts)


@dataclass(slots=True, init=False)
class GroupNode(PromptNode):
    parts: list[PromptNode] = field(default_factory=list)
    delimiter: Literal["brace", "paren", "bracket"] = "brace"

    def __init__(
        self,
        parts: list[PromptNode] | None = None,
        delimiter: Literal["brace", "paren", "bracket"] = "brace",
        meta: GraphMeta | None = None,
    ):
        PromptNode.__init__(self, PromptNodeKind.GROUP, meta or GraphMeta())
        self.parts = parts or []
        self.delimiter = delimiter

    def children(self) -> list[PromptNode]:
        return list(self.parts)


@dataclass(slots=True, init=False)
class WeightNode(PromptNode):
    node: PromptNode
    weight: float = 1.0
    mode: Literal["attention", "branch", "conditioning"] = "attention"

    def __init__(
        self,
        node: PromptNode,
        weight: float = 1.0,
        mode: Literal["attention", "branch", "conditioning"] = "attention",
        meta: GraphMeta | None = None,
    ):
        PromptNode.__init__(self, PromptNodeKind.WEIGHT, meta or GraphMeta())
        self.node = node
        self.weight = float(weight)
        self.mode = mode

    def children(self) -> list[PromptNode]:
        return [self.node]


@dataclass(slots=True, init=False)
class AlternateNode(PromptNode):
    options: list[PromptNode] = field(default_factory=list)
    mode: Literal["cycle", "random", "seeded"] = "cycle"

    def __init__(
        self,
        options: list[PromptNode] | None = None,
        mode: Literal["cycle", "random", "seeded"] = "cycle",
        meta: GraphMeta | None = None,
    ):
        PromptNode.__init__(self, PromptNodeKind.ALTERNATE, meta or GraphMeta())
        self.options = options or []
        self.mode = mode

    def children(self) -> list[PromptNode]:
        return list(self.options)


@dataclass(slots=True)
class ScheduleBoundary:
    value: float
    kind: Literal["step", "fraction", "percent"] = "step"

    def resolve(self, total_steps: int) -> int:
        if self.kind == "step":
            return int(self.value)
        if self.kind == "fraction":
            return int(round(float(self.value) * total_steps))
        if self.kind == "percent":
            return int(round((float(self.value) / 100.0) * total_steps))
        raise ValueError(f"Unknown boundary kind: {self.kind!r}")


@dataclass(slots=True)
class ScheduleSegment:
    node: PromptNode
    start: ScheduleBoundary | None = None
    end: ScheduleBoundary | None = None
    weight: float = 1.0


@dataclass(slots=True, init=False)
class ScheduleNode(PromptNode):
    segments: list[ScheduleSegment] = field(default_factory=list)
    reverse: bool = False

    def __init__(self, segments: list[ScheduleSegment] | None = None, reverse: bool = False, meta: GraphMeta | None = None):
        PromptNode.__init__(self, PromptNodeKind.SCHEDULE, meta or GraphMeta())
        self.segments = segments or []
        self.reverse = bool(reverse)

    def children(self) -> list[PromptNode]:
        return [segment.node for segment in self.segments]


@dataclass(slots=True)
class AndBranch:
    node: PromptNode
    weight: float = 1.0


@dataclass(slots=True, init=False)
class AndNode(PromptNode):
    branches: list[AndBranch] = field(default_factory=list)

    def __init__(self, branches: list[AndBranch] | None = None, meta: GraphMeta | None = None):
        PromptNode.__init__(self, PromptNodeKind.AND, meta or GraphMeta())
        self.branches = branches or []

    def children(self) -> list[PromptNode]:
        return [branch.node for branch in self.branches]


@dataclass(slots=True)
class BackendBranch:
    node: PromptNode
    weight: float = 1.0
    label: str | None = None


@dataclass(slots=True, init=False)
class BackendNode(PromptNode):
    mode: str | None = None
    channel_target: Literal["both", "cross", "pooled", "enc1", "enc2"] = "both"
    intensity: float = 1.0

    def __init__(
        self,
        kind: PromptNodeKind,
        meta: GraphMeta | None = None,
        *,
        mode: str | None = None,
        channel_target: Literal["both", "cross", "pooled", "enc1", "enc2"] = "both",
        intensity: float = 1.0,
    ):
        PromptNode.__init__(self, kind, meta or GraphMeta())
        self.mode = mode
        self.channel_target = channel_target
        self.intensity = float(intensity)

    def backend_children(self) -> list[PromptNode]:
        return []


@dataclass(slots=True, init=False)
class BlendNode(BackendNode):
    branches: list[BackendBranch] = field(default_factory=list)
    blend_mode: Literal["mean", "sum"] = "mean"

    def __init__(
        self,
        branches: list[BackendBranch] | None = None,
        blend_mode: Literal["mean", "sum"] = "mean",
        channel_target: Literal["both", "cross", "pooled", "enc1", "enc2"] = "both",
        intensity: float = 1.0,
        mode: str | None = None,
        meta: GraphMeta | None = None,
    ):
        BackendNode.__init__(self, PromptNodeKind.BLEND, meta, mode=mode, channel_target=channel_target, intensity=intensity)
        self.branches = branches or []
        self.blend_mode = blend_mode

    def children(self) -> list[PromptNode]:
        return [branch.node for branch in self.branches]


@dataclass(slots=True, init=False)
class ChunkNode(BackendNode):
    branches: list[BackendBranch] = field(default_factory=list)
    shared_channel: Literal["none", "cross", "pooled"] = "none"

    def __init__(
        self,
        branches: list[BackendBranch] | None = None,
        shared_channel: Literal["none", "cross", "pooled"] = "none",
        mode: str | None = None,
        meta: GraphMeta | None = None,
    ):
        BackendNode.__init__(self, PromptNodeKind.CHUNK, meta, mode=mode)
        self.branches = branches or []
        self.shared_channel = shared_channel

    def children(self) -> list[PromptNode]:
        return [branch.node for branch in self.branches]


@dataclass(slots=True)
class MorphPoint:
    node: PromptNode
    boundary: ScheduleBoundary | None = None
    weight: float = 1.0


@dataclass(slots=True, init=False)
class MorphNode(BackendNode):
    points: list[MorphPoint] = field(default_factory=list)
    curve: Literal["linear", "bezier", "catmull"] = "linear"
    window_start: ScheduleBoundary | None = None
    window_end: ScheduleBoundary | None = None

    def __init__(
        self,
        points: list[MorphPoint] | None = None,
        curve: Literal["linear", "bezier", "catmull"] = "linear",
        channel_target: Literal["both", "cross", "pooled", "enc1", "enc2"] = "both",
        intensity: float = 1.0,
        mode: str | None = None,
        window_start: ScheduleBoundary | None = None,
        window_end: ScheduleBoundary | None = None,
        meta: GraphMeta | None = None,
    ):
        BackendNode.__init__(self, PromptNodeKind.MORPH, meta, mode=mode, channel_target=channel_target, intensity=intensity)
        self.points = points or []
        self.curve = curve
        self.window_start = window_start
        self.window_end = window_end

    def children(self) -> list[PromptNode]:
        return [point.node for point in self.points]


@dataclass(slots=True, init=False)
class PoolNode(BackendNode):
    node: PromptNode

    def __init__(self, node: PromptNode, mode: str | None = None, meta: GraphMeta | None = None):
        BackendNode.__init__(self, PromptNodeKind.POOL, meta, mode=mode)
        self.node = node

    def children(self) -> list[PromptNode]:
        return [self.node]


@dataclass(slots=True, init=False)
class BindNode(BackendNode):
    owner: PromptNode
    attrs: PromptNode
    weight: float = 1.0

    def __init__(self, owner: PromptNode, attrs: PromptNode, weight: float = 1.0, mode: str | None = None, meta: GraphMeta | None = None):
        BackendNode.__init__(self, PromptNodeKind.BIND, meta, mode=mode)
        self.owner = owner
        self.attrs = attrs
        self.weight = float(weight)

    def children(self) -> list[PromptNode]:
        return [self.owner, self.attrs]


@dataclass(slots=True, init=False)
class AssembleNode(BackendNode):
    enc1: PromptNode
    enc2: PromptNode
    pooled: PromptNode | None = None

    def __init__(self, enc1: PromptNode, enc2: PromptNode, pooled: PromptNode | None = None, mode: str | None = None, meta: GraphMeta | None = None):
        BackendNode.__init__(self, PromptNodeKind.ASSEMBLE, meta, mode=mode)
        self.enc1 = enc1
        self.enc2 = enc2
        self.pooled = pooled

    def children(self) -> list[PromptNode]:
        nodes = [self.enc1, self.enc2]
        if self.pooled is not None:
            nodes.append(self.pooled)
        return nodes


@dataclass(slots=True)
class PromptGraph:
    root: PromptNode
    version: str = "1"
    symbols: dict[str, str] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)

    def walk(self) -> list[PromptNode]:
        out: list[PromptNode] = []

        def visit(node: PromptNode) -> None:
            out.append(node)
            for child in node.children():
                visit(child)

        visit(self.root)
        return out

    def find(self, kind: PromptNodeKind) -> list[PromptNode]:
        return [node for node in self.walk() if node.kind == kind]

    def has_backend_nodes(self) -> bool:
        return any(
            node.kind in {
                PromptNodeKind.BLEND,
                PromptNodeKind.CHUNK,
                PromptNodeKind.MORPH,
                PromptNodeKind.POOL,
                PromptNodeKind.BIND,
                PromptNodeKind.ASSEMBLE,
            }
            for node in self.walk()
        )
