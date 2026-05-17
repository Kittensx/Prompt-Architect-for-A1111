# prompt_execution_plan.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# ============================================================================
# EXECUTION ENUMS
# ============================================================================


class ExecutionStage(str, Enum):
    """
    High-level execution stage.

    Useful for:
    - scheduling
    - debugging
    - caching
    - progress display
    """

    TOKENIZE = "tokenize"
    CONDITION = "condition"
    MERGE = "merge"
    POST_PROCESS = "post_process"


class PromptCallKind(str, Enum):
    PLAIN = "plain"
    BACKEND = "backend"


class MergeMode(str, Enum):
    BLEND = "blend"
    CHUNK = "chunk"
    MORPH = "morph"
    POOL = "pool"
    BIND = "bind"
    ASSEMBLE = "assemble"
    AND = "and"
    SEQUENCE_CONCAT = "sequence_concat"
    WEIGHT = "weight"


class TensorChannel(str, Enum):
    BOTH = "both"
    CROSS = "cross"
    POOLED = "pooled"
    ENC1 = "enc1"
    ENC2 = "enc2"


# ============================================================================
# SOURCE REFERENCES
# ============================================================================


@dataclass(slots=True)
class PlanReference:
    """
    Reference to another call/op output.
    """

    source_id: str
    weight: float = 1.0
    label: str | None = None
    enabled: bool = True


# ============================================================================
# CONDITIONING OUTPUTS
# ============================================================================


@dataclass(slots=True)
class ConditioningOutput:
    """
    Standardized conditioning payload.

    This lets your graph executor become backend-independent later.
    """

    source_id: str

    cross_attention: Any | None = None
    pooled_output: Any | None = None

    extra_channels: dict[str, Any] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# EXECUTION RESULTS
# ============================================================================


@dataclass(slots=True)
class ExecutionResult:
    """
    Runtime result from one call/op execution.
    """

    result_id: str

    conditioning: ConditioningOutput | None = None

    success: bool = True
    error: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# PROMPT CALLS
# ============================================================================


@dataclass(slots=True)
class PromptCall:
    """
    One primitive conditioning call.

    Usually:
    - direct _21 call
    - plain A1111 conditioning
    - backend block conditioning
    """

    call_id: str

    prompt: str

    kind: PromptCallKind = PromptCallKind.PLAIN

    stage: ExecutionStage = ExecutionStage.CONDITION

    negative_prompt: bool = False

    steps: int | None = None
    width: int | None = None
    height: int | None = None

    backend_name: str = "_21"

    cacheable: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# GRAPH OPS
# ============================================================================


@dataclass(slots=True)
class PlanOperation:
    """
    Graph-level operation.

    Examples:
    - blend tensors
    - concatenate chunks
    - morph interpolation
    - pooled replacement
    """

    op_id: str

    mode: MergeMode

    inputs: list[PlanReference] = field(default_factory=list)

    stage: ExecutionStage = ExecutionStage.MERGE

    params: dict[str, Any] = field(default_factory=dict)

    cacheable: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# EXECUTION WINDOWS
# ============================================================================


@dataclass(slots=True)
class ExecutionBoundary:
    """
    Runtime-resolved scheduling boundary.
    """

    value: float

    kind: Literal[
        "step",
        "fraction",
        "percent",
    ] = "step"

    def resolve(self, total_steps: int) -> int:
        if self.kind == "step":
            return int(self.value)

        if self.kind == "fraction":
            return int(round(total_steps * self.value))

        if self.kind == "percent":
            return int(round(total_steps * (self.value / 100.0)))

        raise ValueError(f"Unknown boundary kind: {self.kind!r}")


@dataclass(slots=True)
class ExecutionWindow:
    """
    Optional execution window for scheduled operations.
    """

    start: ExecutionBoundary | None = None
    end: ExecutionBoundary | None = None

    enabled: bool = True


# ============================================================================
# EXECUTION NODES
# ============================================================================


@dataclass(slots=True)
class ExecutionNode:
    """
    Unified node wrapper.

    Allows executor pipelines to treat calls and ops uniformly.
    """

    node_id: str

    kind: Literal["call", "op"]

    payload: PromptCall | PlanOperation

    dependencies: list[str] = field(default_factory=list)

    execution_window: ExecutionWindow | None = None

    completed: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# EXECUTION GRAPH
# ============================================================================


@dataclass(slots=True)
class PromptExecutionPlan:
    """
    Final lowered execution graph.

    Produced by:
        prompt_graph_lowering.py

    Consumed by:
        prompt_graph_executor.py
    """

    calls: list[PromptCall] = field(default_factory=list)

    operations: list[PlanOperation] = field(default_factory=list)

    execution_nodes: list[ExecutionNode] = field(default_factory=list)

    output_id: str = ""

    diagnostics: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    version: str = "1"

    backend_name: str = "_21"

    def all_ids(self) -> set[str]:
        ids = set()

        for call in self.calls:
            ids.add(call.call_id)

        for op in self.operations:
            ids.add(op.op_id)

        return ids

    def validate(self) -> list[str]:
        errors = []

        known_ids = self.all_ids()

        if self.output_id and self.output_id not in known_ids:
            errors.append(
                f"Unknown output_id: {self.output_id!r}"
            )

        for op in self.operations:
            for ref in op.inputs:
                if ref.source_id not in known_ids:
                    errors.append(
                        f"Operation {op.op_id!r} references unknown "
                        f"source_id {ref.source_id!r}"
                    )

        return errors

    def get_call(self, call_id: str) -> PromptCall | None:
        for call in self.calls:
            if call.call_id == call_id:
                return call
        return None

    def get_operation(self, op_id: str) -> PlanOperation | None:
        for op in self.operations:
            if op.op_id == op_id:
                return op
        return None

    def build_execution_nodes(self) -> None:
        """
        Converts flat call/op lists into dependency-aware execution nodes.
        """

        nodes: list[ExecutionNode] = []

        for call in self.calls:
            nodes.append(
                ExecutionNode(
                    node_id=call.call_id,
                    kind="call",
                    payload=call,
                    dependencies=[],
                )
            )

        for op in self.operations:
            dependencies = [
                ref.source_id
                for ref in op.inputs
            ]

            nodes.append(
                ExecutionNode(
                    node_id=op.op_id,
                    kind="op",
                    payload=op,
                    dependencies=dependencies,
                )
            )

        self.execution_nodes = nodes


# ============================================================================
# EXECUTION CACHE
# ============================================================================


@dataclass(slots=True)
class ExecutionCacheEntry:
    """
    Cached conditioning result.
    """

    cache_key: str

    source_id: str

    result: ExecutionResult

    hits: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# EXECUTION TRACE
# ============================================================================


@dataclass(slots=True)
class ExecutionTraceEvent:
    """
    Useful for debugging graph execution.
    """

    event_id: str

    source_id: str

    stage: ExecutionStage

    message: str

    success: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionTrace:
    """
    Full runtime trace.
    """

    events: list[ExecutionTraceEvent] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def add_event(
        self,
        source_id: str,
        stage: ExecutionStage,
        message: str,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            ExecutionTraceEvent(
                event_id=f"evt_{len(self.events)+1:05d}",
                source_id=source_id,
                stage=stage,
                message=message,
                success=success,
                metadata=metadata or {},
            )
        )

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        self.errors.append(message)