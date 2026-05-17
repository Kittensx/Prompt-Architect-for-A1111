# prompt_graph_validator.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite

try:
    from modules.prompt_graph import (
        AlternateNode,
        AndNode,
        AssembleNode,
        BackendBranch,
        BindNode,
        BlendNode,
        ChunkNode,
        GroupNode,
        MorphNode,
        PoolNode,
        PromptGraph,
        PromptNode,
        PromptNodeKind,
        ScheduleNode,
        SequenceNode,
        TextNode,
        WeightNode,
    )
except ImportError:
    from prompt_graph import (
        AlternateNode,
        AndNode,
        AssembleNode,
        BackendBranch,
        BindNode,
        BlendNode,
        ChunkNode,
        GroupNode,
        MorphNode,
        PoolNode,
        PromptGraph,
        PromptNode,
        PromptNodeKind,
        ScheduleNode,
        SequenceNode,
        TextNode,
        WeightNode,
    )


class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(slots=True)
class ValidationIssue:
    severity: ValidationSeverity
    code: str
    message: str
    node_kind: str | None = None
    node_id: str | None = None
    suggestion: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class ValidationResult:
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(
        self,
        severity: ValidationSeverity,
        code: str,
        message: str,
        *,
        node: PromptNode | None = None,
        suggestion: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                severity=severity,
                code=code,
                message=message,
                node_kind=node.kind.value if node else None,
                node_id=node.meta.node_id if node else None,
                suggestion=suggestion,
                metadata=metadata or {},
            )
        )

    @property
    def errors(self) -> list[ValidationIssue]:
        return [
            issue for issue in self.issues
            if issue.severity == ValidationSeverity.ERROR
        ]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [
            issue for issue in self.issues
            if issue.severity == ValidationSeverity.WARNING
        ]

    def has_errors(self) -> bool:
        return bool(self.errors)

    def raise_if_errors(self) -> None:
        if not self.has_errors():
            return

        raise PromptGraphValidationError(
            "\n".join(
                f"[{issue.code}] {issue.message}"
                for issue in self.errors
            )
        )


class PromptGraphValidationError(Exception):
    pass


class PromptGraphValidator:
    VALID_BLEND_MODES = {"mean", "sum"}
    VALID_CHANNEL_TARGETS = {"both", "cross", "pooled", "enc1", "enc2"}
    VALID_CHUNK_SHARED = {"none", "cross", "pooled"}
    VALID_MORPH_CURVES = {"linear", "bezier", "catmull"}
    VALID_ALTERNATE_MODES = {"cycle", "random", "seeded"}

    def __init__(
        self,
        *,
        backend_name: str = "_21",
        strict_backend_rules: bool = False,
        sdxl_mode: bool = False,
        allow_graph_schedule_backends: bool = False,
        allow_graph_alternate_backends: bool = False,
    ) -> None:
        self.backend_name = backend_name
        self.strict_backend_rules = strict_backend_rules
        self.sdxl_mode = sdxl_mode
        self.allow_graph_schedule_backends = allow_graph_schedule_backends
        self.allow_graph_alternate_backends = allow_graph_alternate_backends

    def validate(self, graph: PromptGraph) -> ValidationResult:
        result = ValidationResult()

        self._validate_node(
            graph.root,
            result,
            parent=None,
            backend_depth=0,
        )

        return result

    def _validate_node(
        self,
        node: PromptNode,
        result: ValidationResult,
        *,
        parent: PromptNode | None,
        backend_depth: int,
    ) -> None:
        self._validate_empty_text(node, result)
        self._validate_weight_node(node, result)

        is_backend = _is_backend_node(node)

        if is_backend:
            self._validate_backend_node(
                node,
                result,
                backend_depth=backend_depth,
                parent=parent,
            )
            backend_depth += 1

        if isinstance(node, BlendNode):
            self._validate_blend(node, result)
        elif isinstance(node, ChunkNode):
            self._validate_chunk(node, result)
        elif isinstance(node, MorphNode):
            self._validate_morph(node, result)
        elif isinstance(node, PoolNode):
            self._validate_pool(node, result)
        elif isinstance(node, BindNode):
            self._validate_bind(node, result)
        elif isinstance(node, AssembleNode):
            self._validate_assemble(node, result)
        elif isinstance(node, AlternateNode):
            self._validate_alternate(node, result)
        elif isinstance(node, ScheduleNode):
            self._validate_schedule(node, result)
        elif isinstance(node, AndNode):
            self._validate_and(node, result)

        for child in node.children():
            self._validate_node(
                child,
                result,
                parent=node,
                backend_depth=backend_depth,
            )

    def _validate_empty_text(
        self,
        node: PromptNode,
        result: ValidationResult,
    ) -> None:
        if not isinstance(node, TextNode):
            return

        if node.text.strip():
            return

        result.add(
            ValidationSeverity.WARNING,
            "empty_text",
            "Empty text node",
            node=node,
        )

    def _validate_weight_node(
        self,
        node: PromptNode,
        result: ValidationResult,
    ) -> None:
        if not isinstance(node, WeightNode):
            return

        if not _valid_number(node.weight):
            result.add(
                ValidationSeverity.ERROR,
                "invalid_weight",
                "Weight must be a finite number",
                node=node,
            )
            return

        if abs(node.weight) < 1e-12:
            result.add(
                ValidationSeverity.WARNING,
                "zero_weight",
                "Weight is effectively zero",
                node=node,
            )

        if node.weight < 0:
            result.add(
                ValidationSeverity.WARNING,
                "negative_weight",
                "Negative attention weight",
                node=node,
                suggestion="Ensure the active backend supports negative conditioning weights.",
            )

    def _validate_backend_node(
        self,
        node: PromptNode,
        result: ValidationResult,
        *,
        backend_depth: int,
        parent: PromptNode | None,
    ) -> None:
        if backend_depth <= 0:
            return

        if self.strict_backend_rules:
            result.add(
                ValidationSeverity.ERROR,
                "nested_backend",
                "Nested backend nodes are not allowed in strict backend mode",
                node=node,
                suggestion="Run graph normalization/lowering, or disable strict backend rules.",
            )
            return

        result.add(
            ValidationSeverity.INFO,
            "nested_backend_graph_lowering",
            "Nested backend node detected. Graph lowering should split this into multi-call execution.",
            node=node,
            metadata={
                "parent_kind": parent.kind.value if parent else None,
                "backend_depth": backend_depth,
            },
        )

    def _validate_blend(
        self,
        node: BlendNode,
        result: ValidationResult,
    ) -> None:
        if len(node.branches) < 2:
            result.add(
                ValidationSeverity.ERROR,
                "blend_min_branches",
                "BLEND requires at least 2 branches",
                node=node,
            )

        if node.blend_mode not in self.VALID_BLEND_MODES:
            result.add(
                ValidationSeverity.ERROR,
                "blend_invalid_mode",
                f"Unsupported BLEND mode: {node.blend_mode!r}",
                node=node,
            )

        if node.channel_target not in self.VALID_CHANNEL_TARGETS:
            result.add(
                ValidationSeverity.ERROR,
                "blend_invalid_channel",
                f"Unsupported BLEND channel target: {node.channel_target!r}",
                node=node,
            )

        if not _valid_positive_number(node.intensity):
            result.add(
                ValidationSeverity.ERROR,
                "blend_invalid_intensity",
                "BLEND intensity must be a positive finite number",
                node=node,
            )

        for branch in node.branches:
            self._validate_backend_branch(
                branch,
                result,
                backend_name="BLEND",
            )

    def _validate_chunk(
        self,
        node: ChunkNode,
        result: ValidationResult,
    ) -> None:
        if not node.branches:
            result.add(
                ValidationSeverity.ERROR,
                "chunk_empty",
                "CHUNK requires at least 1 branch",
                node=node,
            )
        elif len(node.branches) < 2:
            result.add(
                ValidationSeverity.WARNING,
                "chunk_single_branch",
                "CHUNK only contains one branch",
                node=node,
            )

        if node.shared_channel not in self.VALID_CHUNK_SHARED:
            result.add(
                ValidationSeverity.ERROR,
                "chunk_invalid_shared_channel",
                f"Unsupported CHUNK shared channel: {node.shared_channel!r}",
                node=node,
            )

        for branch in node.branches:
            self._validate_backend_branch(
                branch,
                result,
                backend_name="CHUNK",
            )

    def _validate_morph(
        self,
        node: MorphNode,
        result: ValidationResult,
    ) -> None:
        if len(node.points) < 2:
            result.add(
                ValidationSeverity.ERROR,
                "morph_min_points",
                "MORPH requires at least 2 points",
                node=node,
            )
            return

        if node.curve not in self.VALID_MORPH_CURVES:
            result.add(
                ValidationSeverity.ERROR,
                "morph_invalid_curve",
                f"Unsupported MORPH curve: {node.curve!r}",
                node=node,
            )

        if node.channel_target not in self.VALID_CHANNEL_TARGETS:
            result.add(
                ValidationSeverity.ERROR,
                "morph_invalid_channel",
                f"Unsupported MORPH channel target: {node.channel_target!r}",
                node=node,
            )

        if not _valid_positive_number(node.intensity):
            result.add(
                ValidationSeverity.ERROR,
                "morph_invalid_intensity",
                "MORPH intensity must be a positive finite number",
                node=node,
            )

        previous = None

        for index, point in enumerate(node.points):
            if not _valid_number(point.weight):
                result.add(
                    ValidationSeverity.ERROR,
                    "morph_invalid_point_weight",
                    "MORPH point weight must be a finite number",
                    node=point.node,
                )

            if point.weight < 0:
                result.add(
                    ValidationSeverity.WARNING,
                    "morph_negative_point_weight",
                    "MORPH point has negative weight",
                    node=point.node,
                )

            boundary = point.boundary

            if index == 0 and boundary is not None:
                result.add(
                    ValidationSeverity.WARNING,
                    "morph_first_boundary",
                    "First MORPH point usually should not define a boundary",
                    node=node,
                )

            if boundary is None:
                continue

            if not _valid_number(boundary.value):
                result.add(
                    ValidationSeverity.ERROR,
                    "morph_invalid_boundary",
                    "MORPH boundary must be finite",
                    node=node,
                )
                continue

            if previous is not None and boundary.value <= previous:
                result.add(
                    ValidationSeverity.ERROR,
                    "morph_boundary_order",
                    "MORPH boundaries must be strictly increasing",
                    node=node,
                )

            previous = boundary.value

        self._validate_window_pair(
            node.window_start,
            node.window_end,
            result,
            node=node,
            code_prefix="morph",
        )

    def _validate_pool(
        self,
        node: PoolNode,
        result: ValidationResult,
    ) -> None:
        if _is_empty_like(node.node):
            result.add(
                ValidationSeverity.ERROR,
                "pool_empty",
                "POOL body cannot be empty",
                node=node,
            )

        if _contains_backend(node.node):
            result.add(
                ValidationSeverity.INFO,
                "pool_nested_backend",
                "POOL contains a backend node and will require graph lowering",
                node=node,
            )

    def _validate_bind(
        self,
        node: BindNode,
        result: ValidationResult,
    ) -> None:
        if _is_empty_like(node.owner):
            result.add(
                ValidationSeverity.ERROR,
                "bind_empty_owner",
                "BIND owner cannot be empty",
                node=node,
            )

        if _is_empty_like(node.attrs):
            result.add(
                ValidationSeverity.ERROR,
                "bind_empty_attrs",
                "BIND attrs cannot be empty",
                node=node,
            )

        if not _valid_number(node.weight):
            result.add(
                ValidationSeverity.ERROR,
                "bind_invalid_weight",
                "BIND weight must be a finite number",
                node=node,
            )
            return

        if abs(node.weight) < 1e-12:
            result.add(
                ValidationSeverity.WARNING,
                "bind_zero_weight",
                "BIND weight is zero",
                node=node,
            )

        if node.weight < 0:
            result.add(
                ValidationSeverity.WARNING,
                "bind_negative_weight",
                "BIND weight is negative",
                node=node,
            )

    def _validate_assemble(
        self,
        node: AssembleNode,
        result: ValidationResult,
    ) -> None:
        if not self.sdxl_mode:
            result.add(
                ValidationSeverity.WARNING,
                "assemble_non_sdxl",
                "ASSEMBLE is typically intended for SDXL dual encoders",
                node=node,
            )

        if _is_empty_like(node.enc1):
            result.add(
                ValidationSeverity.ERROR,
                "assemble_empty_enc1",
                "ASSEMBLE enc1 cannot be empty",
                node=node,
            )

        if _is_empty_like(node.enc2):
            result.add(
                ValidationSeverity.ERROR,
                "assemble_empty_enc2",
                "ASSEMBLE enc2 cannot be empty",
                node=node,
            )

        if node.pooled is not None and _is_empty_like(node.pooled):
            result.add(
                ValidationSeverity.WARNING,
                "assemble_empty_pooled",
                "ASSEMBLE pooled field is empty and will be ignored",
                node=node,
            )

    def _validate_alternate(
        self,
        node: AlternateNode,
        result: ValidationResult,
    ) -> None:
        if not node.options:
            result.add(
                ValidationSeverity.ERROR,
                "alternate_empty",
                "Alternate node has no options",
                node=node,
            )
            return

        if len(node.options) < 2:
            result.add(
                ValidationSeverity.WARNING,
                "alternate_single_option",
                "Alternate node only contains one option",
                node=node,
            )

        if node.mode not in self.VALID_ALTERNATE_MODES:
            result.add(
                ValidationSeverity.ERROR,
                "alternate_invalid_mode",
                f"Unsupported alternate mode: {node.mode!r}",
                node=node,
            )

        if _contains_backend(node) and not self.allow_graph_alternate_backends:
            result.add(
                ValidationSeverity.WARNING,
                "alternate_contains_backend",
                "Alternate contains backend nodes; lowering must support MergeMode.ALTERNATE or resolve the alternate earlier.",
                node=node,
            )

    def _validate_schedule(
        self,
        node: ScheduleNode,
        result: ValidationResult,
    ) -> None:
        if not node.segments:
            result.add(
                ValidationSeverity.ERROR,
                "empty_schedule",
                "Schedule node has no segments",
                node=node,
            )
            return

        previous_end = None

        for index, segment in enumerate(node.segments):
            if _is_empty_like(segment.node):
                result.add(
                    ValidationSeverity.WARNING,
                    "schedule_empty_segment",
                    "Schedule segment is empty",
                    node=segment.node,
                    metadata={"segment_index": index},
                )

            if not _valid_number(segment.weight):
                result.add(
                    ValidationSeverity.ERROR,
                    "schedule_invalid_weight",
                    "Schedule segment weight must be finite",
                    node=segment.node,
                    metadata={"segment_index": index},
                )

            if segment.weight < 0:
                result.add(
                    ValidationSeverity.WARNING,
                    "schedule_negative_weight",
                    "Schedule segment weight is negative",
                    node=segment.node,
                    metadata={"segment_index": index},
                )

            if segment.start is not None and not _valid_number(segment.start.value):
                result.add(
                    ValidationSeverity.ERROR,
                    "schedule_invalid_start",
                    "Schedule segment start boundary must be finite",
                    node=segment.node,
                    metadata={"segment_index": index},
                )

            if segment.end is not None and not _valid_number(segment.end.value):
                result.add(
                    ValidationSeverity.ERROR,
                    "schedule_invalid_end",
                    "Schedule segment end boundary must be finite",
                    node=segment.node,
                    metadata={"segment_index": index},
                )

            if (
                segment.start is not None
                and segment.end is not None
                and segment.end.value <= segment.start.value
            ):
                result.add(
                    ValidationSeverity.ERROR,
                    "schedule_invalid_range",
                    "Schedule segment end must be greater than start",
                    node=segment.node,
                    metadata={"segment_index": index},
                )

            if previous_end is not None and segment.start is not None:
                if segment.start.value < previous_end:
                    result.add(
                        ValidationSeverity.WARNING,
                        "schedule_overlap",
                        "Schedule segment starts before previous segment ended",
                        node=segment.node,
                        metadata={"segment_index": index},
                    )

            if segment.end is not None:
                previous_end = segment.end.value

        if _contains_backend(node) and not self.allow_graph_schedule_backends:
            result.add(
                ValidationSeverity.WARNING,
                "schedule_contains_backend",
                "Schedule contains backend nodes; lowering must support MergeMode.SCHEDULE or resolve the schedule earlier.",
                node=node,
            )

    def _validate_and(
        self,
        node: AndNode,
        result: ValidationResult,
    ) -> None:
        if not node.branches:
            result.add(
                ValidationSeverity.ERROR,
                "and_empty",
                "AND node has no branches",
                node=node,
            )
            return

        if len(node.branches) < 2:
            result.add(
                ValidationSeverity.WARNING,
                "and_single_branch",
                "AND node only contains one branch",
                node=node,
            )

        for branch in node.branches:
            if not _valid_number(branch.weight):
                result.add(
                    ValidationSeverity.ERROR,
                    "and_invalid_branch_weight",
                    "AND branch weight must be finite",
                    node=branch.node,
                )
            elif abs(branch.weight) < 1e-12:
                result.add(
                    ValidationSeverity.WARNING,
                    "and_zero_branch_weight",
                    "AND branch weight is effectively zero",
                    node=branch.node,
                )

    def _validate_backend_branch(
        self,
        branch: BackendBranch,
        result: ValidationResult,
        *,
        backend_name: str,
    ) -> None:
        if _is_empty_like(branch.node):
            result.add(
                ValidationSeverity.WARNING,
                "branch_empty",
                f"{backend_name} branch is empty",
                node=branch.node,
            )

        if not _valid_number(branch.weight):
            result.add(
                ValidationSeverity.ERROR,
                "branch_invalid_weight",
                f"{backend_name} branch weight must be finite",
                node=branch.node,
            )
            return

        if abs(branch.weight) < 1e-12:
            result.add(
                ValidationSeverity.WARNING,
                "branch_zero_weight",
                f"{backend_name} branch weight is effectively zero",
                node=branch.node,
            )

        if branch.weight < 0:
            result.add(
                ValidationSeverity.WARNING,
                "branch_negative_weight",
                f"{backend_name} branch has negative weight",
                node=branch.node,
            )

    def _validate_window_pair(
        self,
        start,
        end,
        result: ValidationResult,
        *,
        node: PromptNode,
        code_prefix: str,
    ) -> None:
        if start is None and end is None:
            return

        if start is None or end is None:
            result.add(
                ValidationSeverity.ERROR,
                f"{code_prefix}_incomplete_window",
                "Window requires both start and end boundaries",
                node=node,
            )
            return

        if not _valid_number(start.value) or not _valid_number(end.value):
            result.add(
                ValidationSeverity.ERROR,
                f"{code_prefix}_invalid_window",
                "Window boundaries must be finite",
                node=node,
            )
            return

        if end.value <= start.value:
            result.add(
                ValidationSeverity.ERROR,
                f"{code_prefix}_invalid_window_order",
                "Window end must be greater than window start",
                node=node,
            )


def _is_backend_node(node: PromptNode) -> bool:
    return node.kind in {
        PromptNodeKind.BLEND,
        PromptNodeKind.CHUNK,
        PromptNodeKind.MORPH,
        PromptNodeKind.POOL,
        PromptNodeKind.BIND,
        PromptNodeKind.ASSEMBLE,
    }


def _contains_backend(node: PromptNode) -> bool:
    if _is_backend_node(node):
        return True

    return any(_contains_backend(child) for child in node.children())


def _is_empty_like(node: PromptNode) -> bool:
    if isinstance(node, TextNode):
        return not node.text.strip()

    if isinstance(node, SequenceNode):
        return not node.parts

    if isinstance(node, GroupNode):
        return not node.parts

    if isinstance(node, AlternateNode):
        return not node.options

    if isinstance(node, ScheduleNode):
        return not node.segments

    if isinstance(node, AndNode):
        return not node.branches

    return False


def _valid_number(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False

    return isfinite(number)


def _valid_positive_number(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False

    return isfinite(number) and number > 0.0


def validate_prompt_graph(
    graph: PromptGraph,
    *,
    backend_name: str = "_21",
    strict_backend_rules: bool = False,
    sdxl_mode: bool = False,
    allow_graph_schedule_backends: bool = False,
    allow_graph_alternate_backends: bool = False,
) -> ValidationResult:
    validator = PromptGraphValidator(
        backend_name=backend_name,
        strict_backend_rules=strict_backend_rules,
        sdxl_mode=sdxl_mode,
        allow_graph_schedule_backends=allow_graph_schedule_backends,
        allow_graph_alternate_backends=allow_graph_alternate_backends,
    )

    return validator.validate(graph)