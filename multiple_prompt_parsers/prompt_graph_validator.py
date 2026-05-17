# prompt_graph_validator.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

try:
    from modules.prompt_graph import (
        AlternateNode,
        AndNode,
        AssembleNode,
        BackendBranch,
        BackendNode,
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
        BackendNode,
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


# ============================================================================
# SEVERITY
# ============================================================================


class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# ============================================================================
# ISSUE
# ============================================================================


@dataclass(slots=True)
class ValidationIssue:
    severity: ValidationSeverity
    code: str
    message: str

    node_kind: str | None = None
    node_id: str | None = None

    suggestion: str | None = None

    metadata: dict = field(default_factory=dict)


# ============================================================================
# RESULT
# ============================================================================


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
            issue
            for issue in self.issues
            if issue.severity == ValidationSeverity.ERROR
        ]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [
            issue
            for issue in self.issues
            if issue.severity == ValidationSeverity.WARNING
        ]

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def raise_if_errors(self) -> None:

        if not self.has_errors():
            return

        lines = []

        for issue in self.errors:
            lines.append(
                f"[{issue.code}] {issue.message}"
            )

        raise PromptGraphValidationError(
            "\n".join(lines)
        )


# ============================================================================
# ERROR
# ============================================================================


class PromptGraphValidationError(Exception):
    pass


# ============================================================================
# VALIDATOR
# ============================================================================


class PromptGraphValidator:

    def __init__(
        self,
        *,
        backend_name: str = "_21",
        strict_backend_rules: bool = False,
        sdxl_mode: bool = False,
    ) -> None:

        self.backend_name = backend_name
        self.strict_backend_rules = strict_backend_rules
        self.sdxl_mode = sdxl_mode

    # ---------------------------------------------------------------------
    # PUBLIC
    # ---------------------------------------------------------------------

    def validate(
        self,
        graph: PromptGraph,
    ) -> ValidationResult:

        result = ValidationResult()

        self._validate_node(
            graph.root,
            result,
            parent=None,
            backend_depth=0,
        )

        return result

    # ---------------------------------------------------------------------
    # CORE WALK
    # ---------------------------------------------------------------------

    def _validate_node(
        self,
        node: PromptNode,
        result: ValidationResult,
        *,
        parent: PromptNode | None,
        backend_depth: int,
    ) -> None:

        # -------------------------------------------------------------
        # generic checks
        # -------------------------------------------------------------

        self._validate_empty_text(
            node,
            result,
        )

        self._validate_weights(
            node,
            result,
        )

        # -------------------------------------------------------------
        # backend nesting
        # -------------------------------------------------------------

        if isinstance(node, BackendNode):

            self._validate_backend_node(
                node,
                result,
                backend_depth=backend_depth,
            )

            backend_depth += 1

        # -------------------------------------------------------------
        # specific node checks
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # recurse
        # -------------------------------------------------------------

        for child in node.children():

            self._validate_node(
                child,
                result,
                parent=node,
                backend_depth=backend_depth,
            )

    # =========================================================================
    # GENERIC CHECKS
    # =========================================================================

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

    def _validate_weights(
        self,
        node: PromptNode,
        result: ValidationResult,
    ) -> None:

        if isinstance(node, WeightNode):

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
                    suggestion=(
                        "Ensure backend supports "
                        "negative conditioning weights"
                    ),
                )

    # =========================================================================
    # BACKEND RULES
    # =========================================================================

    def _validate_backend_node(
        self,
        node: BackendNode,
        result: ValidationResult,
        *,
        backend_depth: int,
    ) -> None:

        if backend_depth > 0:

            if self.strict_backend_rules:

                result.add(
                    ValidationSeverity.ERROR,
                    "nested_backend",
                    (
                        "Nested backend nodes are not "
                        "allowed in strict backend mode"
                    ),
                    node=node,
                )

            else:

                result.add(
                    ValidationSeverity.INFO,
                    "nested_backend_normalizable",
                    (
                        "Nested backend node detected. "
                        "Lowering layer may split this "
                        "into multi-call execution."
                    ),
                    node=node,
                )

    # =========================================================================
    # BLEND
    # =========================================================================

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

        for branch in node.branches:

            self._validate_backend_branch(
                branch,
                result,
                backend_name="BLEND",
            )

    # =========================================================================
    # CHUNK
    # =========================================================================

    def _validate_chunk(
        self,
        node: ChunkNode,
        result: ValidationResult,
    ) -> None:

        if len(node.branches) < 2:

            result.add(
                ValidationSeverity.WARNING,
                "chunk_single_branch",
                "CHUNK only contains one branch",
                node=node,
            )

    # =========================================================================
    # MORPH
    # =========================================================================

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

        last_boundary = None

        for i, point in enumerate(node.points):

            boundary = point.boundary

            # ---------------------------------------------------------
            # first point
            # ---------------------------------------------------------

            if i == 0:

                if boundary is not None:

                    result.add(
                        ValidationSeverity.WARNING,
                        "morph_first_boundary",
                        (
                            "First MORPH point usually "
                            "should not define boundary"
                        ),
                        node=node,
                    )

            # ---------------------------------------------------------
            # ordering
            # ---------------------------------------------------------

            if boundary is not None:

                if (
                    last_boundary is not None
                    and boundary.value <= last_boundary
                ):

                    result.add(
                        ValidationSeverity.ERROR,
                        "morph_boundary_order",
                        (
                            "MORPH boundaries must be "
                            "strictly increasing"
                        ),
                        node=node,
                    )

                last_boundary = boundary.value

    # =========================================================================
    # POOL
    # =========================================================================

    def _validate_pool(
        self,
        node: PoolNode,
        result: ValidationResult,
    ) -> None:

        if isinstance(node.node, BackendNode):

            result.add(
                ValidationSeverity.WARNING,
                "pool_nested_backend",
                (
                    "POOL wrapping backend node may "
                    "require graph lowering"
                ),
                node=node,
            )

    # =========================================================================
    # BIND
    # =========================================================================

    def _validate_bind(
        self,
        node: BindNode,
        result: ValidationResult,
    ) -> None:

        if abs(node.weight) < 1e-12:

            result.add(
                ValidationSeverity.WARNING,
                "bind_zero_weight",
                "BIND weight is zero",
                node=node,
            )

    # =========================================================================
    # ASSEMBLE
    # =========================================================================

    def _validate_assemble(
        self,
        node: AssembleNode,
        result: ValidationResult,
    ) -> None:

        if not self.sdxl_mode:

            result.add(
                ValidationSeverity.WARNING,
                "assemble_non_sdxl",
                (
                    "ASSEMBLE is typically intended "
                    "for SDXL dual encoders"
                ),
                node=node,
            )

    # =========================================================================
    # ALTERNATE
    # =========================================================================

    def _validate_alternate(
        self,
        node: AlternateNode,
        result: ValidationResult,
    ) -> None:

        if len(node.options) < 2:

            result.add(
                ValidationSeverity.WARNING,
                "alternate_single_option",
                "Alternate node only contains one option",
                node=node,
            )

    # =========================================================================
    # SCHEDULE
    # =========================================================================

    def _validate_schedule(
        self,
        node: ScheduleNode,
        result: ValidationResult,
    ) -> None:

        if not node.segments:

            result.add(
                ValidationSeverity.WARNING,
                "empty_schedule",
                "Schedule node has no segments",
                node=node,
            )

    # =========================================================================
    # AND
    # =========================================================================

    def _validate_and(
        self,
        node: AndNode,
        result: ValidationResult,
    ) -> None:

        if len(node.branches) < 2:

            result.add(
                ValidationSeverity.WARNING,
                "and_single_branch",
                "AND node only contains one branch",
                node=node,
            )

    # =========================================================================
    # BRANCH
    # =========================================================================

    def _validate_backend_branch(
        self,
        branch: BackendBranch,
        result: ValidationResult,
        *,
        backend_name: str,
    ) -> None:

        if abs(branch.weight) < 1e-12:

            result.add(
                ValidationSeverity.WARNING,
                "branch_zero_weight",
                (
                    f"{backend_name} branch weight "
                    "is effectively zero"
                ),
                node=branch.node,
            )

        if branch.weight < 0:

            result.add(
                ValidationSeverity.WARNING,
                "branch_negative_weight",
                (
                    f"{backend_name} branch "
                    "has negative weight"
                ),
                node=branch.node,
            )


# ============================================================================
# CONVENIENCE
# ============================================================================


def validate_prompt_graph(
    graph: PromptGraph,
    *,
    backend_name: str = "_21",
    strict_backend_rules: bool = False,
    sdxl_mode: bool = False,
) -> ValidationResult:

    validator = PromptGraphValidator(
        backend_name=backend_name,
        strict_backend_rules=strict_backend_rules,
        sdxl_mode=sdxl_mode,
    )

    return validator.validate(graph)