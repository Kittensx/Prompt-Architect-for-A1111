# prompt_graph_lowering.py
from __future__ import annotations

try:
    from modules.prompt_graph import (
        AlternateNode,
        AndNode,
        AssembleNode,
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
    from modules.prompt_graph_serializer import node_to_canonical_text
    from modules.prompt_execution_plan import (
        MergeMode,
        PlanOperation,
        PlanReference,
        PromptCall,
        PromptCallKind,
        PromptExecutionPlan,
    )
except ImportError:
    from prompt_graph import (
        AlternateNode,
        AndNode,
        AssembleNode,
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
    from prompt_graph_serializer import node_to_canonical_text
    from prompt_execution_plan import (
        MergeMode,
        PlanOperation,
        PlanReference,
        PromptCall,
        PromptCallKind,
        PromptExecutionPlan,
    )


class GraphLoweringError(Exception):
    pass


class LoweringContext:
    def __init__(self, *, backend_name: str = "_21") -> None:
        self.calls: list[PromptCall] = []
        self.operations: list[PlanOperation] = []
        self.counter = 0
        self.diagnostics: list[str] = []
        self.backend_name = backend_name

    def next_id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter:04d}"

    def add_call(
        self,
        prompt: str,
        kind: PromptCallKind = PromptCallKind.PLAIN,
        *,
        metadata: dict | None = None,
    ) -> str:
        call_id = self.next_id("call")

        self.calls.append(
            PromptCall(
                call_id=call_id,
                prompt=prompt,
                kind=kind,
                backend_name=self.backend_name,
                metadata=metadata or {},
            )
        )

        return call_id

    def add_operation(
        self,
        mode: MergeMode,
        inputs: list[PlanReference],
        params: dict | None = None,
        *,
        metadata: dict | None = None,
    ) -> str:
        op_id = self.next_id("op")

        self.operations.append(
            PlanOperation(
                op_id=op_id,
                mode=mode,
                inputs=inputs,
                params=params or {},
                metadata=metadata or {},
            )
        )

        return op_id


def lower_graph(
    graph: PromptGraph,
    *,
    backend_name: str = "_21",
) -> PromptExecutionPlan:
    ctx = LoweringContext(backend_name=backend_name)
    output_id = lower_node(graph.root, ctx)

    plan = PromptExecutionPlan(
        calls=ctx.calls,
        operations=ctx.operations,
        output_id=output_id,
        diagnostics=[*graph.diagnostics, *ctx.diagnostics],
        backend_name=backend_name,
    )

    plan.build_execution_nodes()
    return plan


def lower_node(node: PromptNode, ctx: LoweringContext) -> str:
    if isinstance(node, TextNode):
        return _lower_text(node, ctx)

    if isinstance(node, SequenceNode):
        return _lower_sequence(node, ctx)

    if isinstance(node, GroupNode):
        return _lower_group(node, ctx)

    if isinstance(node, WeightNode):
        return _lower_weight(node, ctx)

    if isinstance(node, AlternateNode):
        return _lower_alternate(node, ctx)

    if isinstance(node, ScheduleNode):
        return _lower_schedule(node, ctx)

    if isinstance(node, AndNode):
        return _lower_and(node, ctx)

    if isinstance(node, BlendNode):
        return _lower_blend(node, ctx)

    if isinstance(node, ChunkNode):
        return _lower_chunk(node, ctx)

    if isinstance(node, MorphNode):
        return _lower_morph(node, ctx)

    if isinstance(node, PoolNode):
        return _lower_pool(node, ctx)

    if isinstance(node, BindNode):
        return _lower_bind(node, ctx)

    if isinstance(node, AssembleNode):
        return _lower_assemble(node, ctx)

    raise GraphLoweringError(
        f"Unsupported node type: {type(node).__name__}"
    )


# ============================================================================
# BASIC NODES
# ============================================================================


def _lower_text(node: TextNode, ctx: LoweringContext) -> str:
    return ctx.add_call(
        node.text,
        PromptCallKind.PLAIN,
        metadata={
            "node_kind": node.kind.value,
            "node_id": node.meta.node_id,
        },
    )


def _lower_as_plain_text(
    node: PromptNode,
    ctx: LoweringContext,
    *,
    kind: PromptCallKind = PromptCallKind.PLAIN,
) -> str:
    return ctx.add_call(
        node_to_canonical_text(node),
        kind,
        metadata={
            "node_kind": node.kind.value,
            "node_id": node.meta.node_id,
        },
    )


def _lower_sequence(node: SequenceNode, ctx: LoweringContext) -> str:
    if not _contains_backend(node):
        return _lower_as_plain_text(node, ctx)

    inputs = [
        PlanReference(
            source_id=lower_node(part, ctx),
            label=f"part_{index}",
        )
        for index, part in enumerate(node.parts)
    ]

    return ctx.add_operation(
        MergeMode.SEQUENCE_CONCAT,
        inputs,
        params={
            "separator": node.separator,
            "node_kind": node.kind.value,
        },
        metadata={"node_id": node.meta.node_id},
    )


def _lower_group(node: GroupNode, ctx: LoweringContext) -> str:
    if not _contains_backend(node):
        return _lower_as_plain_text(node, ctx)

    inputs = [
        PlanReference(
            source_id=lower_node(part, ctx),
            label=f"group_part_{index}",
        )
        for index, part in enumerate(node.parts)
    ]

    return ctx.add_operation(
        MergeMode.SEQUENCE_CONCAT,
        inputs,
        params={
            "separator": " ",
            "delimiter": node.delimiter,
            "node_kind": node.kind.value,
        },
        metadata={"node_id": node.meta.node_id},
    )


def _lower_weight(node: WeightNode, ctx: LoweringContext) -> str:
    if not _contains_backend(node.node):
        return _lower_as_plain_text(node, ctx)

    source_id = lower_node(node.node, ctx)

    return ctx.add_operation(
        MergeMode.WEIGHT,
        [
            PlanReference(
                source_id=source_id,
                weight=node.weight,
            )
        ],
        params={
            "weight": node.weight,
            "weight_mode": node.mode,
            "node_kind": node.kind.value,
        },
        metadata={"node_id": node.meta.node_id},
    )


def _lower_alternate(node: AlternateNode, ctx: LoweringContext) -> str:
    if _contains_backend(node):
        raise GraphLoweringError(
            "AlternateNode containing backend nodes cannot be lowered yet. "
            "Resolve alternates before graph lowering, or add an ALTERNATE merge mode."
        )

    return _lower_as_plain_text(node, ctx)


def _lower_schedule(node: ScheduleNode, ctx: LoweringContext) -> str:
    if _contains_backend(node):
        raise GraphLoweringError(
            "ScheduleNode containing backend nodes cannot be lowered yet. "
            "Resolve schedule windows before graph lowering, or add scheduled execution windows."
        )

    return _lower_as_plain_text(node, ctx)


def _lower_and(node: AndNode, ctx: LoweringContext) -> str:
    inputs: list[PlanReference] = []

    for index, branch in enumerate(node.branches):
        source_id = lower_node(branch.node, ctx)

        inputs.append(
            PlanReference(
                source_id=source_id,
                weight=branch.weight,
                label=f"and_{index}",
            )
        )

    return ctx.add_operation(
        MergeMode.AND,
        inputs,
        params={"node_kind": node.kind.value},
        metadata={"node_id": node.meta.node_id},
    )


# ============================================================================
# BACKEND NODES
# ============================================================================


def _lower_blend(node: BlendNode, ctx: LoweringContext) -> str:
    if _backend_node_is_21_safe(node):
        return _lower_as_plain_text(
            node,
            ctx,
            kind=PromptCallKind.BACKEND,
        )

    inputs: list[PlanReference] = []

    for index, branch in enumerate(node.branches):
        source_id = lower_node(branch.node, ctx)

        inputs.append(
            PlanReference(
                source_id=source_id,
                weight=branch.weight,
                label=branch.label or f"blend_{index}",
            )
        )

    return ctx.add_operation(
        MergeMode.BLEND,
        inputs,
        params={
            "blend_mode": node.blend_mode,
            "channel_target": node.channel_target,
            "intensity": node.intensity,
            "mode": node.mode,
            "node_kind": node.kind.value,
        },
        metadata={"node_id": node.meta.node_id},
    )


def _lower_chunk(node: ChunkNode, ctx: LoweringContext) -> str:
    if _backend_node_is_21_safe(node):
        return _lower_as_plain_text(
            node,
            ctx,
            kind=PromptCallKind.BACKEND,
        )

    inputs: list[PlanReference] = []

    for index, branch in enumerate(node.branches):
        source_id = lower_node(branch.node, ctx)

        inputs.append(
            PlanReference(
                source_id=source_id,
                weight=branch.weight,
                label=branch.label or f"chunk_{index}",
            )
        )

    return ctx.add_operation(
        MergeMode.CHUNK,
        inputs,
        params={
            "shared_channel": node.shared_channel,
            "mode": node.mode,
            "node_kind": node.kind.value,
        },
        metadata={"node_id": node.meta.node_id},
    )


def _lower_morph(node: MorphNode, ctx: LoweringContext) -> str:
    if _backend_node_is_21_safe(node):
        return _lower_as_plain_text(
            node,
            ctx,
            kind=PromptCallKind.BACKEND,
        )

    inputs: list[PlanReference] = []

    for index, point in enumerate(node.points):
        source_id = lower_node(point.node, ctx)

        inputs.append(
            PlanReference(
                source_id=source_id,
                weight=point.weight,
                label=f"point_{index}",
            )
        )

    return ctx.add_operation(
        MergeMode.MORPH,
        inputs,
        params={
            "curve": node.curve,
            "channel_target": node.channel_target,
            "intensity": node.intensity,
            "mode": node.mode,
            "window_start": boundary_to_dict(node.window_start),
            "window_end": boundary_to_dict(node.window_end),
            "point_boundaries": [
                boundary_to_dict(point.boundary)
                for point in node.points
            ],
            "node_kind": node.kind.value,
        },
        metadata={"node_id": node.meta.node_id},
    )


def _lower_pool(node: PoolNode, ctx: LoweringContext) -> str:
    if _backend_node_is_21_safe(node):
        return _lower_as_plain_text(
            node,
            ctx,
            kind=PromptCallKind.BACKEND,
        )

    source_id = lower_node(node.node, ctx)

    return ctx.add_operation(
        MergeMode.POOL,
        [PlanReference(source_id=source_id, label="pool")],
        params={
            "mode": node.mode,
            "node_kind": node.kind.value,
        },
        metadata={"node_id": node.meta.node_id},
    )


def _lower_bind(node: BindNode, ctx: LoweringContext) -> str:
    if _backend_node_is_21_safe(node):
        return _lower_as_plain_text(
            node,
            ctx,
            kind=PromptCallKind.BACKEND,
        )

    owner_id = lower_node(node.owner, ctx)
    attrs_id = lower_node(node.attrs, ctx)

    return ctx.add_operation(
        MergeMode.BIND,
        [
            PlanReference(
                source_id=owner_id,
                label="owner",
            ),
            PlanReference(
                source_id=attrs_id,
                weight=node.weight,
                label="attrs",
            ),
        ],
        params={
            "weight": node.weight,
            "mode": node.mode,
            "node_kind": node.kind.value,
        },
        metadata={"node_id": node.meta.node_id},
    )


def _lower_assemble(node: AssembleNode, ctx: LoweringContext) -> str:
    if _backend_node_is_21_safe(node):
        return _lower_as_plain_text(
            node,
            ctx,
            kind=PromptCallKind.BACKEND,
        )

    inputs = [
        PlanReference(
            source_id=lower_node(node.enc1, ctx),
            label="enc1",
        ),
        PlanReference(
            source_id=lower_node(node.enc2, ctx),
            label="enc2",
        ),
    ]

    if node.pooled is not None:
        inputs.append(
            PlanReference(
                source_id=lower_node(node.pooled, ctx),
                label="pooled",
            )
        )

    return ctx.add_operation(
        MergeMode.ASSEMBLE,
        inputs,
        params={
            "mode": node.mode,
            "node_kind": node.kind.value,
        },
        metadata={"node_id": node.meta.node_id},
    )


# ============================================================================
# SAFETY CHECKS
# ============================================================================


def _backend_node_is_21_safe(node: PromptNode) -> bool:
    """
    True when this backend node can be serialized into one _21 backend prompt.

    Rule:
    - The backend node itself may be the root call.
    - Its immediate bodies may contain regular prompt grammar.
    - Its immediate bodies must not contain another backend node.
    """

    if isinstance(node, BlendNode):
        return all(
            not _contains_backend(branch.node)
            for branch in node.branches
        )

    if isinstance(node, ChunkNode):
        return all(
            not _contains_backend(branch.node)
            for branch in node.branches
        )

    if isinstance(node, MorphNode):
        return all(
            not _contains_backend(point.node)
            for point in node.points
        )

    if isinstance(node, PoolNode):
        return not _contains_backend(node.node)

    if isinstance(node, BindNode):
        return (
            not _contains_backend(node.owner)
            and not _contains_backend(node.attrs)
        )

    if isinstance(node, AssembleNode):
        return (
            not _contains_backend(node.enc1)
            and not _contains_backend(node.enc2)
            and (
                node.pooled is None
                or not _contains_backend(node.pooled)
            )
        )

    return False


def _contains_backend(node: PromptNode) -> bool:
    if node.kind in {
        PromptNodeKind.BLEND,
        PromptNodeKind.CHUNK,
        PromptNodeKind.MORPH,
        PromptNodeKind.POOL,
        PromptNodeKind.BIND,
        PromptNodeKind.ASSEMBLE,
    }:
        return True

    return any(
        _contains_backend(child)
        for child in node.children()
    )


def boundary_to_dict(boundary) -> dict | None:
    if boundary is None:
        return None

    return {
        "kind": boundary.kind,
        "value": boundary.value,
    }