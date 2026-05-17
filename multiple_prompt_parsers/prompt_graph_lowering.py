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
except ImportError:  # allows standalone/local testing outside A1111 modules package
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
    def __init__(self) -> None:
        self.calls: list[PromptCall] = []
        self.operations: list[PlanOperation] = []
        self.counter = 0
        self.diagnostics: list[str] = []

    def next_id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter:04d}"

    def add_call(self, prompt: str, kind: PromptCallKind = PromptCallKind.PLAIN) -> str:
        call_id = self.next_id("call")
        self.calls.append(
            PromptCall(
                call_id=call_id,
                prompt=prompt,
                kind=kind,
            )
        )
        return call_id

    def add_operation(
        self,
        mode: MergeMode,
        inputs: list[PlanReference],
        params: dict | None = None,
    ) -> str:
        op_id = self.next_id("op")
        self.operations.append(
            PlanOperation(
                op_id=op_id,
                mode=mode,
                inputs=inputs,
                params=params or {},
            )
        )
        return op_id


def lower_graph(graph: PromptGraph) -> PromptExecutionPlan:
    """
    Lower a normalized PromptGraph into the shared PromptExecutionPlan schema.

    Important:
    This function intentionally returns prompt_execution_plan.PromptExecutionPlan,
    not a local lightweight plan type. That keeps lowering compatible with
    prompt_graph_executor.PromptGraphExecutor.
    """
    ctx = LoweringContext()
    output_id = lower_node(graph.root, ctx)

    plan = PromptExecutionPlan(
        calls=ctx.calls,
        operations=ctx.operations,
        output_id=output_id,
        diagnostics=[*graph.diagnostics, *ctx.diagnostics],
    )
    plan.build_execution_nodes()
    return plan


def lower_node(node: PromptNode, ctx: LoweringContext) -> str:
    if isinstance(node, TextNode):
        return ctx.add_call(node.text, PromptCallKind.PLAIN)

    if isinstance(node, SequenceNode):
        return _lower_sequence(node, ctx)

    if isinstance(node, GroupNode):
        return _lower_as_plain_text(node, ctx)

    if isinstance(node, WeightNode):
        return _lower_weight(node, ctx)

    if isinstance(node, AlternateNode):
        return _lower_as_plain_text(node, ctx)

    if isinstance(node, ScheduleNode):
        return _lower_as_plain_text(node, ctx)

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

    raise GraphLoweringError(f"Unsupported node type: {type(node).__name__}")


def _lower_as_plain_text(node: PromptNode, ctx: LoweringContext) -> str:
    return ctx.add_call(node_to_canonical_text(node), PromptCallKind.PLAIN)


def _lower_sequence(node: SequenceNode, ctx: LoweringContext) -> str:
    """
    If a sequence is backend-free, serialize it as one plain prompt.

    After normalization, sequences containing backend nodes should usually have
    been lifted. If one remains, lower children and merge their conditioning as
    a graph-level sequence concat op.
    """
    if not _contains_backend(node):
        return ctx.add_call(node_to_canonical_text(node), PromptCallKind.PLAIN)

    inputs = [PlanReference(source_id=lower_node(part, ctx)) for part in node.parts]

    return ctx.add_operation(
        MergeMode.SEQUENCE_CONCAT,
        inputs,
        params={"separator": node.separator},
    )


def _lower_weight(node: WeightNode, ctx: LoweringContext) -> str:
    source_id = lower_node(node.node, ctx)

    return ctx.add_operation(
        MergeMode.WEIGHT,
        [PlanReference(source_id=source_id, weight=node.weight)],
        params={"weight_mode": node.mode},
    )


def _lower_and(node: AndNode, ctx: LoweringContext) -> str:
    inputs: list[PlanReference] = []

    for branch in node.branches:
        source_id = lower_node(branch.node, ctx)
        inputs.append(
            PlanReference(
                source_id=source_id,
                weight=branch.weight,
            )
        )

    return ctx.add_operation(MergeMode.AND, inputs, params={})


def _lower_blend(node: BlendNode, ctx: LoweringContext) -> str:
    """
    Lower BLEND.

    Simple case:
        all branches are _21-safe plain text -> one backend prompt call.

    Complex case:
        nested graph exists inside branches -> lower branches independently,
        then blend as a graph-level merge op.
    """
    if _backend_node_is_21_safe(node):
        return ctx.add_call(node_to_canonical_text(node), PromptCallKind.BACKEND)

    inputs: list[PlanReference] = []

    for branch in node.branches:
        source_id = lower_node(branch.node, ctx)
        inputs.append(
            PlanReference(
                source_id=source_id,
                weight=branch.weight,
                label=branch.label,
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
        },
    )


def _lower_chunk(node: ChunkNode, ctx: LoweringContext) -> str:
    if _backend_node_is_21_safe(node):
        return ctx.add_call(node_to_canonical_text(node), PromptCallKind.BACKEND)

    inputs: list[PlanReference] = []

    for branch in node.branches:
        source_id = lower_node(branch.node, ctx)
        inputs.append(
            PlanReference(
                source_id=source_id,
                weight=branch.weight,
                label=branch.label,
            )
        )

    return ctx.add_operation(
        MergeMode.CHUNK,
        inputs,
        params={"shared_channel": node.shared_channel},
    )


def _lower_morph(node: MorphNode, ctx: LoweringContext) -> str:
    if _backend_node_is_21_safe(node):
        return ctx.add_call(node_to_canonical_text(node), PromptCallKind.BACKEND)

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
            "window_start": boundary_to_dict(node.window_start),
            "window_end": boundary_to_dict(node.window_end),
            "point_boundaries": [boundary_to_dict(point.boundary) for point in node.points],
        },
    )


def _lower_pool(node: PoolNode, ctx: LoweringContext) -> str:
    if _backend_node_is_21_safe(node):
        return ctx.add_call(node_to_canonical_text(node), PromptCallKind.BACKEND)

    source_id = lower_node(node.node, ctx)

    return ctx.add_operation(MergeMode.POOL, [PlanReference(source_id=source_id)], params={})


def _lower_bind(node: BindNode, ctx: LoweringContext) -> str:
    if _backend_node_is_21_safe(node):
        return ctx.add_call(node_to_canonical_text(node), PromptCallKind.BACKEND)

    owner_id = lower_node(node.owner, ctx)
    attrs_id = lower_node(node.attrs, ctx)

    return ctx.add_operation(
        MergeMode.BIND,
        [
            PlanReference(source_id=owner_id, label="owner"),
            PlanReference(source_id=attrs_id, weight=node.weight, label="attrs"),
        ],
        params={"weight": node.weight},
    )


def _lower_assemble(node: AssembleNode, ctx: LoweringContext) -> str:
    if _backend_node_is_21_safe(node):
        return ctx.add_call(node_to_canonical_text(node), PromptCallKind.BACKEND)

    inputs = [
        PlanReference(source_id=lower_node(node.enc1, ctx), label="enc1"),
        PlanReference(source_id=lower_node(node.enc2, ctx), label="enc2"),
    ]

    if node.pooled is not None:
        inputs.append(PlanReference(source_id=lower_node(node.pooled, ctx), label="pooled"))

    return ctx.add_operation(MergeMode.ASSEMBLE, inputs, params={})


def _backend_node_is_21_safe(node: PromptNode) -> bool:
    """
    True when the whole backend can be serialized into one _21-safe backend prompt.

    Conservative rule:
    - backend root is allowed
    - direct branch/control bodies must not contain backend nodes
    - plain schedules/weights/alternates are okay because _21 already handles them
    """
    if isinstance(node, BlendNode):
        return all(not _contains_backend(branch.node) for branch in node.branches)

    if isinstance(node, ChunkNode):
        return all(not _contains_disallowed_chunk_backend(branch.node) for branch in node.branches)

    if isinstance(node, MorphNode):
        return all(not _contains_disallowed_morph_backend(point.node) for point in node.points)

    if isinstance(node, PoolNode):
        return not _contains_backend(node.node)

    if isinstance(node, BindNode):
        return not _contains_backend(node.owner) and not _contains_backend(node.attrs)

    if isinstance(node, AssembleNode):
        return (
            not _contains_backend(node.enc1)
            and not _contains_backend(node.enc2)
            and (node.pooled is None or not _contains_backend(node.pooled))
        )

    return False


def _contains_backend(node: PromptNode) -> bool:
    if isinstance(node, (BlendNode, ChunkNode, MorphNode, PoolNode, BindNode, AssembleNode)):
        return True

    return any(_contains_backend(child) for child in node.children())


def _contains_disallowed_chunk_backend(node: PromptNode) -> bool:
    """
    _21 can tolerate special CHUNK + MORPH sugar in some cases, but this lowering
    layer is intentionally conservative. Let graph-level execution handle nested
    complex cases.
    """
    return _contains_backend(node)


def _contains_disallowed_morph_backend(node: PromptNode) -> bool:
    return _contains_backend(node)


def boundary_to_dict(boundary) -> dict | None:
    if boundary is None:
        return None

    return {
        "kind": boundary.kind,
        "value": boundary.value,
    }
