# prompt_graph_normalizer.py
from __future__ import annotations

from copy import deepcopy

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
        PromptNodeKind,
        ScheduleNode,
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
        PromptNodeKind,
        ScheduleNode,
        SequenceNode,
        TextNode,
        WeightNode,
    )


class GraphNormalizationError(Exception):
    pass


def normalize_graph(graph: PromptGraph) -> PromptGraph:
    normalized_root = normalize_node(graph.root)

    return PromptGraph(
        root=normalized_root,
        version=graph.version,
        symbols=dict(graph.symbols),
        diagnostics=list(graph.diagnostics),
    )


def normalize_node(node: PromptNode) -> PromptNode:
    if isinstance(node, TextNode):
        return _normalize_text(node)

    if isinstance(node, SequenceNode):
        return _normalize_sequence(node)

    if isinstance(node, GroupNode):
        return _normalize_group(node)

    if isinstance(node, WeightNode):
        return _normalize_weight(node)

    if isinstance(node, AlternateNode):
        return _normalize_alternate(node)

    if isinstance(node, ScheduleNode):
        return _normalize_schedule(node)

    if isinstance(node, AndNode):
        return _normalize_and(node)

    if isinstance(node, BlendNode):
        return _normalize_blend(node)

    if isinstance(node, ChunkNode):
        return _normalize_chunk(node)

    if isinstance(node, MorphNode):
        return _normalize_morph(node)

    if isinstance(node, PoolNode):
        return _normalize_pool(node)

    if isinstance(node, BindNode):
        return _normalize_bind(node)

    if isinstance(node, AssembleNode):
        return _normalize_assemble(node)

    raise TypeError(f"Unsupported node type: {type(node).__name__}")


# ============================================================================
# BASIC CLEANUP
# ============================================================================


def _normalize_text(node: TextNode) -> TextNode:
    text = " ".join(str(node.text).split())

    return TextNode(
        text=text,
        meta=node.meta,
    )


def _normalize_sequence(node: SequenceNode) -> PromptNode:
    normalized_parts: list[PromptNode] = []

    for part in node.parts:
        normalized = normalize_node(part)

        if _is_empty_node(normalized):
            continue

        if isinstance(normalized, SequenceNode):
            normalized_parts.extend(normalized.parts)
            continue

        normalized_parts.append(normalized)

    if not normalized_parts:
        return TextNode("")

    if len(normalized_parts) == 1:
        return normalized_parts[0]

    sequence = SequenceNode(
        parts=normalized_parts,
        separator=node.separator,
        meta=node.meta,
    )

    return _lift_backend_nodes(sequence)


def _normalize_group(node: GroupNode) -> PromptNode:
    normalized_parts = [
        normalize_node(part)
        for part in node.parts
    ]

    normalized_parts = [
        part for part in normalized_parts
        if not _is_empty_node(part)
    ]

    if not normalized_parts:
        return TextNode("")

    if len(normalized_parts) == 1:
        child = normalized_parts[0]

        if isinstance(child, GroupNode):
            return child

        return child

    group = GroupNode(
        parts=normalized_parts,
        delimiter=node.delimiter,
        meta=node.meta,
    )

    return _lift_backend_nodes(group)


def _normalize_weight(node: WeightNode) -> PromptNode:
    normalized_child = normalize_node(node.node)

    if _is_empty_node(normalized_child):
        return TextNode("")

    if abs(node.weight - 1.0) < 1e-8:
        return normalized_child

    if isinstance(normalized_child, WeightNode):
        return WeightNode(
            node=normalized_child.node,
            weight=node.weight * normalized_child.weight,
            mode=node.mode,
            meta=node.meta,
        )

    return WeightNode(
        node=normalized_child,
        weight=node.weight,
        mode=node.mode,
        meta=node.meta,
    )


# ============================================================================
# STRUCTURAL NODES
# ============================================================================


def _normalize_alternate(node: AlternateNode) -> PromptNode:
    options = [
        normalize_node(option)
        for option in node.options
    ]

    options = [
        option for option in options
        if not _is_empty_node(option)
    ]

    if not options:
        return TextNode("")

    if len(options) == 1:
        return options[0]

    return AlternateNode(
        options=options,
        mode=node.mode,
        meta=node.meta,
    )


def _normalize_schedule(node: ScheduleNode) -> PromptNode:
    segments = []

    for segment in node.segments:
        normalized_child = normalize_node(segment.node)

        if _is_empty_node(normalized_child):
            continue

        segments.append(
            segment.__class__(
                node=normalized_child,
                start=segment.start,
                end=segment.end,
                weight=segment.weight,
            )
        )

    if not segments:
        return TextNode("")

    return ScheduleNode(
        segments=segments,
        reverse=node.reverse,
        meta=node.meta,
    )


def _normalize_and(node: AndNode) -> PromptNode:
    branches = []

    for branch in node.branches:
        normalized_child = normalize_node(branch.node)

        if _is_empty_node(normalized_child):
            continue

        branches.append(
            AndBranch(
                node=normalized_child,
                weight=branch.weight,
            )
        )

    if not branches:
        return TextNode("")

    if len(branches) == 1:
        return branches[0].node

    return AndNode(
        branches=branches,
        meta=node.meta,
    )


# ============================================================================
# BACKEND NODES
# ============================================================================


def _normalize_blend(node: BlendNode) -> PromptNode:
    branches = []

    for branch in node.branches:
        normalized = normalize_node(branch.node)

        if _is_empty_node(normalized):
            continue

        branches.append(
            BackendBranch(
                node=normalized,
                weight=branch.weight,
                label=branch.label,
            )
        )

    if len(branches) < 2:
        raise GraphNormalizationError(
            "BLEND requires at least 2 non-empty branches"
        )

    return BlendNode(
        branches=branches,
        blend_mode=node.blend_mode,
        intensity=node.intensity,
        channel_target=node.channel_target,
        mode=node.mode,
        meta=node.meta,
    )


def _normalize_chunk(node: ChunkNode) -> PromptNode:
    branches = []

    for branch in node.branches:
        normalized = normalize_node(branch.node)

        if _is_empty_node(normalized):
            continue

        branches.append(
            BackendBranch(
                node=normalized,
                weight=branch.weight,
                label=branch.label,
            )
        )

    if not branches:
        return TextNode("")

    return ChunkNode(
        branches=branches,
        shared_channel=node.shared_channel,
        mode=node.mode,
        meta=node.meta,
    )


def _normalize_morph(node: MorphNode) -> PromptNode:
    points = []

    for point in node.points:
        normalized = normalize_node(point.node)

        if _is_empty_node(normalized):
            continue

        points.append(
            MorphPoint(
                node=normalized,
                boundary=point.boundary,
                weight=point.weight,
            )
        )

    if len(points) < 2:
        raise GraphNormalizationError(
            "MORPH requires at least 2 non-empty points"
        )

    return MorphNode(
        points=points,
        curve=node.curve,
        intensity=node.intensity,
        channel_target=node.channel_target,
        mode=node.mode,
        window_start=node.window_start,
        window_end=node.window_end,
        meta=node.meta,
    )


def _normalize_pool(node: PoolNode) -> PromptNode:
    normalized = normalize_node(node.node)

    if _is_empty_node(normalized):
        return TextNode("")

    return PoolNode(
        node=normalized,
        mode=node.mode,
        meta=node.meta,
    )


def _normalize_bind(node: BindNode) -> PromptNode:
    owner = normalize_node(node.owner)
    attrs = normalize_node(node.attrs)

    if _is_empty_node(owner) or _is_empty_node(attrs):
        raise GraphNormalizationError(
            "BIND requires non-empty owner and attrs"
        )

    return BindNode(
        owner=owner,
        attrs=attrs,
        weight=node.weight,
        mode=node.mode,
        meta=node.meta,
    )


def _normalize_assemble(node: AssembleNode) -> PromptNode:
    enc1 = normalize_node(node.enc1)
    enc2 = normalize_node(node.enc2)

    if _is_empty_node(enc1) or _is_empty_node(enc2):
        raise GraphNormalizationError(
            "ASSEMBLE requires non-empty enc1 and enc2"
        )

    pooled = None
    if node.pooled is not None:
        pooled = normalize_node(node.pooled)
        if _is_empty_node(pooled):
            pooled = None

    return AssembleNode(
        enc1=enc1,
        enc2=enc2,
        pooled=pooled,
        mode=node.mode,
        meta=node.meta,
    )


# ============================================================================
# BACKEND LIFTING
# ============================================================================


def _lift_backend_nodes(node: PromptNode) -> PromptNode:
    """
    Lift branch-style backend nodes out of regular prompt context.

    Example:
        {portrait BLEND{realism | oil}}

    Becomes:
        BLEND{portrait realism | portrait oil}

    Only branch-style backends are lifted:
        BLEND
        CHUNK
        MORPH

    POOL, BIND, and ASSEMBLE are deliberately not context-lifted here because
    moving surrounding text into those nodes can change channel semantics. Those
    cases should be handled by graph lowering as multi-call execution.
    """

    if isinstance(node, SequenceNode):
        parts = node.parts

    elif isinstance(node, GroupNode):
        parts = node.parts

    else:
        return node

    liftable_indices = [
        index
        for index, part in enumerate(parts)
        if _is_context_liftable_backend_node(part)
    ]

    if not liftable_indices:
        return node

    if len(liftable_indices) > 1:
        raise GraphNormalizationError(
            "Multiple liftable backend nodes in the same branch are ambiguous"
        )

    backend_index = liftable_indices[0]
    backend = parts[backend_index]

    prefix = parts[:backend_index]
    suffix = parts[backend_index + 1:]

    if isinstance(backend, BlendNode):
        return _lift_blend(
            backend,
            prefix,
            suffix,
        )

    if isinstance(backend, ChunkNode):
        return _lift_chunk(
            backend,
            prefix,
            suffix,
        )

    if isinstance(backend, MorphNode):
        return _lift_morph(
            backend,
            prefix,
            suffix,
        )

    return node


def _lift_blend(
    backend: BlendNode,
    prefix: list[PromptNode],
    suffix: list[PromptNode],
) -> BlendNode:
    lifted_branches = []

    for branch in backend.branches:
        lifted_branches.append(
            BackendBranch(
                node=_normalize_lifted_context(
                    prefix,
                    branch.node,
                    suffix,
                ),
                weight=branch.weight,
                label=branch.label,
            )
        )

    return BlendNode(
        branches=lifted_branches,
        blend_mode=backend.blend_mode,
        intensity=backend.intensity,
        channel_target=backend.channel_target,
        mode=backend.mode,
        meta=backend.meta,
    )


def _lift_chunk(
    backend: ChunkNode,
    prefix: list[PromptNode],
    suffix: list[PromptNode],
) -> ChunkNode:
    lifted_branches = []

    for branch in backend.branches:
        lifted_branches.append(
            BackendBranch(
                node=_normalize_lifted_context(
                    prefix,
                    branch.node,
                    suffix,
                ),
                weight=branch.weight,
                label=branch.label,
            )
        )

    return ChunkNode(
        branches=lifted_branches,
        shared_channel=backend.shared_channel,
        mode=backend.mode,
        meta=backend.meta,
    )


def _lift_morph(
    backend: MorphNode,
    prefix: list[PromptNode],
    suffix: list[PromptNode],
) -> MorphNode:
    lifted_points = []

    for point in backend.points:
        lifted_points.append(
            MorphPoint(
                node=_normalize_lifted_context(
                    prefix,
                    point.node,
                    suffix,
                ),
                boundary=point.boundary,
                weight=point.weight,
            )
        )

    return MorphNode(
        points=lifted_points,
        curve=backend.curve,
        intensity=backend.intensity,
        channel_target=backend.channel_target,
        mode=backend.mode,
        window_start=backend.window_start,
        window_end=backend.window_end,
        meta=backend.meta,
    )


# ============================================================================
# HELPERS
# ============================================================================


def _normalize_lifted_context(
    prefix: list[PromptNode],
    center: PromptNode,
    suffix: list[PromptNode],
) -> PromptNode:
    merged = _merge_context_into_branch(
        prefix,
        center,
        suffix,
    )

    return normalize_node(merged)


def _merge_context_into_branch(
    prefix: list[PromptNode],
    center: PromptNode,
    suffix: list[PromptNode],
) -> PromptNode:
    parts: list[PromptNode] = []

    for item in prefix:
        parts.append(deepcopy(item))

    parts.append(deepcopy(center))

    for item in suffix:
        parts.append(deepcopy(item))

    parts = [
        part for part in parts
        if not _is_empty_node(part)
    ]

    if not parts:
        return TextNode("")

    if len(parts) == 1:
        return parts[0]

    return SequenceNode(
        parts=parts,
        separator=" ",
    )


def _is_context_liftable_backend_node(node: PromptNode) -> bool:
    return isinstance(
        node,
        (
            BlendNode,
            ChunkNode,
            MorphNode,
        ),
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


def _contains_backend_node(node: PromptNode) -> bool:
    if _is_backend_node(node):
        return True

    return any(
        _contains_backend_node(child)
        for child in node.children()
    )


def _is_empty_node(node: PromptNode) -> bool:
    if isinstance(node, TextNode):
        return not node.text.strip()

    if isinstance(node, SequenceNode):
        return len(node.parts) == 0

    if isinstance(node, GroupNode):
        return len(node.parts) == 0

    if isinstance(node, AlternateNode):
        return len(node.options) == 0

    if isinstance(node, ScheduleNode):
        return len(node.segments) == 0

    if isinstance(node, AndNode):
        return len(node.branches) == 0

    return False