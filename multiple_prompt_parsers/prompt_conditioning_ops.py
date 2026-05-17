# prompt_conditioning_ops.py
from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from prompt_execution_plan import ConditioningOutput


# ============================================================================
# HELPERS
# ============================================================================


def _clone_tensor(value):
    if value is None:
        return None

    if torch.is_tensor(value):
        return value.clone()

    return deepcopy(value)


def _weighted_sum_tensors(
    tensors: list[torch.Tensor],
    weights: list[float],
) -> torch.Tensor:

    if not tensors:
        raise RuntimeError("No tensors provided")

    if len(tensors) != len(weights):
        raise RuntimeError("Tensor/weight mismatch")

    total_weight = sum(weights)

    if abs(total_weight) < 1e-12:
        raise RuntimeError("Total weight is zero")

    normalized = [
        float(w) / total_weight
        for w in weights
    ]

    result = None

    for tensor, weight in zip(tensors, normalized):

        weighted = tensor * weight

        if result is None:
            result = weighted
        else:
            result = result + weighted

    return result


def _get_weight(output: ConditioningOutput) -> float:
    return float(
        output.metadata.get("graph_weight", 1.0)
    )


def _get_label(output: ConditioningOutput) -> str | None:
    return output.metadata.get("graph_label")


def _copy_metadata(
    outputs: list[ConditioningOutput],
) -> dict[str, Any]:

    merged: dict[str, Any] = {}

    for output in outputs:
        merged.update(output.metadata)

    return merged


def _new_output(
    *,
    source_id: str,
    cross_attention=None,
    pooled_output=None,
    metadata: dict[str, Any] | None = None,
) -> ConditioningOutput:

    return ConditioningOutput(
        source_id=source_id,
        cross_attention=cross_attention,
        pooled_output=pooled_output,
        metadata=metadata or {},
    )


# ============================================================================
# BLEND
# ============================================================================


def conditioning_blend(
    outputs: list[ConditioningOutput],
    *,
    blend_mode: str = "mean",
    channel_target: str = "both",
    intensity: float = 1.0,
    mode: str | None = None,
) -> ConditioningOutput:

    if len(outputs) < 2:
        raise RuntimeError(
            "Blend requires at least 2 inputs"
        )

    metadata = _copy_metadata(outputs)

    weights = [
        _get_weight(output)
        for output in outputs
    ]

    result_cross = None
    result_pooled = None

    if channel_target in ("both", "cross"):

        tensors = [
            output.cross_attention
            for output in outputs
            if output.cross_attention is not None
        ]

        if tensors:
            result_cross = _weighted_sum_tensors(
                tensors,
                weights,
            )

            result_cross = result_cross * float(intensity)

    if channel_target in ("both", "pooled"):

        tensors = [
            output.pooled_output
            for output in outputs
            if output.pooled_output is not None
        ]

        if tensors:
            result_pooled = _weighted_sum_tensors(
                tensors,
                weights,
            )

            result_pooled = result_pooled * float(intensity)

    metadata["conditioning_op"] = "blend"

    return _new_output(
        source_id="blend_result",
        cross_attention=result_cross,
        pooled_output=result_pooled,
        metadata=metadata,
    )


# ============================================================================
# CHUNK
# ============================================================================


def conditioning_chunk(
    outputs: list[ConditioningOutput],
    *,
    shared_channel: str = "none",
) -> ConditioningOutput:

    if len(outputs) < 2:
        raise RuntimeError(
            "Chunk requires at least 2 inputs"
        )

    metadata = _copy_metadata(outputs)

    cross_tensors = [
        output.cross_attention
        for output in outputs
        if output.cross_attention is not None
    ]

    pooled_tensors = [
        output.pooled_output
        for output in outputs
        if output.pooled_output is not None
    ]

    result_cross = None
    result_pooled = None

    if cross_tensors:
        result_cross = torch.cat(
            cross_tensors,
            dim=1,
        )

    if pooled_tensors:

        if shared_channel == "pooled":
            result_pooled = pooled_tensors[0]
        else:
            result_pooled = torch.cat(
                pooled_tensors,
                dim=-1,
            )

    metadata["conditioning_op"] = "chunk"

    return _new_output(
        source_id="chunk_result",
        cross_attention=result_cross,
        pooled_output=result_pooled,
        metadata=metadata,
    )


# ============================================================================
# MORPH
# ============================================================================


def conditioning_morph(
    outputs: list[ConditioningOutput],
    *,
    curve: str = "linear",
    channel_target: str = "both",
    intensity: float = 1.0,
    point_boundaries=None,
    window_start=None,
    window_end=None,
) -> ConditioningOutput:

    if len(outputs) < 2:
        raise RuntimeError(
            "Morph requires at least 2 points"
        )

    metadata = _copy_metadata(outputs)

    weights = [
        _get_weight(output)
        for output in outputs
    ]

    result_cross = None
    result_pooled = None

    # simplified initial implementation
    if channel_target in ("both", "cross"):

        tensors = [
            output.cross_attention
            for output in outputs
            if output.cross_attention is not None
        ]

        if tensors:
            result_cross = _weighted_sum_tensors(
                tensors,
                weights,
            )

            result_cross = result_cross * float(intensity)

    if channel_target in ("both", "pooled"):

        tensors = [
            output.pooled_output
            for output in outputs
            if output.pooled_output is not None
        ]

        if tensors:
            result_pooled = _weighted_sum_tensors(
                tensors,
                weights,
            )

            result_pooled = result_pooled * float(intensity)

    metadata["conditioning_op"] = "morph"
    metadata["curve"] = curve

    return _new_output(
        source_id="morph_result",
        cross_attention=result_cross,
        pooled_output=result_pooled,
        metadata=metadata,
    )


# ============================================================================
# POOL
# ============================================================================


def conditioning_pool(
    outputs: list[ConditioningOutput],
    **kwargs,
) -> ConditioningOutput:

    if len(outputs) != 1:
        raise RuntimeError(
            "Pool expects exactly 1 input"
        )

    output = outputs[0]

    metadata = dict(output.metadata)
    metadata["conditioning_op"] = "pool"

    return _new_output(
        source_id="pool_result",
        cross_attention=_clone_tensor(
            output.cross_attention
        ),
        pooled_output=_clone_tensor(
            output.pooled_output
        ),
        metadata=metadata,
    )


# ============================================================================
# BIND
# ============================================================================


def conditioning_bind(
    outputs: list[ConditioningOutput],
    *,
    weight: float = 1.0,
) -> ConditioningOutput:

    if len(outputs) != 2:
        raise RuntimeError(
            "Bind requires exactly 2 inputs"
        )

    owner = outputs[0]
    attrs = outputs[1]

    metadata = _copy_metadata(outputs)

    owner_cross = owner.cross_attention
    attrs_cross = attrs.cross_attention

    result_cross = None

    if owner_cross is not None and attrs_cross is not None:

        result_cross = torch.cat(
            [
                owner_cross,
                attrs_cross * float(weight),
            ],
            dim=1,
        )

    result_pooled = (
        owner.pooled_output
        if owner.pooled_output is not None
        else attrs.pooled_output
    )

    metadata["conditioning_op"] = "bind"

    return _new_output(
        source_id="bind_result",
        cross_attention=result_cross,
        pooled_output=result_pooled,
        metadata=metadata,
    )


# ============================================================================
# ASSEMBLE
# ============================================================================


def conditioning_assemble(
    outputs: list[ConditioningOutput],
    **kwargs,
) -> ConditioningOutput:

    if len(outputs) < 2:
        raise RuntimeError(
            "Assemble requires enc1 and enc2"
        )

    metadata = _copy_metadata(outputs)

    enc1 = outputs[0]
    enc2 = outputs[1]

    pooled = None

    if len(outputs) >= 3:
        pooled = outputs[2].pooled_output

    extra_channels = {
        "enc1": enc1.cross_attention,
        "enc2": enc2.cross_attention,
    }

    metadata["conditioning_op"] = "assemble"

    return ConditioningOutput(
        source_id="assemble_result",
        cross_attention=enc1.cross_attention,
        pooled_output=pooled,
        extra_channels=extra_channels,
        metadata=metadata,
    )


# ============================================================================
# AND
# ============================================================================


def conditioning_and(
    outputs: list[ConditioningOutput],
    **kwargs,
) -> ConditioningOutput:

    if len(outputs) < 2:
        raise RuntimeError(
            "AND requires at least 2 inputs"
        )

    metadata = _copy_metadata(outputs)

    cross_tensors = []
    pooled_tensors = []

    for output in outputs:

        weight = _get_weight(output)

        if output.cross_attention is not None:
            cross_tensors.append(
                output.cross_attention * weight
            )

        if output.pooled_output is not None:
            pooled_tensors.append(
                output.pooled_output * weight
            )

    result_cross = None
    result_pooled = None

    if cross_tensors:
        result_cross = torch.cat(
            cross_tensors,
            dim=1,
        )

    if pooled_tensors:
        result_pooled = torch.cat(
            pooled_tensors,
            dim=-1,
        )

    metadata["conditioning_op"] = "and"

    return _new_output(
        source_id="and_result",
        cross_attention=result_cross,
        pooled_output=result_pooled,
        metadata=metadata,
    )


# ============================================================================
# SEQUENCE CONCAT
# ============================================================================


def conditioning_sequence_concat(
    outputs: list[ConditioningOutput],
    *,
    separator: str = " ",
) -> ConditioningOutput:

    if len(outputs) == 1:
        return outputs[0]

    metadata = _copy_metadata(outputs)

    cross_tensors = [
        output.cross_attention
        for output in outputs
        if output.cross_attention is not None
    ]

    pooled_tensors = [
        output.pooled_output
        for output in outputs
        if output.pooled_output is not None
    ]

    result_cross = None
    result_pooled = None

    if cross_tensors:
        result_cross = torch.cat(
            cross_tensors,
            dim=1,
        )

    if pooled_tensors:
        result_pooled = torch.cat(
            pooled_tensors,
            dim=-1,
        )

    metadata["conditioning_op"] = "sequence_concat"

    return _new_output(
        source_id="sequence_result",
        cross_attention=result_cross,
        pooled_output=result_pooled,
        metadata=metadata,
    )


# ============================================================================
# WEIGHT
# ============================================================================


def conditioning_weight(
    outputs: list[ConditioningOutput],
    *,
    weight_mode: str = "attention",
) -> ConditioningOutput:

    if len(outputs) != 1:
        raise RuntimeError(
            "Weight op expects exactly 1 input"
        )

    output = outputs[0]

    weight = _get_weight(output)

    metadata = dict(output.metadata)
    metadata["conditioning_op"] = "weight"
    metadata["weight_mode"] = weight_mode

    result_cross = output.cross_attention
    result_pooled = output.pooled_output

    if result_cross is not None:
        result_cross = result_cross * weight

    if result_pooled is not None:
        result_pooled = result_pooled * weight

    return _new_output(
        source_id="weight_result",
        cross_attention=result_cross,
        pooled_output=result_pooled,
        metadata=metadata,
    )