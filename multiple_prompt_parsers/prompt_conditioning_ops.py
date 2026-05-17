# prompt_conditioning_ops.py
from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

try:
    from modules.prompt_execution_plan import ConditioningOutput
except ImportError:
    from prompt_execution_plan import ConditioningOutput


def _clone_value(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.clone()
    return deepcopy(value)


def _clone_output(
    output: ConditioningOutput,
    *,
    source_id: str,
    metadata: dict[str, Any] | None = None,
) -> ConditioningOutput:
    merged_metadata = dict(output.metadata)
    if metadata:
        merged_metadata.update(metadata)

    return ConditioningOutput(
        source_id=source_id,
        cross_attention=_clone_value(output.cross_attention),
        pooled_output=_clone_value(output.pooled_output),
        extra_channels=deepcopy(output.extra_channels),
        metadata=merged_metadata,
    )


def _new_output(
    *,
    source_id: str,
    cross_attention=None,
    pooled_output=None,
    extra_channels: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ConditioningOutput:
    return ConditioningOutput(
        source_id=source_id,
        cross_attention=cross_attention,
        pooled_output=pooled_output,
        extra_channels=extra_channels or {},
        metadata=metadata or {},
    )


def _get_weight(output: ConditioningOutput) -> float:
    return float(output.metadata.get("graph_weight", 1.0))


def _get_label(output: ConditioningOutput) -> str | None:
    label = output.metadata.get("graph_label")
    return None if label is None else str(label)


def _copy_metadata(outputs: list[ConditioningOutput]) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    for index, output in enumerate(outputs):
        merged.update(output.metadata)
        merged[f"input_{index}_source_id"] = output.source_id

        label = _get_label(output)
        if label is not None:
            merged[f"input_{index}_label"] = label

    return merged


def _weighted_sum_channel(
    outputs: list[ConditioningOutput],
    getter,
    *,
    normalize: bool = True,
):
    pairs = [
        (getter(output), _get_weight(output))
        for output in outputs
        if getter(output) is not None
    ]

    if not pairs:
        return None

    tensors = [tensor for tensor, _weight in pairs]
    weights = [float(weight) for _tensor, weight in pairs]

    if len(tensors) != len(weights):
        raise RuntimeError("Tensor/weight mismatch")

    total = sum(weights)

    if normalize:
        if abs(total) < 1e-12:
            raise RuntimeError("Total weight is zero")
        weights = [weight / total for weight in weights]

    result = None

    for tensor, weight in zip(tensors, weights):
        weighted = tensor * weight
        result = weighted if result is None else result + weighted

    return result


def _merge_extra_channels_weighted(
    outputs: list[ConditioningOutput],
    *,
    normalize: bool = True,
) -> dict[str, Any]:
    keys: set[str] = set()

    for output in outputs:
        keys.update(output.extra_channels.keys())

    merged: dict[str, Any] = {}

    for key in keys:
        pairs = [
            (output.extra_channels.get(key), _get_weight(output))
            for output in outputs
            if output.extra_channels.get(key) is not None
        ]

        if not pairs:
            continue

        values = [value for value, _weight in pairs]

        if all(torch.is_tensor(value) for value in values):
            fake_outputs = [
                ConditioningOutput(
                    source_id=output.source_id,
                    cross_attention=output.extra_channels.get(key),
                    metadata={"graph_weight": _get_weight(output)},
                )
                for output in outputs
                if output.extra_channels.get(key) is not None
            ]
            merged[key] = _weighted_sum_channel(
                fake_outputs,
                lambda item: item.cross_attention,
                normalize=normalize,
            )
        else:
            merged[key] = _clone_value(values[0])

    return merged


def _cat_channel(outputs: list[ConditioningOutput], getter, *, dim: int):
    tensors = [
        getter(output)
        for output in outputs
        if getter(output) is not None
    ]

    if not tensors:
        return None

    if len(tensors) == 1:
        return _clone_value(tensors[0])

    return torch.cat(tensors, dim=dim)


def conditioning_blend(
    outputs: list[ConditioningOutput],
    *,
    blend_mode: str = "mean",
    channel_target: str = "both",
    intensity: float = 1.0,
    mode: str | None = None,
) -> ConditioningOutput:
    if len(outputs) < 2:
        raise RuntimeError("Blend requires at least 2 inputs")

    blend_mode = mode or blend_mode
    normalize = blend_mode != "sum"

    metadata = _copy_metadata(outputs)
    metadata["conditioning_op"] = "blend"
    metadata["blend_mode"] = blend_mode
    metadata["channel_target"] = channel_target
    metadata["intensity"] = float(intensity)

    result_cross = None
    result_pooled = None

    if channel_target in ("both", "cross", "enc1", "enc2"):
        result_cross = _weighted_sum_channel(
            outputs,
            lambda output: output.cross_attention,
            normalize=normalize,
        )
        if result_cross is not None:
            result_cross = result_cross * float(intensity)

    if channel_target in ("both", "pooled"):
        result_pooled = _weighted_sum_channel(
            outputs,
            lambda output: output.pooled_output,
            normalize=normalize,
        )
        if result_pooled is not None:
            result_pooled = result_pooled * float(intensity)

    extra_channels = _merge_extra_channels_weighted(
        outputs,
        normalize=normalize,
    )

    return _new_output(
        source_id="blend_result",
        cross_attention=result_cross,
        pooled_output=result_pooled,
        extra_channels=extra_channels,
        metadata=metadata,
    )


def conditioning_chunk(
    outputs: list[ConditioningOutput],
    *,
    shared_channel: str = "none",
) -> ConditioningOutput:
    if len(outputs) < 2:
        raise RuntimeError("Chunk requires at least 2 inputs")

    metadata = _copy_metadata(outputs)
    metadata["conditioning_op"] = "chunk"
    metadata["shared_channel"] = shared_channel

    if shared_channel == "cross":
        result_cross = _clone_value(outputs[0].cross_attention)
    else:
        result_cross = _cat_channel(
            outputs,
            lambda output: output.cross_attention,
            dim=1,
        )

    if shared_channel == "pooled":
        result_pooled = _clone_value(outputs[0].pooled_output)
    else:
        result_pooled = _cat_channel(
            outputs,
            lambda output: output.pooled_output,
            dim=-1,
        )

    extra_channels: dict[str, Any] = {}

    for output in outputs:
        for key, value in output.extra_channels.items():
            extra_channels.setdefault(key, [])
            extra_channels[key].append(value)

    for key, values in list(extra_channels.items()):
        if all(torch.is_tensor(value) for value in values):
            extra_channels[key] = (
                _clone_value(values[0])
                if len(values) == 1
                else torch.cat(values, dim=1)
            )
        else:
            extra_channels[key] = deepcopy(values)

    return _new_output(
        source_id="chunk_result",
        cross_attention=result_cross,
        pooled_output=result_pooled,
        extra_channels=extra_channels,
        metadata=metadata,
    )


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
        raise RuntimeError("Morph requires at least 2 points")

    metadata = _copy_metadata(outputs)
    metadata["conditioning_op"] = "morph"
    metadata["curve"] = curve
    metadata["channel_target"] = channel_target
    metadata["intensity"] = float(intensity)
    metadata["point_boundaries"] = point_boundaries
    metadata["window_start"] = window_start
    metadata["window_end"] = window_end

    # Current graph-level implementation is weighted interpolation.
    # Time-window/curve-aware morphing can be added later when the executor
    # passes current step metadata.
    result_cross = None
    result_pooled = None

    if channel_target in ("both", "cross", "enc1", "enc2"):
        result_cross = _weighted_sum_channel(
            outputs,
            lambda output: output.cross_attention,
            normalize=True,
        )
        if result_cross is not None:
            result_cross = result_cross * float(intensity)

    if channel_target in ("both", "pooled"):
        result_pooled = _weighted_sum_channel(
            outputs,
            lambda output: output.pooled_output,
            normalize=True,
        )
        if result_pooled is not None:
            result_pooled = result_pooled * float(intensity)

    extra_channels = _merge_extra_channels_weighted(
        outputs,
        normalize=True,
    )

    return _new_output(
        source_id="morph_result",
        cross_attention=result_cross,
        pooled_output=result_pooled,
        extra_channels=extra_channels,
        metadata=metadata,
    )


def conditioning_pool(
    outputs: list[ConditioningOutput],
    *,
    mode: str | None = None,
) -> ConditioningOutput:
    if not outputs:
        raise RuntimeError("Pool requires at least 1 input")

    source = outputs[0]

    metadata = dict(source.metadata)
    metadata["conditioning_op"] = "pool"
    metadata["mode"] = mode

    return _new_output(
        source_id="pool_result",
        cross_attention=_clone_value(source.cross_attention),
        pooled_output=_clone_value(source.pooled_output),
        extra_channels=deepcopy(source.extra_channels),
        metadata=metadata,
    )


def conditioning_bind(
    outputs: list[ConditioningOutput],
    *,
    weight: float = 1.0,
    mode: str | None = None,
) -> ConditioningOutput:
    if len(outputs) < 2:
        raise RuntimeError("Bind requires owner and attrs inputs")

    owner = outputs[0]
    attrs = outputs[1]

    metadata = _copy_metadata(outputs)
    metadata["conditioning_op"] = "bind"
    metadata["bind_weight"] = float(weight)
    metadata["mode"] = mode

    result_cross = _weighted_sum_channel(
        [
            ConditioningOutput(
                source_id=owner.source_id,
                cross_attention=owner.cross_attention,
                metadata={"graph_weight": 1.0},
            ),
            ConditioningOutput(
                source_id=attrs.source_id,
                cross_attention=attrs.cross_attention,
                metadata={"graph_weight": float(weight)},
            ),
        ],
        lambda output: output.cross_attention,
        normalize=True,
    )

    result_pooled = _weighted_sum_channel(
        [
            ConditioningOutput(
                source_id=owner.source_id,
                pooled_output=owner.pooled_output,
                metadata={"graph_weight": 1.0},
            ),
            ConditioningOutput(
                source_id=attrs.source_id,
                pooled_output=attrs.pooled_output,
                metadata={"graph_weight": float(weight)},
            ),
        ],
        lambda output: output.pooled_output,
        normalize=True,
    )

    extra_channels = _merge_extra_channels_weighted(
        [
            ConditioningOutput(
                source_id=owner.source_id,
                extra_channels=owner.extra_channels,
                metadata={"graph_weight": 1.0},
            ),
            ConditioningOutput(
                source_id=attrs.source_id,
                extra_channels=attrs.extra_channels,
                metadata={"graph_weight": float(weight)},
            ),
        ],
        normalize=True,
    )

    return _new_output(
        source_id="bind_result",
        cross_attention=result_cross,
        pooled_output=result_pooled,
        extra_channels=extra_channels,
        metadata=metadata,
    )


def conditioning_assemble(
    outputs: list[ConditioningOutput],
    *,
    mode: str | None = None,
) -> ConditioningOutput:
    if len(outputs) < 2:
        raise RuntimeError("Assemble requires enc1 and enc2 inputs")

    by_label = {
        _get_label(output): output
        for output in outputs
        if _get_label(output) is not None
    }

    enc1 = by_label.get("enc1", outputs[0])
    enc2 = by_label.get("enc2", outputs[1])
    pooled = by_label.get("pooled")

    metadata = _copy_metadata(outputs)
    metadata["conditioning_op"] = "assemble"
    metadata["mode"] = mode

    cross_parts = [
        item.cross_attention
        for item in (enc1, enc2)
        if item.cross_attention is not None
    ]

    result_cross = None
    if cross_parts:
        result_cross = (
            _clone_value(cross_parts[0])
            if len(cross_parts) == 1
            else torch.cat(cross_parts, dim=-1)
        )

    result_pooled = None
    if pooled is not None:
        result_pooled = _clone_value(pooled.pooled_output)
    elif enc2.pooled_output is not None:
        result_pooled = _clone_value(enc2.pooled_output)
    elif enc1.pooled_output is not None:
        result_pooled = _clone_value(enc1.pooled_output)

    extra_channels = {}
    extra_channels.update(deepcopy(enc1.extra_channels))
    extra_channels.update(deepcopy(enc2.extra_channels))
    if pooled is not None:
        extra_channels.update(deepcopy(pooled.extra_channels))

    return _new_output(
        source_id="assemble_result",
        cross_attention=result_cross,
        pooled_output=result_pooled,
        extra_channels=extra_channels,
        metadata=metadata,
    )


def conditioning_and(
    outputs: list[ConditioningOutput],
) -> ConditioningOutput:
    if not outputs:
        raise RuntimeError("AND requires at least 1 input")

    metadata = _copy_metadata(outputs)
    metadata["conditioning_op"] = "and"

    result_cross = _weighted_sum_channel(
        outputs,
        lambda output: output.cross_attention,
        normalize=False,
    )

    result_pooled = _weighted_sum_channel(
        outputs,
        lambda output: output.pooled_output,
        normalize=False,
    )

    extra_channels = _merge_extra_channels_weighted(
        outputs,
        normalize=False,
    )

    return _new_output(
        source_id="and_result",
        cross_attention=result_cross,
        pooled_output=result_pooled,
        extra_channels=extra_channels,
        metadata=metadata,
    )


def conditioning_sequence_concat(
    outputs: list[ConditioningOutput],
    *,
    separator: str = " ",
) -> ConditioningOutput:
    if not outputs:
        raise RuntimeError("Sequence concat requires at least 1 input")

    metadata = _copy_metadata(outputs)
    metadata["conditioning_op"] = "sequence_concat"
    metadata["separator"] = separator

    result_cross = _cat_channel(
        outputs,
        lambda output: output.cross_attention,
        dim=1,
    )

    result_pooled = _weighted_sum_channel(
        outputs,
        lambda output: output.pooled_output,
        normalize=True,
    )

    extra_channels = _merge_extra_channels_weighted(
        outputs,
        normalize=True,
    )

    return _new_output(
        source_id="sequence_concat_result",
        cross_attention=result_cross,
        pooled_output=result_pooled,
        extra_channels=extra_channels,
        metadata=metadata,
    )


def conditioning_weight(
    outputs: list[ConditioningOutput],
    *,
    weight: float = 1.0,
    weight_mode: str = "conditioning",
) -> ConditioningOutput:
    if len(outputs) != 1:
        raise RuntimeError("Weight operation requires exactly 1 input")

    source = outputs[0]
    weight = float(weight)

    metadata = dict(source.metadata)
    metadata["conditioning_op"] = "weight"
    metadata["weight"] = weight
    metadata["weight_mode"] = weight_mode

    cross = _clone_value(source.cross_attention)
    pooled = _clone_value(source.pooled_output)

    if cross is not None:
        cross = cross * weight

    if pooled is not None:
        pooled = pooled * weight

    extra_channels = {}
    for key, value in source.extra_channels.items():
        cloned = _clone_value(value)
        if torch.is_tensor(cloned):
            cloned = cloned * weight
        extra_channels[key] = cloned

    return _new_output(
        source_id="weight_result",
        cross_attention=cross,
        pooled_output=pooled,
        extra_channels=extra_channels,
        metadata=metadata,
    )