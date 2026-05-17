# prompt_graph_serializer.py
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

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
        ScheduleBoundary,
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
        ScheduleBoundary,
        ScheduleNode,
        SequenceNode,
        TextNode,
        WeightNode,
    )


BACKEND_KINDS = {
    PromptNodeKind.BLEND,
    PromptNodeKind.CHUNK,
    PromptNodeKind.MORPH,
    PromptNodeKind.POOL,
    PromptNodeKind.BIND,
    PromptNodeKind.ASSEMBLE,
}


def graph_to_debug_text(graph: PromptGraph) -> str:
    return node_to_debug_text(graph.root)


def node_to_debug_text(node: PromptNode, indent: int = 0) -> str:
    pad = "  " * indent
    label = node.kind.value

    if isinstance(node, TextNode):
        return f'{pad}Text({node.text!r})'

    lines = [f"{pad}{label}"]

    for child in node.children():
        lines.append(node_to_debug_text(child, indent + 1))

    return "\n".join(lines)


def graph_to_canonical_text(graph: PromptGraph) -> str:
    return node_to_canonical_text(graph.root)


def node_to_canonical_text(node: PromptNode) -> str:
    if isinstance(node, TextNode):
        return node.text

    if isinstance(node, SequenceNode):
        return _join_sequence([node_to_canonical_text(part) for part in node.parts], node.separator)

    if isinstance(node, GroupNode):
        body = _join_sequence([node_to_canonical_text(part) for part in node.parts], " ")
        if node.delimiter == "paren":
            return f"({body})"
        if node.delimiter == "bracket":
            return f"[{body}]"
        return f"{{{body}}}"

    if isinstance(node, WeightNode):
        body = node_to_canonical_text(node.node)
        if node.mode == "attention":
            return f"({body}:{_fmt_float(node.weight)})"
        return f"{body}*{_fmt_float(node.weight)}"

    if isinstance(node, AlternateNode):
        return "[" + " | ".join(node_to_canonical_text(option) for option in node.options) + "]"

    if isinstance(node, ScheduleNode):
        parts = [node_to_canonical_text(segment.node) for segment in node.segments]
        tail = ""

        last_end = node.segments[-1].end if node.segments else None
        if last_end is not None:
            tail = ":" + boundary_to_text(last_end)

        reverse = " reverse" if node.reverse else ""
        return "[" + " : ".join(parts) + tail + "]" + reverse

    if isinstance(node, AndNode):
        parts = []
        for branch in node.branches:
            text = node_to_canonical_text(branch.node)
            if abs(branch.weight - 1.0) > 1e-8:
                text = f"{text}:{_fmt_float(branch.weight)}"
            parts.append(text)
        return " AND ".join(parts)

    if isinstance(node, BlendNode):
        header = "BLEND"
        if abs(node.intensity - 1.0) > 1e-8:
            header += f"^{_fmt_float(node.intensity)}"
        if node.blend_mode != "mean" or node.channel_target != "both":
            mode = node.blend_mode
            if node.channel_target != "both":
                mode += f"@{node.channel_target}"
            header += f"[{mode}]"
        return header + "{" + _serialize_backend_branches(node.branches) + "}"

    if isinstance(node, ChunkNode):
        header = "CHUNK"
        if node.shared_channel == "pooled":
            header += "[share-pooled]"
        elif node.shared_channel == "cross":
            header += "[share-cross]"
        return header + "{" + _serialize_backend_branches(node.branches) + "}"

    if isinstance(node, MorphNode):
        header = "MORPH"
        if abs(node.intensity - 1.0) > 1e-8:
            header += f"^{_fmt_float(node.intensity)}"
        if node.channel_target != "both":
            header += f"@{node.channel_target}"
        if node.window_start is not None and node.window_end is not None:
            header += f"[{boundary_to_text(node.window_start)}-{boundary_to_text(node.window_end)}]"

        points = []
        for point in node.points:
            text = node_to_canonical_text(point.node)
            if abs(point.weight - 1.0) > 1e-8:
                text += f"*{_fmt_float(point.weight)}"
            if point.boundary is not None:
                text += f"@{boundary_to_text(point.boundary)}"
            points.append(text)

        body = " => ".join(points)
        if node.curve != "linear":
            body += f" ~ {node.curve}"

        return header + "{" + body + "}"

    if isinstance(node, PoolNode):
        return "POOL{" + node_to_canonical_text(node.node) + "}"

    if isinstance(node, BindNode):
        body = f"{node_to_canonical_text(node.owner)} => {node_to_canonical_text(node.attrs)}"
        if abs(node.weight - 1.0) > 1e-8:
            return f"BIND^{_fmt_float(node.weight)}" + "{" + body + "}"
        return "BIND{" + body + "}"

    if isinstance(node, AssembleNode):
        fields = [
            "enc1=" + node_to_canonical_text(node.enc1),
            "enc2=" + node_to_canonical_text(node.enc2),
        ]
        if node.pooled is not None:
            fields.append("pooled=" + node_to_canonical_text(node.pooled))
        return "ASSEMBLE{" + "; ".join(fields) + "}"

    raise TypeError(f"Unsupported prompt node: {type(node).__name__}")


def graph_to_json_dict(graph: PromptGraph) -> dict[str, Any]:
    return {
        "version": graph.version,
        "symbols": dict(graph.symbols),
        "diagnostics": list(graph.diagnostics),
        "root": node_to_json_dict(graph.root),
    }


def graph_to_json(graph: PromptGraph, *, indent: int = 2) -> str:
    return json.dumps(graph_to_json_dict(graph), indent=indent, ensure_ascii=False)


def node_to_json_dict(node: PromptNode) -> dict[str, Any]:
    data: dict[str, Any] = {
        "kind": node.kind.value,
        "meta": _safe_dataclass_to_dict(node.meta),
    }

    if isinstance(node, TextNode):
        data["text"] = node.text

    elif isinstance(node, SequenceNode):
        data["separator"] = node.separator
        data["parts"] = [node_to_json_dict(part) for part in node.parts]

    elif isinstance(node, GroupNode):
        data["delimiter"] = node.delimiter
        data["parts"] = [node_to_json_dict(part) for part in node.parts]

    elif isinstance(node, WeightNode):
        data["weight"] = node.weight
        data["mode"] = node.mode
        data["node"] = node_to_json_dict(node.node)

    elif isinstance(node, AlternateNode):
        data["mode"] = node.mode
        data["options"] = [node_to_json_dict(option) for option in node.options]

    elif isinstance(node, ScheduleNode):
        data["reverse"] = node.reverse
        data["segments"] = [
            {
                "node": node_to_json_dict(segment.node),
                "start": boundary_to_json(segment.start),
                "end": boundary_to_json(segment.end),
                "weight": segment.weight,
            }
            for segment in node.segments
        ]

    elif isinstance(node, AndNode):
        data["branches"] = [
            {
                "node": node_to_json_dict(branch.node),
                "weight": branch.weight,
            }
            for branch in node.branches
        ]

    elif isinstance(node, BlendNode):
        data.update(_backend_common_json(node))
        data["blend_mode"] = node.blend_mode
        data["branches"] = [backend_branch_to_json(branch) for branch in node.branches]

    elif isinstance(node, ChunkNode):
        data["shared_channel"] = node.shared_channel
        data["branches"] = [backend_branch_to_json(branch) for branch in node.branches]

    elif isinstance(node, MorphNode):
        data.update(_backend_common_json(node))
        data["curve"] = node.curve
        data["window_start"] = boundary_to_json(node.window_start)
        data["window_end"] = boundary_to_json(node.window_end)
        data["points"] = [
            {
                "node": node_to_json_dict(point.node),
                "boundary": boundary_to_json(point.boundary),
                "weight": point.weight,
            }
            for point in node.points
        ]

    elif isinstance(node, PoolNode):
        data["node"] = node_to_json_dict(node.node)

    elif isinstance(node, BindNode):
        data["owner"] = node_to_json_dict(node.owner)
        data["attrs"] = node_to_json_dict(node.attrs)
        data["weight"] = node.weight

    elif isinstance(node, AssembleNode):
        data["enc1"] = node_to_json_dict(node.enc1)
        data["enc2"] = node_to_json_dict(node.enc2)
        data["pooled"] = node_to_json_dict(node.pooled) if node.pooled is not None else None

    else:
        raise TypeError(f"Unsupported prompt node: {type(node).__name__}")

    return data


def boundary_to_text(boundary: ScheduleBoundary) -> str:
    value = _fmt_float(boundary.value)

    if boundary.kind == "percent":
        return f"{value}%"
    if boundary.kind == "fraction":
        return value
    if boundary.kind == "step":
        return value

    raise ValueError(f"Unknown boundary kind: {boundary.kind!r}")


def boundary_to_json(boundary: ScheduleBoundary | None) -> dict[str, Any] | None:
    if boundary is None:
        return None
    return {
        "value": boundary.value,
        "kind": boundary.kind,
    }


def backend_branch_to_json(branch: BackendBranch) -> dict[str, Any]:
    return {
        "node": node_to_json_dict(branch.node),
        "weight": branch.weight,
        "label": branch.label,
    }


def _serialize_backend_branches(branches: list[BackendBranch]) -> str:
    parts = []

    for branch in branches:
        text = node_to_canonical_text(branch.node)

        if abs(branch.weight - 1.0) > 1e-8:
            text += f"*{_fmt_float(branch.weight)}"

        parts.append(text)

    return " | ".join(parts)


def _backend_common_json(node: Any) -> dict[str, Any]:
    return {
        "mode": getattr(node, "mode", None),
        "channel_target": getattr(node, "channel_target", "both"),
        "intensity": getattr(node, "intensity", 1.0),
    }


def _safe_dataclass_to_dict(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value):
        return {
            key: _safe_dataclass_to_dict(item)
            for key, item in asdict(value).items()
        }

    if isinstance(value, dict):
        return {
            str(key): _safe_dataclass_to_dict(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_safe_dataclass_to_dict(item) for item in value]

    return value


def _join_sequence(parts: list[str], separator: str = " ") -> str:
    cleaned = [part.strip() for part in parts if str(part).strip()]
    if not cleaned:
        return ""

    if separator == "":
        return "".join(cleaned)

    return separator.join(cleaned)


def _fmt_float(value: float) -> str:
    value = float(value)

    if value.is_integer():
        return str(int(value))

    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"