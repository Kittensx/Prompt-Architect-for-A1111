# prompt_graph_executor.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from modules.prompt_execution_plan import (
        ConditioningOutput,
        ExecutionResult,
        ExecutionTrace,
        ExecutionStage,
        MergeMode,
        PlanOperation,
        PlanReference,
        PromptCall,
        PromptCallKind,
        PromptExecutionPlan,
    )
except ImportError:
    from prompt_execution_plan import (
        ConditioningOutput,
        ExecutionResult,
        ExecutionTrace,
        ExecutionStage,
        MergeMode,
        PlanOperation,
        PlanReference,
        PromptCall,
        PromptCallKind,
        PromptExecutionPlan,
    )

try:
    from modules.prompt_conditioning_ops import (
        conditioning_and,
        conditioning_assemble,
        conditioning_bind,
        conditioning_blend,
        conditioning_chunk,
        conditioning_morph,
        conditioning_pool,
        conditioning_sequence_concat,
        conditioning_weight,
    )
except ImportError:
    from prompt_conditioning_ops import (
        conditioning_and,
        conditioning_assemble,
        conditioning_bind,
        conditioning_blend,
        conditioning_chunk,
        conditioning_morph,
        conditioning_pool,
        conditioning_sequence_concat,
        conditioning_weight,
    )


# ============================================================================
# BACKEND INTERFACE
# ============================================================================


class PromptBackend:
    """
    Abstract backend interface.

    Adapters:
        _21Backend
        A1111Backend
        ComfyBackend
        SDNextBackend
        etc.
    """

    backend_name = "base"

    def condition_prompt(
        self,
        prompt: str,
        *,
        negative_prompt: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ConditioningOutput:
        raise NotImplementedError


# ============================================================================
# EXECUTION CONTEXT
# ============================================================================


@dataclass(slots=True)
class ExecutionContext:
    """
    Runtime state.
    """

    backend: PromptBackend

    trace: ExecutionTrace = field(default_factory=ExecutionTrace)

    results: dict[str, ExecutionResult] = field(default_factory=dict)

    cache: dict[str, ExecutionResult] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    enable_cache: bool = True

    strict_mode: bool = True


# ============================================================================
# EXECUTOR
# ============================================================================


class PromptGraphExecutor:
    """
    Executes lowered execution plans.

    Responsibilities:
    - execute primitive prompt calls
    - execute graph merge ops
    - cache reuse
    - dependency resolution
    - execution tracing
    """

    def __init__(
        self,
        backend: PromptBackend,
        *,
        enable_cache: bool = True,
        strict_mode: bool = True,
    ) -> None:

        self.ctx = ExecutionContext(
            backend=backend,
            enable_cache=enable_cache,
            strict_mode=strict_mode,
        )

    # ---------------------------------------------------------------------
    # PUBLIC
    # ---------------------------------------------------------------------

    def execute_plan(
        self,
        plan: PromptExecutionPlan,
    ) -> ExecutionResult:

        self.ctx.trace.add_event(
            source_id="plan",
            stage=ExecutionStage.CONDITION,
            message="Starting execution plan",
        )

        validation_errors = plan.validate()

        if validation_errors:
            raise RuntimeError(
                "Invalid execution plan:\n"
                + "\n".join(validation_errors)
            )

        # Build dependency graph if needed
        if not plan.execution_nodes:
            plan.build_execution_nodes()

        completed = set()

        while len(completed) < len(plan.execution_nodes):

            progress = False

            for node in plan.execution_nodes:

                if node.node_id in completed:
                    continue

                if not self._dependencies_satisfied(
                    node.dependencies,
                    completed,
                ):
                    continue

                self._execute_node(node)

                completed.add(node.node_id)

                progress = True

            if not progress:
                unresolved = [
                    node.node_id
                    for node in plan.execution_nodes
                    if node.node_id not in completed
                ]

                raise RuntimeError(
                    "Execution deadlock. Remaining nodes:\n"
                    + "\n".join(unresolved)
                )

        output = self.ctx.results.get(plan.output_id)

        if output is None:
            raise RuntimeError(
                f"Missing final output: {plan.output_id!r}"
            )

        self.ctx.trace.add_event(
            source_id=plan.output_id,
            stage=ExecutionStage.POST_PROCESS,
            message="Execution completed successfully",
        )

        return output

    # ---------------------------------------------------------------------
    # NODE EXECUTION
    # ---------------------------------------------------------------------

    def _execute_node(self, node) -> None:

        payload = node.payload

        try:

            if isinstance(payload, PromptCall):
                result = self._execute_prompt_call(payload)

            elif isinstance(payload, PlanOperation):
                result = self._execute_operation(payload)

            else:
                raise TypeError(
                    f"Unsupported execution payload: "
                    f"{type(payload).__name__}"
                )

            self.ctx.results[node.node_id] = result

            node.completed = True

        except Exception as exc:

            self.ctx.trace.add_error(
                f"{node.node_id}: {exc}"
            )

            if self.ctx.strict_mode:
                raise

            self.ctx.results[node.node_id] = ExecutionResult(
                result_id=node.node_id,
                success=False,
                error=str(exc),
            )

    # ---------------------------------------------------------------------
    # PROMPT CALLS
    # ---------------------------------------------------------------------

    def _execute_prompt_call(
        self,
        call: PromptCall,
    ) -> ExecutionResult:

        cache_key = self._build_cache_key(call)

        if self.ctx.enable_cache and cache_key in self.ctx.cache:

            cached = self.ctx.cache[cache_key]

            self.ctx.trace.add_event(
                source_id=call.call_id,
                stage=ExecutionStage.CONDITION,
                message="Cache hit",
            )

            return cached

        self.ctx.trace.add_event(
            source_id=call.call_id,
            stage=ExecutionStage.CONDITION,
            message=f"Conditioning prompt ({call.kind.value})",
            metadata={
                "prompt": call.prompt,
            },
        )

        conditioning = self.ctx.backend.condition_prompt(
            call.prompt,
            negative_prompt=call.negative_prompt,
            metadata=call.metadata,
        )

        conditioning.source_id = call.call_id

        conditioning.metadata.setdefault(
            "prompt",
            call.prompt,
        )

        conditioning.metadata.setdefault(
            "prompt_call_kind",
            call.kind.value,
        )

        conditioning.metadata.setdefault(
            "backend_name",
            self.ctx.backend.backend_name,
        )

        result = ExecutionResult(
            result_id=call.call_id,
            conditioning=conditioning,
            success=True,
            metadata={
                "backend": self.ctx.backend.backend_name,
                "prompt_kind": call.kind.value,
            },
        )

        if self.ctx.enable_cache and call.cacheable:
            self.ctx.cache[cache_key] = result

        return result

    # ---------------------------------------------------------------------
    # OPS
    # ---------------------------------------------------------------------

    def _execute_operation(
        self,
        op: PlanOperation,
    ) -> ExecutionResult:

        self.ctx.trace.add_event(
            source_id=op.op_id,
            stage=ExecutionStage.MERGE,
            message=f"Executing op: {op.mode.value}",
        )

        inputs = self._resolve_inputs(op.inputs)

        if op.mode == MergeMode.BLEND:
            conditioning = conditioning_blend(
                inputs,
                **op.params,
            )

        elif op.mode == MergeMode.CHUNK:
            conditioning = conditioning_chunk(
                inputs,
                **op.params,
            )

        elif op.mode == MergeMode.MORPH:
            conditioning = conditioning_morph(
                inputs,
                **op.params,
            )

        elif op.mode == MergeMode.POOL:
            conditioning = conditioning_pool(
                inputs,
                **op.params,
            )

        elif op.mode == MergeMode.BIND:
            conditioning = conditioning_bind(
                inputs,
                **op.params,
            )

        elif op.mode == MergeMode.ASSEMBLE:
            conditioning = conditioning_assemble(
                inputs,
                **op.params,
            )

        elif op.mode == MergeMode.AND:
            conditioning = conditioning_and(
                inputs,
                **op.params,
            )

        elif op.mode == MergeMode.SEQUENCE_CONCAT:
            conditioning = conditioning_sequence_concat(
                inputs,
                **op.params,
            )

        elif op.mode == MergeMode.WEIGHT:
            conditioning = conditioning_weight(
                inputs,
                **op.params,
            )

        else:
            raise RuntimeError(
                f"Unsupported merge op: {op.mode!r}"
            )

        conditioning.source_id = op.op_id

        conditioning.metadata.setdefault(
            "merge_mode",
            op.mode.value,
        )

        conditioning.metadata.update(op.metadata)

        return ExecutionResult(
            result_id=op.op_id,
            conditioning=conditioning,
            success=True,
            metadata={
                "op_mode": op.mode.value,
                **op.metadata,
            },
        )

    # ---------------------------------------------------------------------
    # INPUT RESOLUTION
    # ---------------------------------------------------------------------

    def _resolve_inputs(
        self,
        refs: list[PlanReference],
    ) -> list[ConditioningOutput]:

        outputs = []

        for ref in refs:
            if not ref.enabled:
                continue

            result = self.ctx.results.get(ref.source_id)

            if result is None:
                raise RuntimeError(
                    f"Missing dependency result: "
                    f"{ref.source_id!r}"
                )

            if not result.success:
                raise RuntimeError(
                    f"Dependency failed: "
                    f"{ref.source_id!r}"
                )
            
            conditioning = result.conditioning

            if conditioning is None:
                raise RuntimeError(
                    f"Missing conditioning output: "
                    f"{ref.source_id!r}"
                )

            metadata = dict(conditioning.metadata)

            metadata["graph_weight"] = ref.weight

            if ref.label is not None:
                metadata["graph_label"] = ref.label

            outputs.append(
                ConditioningOutput(
                    source_id=conditioning.source_id,
                    cross_attention=conditioning.cross_attention,
                    pooled_output=conditioning.pooled_output,
                    extra_channels=dict(conditioning.extra_channels),
                    metadata=metadata,
                )
            )

        return outputs

    # ---------------------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------------------

    def _dependencies_satisfied(
        self,
        dependencies: list[str],
        completed: set[str],
    ) -> bool:

        for dep in dependencies:
            if dep not in completed:
                return False

        return True

    def _build_cache_key(
        self,
        call: PromptCall,
    ) -> str:

        return "|".join(
            [
                str(self.ctx.backend.backend_name),
                str(call.kind.value),
                str(call.negative_prompt),
                str(call.prompt),
                str(call.steps),
                str(call.width),
                str(call.height),
                repr(sorted(call.metadata.items())),
            ]
        )


# ============================================================================
# SIMPLE _21 BACKEND ADAPTER
# ============================================================================


class Prompt21Backend(PromptBackend):
    """
    Thin adapter around _21.

    This is intentionally small.

    _21 becomes:
        a primitive conditioning provider

    NOT:
        the graph compiler.
    """

    backend_name = "_21"

    def __init__(
        self,
        condition_fn,
    ) -> None:
        """
        Parameters
        ----------
        condition_fn:
            Callable that performs actual _21 conditioning.

        Expected signature:

            condition_fn(prompt: str, negative_prompt=False)
                -> ConditioningOutput-compatible payload
        """

        self.condition_fn = condition_fn

    def condition_prompt(
        self,
        prompt: str,
        *,
        negative_prompt: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ConditioningOutput:

        result = self.condition_fn(
            prompt,
            negative_prompt=negative_prompt,
        )

        # already normalized
        if isinstance(result, ConditioningOutput):
            return result

        # tuple style
        if isinstance(result, tuple):

            cross_attention = None
            pooled_output = None

            if len(result) >= 1:
                cross_attention = result[0]

            if len(result) >= 2:
                pooled_output = result[1]

            return ConditioningOutput(
                source_id="backend_result",
                cross_attention=cross_attention,
                pooled_output=pooled_output,
            )

        # dict style
        if isinstance(result, dict):

            return ConditioningOutput(
                source_id="backend_result",
                cross_attention=result.get("cross_attention"),
                pooled_output=result.get("pooled_output"),
                extra_channels=result.get(
                    "extra_channels",
                    {},
                ),
                metadata=result.get(
                    "metadata",
                    {},
                ),
            )

        # raw tensor fallback
        return ConditioningOutput(
            source_id="backend_result",
            cross_attention=result,
        )