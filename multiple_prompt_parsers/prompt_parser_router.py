# modules/prompt_parser_router.py
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import modules.prompt_parser as stable_parser
import modules.prompt_parser_21 as advanced_parser

from modules.prompt_backend_normalizer import (
    normalize_backend_prompt,
)

from modules.prompt_symbol_interpreter import (
    PromptSymbolConfig,
    to_canonical_prompt,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_SYMBOL_CONFIG_PATH = (
    Path(__file__).resolve().parent / "prompt_symbols.yaml"
)

try:
    GLOBAL_SYMBOL_CONFIG = PromptSymbolConfig.from_yaml(
        DEFAULT_SYMBOL_CONFIG_PATH
    )
except Exception:
    logger.exception(
        "Failed to load prompt_symbols.yaml; using defaults."
    )
    GLOBAL_SYMBOL_CONFIG = PromptSymbolConfig.default()


# ============================================================================
# ADVANCED BACKEND DETECTION
# ============================================================================

ADVANCED_BACKEND_PATTERNS = (
    re.compile(
        r"(?<![\w\\])CHUNK(?:\s*\[[^\]]*\])?\s*\{"
    ),
    re.compile(
        r"(?<![\w\\])BLEND(?:\s*\^[^\[\{]*)?(?:\s*\[[^\]]*\])?\s*\{"
    ),
    re.compile(
        r"(?<![\w\\])MORPH(?:\s*\^[^\[\{@]*)?(?:\s*@[^\[\{]*)?(?:\s*\[[^\]]*\])?\s*\{"
    ),
    re.compile(
        r"(?<![\w\\])ASSEMBLE\s*\{"
    ),
    re.compile(
        r"(?<![\w\\])BIND(?:\s*\^[^\{]*)?\s*\{"
    ),
    re.compile(
        r"(?<![\w\\])POOL\s*\{"
    ),
)


# ============================================================================
# CONFIG HELPERS
# ============================================================================

def load_symbol_config(
    path: str | Path | None = None,
) -> PromptSymbolConfig:

    if path is None:
        return GLOBAL_SYMBOL_CONFIG

    return PromptSymbolConfig.from_yaml(path)


# ============================================================================
# CANONICALIZATION
# ============================================================================

def canonicalize_prompt(
    prompt: str,
    symbol_config: (
        PromptSymbolConfig
        | dict[str, Any]
        | None
    ) = None,
    *,
    normalize_backend: bool = True,
) -> str:

    if symbol_config is None:
        symbol_config = GLOBAL_SYMBOL_CONFIG

    # -------------------------------------------------------------
    # symbol aliases -> canonical backend syntax
    # -------------------------------------------------------------

    canonical = to_canonical_prompt(
        str(prompt),
        symbol_config,
    )

    # -------------------------------------------------------------
    # lightweight backend lifting / normalization
    # -------------------------------------------------------------

    if normalize_backend:
        canonical = normalize_backend_prompt(
            canonical
        )

    return canonical


def canonicalize_prompts(
    prompts,
    symbol_config: (
        PromptSymbolConfig
        | dict[str, Any]
        | None
    ) = None,
    *,
    normalize_backend: bool = True,
):

    return [
        canonicalize_prompt(
            prompt,
            symbol_config,
            normalize_backend=normalize_backend,
        )
        for prompt in prompts
    ]


# ============================================================================
# BACKEND ROUTING
# ============================================================================

def uses_advanced_backend(
    prompt: str,
) -> bool:

    return any(
        pattern.search(str(prompt))
        for pattern in ADVANCED_BACKEND_PATTERNS
    )


def any_advanced_backend(
    prompts,
) -> bool:

    return any(
        uses_advanced_backend(prompt)
        for prompt in prompts
    )


# ============================================================================
# STABLE PREPROCESSING
# ============================================================================

def stable_text_schedule(
    prompt: str,
    steps: int,
    hires_steps=None,
    use_old_scheduling: bool = False,
):

    return stable_parser.get_learned_conditioning_prompt_schedules(
        [prompt],
        steps,
        hires_steps=hires_steps,
        use_old_scheduling=use_old_scheduling,
    )[0]


def stable_text_at_final_step(
    prompt: str,
    steps: int,
    hires_steps=None,
    use_old_scheduling: bool = False,
) -> str:

    schedule = stable_text_schedule(
        prompt,
        steps,
        hires_steps=hires_steps,
        use_old_scheduling=use_old_scheduling,
    )

    if not schedule:
        return prompt

    return str(schedule[-1][1])


def stable_preprocess_prompt_for_advanced(
    prompt: str,
    steps: int,
    hires_steps=None,
    use_old_scheduling: bool = False,
    mode: str = "final",
) -> str:
    """
    Preprocess stable-parser syntax before sending to _21.

    mode="none"
        No stable preprocessing.

    mode="final"
        Expand stable scheduling and send final text to _21.

    Future:
        graph
        layered
        schedule-preserving
    """

    if mode == "none":
        return prompt

    if mode == "final":

        return stable_text_at_final_step(
            prompt,
            steps,
            hires_steps=hires_steps,
            use_old_scheduling=use_old_scheduling,
        )

    raise ValueError(
        f"Unsupported stable preprocessing mode: {mode!r}"
    )


# ============================================================================
# PREPARE
# ============================================================================

def prepare_prompts_for_parser(
    prompts,
    steps: int,
    hires_steps=None,
    use_old_scheduling: bool = False,
    symbol_config: (
        PromptSymbolConfig
        | dict[str, Any]
        | None
    ) = None,
    stable_preprocess_mode: str = "final",
):

    canonical_prompts = canonicalize_prompts(
        prompts,
        symbol_config,
    )

    use_advanced = any_advanced_backend(
        canonical_prompts
    )

    if not use_advanced:
        return canonical_prompts, False

    preprocessed = [
        stable_preprocess_prompt_for_advanced(
            prompt,
            steps,
            hires_steps=hires_steps,
            use_old_scheduling=use_old_scheduling,
            mode=stable_preprocess_mode,
        )
        for prompt in canonical_prompts
    ]

    return preprocessed, True


# ============================================================================
# MAIN ROUTING API
# ============================================================================

def get_learned_conditioning_prompt_schedules(
    prompts,
    base_steps,
    hires_steps=None,
    use_old_scheduling=False,
    symbol_config: (
        PromptSymbolConfig
        | dict[str, Any]
        | None
    ) = None,
    prefer_advanced: bool | None = None,
    stable_preprocess_mode: str = "final",
):

    canonical_prompts = canonicalize_prompts(
        prompts,
        symbol_config,
    )

    use_advanced = (
        any_advanced_backend(canonical_prompts)
        if prefer_advanced is None
        else bool(prefer_advanced)
    )

    if use_advanced:

        advanced_prompts = [
            stable_preprocess_prompt_for_advanced(
                prompt,
                base_steps,
                hires_steps=hires_steps,
                use_old_scheduling=use_old_scheduling,
                mode=stable_preprocess_mode,
            )
            for prompt in canonical_prompts
        ]

        return advanced_parser.get_learned_conditioning_prompt_schedules(
            advanced_prompts,
            base_steps,
            hires_steps=hires_steps,
            use_old_scheduling=use_old_scheduling,
        )

    return stable_parser.get_learned_conditioning_prompt_schedules(
        canonical_prompts,
        base_steps,
        hires_steps=hires_steps,
        use_old_scheduling=use_old_scheduling,
    )


def get_learned_conditioning(
    model,
    prompts,
    steps,
    hires_steps=None,
    use_old_scheduling=False,
    symbol_config: (
        PromptSymbolConfig
        | dict[str, Any]
        | None
    ) = None,
    prefer_advanced: bool | None = None,
    fallback_to_stable: bool = True,
    stable_preprocess_mode: str = "final",
):

    canonical_prompts = canonicalize_prompts(
        prompts,
        symbol_config,
    )

    use_advanced = (
        any_advanced_backend(canonical_prompts)
        if prefer_advanced is None
        else bool(prefer_advanced)
    )

    if use_advanced:

        advanced_prompts = [
            stable_preprocess_prompt_for_advanced(
                prompt,
                steps,
                hires_steps=hires_steps,
                use_old_scheduling=use_old_scheduling,
                mode=stable_preprocess_mode,
            )
            for prompt in canonical_prompts
        ]

        try:

            return advanced_parser.get_learned_conditioning(
                model,
                advanced_prompts,
                steps,
                hires_steps=hires_steps,
                use_old_scheduling=use_old_scheduling,
            )

        except Exception:

            if not fallback_to_stable:
                raise

            logger.exception(
                "Advanced parser failed; "
                "falling back to stable parser."
            )

    return stable_parser.get_learned_conditioning(
        model,
        canonical_prompts,
        steps,
        hires_steps=hires_steps,
        use_old_scheduling=use_old_scheduling,
    )


def get_multicond_learned_conditioning(
    model,
    prompts,
    steps,
    hires_steps=None,
    use_old_scheduling=False,
    symbol_config: (
        PromptSymbolConfig
        | dict[str, Any]
        | None
    ) = None,
    prefer_advanced: bool | None = None,
    fallback_to_stable: bool = True,
    stable_preprocess_mode: str = "final",
):

    canonical_prompts = canonicalize_prompts(
        prompts,
        symbol_config,
    )

    use_advanced = (
        any_advanced_backend(canonical_prompts)
        if prefer_advanced is None
        else bool(prefer_advanced)
    )

    if use_advanced:

        advanced_prompts = [
            stable_preprocess_prompt_for_advanced(
                prompt,
                steps,
                hires_steps=hires_steps,
                use_old_scheduling=use_old_scheduling,
                mode=stable_preprocess_mode,
            )
            for prompt in canonical_prompts
        ]

        try:

            return advanced_parser.get_multicond_learned_conditioning(
                model,
                advanced_prompts,
                steps,
                hires_steps=hires_steps,
                use_old_scheduling=use_old_scheduling,
            )

        except Exception:

            if not fallback_to_stable:
                raise

            logger.exception(
                "Advanced multicond parser failed; "
                "falling back to stable parser."
            )

    return stable_parser.get_multicond_learned_conditioning(
        model,
        canonical_prompts,
        steps,
        hires_steps=hires_steps,
        use_old_scheduling=use_old_scheduling,
    )


# ============================================================================
# SIMPLE UTILITIES
# ============================================================================

def parse_prompt_attention(
    text: str,
    symbol_config: (
        PromptSymbolConfig
        | dict[str, Any]
        | None
    ) = None,
):

    canonical = canonicalize_prompt(
        text,
        symbol_config,
    )

    return stable_parser.parse_prompt_attention(
        canonical
    )


def get_multicond_prompt_list(
    prompts,
    symbol_config=None,
    prefer_advanced: bool | None = None,
):

    canonical_prompts = canonicalize_prompts(
        prompts,
        symbol_config,
    )

    use_advanced = (
        any_advanced_backend(canonical_prompts)
        if prefer_advanced is None
        else bool(prefer_advanced)
    )

    parser = (
        advanced_parser
        if use_advanced
        else stable_parser
    )

    return parser.get_multicond_prompt_list(
        canonical_prompts
    )


# ============================================================================
# RECONSTRUCTION HELPERS
# ============================================================================

def reconstruct_cond_batch(c, current_step):

    if _looks_like_advanced_conditioning(c):
        return advanced_parser.reconstruct_cond_batch(
            c,
            current_step,
        )

    return stable_parser.reconstruct_cond_batch(
        c,
        current_step,
    )


def reconstruct_multicond_batch(c, current_step):

    if _looks_like_advanced_multicond(c):
        return advanced_parser.reconstruct_multicond_batch(
            c,
            current_step,
        )

    return stable_parser.reconstruct_multicond_batch(
        c,
        current_step,
    )


def _looks_like_advanced_multicond(c) -> bool:

    if c is None:
        return False

    if c.__class__.__module__ == advanced_parser.__name__:
        return True

    batch = getattr(c, "batch", None)

    if not batch:
        return False

    text = repr(type(c)) + " " + repr(batch[:1])

    return any(
        marker in text
        for marker in (
            "Chunk",
            "Blend",
            "Morph",
            "Pool",
            "Bind",
            "Assemble",
            "Backend",
            "Graph",
            "ExecutionPlan",
            "ConditioningOutput",
        )
    )


def _looks_like_advanced_conditioning(c) -> bool:

    if c is None:
        return False

    if c.__class__.__module__ == advanced_parser.__name__:
        return True

    preview = c[:1] if isinstance(c, list) else c

    text = repr(type(c)) + " " + repr(preview)

    return any(
        marker in text
        for marker in (
            "Chunk",
            "Blend",
            "Morph",
            "Pool",
            "Bind",
            "Assemble",
            "Backend",
            "Graph",
            "ExecutionPlan",
            "ConditioningOutput",
        )
    )
# ============================================================================
# Helper Functions
# ============================================================================
def may_need_graph_pipeline(prompt: str) -> bool:
    """
    Conservative detector for cases _21 may reject.

    This is intentionally only a hint. The real decision should come from
    graph parse/normalize/validate/lower.
    """
    text = str(prompt)
    return bool(NESTED_BACKEND_PATTERN.search(text))


def any_may_need_graph_pipeline(prompts) -> bool:
    return any(may_need_graph_pipeline(prompt) for prompt in prompts)
    
def build_prompt_graph(prompt: str):
    from modules.prompt_graph_parser import parse_prompt_graph
    from modules.prompt_graph_normalizer import normalize_graph
    from modules.prompt_graph_validator import validate_prompt_graph

    graph = parse_prompt_graph(prompt)
    graph = normalize_graph(graph)

    validation = validate_prompt_graph(
        graph,
        backend_name="_21",
        strict_backend_rules=False,
    )
    validation.raise_if_errors()

    return graph

def lower_prompt_graph(prompt: str):
    from modules.prompt_graph_lowering import lower_graph

    graph = build_prompt_graph(prompt)
    return lower_graph(graph, backend_name="_21")

# ============================================================================
# RE-EXPORTS
# ============================================================================

ScheduledPromptConditioning = (
    stable_parser.ScheduledPromptConditioning
)

SdConditioning = (
    stable_parser.SdConditioning
)

ComposableScheduledPromptConditioning = (
    stable_parser.ComposableScheduledPromptConditioning
)

MulticondLearnedConditioning = (
    stable_parser.MulticondLearnedConditioning
)

DictWithShape = getattr(stable_parser, "Dict", dict)

stack_conds = stable_parser.stack_conds




__all__ = [
    "ADVANCED_BACKEND_PATTERNS",
    "GLOBAL_SYMBOL_CONFIG",
    "PromptSymbolConfig",
    "ScheduledPromptConditioning",
    "SdConditioning",
    "ComposableScheduledPromptConditioning",
    "MulticondLearnedConditioning",
    "DictWithShape",
    "stack_conds",
    "load_symbol_config",
    "canonicalize_prompt",
    "canonicalize_prompts",
    "uses_advanced_backend",
    "any_advanced_backend",
    "may_need_graph_pipeline",
    "any_may_need_graph_pipeline",
    "stable_text_schedule",
    "stable_text_at_final_step",
    "stable_preprocess_prompt_for_advanced",
    "build_prompt_graph",
    "lower_prompt_graph",
    "prepare_prompts_for_parser",
    "get_learned_conditioning_prompt_schedules",
    "get_learned_conditioning",
    "get_multicond_learned_conditioning",
    "parse_prompt_attention",
    "get_multicond_prompt_list",
    "reconstruct_cond_batch",
    "reconstruct_multicond_batch",
]