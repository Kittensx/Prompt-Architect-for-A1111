# modules/prompt_combined_dispatcher.py
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import modules.prompt_parser as stable_parser
import modules.prompt_parser_21 as advanced_parser

from modules.prompt_symbol_interpreter import (
    PromptSymbolConfig,
    to_canonical_prompt,
)

logger = logging.getLogger(__name__)


DEFAULT_SYMBOL_CONFIG_PATH = Path(__file__).resolve().parent / "prompt_symbols.yaml"

try:
    GLOBAL_SYMBOL_CONFIG = PromptSymbolConfig.from_yaml(DEFAULT_SYMBOL_CONFIG_PATH)
except Exception:
    logger.exception("Failed to load prompt_symbols.yaml; using defaults.")
    GLOBAL_SYMBOL_CONFIG = PromptSymbolConfig.default()


ADVANCED_BACKEND_PATTERNS = (
    re.compile(r"(?<![\w\\])CHUNK(?:\s*\[[^\]]*\])?\s*\{"),
    re.compile(r"(?<![\w\\])BLEND(?:\s*\^[^\[\{]*)?(?:\s*\[[^\]]*\])?\s*\{"),
    re.compile(r"(?<![\w\\])MORPH(?:\s*\^[^\[\{@]*)?(?:\s*@[^\[\{]*)?(?:\s*\[[^\]]*\])?\s*\{"),
    re.compile(r"(?<![\w\\])ASSEMBLE\s*\{"),
    re.compile(r"(?<![\w\\])BIND(?:\s*\^[^\{]*)?\s*\{"),
    re.compile(r"(?<![\w\\])POOL\s*\{"),
)


def load_symbol_config(path: str | Path | None = None) -> PromptSymbolConfig:
    if path is None:
        return GLOBAL_SYMBOL_CONFIG
    return PromptSymbolConfig.from_yaml(path)


def canonicalize_prompt(
    prompt: str,
    symbol_config: PromptSymbolConfig | dict[str, Any] | None = None,
) -> str:
    if symbol_config is None:
        symbol_config = GLOBAL_SYMBOL_CONFIG

    return to_canonical_prompt(str(prompt), symbol_config)


def canonicalize_prompts(
    prompts,
    symbol_config: PromptSymbolConfig | dict[str, Any] | None = None,
):
    return [canonicalize_prompt(prompt, symbol_config) for prompt in prompts]


def uses_advanced_backend(prompt: str) -> bool:
    return any(pattern.search(str(prompt)) for pattern in ADVANCED_BACKEND_PATTERNS)


def any_advanced_backend(prompts) -> bool:
    return any(uses_advanced_backend(prompt) for prompt in prompts)


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

    mode="none":
        Do not run stable text preprocessing.

    mode="final":
        Run stable parser schedule expansion and pass the final scheduled text
        into _21. This is simple and safe, but it flattens stable schedules.

    Important:
        This does not pipe conditioning tensors between parsers.
        It only converts prompt text before _21 builds conditioning.
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

    raise ValueError(f"Unsupported stable preprocessing mode: {mode!r}")


def prepare_prompts_for_parser(
    prompts,
    steps: int,
    hires_steps=None,
    use_old_scheduling: bool = False,
    symbol_config: PromptSymbolConfig | dict[str, Any] | None = None,
    stable_preprocess_mode: str = "final",
):
    canonical_prompts = canonicalize_prompts(prompts, symbol_config)
    use_advanced = any_advanced_backend(canonical_prompts)

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


def get_learned_conditioning_prompt_schedules(
    prompts,
    base_steps,
    hires_steps=None,
    use_old_scheduling=False,
    symbol_config: PromptSymbolConfig | dict[str, Any] | None = None,
    prefer_advanced: bool | None = None,
    stable_preprocess_mode: str = "final",
):
    canonical_prompts = canonicalize_prompts(prompts, symbol_config)

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
    symbol_config: PromptSymbolConfig | dict[str, Any] | None = None,
    prefer_advanced: bool | None = None,
    fallback_to_stable: bool = True,
    stable_preprocess_mode: str = "final",
):
    canonical_prompts = canonicalize_prompts(prompts, symbol_config)

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
            logger.exception("Combined advanced parser failed; falling back to stable parser.")

    return stable_parser.get_learned_conditioning(
        model,
        canonical_prompts,
        steps,
        hires_steps=hires_steps,
        use_old_scheduling=use_old_scheduling,
    )


def get_multicond_prompt_list(
    prompts,
    symbol_config: PromptSymbolConfig | dict[str, Any] | None = None,
    prefer_advanced: bool | None = None,
):
    canonical_prompts = canonicalize_prompts(prompts, symbol_config)

    use_advanced = (
        any_advanced_backend(canonical_prompts)
        if prefer_advanced is None
        else bool(prefer_advanced)
    )

    parser = advanced_parser if use_advanced else stable_parser
    return parser.get_multicond_prompt_list(canonical_prompts)


def get_multicond_learned_conditioning(
    model,
    prompts,
    steps,
    hires_steps=None,
    use_old_scheduling=False,
    symbol_config: PromptSymbolConfig | dict[str, Any] | None = None,
    prefer_advanced: bool | None = None,
    fallback_to_stable: bool = True,
    stable_preprocess_mode: str = "final",
):
    canonical_prompts = canonicalize_prompts(prompts, symbol_config)

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
            logger.exception("Combined advanced multicond parser failed; falling back to stable parser.")

    return stable_parser.get_multicond_learned_conditioning(
        model,
        canonical_prompts,
        steps,
        hires_steps=hires_steps,
        use_old_scheduling=use_old_scheduling,
    )


def parse_prompt_attention(
    text: str,
    symbol_config: PromptSymbolConfig | dict[str, Any] | None = None,
):
    canonical = canonicalize_prompt(text, symbol_config)

    # Keep A1111 CLIP token emphasis conservative.
    # Advanced backend syntax should be handled by conditioning paths, not CLIP tokenization.
    return stable_parser.parse_prompt_attention(canonical)


def reconstruct_cond_batch(c, current_step):
    if _looks_like_advanced_conditioning(c):
        return advanced_parser.reconstruct_cond_batch(c, current_step)

    return stable_parser.reconstruct_cond_batch(c, current_step)


def reconstruct_multicond_batch(c, current_step):
    if _looks_like_advanced_multicond(c):
        return advanced_parser.reconstruct_multicond_batch(c, current_step)

    return stable_parser.reconstruct_multicond_batch(c, current_step)


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
        )
    )


# Re-export stable shared classes/helpers so old call sites continue working.
ScheduledPromptConditioning = stable_parser.ScheduledPromptConditioning
SdConditioning = stable_parser.SdConditioning
ComposableScheduledPromptConditioning = stable_parser.ComposableScheduledPromptConditioning
MulticondLearnedConditioning = stable_parser.MulticondLearnedConditioning
DictWithShape = stable_parser.DictWithShape
stack_conds = stable_parser.stack_conds