# modules/prompt_dispatcher.py
from __future__ import annotations

"""
Backward-compatible dispatcher wrapper.

This module now forwards all functionality to:

    modules.prompt_parser_router

Reason:
    prompt_parser_router.py is now the canonical routing layer for:
    - stable prompt parser
    - prompt_parser_21
    - backend normalization
    - future graph routing/lowering

Existing integrations importing:

    modules.prompt_dispatcher

will continue working without modification.

New integrations should prefer:

    modules.prompt_parser_router
"""

from modules.prompt_parser_router import (
    ADVANCED_BACKEND_PATTERNS,
    GLOBAL_SYMBOL_CONFIG,
    ComposableScheduledPromptConditioning,
    DictWithShape,
    MulticondLearnedConditioning,
    PromptSymbolConfig,
    ScheduledPromptConditioning,
    SdConditioning,
    any_advanced_backend,
    canonicalize_prompt,
    canonicalize_prompts,
    get_learned_conditioning,
    get_learned_conditioning_prompt_schedules,
    get_multicond_learned_conditioning,
    get_multicond_prompt_list,
    load_symbol_config,
    parse_prompt_attention,
    prepare_prompts_for_parser,
    stable_preprocess_prompt_for_advanced,
    stable_text_at_final_step,
    stable_text_schedule,
    uses_advanced_backend,
)

__all__ = [
    "ADVANCED_BACKEND_PATTERNS",
    "GLOBAL_SYMBOL_CONFIG",
    "ComposableScheduledPromptConditioning",
    "DictWithShape",
    "MulticondLearnedConditioning",
    "PromptSymbolConfig",
    "ScheduledPromptConditioning",
    "SdConditioning",
    "any_advanced_backend",
    "canonicalize_prompt",
    "canonicalize_prompts",
    "get_learned_conditioning",
    "get_learned_conditioning_prompt_schedules",
    "get_multicond_learned_conditioning",
    "get_multicond_prompt_list",
    "load_symbol_config",
    "parse_prompt_attention",
    "prepare_prompts_for_parser",
    "stable_preprocess_prompt_for_advanced",
    "stable_text_at_final_step",
    "stable_text_schedule",
    "uses_advanced_backend",
]