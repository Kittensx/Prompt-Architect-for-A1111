from __future__ import annotations

DEFAULT_SYMBOL_CONFIG = {
    "version": 2,
    "reserved_symbols": {
        "semantic_prompt": {
            "canonical": "SEMANTIC_PROMPT",
            "default_symbol": "%%",
            "aliases": ["%%"],
        },
    },
    "backend_operators": {
        "blend": {
            "canonical": "BLEND",
            "default_symbol": "<+>",
            "aliases": ["<+>", "BLEND"],
        },
        "chunk": {
            "canonical": "CHUNK",
            "default_symbol": "&&",
            "aliases": ["&&", "CHUNK"],
        },
        "morph": {
            "canonical": "MORPH",
            "default_symbol": ">>",
            "aliases": [">>", "MORPH"],
        },
        "pool": {
            "canonical": "POOL",
            "default_symbol": "$$",
            "aliases": ["$$", "POOL"],
        },
        "bind": {
            "canonical": "BIND",
            "default_symbol": "=>",
            "aliases": ["=>", "BIND"],
        },
        "assemble": {
            "canonical": "ASSEMBLE",
            "default_symbol": "@@",
            "aliases": ["@@", "ASSEMBLE"],
        },
    },
    "sequence_operators": {
        "sequence": {
            "canonical": "SEQUENCE",
            "default_symbol": "::",
            "aliases": ["::", "SEQUENCE"],
        },
        "deep_sequence": {
            "canonical": "DEEP_SEQUENCE",
            "default_symbol": ":::",
            "aliases": [":::", "DEEP_SEQUENCE"],
        },
        "close": {
            "canonical": "CLOSE",
            "default_symbol": "!",
            "aliases": ["!", "CLOSE"],
        },
        "top_close": {
            "canonical": "TOP_CLOSE",
            "default_symbol": "!!",
            "aliases": ["!!", "TOP_CLOSE"],
        },
    },
    "grouping_operators": {
        "open_group": {
            "canonical": "OPEN_GROUP",
            "default_symbol": "{",
            "aliases": ["{"],
        },
        "close_group": {
            "canonical": "CLOSE_GROUP",
            "default_symbol": "}",
            "aliases": ["}"],
        },
        "open_attention": {
            "canonical": "OPEN_ATTENTION",
            "default_symbol": "(",
            "aliases": ["("],
        },
        "close_attention": {
            "canonical": "CLOSE_ATTENTION",
            "default_symbol": ")",
            "aliases": [")"],
        },
        "open_schedule": {
            "canonical": "OPEN_SCHEDULE",
            "default_symbol": "[",
            "aliases": ["["],
        },
        "close_schedule": {
            "canonical": "CLOSE_SCHEDULE",
            "default_symbol": "]",
            "aliases": ["]"],
        },
    },
    "branch_operators": {
        "branch_separator": {
            "canonical": "BRANCH_SEPARATOR",
            "default_symbol": "|",
            "aliases": ["|"],
        },
        "and_operator": {
            "canonical": "AND",
            "default_symbol": "AND",
            "aliases": ["AND"],
        },
    },
    "weight_operators": {
        "weight": {
            "canonical": "WEIGHT",
            "default_symbol": "*",
            "aliases": ["*"],
        },
        "attention_weight": {
            "canonical": "ATTENTION_WEIGHT",
            "default_symbol": ":",
            "aliases": [":"],
        },
        "morph_boundary": {
            "canonical": "MORPH_BOUNDARY",
            "default_symbol": "@",
            "aliases": ["@"],
        },
    },
    "transition_operators": {
        "morph_transition": {
            "canonical": "MORPH_TRANSITION",
            "default_symbol": "=>",
            "aliases": ["=>"],
        },
    },
    "escaping": {
        "escape_character": "\\",
    },
    "serialization": {
        "serializer_mode": "canonical",
    },
    "validation": {
        "allow_alias_collisions": False,
        "allow_reserved_symbol_override": False,
        "strict_backend_keywords": False,
    },
}