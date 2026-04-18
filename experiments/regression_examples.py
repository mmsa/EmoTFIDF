"""
Curated regression strings for V1 vs V2 evidence checks (not full paper benchmarks).

Used by ``experiments/benchmark_v1_v2_regression.py`` and pytest smoke tests.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Small corpus for fitting both systems (same as compare script, slightly expanded).
CURATED_CORPUS: List[str] = [
    "I am happy today and everything feels great and wonderful.",
    "I am not happy today and everything feels wrong and miserable.",
    "I feel sad and disappointed about the terrible news.",
    "This makes me angry and frustrated with the situation.",
    "I feel furious and angry about the problem.",
    "I am surprised and curious about what happened next.",
    "The meeting was neutral and boring without much emotion.",
]

# Each case: id, text, tags for reporting, and expected V2 behavior (lightweight contracts).
CURATED_EXAMPLES: List[Dict[str, Any]] = [
    {
        "id": "joy_clear",
        "text": "I am very happy and joyful today!",
        "tags": ["joy", "intensifier"],
        "expect_v2_primary": "joy",
        "expect_abstain": False,
    },
    {
        "id": "sadness_clear",
        "text": "She was sad and crying yesterday.",
        "tags": ["sadness", "temporal_noise"],
        "expect_v2_primary": "sadness",
        "expect_abstain": False,
        "expect_top_sadness_term_not_weak_contextual": True,
    },
    {
        "id": "anger_clear",
        "text": "I feel furious and angry about this.",
        "tags": ["anger"],
        "expect_v2_primary": "anger",
        "expect_abstain": False,
    },
    {
        "id": "negation_not_happy",
        "text": "I am not happy today.",
        "tags": ["negation", "valence_sink"],
        "expect_v2_primary": "sadness",
        "expect_abstain": False,
        "expect_no_anger_dominant": True,
    },
    {
        "id": "no_signal_punct",
        "text": "!!! ...",
        "tags": ["no_signal"],
        "expect_v2_primary": None,
        "expect_abstain": True,
    },
    {
        "id": "no_signal_digits",
        "text": "12345",
        "tags": ["no_signal"],
        "expect_v2_primary": None,
        "expect_abstain": True,
    },
    {
        "id": "mixed_evidence",
        "text": "I am happy but also nervous about the angry crowd.",
        "tags": ["mixed"],
        "expect_abstain": False,
        "expect_meaningful": True,
        "expect_min_affect_terms": 2,
    },
]
