"""
Lexeme-level priors for affect scoring and explanation ranking.

The NRC lexicon tags many **temporal / discourse** words with emotions (e.g. *yesterday* →
sadness). Those tags are lexicon-grounded but often **not** the emotional focus of a
sentence. We downweight their *mass* in the scorer and their *rank* in explanations
using a small, documented set—not sentence-specific hacks.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet

from EmoTFIDF.evidence.lexicon import inverse_count_emotion_shares

# High-frequency English tokens that are primarily temporal, spatial, or procedural
# in modern usage but still carry NRC emotion tags. Used only as a *soft* multiplier.
WEAK_CONTEXTUAL_AFFECT_LEXEMES: FrozenSet[str] = frozenset(
    {
        "yesterday",
        "today",
        "tomorrow",
        "tonight",
        "morning",
        "afternoon",
        "evening",
        "midnight",
        "week",
        "month",
        "year",
        "decade",
        "moment",
        "minute",
        "hour",
        "daily",
        "weekly",
        "monthly",
        "recently",
        "currently",
        "formerly",
        "previously",
        "later",
        "soon",
        "ago",
        "now",
        "date",
        "calendar",
        "schedule",
    }
)


def contextual_affect_multiplier(token: str) -> float:
    """
    Multiply lexicon-derived affect mass for *token*.

    Returns ``1.0`` for typical affect-bearing words, and a fraction in ``(0, 1)`` for
    tokens in :data:`WEAK_CONTEXTUAL_AFFECT_LEXEMES` so they rarely dominate sadness/joy
    over clearer affect words (*crying*, *happy*, etc.).
    """
    if token in WEAK_CONTEXTUAL_AFFECT_LEXEMES:
        return 0.22
    return 1.0


def is_weak_contextual_lexeme(token: str) -> bool:
    return token in WEAK_CONTEXTUAL_AFFECT_LEXEMES


def explanation_rank_multiplier(token: str, lexicon: Dict[str, Any]) -> float:
    """
    Downrank weak contextual tokens and uprank lexicon entries with multiple seven-way
    emotions (broader affect profile) for **explanation ordering only**.
    """
    raw = lexicon.get(token)
    if not isinstance(raw, list):
        return 1.0
    u = len(inverse_count_emotion_shares(raw))
    breadth = 0.52 + 0.16 * min(max(u, 1), 4)
    weak = 0.11 if is_weak_contextual_lexeme(token) else 1.0
    return float(breadth * weak)
