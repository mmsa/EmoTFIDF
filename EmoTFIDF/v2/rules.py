"""Transparent negation and intensifier / downtoner rules (local window heuristics)."""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Tuple

# Negation cues (lowercased, match stripped tokens).
NEGATION_CUES: FrozenSet[str] = frozenset(
    {
        "not",
        "never",
        "no",
        "hardly",
        "barely",
        "cannot",
        "can't",
        "cant",
        "don't",
        "dont",
        "didn't",
        "didnt",
        "isn't",
        "isnt",
        "wasn't",
        "wasnt",
        "won't",
        "wont",
        "n't",  # rare standalone after tokenizer splits
    }
)

INTENSIFIERS_UP: FrozenSet[str] = frozenset(
    {"very", "really", "extremely", "so", "too", "highly", "incredibly", "absolutely"}
)

# Downtoners omit "barely"/"hardly" because those are modeled as negation cues above.
INTENSIFIERS_DOWN: FrozenSet[str] = frozenset(
    {"slightly", "somewhat", "little", "marginally", "comparatively"}
)


def find_negation_in_window(
    tokens: List[str],
    affect_index: int,
    window: int,
) -> Optional[Tuple[str, int]]:
    """
    Return (cue, cue_index) for the closest negation cue within *window* tokens before *affect_index*.
    """
    start = max(0, affect_index - window)
    best: Optional[Tuple[str, int]] = None
    for j in range(affect_index - 1, start - 1, -1):
        w = tokens[j]
        if w in NEGATION_CUES:
            best = (w, j)
            break
    return best


def intensifier_multiplier_in_window(
    tokens: List[str],
    affect_index: int,
    window: int,
    default: float = 1.0,
) -> Tuple[float, Optional[str], Optional[str], int]:
    """
    Combine multipliers from the closest up/down intensifiers before the affect token.

    Returns (multiplier, cue_token or None, direction 'up'|'down'|None, cue_index or -1).
    """
    start = max(0, affect_index - window)
    mult = default
    chosen: Optional[str] = None
    direction: Optional[str] = None
    cue_index = -1
    for j in range(affect_index - 1, start - 1, -1):
        w = tokens[j]
        if w in INTENSIFIERS_UP:
            mult *= 1.35
            chosen, direction, cue_index = w, "up", j
            break
        if w in INTENSIFIERS_DOWN:
            mult *= 0.72
            chosen, direction, cue_index = w, "down", j
            break
    return mult, chosen, direction, cue_index
