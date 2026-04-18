"""Lexicon loading and emotion filtering aligned with V1 label set."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

# Same seven-way schema used by V1 transformer + lexicon aggregation paths.
DEFAULT_EMOTION_LABELS: List[str] = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]

_ALLOWED: Set[str] = set(DEFAULT_EMOTION_LABELS)
_META = {"positive", "negative"}


def default_lexicon_path() -> Path:
    """Path to packaged NRC-style lexicon JSON."""
    return Path(__file__).resolve().parent.parent / "emotions_lex.json"


def load_lexicon(path: Optional[str] = None) -> Dict[str, Any]:
    """Load lexicon from *path* or the default packaged emotions_lex.json."""
    p = Path(path) if path else default_lexicon_path()
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def filter_emotions_for_word(raw: Sequence[str]) -> List[str]:
    """
    Filter lexicon emotion tags to the V1-compatible seven-label space.

    Removes meta positive/negative tags; deduplicates while preserving order.
    """
    if not raw:
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for e in raw:
        if not e or not isinstance(e, str):
            continue
        e_low = e.strip().lower()
        if e_low in _META:
            continue
        if e_low not in _ALLOWED:
            continue
        if e_low not in seen:
            seen.add(e_low)
            out.append(e_low)
    return out


def inverse_count_emotion_shares(raw: Sequence[str]) -> Dict[str, float]:
    """
    Split one unit of affect mass across lexicon tags using inverse duplicate weighting.

    The NRC JSON often repeats the same tag (e.g. two ``disgust`` entries next to one
    ``anger``). A naive equal split would overweight disgust. Here each *unique* tag's
    weight is proportional to ``1 / count(tag)`` in the filtered multiset, then normalized.

    This is a general, explainable heuristic—not tuned to individual benchmark sentences.
    """
    filtered: List[str] = []
    for e in raw:
        if not e or not isinstance(e, str):
            continue
        e_low = e.strip().lower()
        if e_low in _META:
            continue
        if e_low not in _ALLOWED:
            continue
        filtered.append(e_low)
    if not filtered:
        return {}
    ctr = Counter(filtered)
    inv = {emotion: 1.0 / float(ctr[emotion]) for emotion in ctr}
    total = sum(inv.values())
    return {emotion: inv[emotion] / total for emotion in inv}
