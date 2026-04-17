"""Lexicon loading and emotion filtering aligned with V1 label set."""

from __future__ import annotations

import json
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
