"""
Classical baselines: NRC lexicon counting and TF-IDF + logistic regression.

The lexicon baseline aggregates NRC Emotion Lexicon associations (via the
``nrclex`` helper) into the **same** target label space as the GoEmotions
slice by using a fixed many-to-one affinity table from each dataset class to
underlying NRC dimensions.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from nrclex import NRCLex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# NRCLex / NRC Emotion Lexicon primary affect channels used as an 8-d basis.
NRC_KEYS: List[str] = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]


def _unit(vec: List[float]) -> np.ndarray:
    v = np.array(vec, dtype=np.float64)
    s = v.sum()
    if s <= 0:
        return np.ones(len(vec), dtype=np.float64) / len(vec)
    return v / s


# Hand-crafted affinity: each GoEmotions simplified label -> weights over NRC_KEYS.
# Rows need not sum to one; they are normalized at class matrix construction time.
_GO_EMOTION_NRC_AFFINITY: Dict[str, List[float]] = {
    "neutral": [0.2, 0.1, 0.05, 0.1, 0.15, 0.1, 0.15, 0.15],
    "admiration": [0.0, 0.15, 0.0, 0.0, 0.45, 0.0, 0.0, 0.4],
    "amusement": [0.0, 0.1, 0.0, 0.0, 0.65, 0.0, 0.25, 0.0],
    "anger": [0.85, 0.05, 0.05, 0.0, 0.0, 0.05, 0.0, 0.0],
    "annoyance": [0.65, 0.1, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
    "approval": [0.0, 0.2, 0.0, 0.0, 0.35, 0.0, 0.0, 0.45],
    "caring": [0.0, 0.1, 0.0, 0.05, 0.45, 0.05, 0.0, 0.35],
    "confusion": [0.05, 0.15, 0.0, 0.15, 0.0, 0.05, 0.45, 0.15],
    "curiosity": [0.0, 0.45, 0.0, 0.05, 0.15, 0.0, 0.25, 0.1],
    "desire": [0.0, 0.55, 0.0, 0.0, 0.35, 0.0, 0.0, 0.1],
    "disappointment": [0.1, 0.0, 0.05, 0.1, 0.0, 0.65, 0.0, 0.1],
    "disapproval": [0.45, 0.0, 0.25, 0.0, 0.0, 0.15, 0.0, 0.15],
    "disgust": [0.15, 0.0, 0.75, 0.0, 0.0, 0.1, 0.0, 0.0],
    "embarrassment": [0.05, 0.0, 0.1, 0.45, 0.0, 0.25, 0.15, 0.0],
    "excitement": [0.0, 0.35, 0.0, 0.05, 0.45, 0.0, 0.15, 0.0],
    "fear": [0.1, 0.1, 0.05, 0.7, 0.0, 0.05, 0.0, 0.0],
    "gratitude": [0.0, 0.1, 0.0, 0.0, 0.45, 0.0, 0.0, 0.45],
    "grief": [0.05, 0.0, 0.0, 0.1, 0.0, 0.8, 0.0, 0.05],
    "joy": [0.0, 0.1, 0.0, 0.0, 0.85, 0.0, 0.05, 0.0],
    "love": [0.0, 0.15, 0.0, 0.0, 0.55, 0.0, 0.0, 0.3],
    "nervousness": [0.1, 0.2, 0.0, 0.55, 0.0, 0.15, 0.0, 0.0],
    "optimism": [0.0, 0.35, 0.0, 0.0, 0.55, 0.0, 0.0, 0.1],
    "pride": [0.0, 0.15, 0.0, 0.0, 0.55, 0.0, 0.0, 0.3],
    "realization": [0.0, 0.25, 0.0, 0.05, 0.1, 0.05, 0.45, 0.1],
    "relief": [0.0, 0.1, 0.0, 0.05, 0.45, 0.1, 0.0, 0.3],
    "remorse": [0.15, 0.0, 0.1, 0.1, 0.0, 0.6, 0.0, 0.05],
    "sadness": [0.05, 0.0, 0.0, 0.1, 0.0, 0.8, 0.0, 0.05],
    "surprise": [0.05, 0.15, 0.0, 0.1, 0.15, 0.05, 0.45, 0.05],
}


def affinity_for_label(name: str) -> np.ndarray:
    """Return a normalized affinity vector over ``NRC_KEYS`` for a GoEmotions name."""
    key = name.lower().strip()
    raw = _GO_EMOTION_NRC_AFFINITY.get(key)
    if raw is None:
        return np.ones(len(NRC_KEYS), dtype=np.float64) / len(NRC_KEYS)
    return _unit(raw)


class NRCLexiconBaseline:
    """
    Lexicon baseline using NRCLex document-level affect frequencies.

    For each target class, a fixed affinity profile over NRC channels is
    defined. The predicted class maximizes the cosine-like similarity between
    the profile and the observed NRC count vector (dot product after L2 norm).
    """

    def __init__(self, class_names: Sequence[str]) -> None:
        self.class_names = [str(c) for c in class_names]
        profiles = np.vstack([affinity_for_label(c) for c in self.class_names])
        # L2-normalize rows for stable dot products with nonnegative NRC counts.
        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._profiles = profiles / norms

    def _nrc_counts(self, text: str) -> np.ndarray:
        obj = NRCLex(str(text))
        freq = getattr(obj, "raw_emotion_scores", None) or getattr(
            obj, "affect_frequencies", {}
        )
        vec = np.array([float(freq.get(k, 0.0)) for k in NRC_KEYS], dtype=np.float64)
        if vec.sum() == 0:
            vec = np.ones(len(NRC_KEYS), dtype=np.float64) / len(NRC_KEYS)
        return vec

    def predict_one(self, text: str) -> int:
        nrc = self._nrc_counts(text)
        scores = self._profiles @ nrc
        return int(np.argmax(scores))

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([self.predict_one(t) for t in texts], dtype=np.int64)


def build_tfidf_logistic(
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    C: float = 4.0,
) -> Pipeline:
    """
    Construct a ``TfidfVectorizer`` + ``LogisticRegression`` multiclass pipeline.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=C,
                    solver="lbfgs",
                ),
            ),
        ]
    )
