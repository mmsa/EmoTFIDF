"""
Seven-dimensional evidence vectors from EmoTFIDF V2 (``EmoTFIDF.evidence``).

Used the same way as :class:`emotfidf_wrapper.EmoTFIDFVectorizer` outputs for a
probability-fusion head: positive-mass normalized scores, fixed label order.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from utils import ensure_repo_on_path

ensure_repo_on_path()

from EmoTFIDF.evidence import EmoTFIDFv2
from EmoTFIDF.evidence.lexicon import DEFAULT_EMOTION_LABELS


def v2_normalized_emotion_matrix(
    texts: Sequence[str],
    corpus_for_tfidf: Sequence[str],
    *,
    lexicon_path: str | None = None,
) -> np.ndarray:
    """
    ``fit`` V2 TF-IDF on ``corpus_for_tfidf``, then stack ``normalized_emotion_scores`` rows.

    Shape ``(len(texts), 7)``. Rows are non-negative and L1-normalize to ~1 when there is
    lexical positive mass; otherwise all zeros (same semantics as analysis).
    """
    v2 = EmoTFIDFv2(lexicon_path=lexicon_path)
    v2.fit([str(t) for t in corpus_for_tfidf])
    rows: List[List[float]] = []
    order = list(DEFAULT_EMOTION_LABELS)
    for t in texts:
        r = v2.analyze(str(t))
        rows.append([float(r.normalized_emotion_scores.get(e, 0.0)) for e in order])
    return np.asarray(rows, dtype=np.float64)
