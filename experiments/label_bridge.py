"""
Map GoEmotions (benchmark) class names to the seven-way evidence label space.

The V2 verifier (:func:`EmoTFIDF.evidence.verifier.verify_label`) expects one of
``EmoTFIDF.evidence.lexicon.DEFAULT_EMOTION_LABELS``. GoEmotions has 28+ fine labels;
we reuse the same NRC affinity row as ``baselines.affinity_for_label`` and collapse
the **dominant NRC channel** into seven-way space for a transparent, reproducible bridge.
"""

from __future__ import annotations

import numpy as np

from baselines import NRC_KEYS, affinity_for_label

# Collapse 8-d NRC basis (NRCLex channels) into the seven library emotions.
_NRC_PRIMARY_TO_SEVEN = {
    "anger": "anger",
    "anticipation": "surprise",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "sadness": "sadness",
    "surprise": "surprise",
    "trust": "joy",
}


def goemotion_class_to_evidence_label(class_name: str) -> str:
    """
    Return the seven-way label used for lexical verification of a GoEmotions gold/pred name.

    Uses ``argmax`` over the fixed NRC affinity profile for that class, then maps the
    winning NRC channel through :data:`_NRC_PRIMARY_TO_SEVEN`.

    The simplified **neutral** class is treated explicitly: its affinity row peaks on
    *anger* only because the NRC basis has no ``neutral`` channel, but the verifier
    should audit the neutral hypothesis in seven-way space.
    """
    key = str(class_name).strip().lower()
    if key == "neutral":
        return "neutral"
    v = affinity_for_label(class_name)
    idx = int(np.argmax(v))
    nrc = NRC_KEYS[idx]
    return _NRC_PRIMARY_TO_SEVEN.get(nrc, "neutral")
