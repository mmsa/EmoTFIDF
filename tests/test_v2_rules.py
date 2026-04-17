"""Negation and intensifier window behavior."""

from __future__ import annotations

import pytest

from EmoTFIDF.evidence import EmoTFIDFv2


@pytest.fixture(scope="module")
def fitted():
    corpus = [
        "I am happy today and everything feels great.",
        "I am not happy today and everything feels wrong.",
        "I am very happy today and everything feels amazing.",
        "I am slightly happy today but still concerned.",
    ]
    v2 = EmoTFIDFv2()
    v2.fit(corpus)
    return v2


def test_negation_changes_joy(fitted):
    pos = fitted.analyze("I am happy today")
    neg = fitted.analyze("I am not happy today")
    assert pos.raw_emotion_scores["joy"] > 0
    assert neg.raw_emotion_scores["joy"] < 0
    assert len(neg.negation_hits) >= 1
    expl = fitted.explain("I am not happy today")
    assert expl["adjustment_notes"] or expl["warnings"]
    assert any("negation" in n.lower() for n in expl["adjustment_notes"])


def test_intensifier_boosts_joy(fitted):
    base = fitted.analyze("I am happy today")
    boosted = fitted.analyze("I am very happy today")
    assert boosted.raw_emotion_scores["joy"] > base.raw_emotion_scores["joy"]
    assert len(boosted.intensifier_hits) >= 1


def test_downtoner_reduces_vs_baseline(fitted):
    base = fitted.analyze("I am happy today")
    down = fitted.analyze("I am slightly happy today")
    assert down.raw_emotion_scores["joy"] < base.raw_emotion_scores["joy"]
