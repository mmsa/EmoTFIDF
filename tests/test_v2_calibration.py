"""Calibration: no-signal, negation, and strong-emotion dominance (evidence API)."""

from __future__ import annotations

import json

import pytest

from EmoTFIDF.evidence import EmoTFIDFv2


@pytest.fixture(scope="module")
def fitted():
    corpus = [
        "I am happy today and everything feels great.",
        "I am not happy today and everything feels wrong.",
        "I feel sad and disappointed about the news.",
        "This makes me angry and frustrated with the situation.",
        "I feel furious and angry about the problem.",
        "I am extremely upset and mad about the delay.",
    ]
    v2 = EmoTFIDFv2()
    v2.fit(corpus)
    return v2


@pytest.mark.parametrize(
    "text",
    ["", "!!! ...", "12345", "   ", "###"],
)
def test_no_signal_no_uniform_distribution(fitted, text):
    r = fitted.analyze(text)
    assert r.has_low_evidence is True
    assert r.has_meaningful_signal is False
    assert sum(r.normalized_emotion_scores.values()) == 0.0
    assert r.dominant_emotions == []
    expl = fitted.explain(text)
    assert expl["no_signal_detected"] is True
    pf = fitted.to_prompt_features(text)
    json.dumps(pf)
    assert pf["has_meaningful_signal"] is False


def test_negation_reduces_joy_not_arbitrary_anger(fitted):
    pos = fitted.analyze("I am happy today")
    neg = fitted.analyze("I am not happy today")
    assert pos.raw_emotion_scores["joy"] > neg.raw_emotion_scores["joy"]
    assert "anger" not in neg.dominant_emotions
    assert neg.dominant_emotions and neg.dominant_emotions[0] == "sadness"


def test_negation_never_happy_variants(fitted):
    for text in ("I am never happy today.", "I am not very happy today."):
        r = fitted.analyze(text)
        assert r.raw_emotion_scores["joy"] < 0.0 or r.normalized_emotion_scores.get("joy", 0) == 0.0
        assert "anger" not in r.dominant_emotions


def test_strong_anger_clear_dominance_no_weak_tie(fitted):
    for text in (
        "I feel furious and angry about this.",
        "I am extremely angry.",
    ):
        r = fitted.analyze(text)
        assert r.has_meaningful_signal is True
        assert r.dominant_emotions, text
        assert r.dominant_emotions[0] == "anger", text
        assert len(r.dominant_emotions) == 1 or r.dominance_margin >= 0.05, text


def test_explain_reflects_no_signal(fitted):
    expl = fitted.explain("!!!")
    assert expl["no_signal_detected"] is True
    assert any("No emotional evidence" in w for w in expl["warnings"])


def test_verifier_unsupported_on_no_signal(fitted):
    out = fitted.verify_label("!!!", "joy")
    assert out["support_level"] == "unsupported"
    assert out["support_score"] == 0.0
