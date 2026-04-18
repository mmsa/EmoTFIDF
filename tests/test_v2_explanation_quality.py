"""Explanation salience, weak-contextual handling, and verifier calibration fields."""

from __future__ import annotations

import pytest

from EmoTFIDF.evidence import EmoTFIDFv2
from EmoTFIDF.evidence.lexeme_prior import is_weak_contextual_lexeme


@pytest.fixture(scope="module")
def fitted():
    corpus = [
        "I am happy today and everything feels great.",
        "I am not happy today and everything feels wrong.",
        "She was sad and crying yesterday about the loss.",
        "I feel furious and angry about the situation.",
    ]
    v2 = EmoTFIDFv2()
    v2.fit(corpus)
    return v2


def test_yesterday_not_first_sadness_top_term(fitted):
    """NRC tags *yesterday* with sadness; it should not outrank clearer affect words."""
    r = fitted.analyze("She was sad and crying yesterday.")
    sad_terms = [row["term"] for row in r.top_terms_by_emotion.get("sadness", [])]
    assert sad_terms, "expected sadness lexicon hits"
    assert sad_terms[0] != "yesterday"
    assert "crying" in sad_terms[:2] or "sad" in sad_terms[:2]


def test_explain_omits_weak_contextual_when_alternatives_exist(fitted):
    expl = fitted.explain("She was sad and crying yesterday.")
    tokens = [w["token"] for w in expl["top_contributing_words"]]
    if any(not is_weak_contextual_lexeme(t) for t in tokens):
        assert "yesterday" not in tokens


def test_no_signal_normalized_zero_and_explain(fitted):
    r = fitted.analyze("!!!")
    assert sum(r.normalized_emotion_scores.values()) == 0.0
    expl = fitted.explain("!!!")
    assert expl["no_signal_detected"] is True
    assert expl["top_contributing_words"] == []


def test_verifier_reports_calibration_fields(fitted):
    out = fitted.verify_label("I am happy today!", "joy")
    for key in (
        "support_score",
        "support_level",
        "dominance_margin",
        "coverage_score",
        "evidence_term_count",
    ):
        assert key in out
    assert out["evidence_term_count"] >= 1
    assert isinstance(out["coverage_score"], float)


def test_verifier_no_signal_evidence_count_zero(fitted):
    out = fitted.verify_label("!!!", "joy")
    assert out["evidence_term_count"] == 0
    assert out["support_level"] == "unsupported"
