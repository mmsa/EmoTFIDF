"""Verifier structured output and simple ranking checks."""

from __future__ import annotations

import pytest

from EmoTFIDF.evidence import EmoTFIDFv2


@pytest.fixture(scope="module")
def fitted():
    corpus = [
        "I am joyful and happy about the wonderful celebration.",
        "I feel miserable and sad about the terrible loss.",
    ]
    v2 = EmoTFIDFv2()
    v2.fit(corpus)
    return v2


def test_verify_structure(fitted):
    out = fitted.verify_label("I am happy today!", "joy")
    assert set(out.keys()) >= {
        "predicted_label",
        "support_score",
        "support_level",
        "supporting_terms",
        "conflicting_emotions",
        "notes",
        "dominance_margin",
        "coverage_score",
        "evidence_term_count",
    }
    assert out["predicted_label"] == "joy"
    assert out["support_level"] in {"strong", "moderate", "weak", "unsupported"}


def test_supported_beats_mismatched(fitted):
    good = fitted.verify_label("I am happy and joyful today!", "joy")
    bad = fitted.verify_label("I am happy and joyful today!", "sadness")
    assert good["support_score"] > bad["support_score"]
