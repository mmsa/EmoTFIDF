"""
Regression-style checks: V1 vs V2 (evidence) on the same corpus and texts.

V1 and V2 use different scoring formulas; we check that both behave sensibly
on clear examples and record dominant-label overlap where reasonable.
"""

from __future__ import annotations

import pytest

from experiments.compare_v1_v2 import (
    CORPUS,
    LABELS,
    LEX_PATH,
    TEST_TEXTS,
    run_rows,
    _v1_em_tfidf,
    _v2_normalized,
)


@pytest.mark.skipif(not LEX_PATH.is_file(), reason="Packaged lexicon missing")
def test_lexicon_file_exists():
    assert LEX_PATH.is_file()


@pytest.mark.skipif(not LEX_PATH.is_file(), reason="Packaged lexicon missing")
def test_v1_v2_both_produce_seven_scores():
    text = "I am happy today!"
    s1 = _v1_em_tfidf(CORPUS, text)
    s2 = _v2_normalized(CORPUS, text)
    assert set(s1.keys()) == set(LABELS)
    assert set(s2.keys()) == set(LABELS)
    assert abs(sum(s2.values()) - 1.0) < 1e-5


@pytest.mark.skipif(not LEX_PATH.is_file(), reason="Packaged lexicon missing")
def test_clear_happy_text_joy_dominates_both():
    text = "I am very happy joyful wonderful today!"
    s1 = _v1_em_tfidf(CORPUS, text)
    s2 = _v2_normalized(CORPUS, text)
    assert max(s1, key=s1.get) == "joy"
    assert max(s2, key=s2.get) == "joy"


@pytest.mark.skipif(not LEX_PATH.is_file(), reason="Packaged lexicon missing")
def test_clear_anger_text_anger_dominates_both():
    text = "I am furious angry rage about this attack."
    s1 = _v1_em_tfidf(CORPUS, text)
    s2 = _v2_normalized(CORPUS, text)
    assert max(s1, key=s1.get) == "anger"
    assert max(s2, key=s2.get) == "anger"


@pytest.mark.skipif(not LEX_PATH.is_file(), reason="Packaged lexicon missing")
def test_compare_script_rows_have_metrics():
    rows = run_rows(CORPUS, TEST_TEXTS)
    assert len(rows) == len(TEST_TEXTS)
    for r in rows:
        assert "v1_dominant" in r and "v2_dominant" in r
        assert "agree" in r and "l1_dist" in r and "cosine" in r
        assert r["v1_scores"]["joy"] >= 0.0
        assert r["v2_scores"]["joy"] >= 0.0
