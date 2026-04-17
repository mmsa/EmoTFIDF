"""Analyzer API smoke and structure tests for EmoTFIDF V2."""

from __future__ import annotations

import json

import pytest

from EmoTFIDF.v2 import EmoTFIDFv2


@pytest.fixture(scope="module")
def small_corpus():
    return [
        "I am happy today because everything went well.",
        "I feel sad and disappointed about the news.",
        "This makes me angry and frustrated with the situation.",
        "I am surprised and curious about what happened next.",
    ]


@pytest.fixture(scope="module")
def analyzer(small_corpus):
    v2 = EmoTFIDFv2()
    v2.fit(small_corpus)
    return v2


def test_fit_analyze_keys(analyzer):
    r = analyzer.analyze("I am very happy today!")
    d = r.to_dict()
    expected_keys = {
        "raw_emotion_scores",
        "normalized_emotion_scores",
        "top_terms_by_emotion",
        "term_contributions",
        "coverage",
        "matched_terms",
        "unmatched_terms",
        "negation_hits",
        "intensifier_hits",
        "dominant_emotions",
        "support_summary",
        "feature_vector",
        "feature_names",
    }
    assert expected_keys <= set(d.keys())
    assert len(r.feature_vector) == len(r.feature_names)
    assert len(r.feature_vector) == 35


def test_dominant_emotion_stable(analyzer):
    text = "I am happy today!"
    first = analyzer.analyze(text).dominant_emotions
    second = analyzer.analyze(text).dominant_emotions
    assert first == second
    assert first[0] == "joy"


def test_batch_and_empty(analyzer):
    batch = analyzer.analyze_batch(["hello", "", "!!!"])
    assert len(batch) == 3
    assert isinstance(batch[0], dict)
    empty = analyzer.analyze("")
    assert empty.to_dict() == json.loads(json.dumps(empty.to_dict()))


def test_punctuation_short_text(analyzer):
    r = analyzer.analyze("!!! ... ")
    assert r.to_dict() == json.loads(json.dumps(r.to_dict()))


def test_get_feature_vector(analyzer):
    vec, names = analyzer.get_feature_vector("happy joyful day")
    assert len(vec) == len(names) == 35
