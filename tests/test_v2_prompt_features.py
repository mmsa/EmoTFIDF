"""Prompt feature export tests."""

from __future__ import annotations

import json

import pytest

from EmoTFIDF.v2 import EmoTFIDFv2


@pytest.fixture(scope="module")
def fitted():
    v2 = EmoTFIDFv2()
    v2.fit(
        [
            "happy joyful day",
            "sad miserable afternoon",
        ]
    )
    return v2


def test_prompt_features_json_serializable(fitted):
    pf = fitted.to_prompt_features("I am very happy today!")
    json.dumps(pf)
    assert "dominant_emotions" in pf
    assert "support_summary" in pf
    assert isinstance(pf["dominant_emotions"], list)
    assert isinstance(pf["support_summary"], str)
