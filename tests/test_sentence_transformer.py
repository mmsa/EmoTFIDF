import pytest

from EmoTFIDF.sentence import SentenceEmotionTransformer


def test_bundled_checkpoint_matches_reference():
    scorer = SentenceEmotionTransformer()
    scores = scorer.score("I want to go have a party like hell")
    assert list(scores) == ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
    assert scores["joy"] == pytest.approx(0.7927997, abs=1e-5)
    assert all(0 <= value <= 1 for value in scores.values())
