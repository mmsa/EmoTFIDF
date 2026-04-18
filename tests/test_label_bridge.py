"""Tests for GoEmotions → seven-way evidence label bridge."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "experiments"
for p in (str(REPO_ROOT), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from label_bridge import goemotion_class_to_evidence_label  # noqa: E402


@pytest.mark.parametrize(
    ("goe", "expected_seven"),
    [
        ("anger", "anger"),
        ("disgust", "disgust"),
        ("fear", "fear"),
        ("joy", "joy"),
        ("sadness", "sadness"),
        ("surprise", "surprise"),
        ("neutral", "neutral"),
        ("approval", "joy"),  # NRC trust → joy in bridge
        ("curiosity", "surprise"),  # anticipation → surprise
    ],
)
def test_goemotion_class_to_evidence_label(goe: str, expected_seven: str) -> None:
    assert goemotion_class_to_evidence_label(goe) == expected_seven
