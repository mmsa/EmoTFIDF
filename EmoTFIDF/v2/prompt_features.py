"""Compact, JSON-ready features for LLM / guardrail / hybrid pipelines."""

from __future__ import annotations

from typing import Any, Dict, List

from EmoTFIDF.v2.schemas import AnalysisResult


def _weighted_terms(analysis: AnalysisResult, limit: int = 12) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for c in analysis.term_contributions[:limit]:
        pos = {k: v for k, v in c.per_emotion_contribution.items() if v > 0.0}
        if not pos and c.negated:
            pos = {k: v for k, v in c.per_emotion_contribution.items()}
        rows.append(
            {
                "token": c.token,
                "weights_by_emotion": dict(sorted(pos.items())),
                "negated": c.negated,
                "intensifier_multiplier": c.intensifier_multiplier,
            }
        )
    return rows


def _nl_summary(analysis: AnalysisResult) -> str:
    dom = ", ".join(analysis.dominant_emotions[:2])
    neg = "negation present" if analysis.negation_hits else "no negation cues"
    ints = (
        "intensifier/downtoner present"
        if analysis.intensifier_hits
        else "no intensifier cues"
    )
    return (
        f"Lexical emotions lean toward {dom}. Coverage "
        f"{analysis.coverage.coverage_ratio:.2f}; {neg}; {ints}. "
        f"{analysis.support_summary}"
    )


def build_prompt_features(analysis: AnalysisResult) -> Dict[str, Any]:
    """
    Return a small dict suitable for JSON serialization and prompt injection.

    Fields are explicit strings and numbers so downstream templates stay stable.
    """
    return {
        "dominant_emotions": list(analysis.dominant_emotions),
        "normalized_emotion_scores": dict(
            sorted(analysis.normalized_emotion_scores.items())
        ),
        "weighted_terms": _weighted_terms(analysis),
        "support_summary": analysis.support_summary,
        "coverage": {
            "ratio": analysis.coverage.coverage_ratio,
            "matched_terms": analysis.coverage.matched_terms[:20],
            "matched_count": analysis.coverage.matched_term_count,
        },
        "negation_markers": [
            {"cue": n.cue, "target": n.target_token} for n in analysis.negation_hits[:8]
        ],
        "intensity_markers": [
            {"cue": h.cue, "direction": h.direction, "target": h.target_token}
            for h in analysis.intensifier_hits[:8]
        ],
        "natural_language_summary": _nl_summary(analysis),
    }
