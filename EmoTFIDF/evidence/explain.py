"""Human-readable explanations from a V2 :class:`~EmoTFIDF.evidence.schemas.AnalysisResult`."""

from __future__ import annotations

from typing import Any, Dict, List

from EmoTFIDF.evidence.lexicon import DEFAULT_EMOTION_LABELS
from EmoTFIDF.evidence.schemas import AnalysisResult, ExplanationBundle


def build_explanation(analysis: AnalysisResult) -> ExplanationBundle:
    """
    Produce a concise explanation: dominant emotions, drivers, cue adjustments, and caveats.

    Intended for debugging, demos, and documentation — not as a gold truth labeler.
    """
    top_words: List[Dict[str, Any]] = []
    seen: set = set()
    for c in analysis.term_contributions:
        key = (c.token, tuple(sorted(c.emotions)))
        if key in seen:
            continue
        seen.add(key)
        top_words.append(
            {
                "token": c.token,
                "emotions": list(c.emotions),
                "base_tfidf_mass": c.base_weight,
                "intensifier_multiplier": c.intensifier_multiplier,
                "negated": c.negated,
            }
        )
    top_words = top_words[:12]

    adjustments: List[str] = []
    for n in analysis.negation_hits:
        adj = (
            f"Negation '{n.cue}' scopes '{n.target_token}' "
            f"({', '.join(n.emotions_affected)} contribution inverted/scaled)."
        )
        adjustments.append(adj)
    for h in analysis.intensifier_hits:
        adjustments.append(
            f"{h.direction.title()}toner '{h.cue}' near '{h.target_token}' "
            f"(×{h.multiplier:.3f})."
        )

    warnings: List[str] = []
    if analysis.coverage.coverage_ratio < 0.12:
        warnings.append("Lexical coverage is low; treat scores as weak evidence.")
    if not analysis.matched_terms:
        warnings.append("No lexicon hits in the seven-label space; emotions are baseline.")

    if analysis.negation_hits:
        warnings.append("Negation handling is heuristic (local window, no full parsing).")

    # Commentary from softmax-like margin using normalized dict already on analysis
    ranked = sorted(
        ((analysis.normalized_emotion_scores.get(e, 0.0), e) for e in DEFAULT_EMOTION_LABELS),
        reverse=True,
    )
    top_s, top_e = ranked[0]
    second_s = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = top_s - second_s
    if margin < 0.08 and analysis.coverage.coverage_ratio >= 0.12:
        commentary = (
            f"Top emotion {top_e} is only marginally ahead of alternatives "
            f"(flat distribution, margin {margin:.3f})."
        )
    elif analysis.coverage.coverage_ratio < 0.12:
        commentary = "Low lexical grounding; prefer qualitative reading over sharp scores."
    else:
        commentary = (
            f"Lexical mass concentrates on {top_e} "
            f"(normalized share {top_s:.3f}, margin {margin:.3f})."
        )

    return ExplanationBundle(
        dominant_emotions=list(analysis.dominant_emotions),
        top_contributing_words=top_words,
        adjustment_notes=adjustments,
        confidence_commentary=commentary,
        warnings=warnings,
        raw_support_summary=analysis.support_summary,
    )
