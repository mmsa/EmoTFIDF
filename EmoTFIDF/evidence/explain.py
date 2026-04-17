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
    no_signal = not analysis.has_meaningful_signal
    low_ev = analysis.has_low_evidence

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
        adjustments.append(
            f"Negation '{n.cue}' near '{n.target_token}' reduced or inverted "
            f"{', '.join(n.emotions_affected)}; suppressed joy is partially attributed to sadness "
            f"(see NEGATION_SUPPRESSED_JOY_TO_SADNESS_FRACTION in rules.py)."
        )
    for h in analysis.intensifier_hits:
        adjustments.append(
            f"{h.direction.title()}toner '{h.cue}' near '{h.target_token}' "
            f"(×{h.multiplier:.3f})."
        )

    warnings: List[str] = []
    if no_signal:
        warnings.append("No emotional evidence detected from lexicon or phrase rules.")
    elif low_ev:
        warnings.append("Low positive affect mass; dominant emotions may be absent or unstable.")
    if analysis.coverage.coverage_ratio < 0.12 and not no_signal:
        warnings.append("Lexical coverage is low; treat scores as weak evidence.")
    if analysis.negation_hits and not no_signal:
        warnings.append("Negation handling is heuristic (local window, no full parsing).")

    if no_signal:
        commentary = (
            "No lexicon hits in the seven-label space and no affect-bearing terms; "
            "normalized scores are zeroed (no uniform pseudo-distribution)."
        )
    elif low_ev:
        commentary = (
            "Matched terms did not yield positive summarized mass after cue handling; "
            "see raw scores and negation notes if present."
        )
    else:
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
                f"(margin {margin:.3f} on positive-mass normalization)."
            )
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
        no_signal_detected=no_signal,
        has_low_evidence=low_ev,
    )
