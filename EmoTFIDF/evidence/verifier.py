"""Lexical support / contradiction hints for a proposed emotion label (evidence only)."""

from __future__ import annotations

from typing import List

from EmoTFIDF.evidence.lexicon import DEFAULT_EMOTION_LABELS
from EmoTFIDF.evidence.schemas import AnalysisResult, VerificationResult


def _support_level(score: float) -> str:
    if score >= 0.38:
        return "strong"
    if score >= 0.22:
        return "moderate"
    if score >= 0.08:
        return "weak"
    return "unsupported"


def verify_label(analysis: AnalysisResult, predicted_label: str) -> VerificationResult:
    """
    Assess how well *predicted_label* is supported by lexicon + TF-IDF evidence.

    This does **not** certify semantic correctness; it surfaces lexical alignment
    and simple conflicts with other high-scoring emotions.
    """
    label = predicted_label.strip().lower()
    notes: List[str] = []

    if label not in DEFAULT_EMOTION_LABELS:
        notes.append(
            f"Unknown label '{predicted_label}'; expected one of {DEFAULT_EMOTION_LABELS}."
        )
        return VerificationResult(
            predicted_label=label,
            support_score=0.0,
            support_level="unsupported",
            supporting_terms=[],
            conflicting_emotions=list(
                e for e in analysis.dominant_emotions if e != label
            )[:3],
            notes=notes,
        )

    if not analysis.has_meaningful_signal or analysis.has_low_evidence:
        notes.append("No reliable lexical emotional evidence; verifier scores are not meaningful.")
        return VerificationResult(
            predicted_label=label,
            support_score=0.0,
            support_level="unsupported",
            supporting_terms=[],
            conflicting_emotions=[],
            notes=notes,
            dominance_margin=float(analysis.dominance_margin),
            coverage_score=float(analysis.coverage.coverage_ratio),
            evidence_term_count=0,
        )

    norm = analysis.normalized_emotion_scores
    label_share = float(norm.get(label, 0.0))

    supporting_terms: List[str] = []
    for row in analysis.top_terms_by_emotion.get(label, [])[:8]:
        t = row.get("term")
        w = float(row.get("weight", 0.0))
        if t and w > 0.0:
            supporting_terms.append(str(t))

    # Lexical mass from term contributions for this label
    mass = 0.0
    for c in analysis.term_contributions:
        mass += max(0.0, c.per_emotion_contribution.get(label, 0.0))
    mass_norm = mass / (sum(max(0.0, x) for x in analysis.raw_emotion_scores.values()) + 1e-9)

    support_score = round(0.65 * label_share + 0.35 * min(1.0, mass_norm), 6)
    level = _support_level(support_score)

    evidence_term_count = sum(
        1
        for c in analysis.term_contributions
        if sum(max(0.0, float(v)) for v in c.per_emotion_contribution.values()) > 1e-12
    )

    ranked = sorted(((norm[e], e) for e in DEFAULT_EMOTION_LABELS), reverse=True)
    top_e = ranked[0][1]
    conflicting: List[str] = []
    if top_e != label:
        conflicting.append(top_e)
    for _, e in ranked[1:4]:
        if e != label and e not in conflicting:
            conflicting.append(e)

    if analysis.negation_hits:
        notes.append("Negation cues present; verify polarity against the proposed label.")
    if analysis.coverage.coverage_ratio < 0.12:
        notes.append("Low lexicon coverage; support score is indicative only.")

    if label_share < 0.05 and mass < 1e-8:
        notes.append("Little direct lexical mass for this label in the current windowing rules.")

    return VerificationResult(
        predicted_label=label,
        support_score=support_score,
        support_level=level,
        supporting_terms=supporting_terms[:10],
        conflicting_emotions=conflicting[:4],
        notes=notes,
        dominance_margin=float(analysis.dominance_margin),
        coverage_score=float(analysis.coverage.coverage_ratio),
        evidence_term_count=int(evidence_term_count),
    )
