"""Structured types for EmoTFIDF V2 analysis outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class NegationHit:
    """A negation cue affecting a downstream affect-bearing token."""

    cue: str
    cue_position: int
    target_token: str
    target_position: int
    emotions_affected: List[str]


@dataclass
class IntensifierHit:
    """An intensifier or downtoner modifying a downstream affect token."""

    cue: str
    cue_position: int
    direction: str  # "up" | "down"
    target_token: str
    target_position: int
    multiplier: float


@dataclass
class TermContribution:
    """Per-token weighted contribution to emotion dimensions."""

    token: str
    position: int
    emotions: List[str]
    base_weight: float
    intensifier_multiplier: float
    negated: bool
    negation_cue: Optional[str]
    intensifier_cue: Optional[str]
    per_emotion_contribution: Dict[str, float]


@dataclass
class Coverage:
    """Lexicon coverage statistics for the analyzed text."""

    matched_term_count: int
    total_tokens_considered: int
    coverage_ratio: float
    matched_terms: List[str]
    unmatched_terms: List[str]


@dataclass
class VerificationResult:
    """Lexical support assessment for a proposed emotion label."""

    predicted_label: str
    support_score: float
    support_level: str
    supporting_terms: List[str]
    conflicting_emotions: List[str]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    """Rich, interpretable analysis for one document."""

    raw_emotion_scores: Dict[str, float]
    normalized_emotion_scores: Dict[str, float]
    top_terms_by_emotion: Dict[str, List[Dict[str, Any]]]
    term_contributions: List[TermContribution]
    coverage: Coverage
    matched_terms: List[str]
    unmatched_terms: List[str]
    negation_hits: List[NegationHit]
    intensifier_hits: List[IntensifierHit]
    dominant_emotions: List[str]
    support_summary: str
    feature_vector: List[float]
    feature_names: List[str]
    # Evidence calibration (positive-mass normalization; no fake uniform on empty input).
    total_evidence: float
    total_positive_evidence: float
    top1_score: float
    top2_score: float
    dominance_margin: float
    has_meaningful_signal: bool
    has_low_evidence: bool

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly nested dict (lists of primitives and dicts)."""

        def contrib(c: TermContribution) -> Dict[str, Any]:
            d = asdict(c)
            d["per_emotion_contribution"] = dict(
                sorted(c.per_emotion_contribution.items())
            )
            return d

        return {
            "raw_emotion_scores": dict(sorted(self.raw_emotion_scores.items())),
            "normalized_emotion_scores": dict(
                sorted(self.normalized_emotion_scores.items())
            ),
            "top_terms_by_emotion": {
                k: list(v) for k, v in sorted(self.top_terms_by_emotion.items())
            },
            "term_contributions": [contrib(t) for t in self.term_contributions],
            "coverage": asdict(self.coverage),
            "matched_terms": list(self.matched_terms),
            "unmatched_terms": list(self.unmatched_terms),
            "negation_hits": [asdict(n) for n in self.negation_hits],
            "intensifier_hits": [asdict(i) for i in self.intensifier_hits],
            "dominant_emotions": list(self.dominant_emotions),
            "support_summary": self.support_summary,
            "feature_vector": list(self.feature_vector),
            "feature_names": list(self.feature_names),
            "total_evidence": float(self.total_evidence),
            "total_positive_evidence": float(self.total_positive_evidence),
            "top1_score": float(self.top1_score),
            "top2_score": float(self.top2_score),
            "dominance_margin": float(self.dominance_margin),
            "has_meaningful_signal": bool(self.has_meaningful_signal),
            "has_low_evidence": bool(self.has_low_evidence),
        }


@dataclass
class ExplanationBundle:
    """Human-readable explanation for debugging, demos, and papers."""

    dominant_emotions: List[str]
    top_contributing_words: List[Dict[str, Any]]
    adjustment_notes: List[str]
    confidence_commentary: str
    warnings: List[str]
    raw_support_summary: str
    no_signal_detected: bool
    has_low_evidence: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
