"""Main EmoTFIDF V2 analyzer: lexicon + corpus TF-IDF + transparent cue windows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from EmoTFIDF.evidence.lexeme_prior import (
    contextual_affect_multiplier,
    explanation_rank_multiplier,
)
from EmoTFIDF.evidence.lexicon import (
    DEFAULT_EMOTION_LABELS,
    inverse_count_emotion_shares,
    load_lexicon,
)
from EmoTFIDF.evidence.preprocessing import (
    content_tokens_for_coverage,
    process_message_for_tfidf,
    strip_edges,
    tokenize_raw_sequence,
)
from EmoTFIDF.evidence.rules import (
    INTENSIFIERS_DOWN,
    INTENSIFIERS_UP,
    NEGATION_CUES,
    NEGATION_SUPPRESSED_JOY_TO_SADNESS_FRACTION,
    find_negation_in_window,
    intensifier_multiplier_in_window,
)
from EmoTFIDF.evidence.schemas import (
    AnalysisResult,
    Coverage,
    IntensifierHit,
    NegationHit,
    TermContribution,
)
from EmoTFIDF.evidence.weighting import (
    build_feature_vector,
    distribution_entropy,
    dominant_margin,
    emotion_dict_zeros,
    normalize_positive_l1,
    per_emotion_max_and_topk,
    select_dominant_emotions,
    softmax_positive_or_zeros,
    top_terms_by_emotion_from_contribs,
)


def _tfidf_weight_dict(vectorizer: Optional[TfidfVectorizer], processed_doc: str) -> Dict[str, float]:
    if vectorizer is None or not processed_doc.strip():
        return {}
    mat = vectorizer.transform([processed_doc])
    row = mat[0]
    names = vectorizer.get_feature_names_out()
    coo = row.tocoo()
    return {str(names[j]): float(v) for v, j in zip(coo.data, coo.col)}


def _median_nonzero_weight(vectorizer: TfidfVectorizer, processed_docs: List[str]) -> float:
    vals: List[float] = []
    for d in processed_docs[:200]:
        vals.extend(_tfidf_weight_dict(vectorizer, d).values())
    if not vals:
        return 0.25
    vals.sort()
    mid = len(vals) // 2
    if len(vals) % 2:
        return max(vals[mid], 1e-6)
    return max(0.5 * (vals[mid - 1] + vals[mid]), 1e-6)


class EmoTFIDFv2:
    """
    Interpretable emotional evidence layer: lexicon hits, corpus TF-IDF weights,
    and lightweight negation / intensifier windows.

    Public methods
    --------------
    fit(corpus_texts)
        Fit the TF-IDF vectorizer on processed documents (same preprocessing as V1).
    analyze(text)
        Return a rich :class:`AnalysisResult`.
    analyze_batch(texts)
        List of analysis dicts / results.
    get_feature_vector(text)
        Numeric feature vector (see ``feature_names`` on the analysis object).
    explain(text)
        Human-oriented explanation bundle as dict.
    verify_label(text, predicted_label)
        Lexical support for a proposed emotion label.
    to_prompt_features(text)
        Compact JSON-friendly features for LLM prompts.
    """

    def __init__(
        self,
        lexicon_path: Optional[str] = None,
        *,
        negation_window: int = 4,
        intensifier_window: int = 3,
        negation_factor: float = -0.55,
        max_tfidf_features: int = 200,
    ) -> None:
        self.labels: List[str] = list(DEFAULT_EMOTION_LABELS)
        self._lexicon: Dict[str, Any] = load_lexicon(lexicon_path)
        self.negation_window = negation_window
        self.intensifier_window = intensifier_window
        self.negation_factor = negation_factor
        self.max_tfidf_features = max_tfidf_features
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._fallback_tfidf_weight: float = 0.25

    def fit(self, corpus_texts: Sequence[str]) -> "EmoTFIDFv2":
        """Fit TF-IDF on *corpus_texts* using V1-compatible preprocessing."""
        if corpus_texts is None or len(corpus_texts) == 0:
            raise ValueError("corpus_texts must be a non-empty sequence of strings.")
        processed = [process_message_for_tfidf(str(t)) for t in corpus_texts]
        sw = stopwords.words("english")
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            stop_words=sw,
            token_pattern=r"(?u)\b[A-Za-z]+\b",
        )
        self._vectorizer.fit(processed)
        self._fallback_tfidf_weight = _median_nonzero_weight(self._vectorizer, processed)
        return self

    def analyze(self, text: str) -> AnalysisResult:
        """Run the full V2 pipeline on a single string."""
        if text is None:
            text = ""
        raw_tokens = [strip_edges(t) for t in tokenize_raw_sequence(str(text))]
        raw_tokens = [t for t in raw_tokens if t]

        processed = process_message_for_tfidf(str(text))
        weights = _tfidf_weight_dict(self._vectorizer, processed)
        proc_tokens = content_tokens_for_coverage(processed)
        total_cov = len(proc_tokens) if proc_tokens else 0

        negation_hits: List[NegationHit] = []
        intensifier_hits: List[IntensifierHit] = []
        term_contributions: List[TermContribution] = []
        matched_in_order: List[str] = []
        matched_set = set()

        raw_scores = emotion_dict_zeros(self.labels)

        for i, tok in enumerate(raw_tokens):
            # Intensifier / downtoner tokens are cue-only (e.g. "extremely" wrongly tags joy in NRC).
            if tok in INTENSIFIERS_UP or tok in INTENSIFIERS_DOWN or tok in NEGATION_CUES:
                continue
            if tok not in self._lexicon:
                continue
            # Inverse duplicate-count shares so repeated lexicon tags (e.g. two disgust)
            # do not drown a single anger tag—general fix for multi-label NRC entries.
            emotion_shares = inverse_count_emotion_shares(self._lexicon[tok])
            if not emotion_shares:
                continue

            base = float(weights.get(tok, self._fallback_tfidf_weight))
            neg_pair = find_negation_in_window(raw_tokens, i, self.negation_window)
            negated = neg_pair is not None
            neg_cue = neg_pair[0] if neg_pair else None
            imult, icue, idir, icue_pos = intensifier_multiplier_in_window(
                raw_tokens, i, self.intensifier_window
            )

            # Negation scales the whole affect token mass (same cue for all emotions on token).
            emotional_mult = self.negation_factor if negated else 1.0
            # Downweight temporal/discourse tokens that NRC still tags with affect (see lexeme_prior).
            ctx_pen = contextual_affect_multiplier(tok)
            mass = base * imult * emotional_mult * ctx_pen
            rank_hint = explanation_rank_multiplier(tok, self._lexicon)
            per_e: Dict[str, float] = emotion_dict_zeros(self.labels)
            for e, w_share in emotion_shares.items():
                contrib = mass * float(w_share)
                per_e[e] = contrib
                raw_scores[e] = raw_scores.get(e, 0.0) + contrib

            emotions = sorted(emotion_shares.keys())

            if neg_pair:
                negation_hits.append(
                    NegationHit(
                        cue=neg_pair[0],
                        cue_position=neg_pair[1],
                        target_token=tok,
                        target_position=i,
                        emotions_affected=list(emotions),
                    )
                )
            if icue and idir and icue_pos >= 0 and abs(imult - 1.0) > 1e-9:
                intensifier_hits.append(
                    IntensifierHit(
                        cue=icue,
                        cue_position=icue_pos,
                        direction=idir,
                        target_token=tok,
                        target_position=i,
                        multiplier=round(imult, 6),
                    )
                )

            term_contributions.append(
                TermContribution(
                    token=tok,
                    position=i,
                    emotions=list(emotions),
                    base_weight=round(base, 6),
                    intensifier_multiplier=round(imult, 6),
                    negated=negated,
                    negation_cue=neg_cue,
                    intensifier_cue=icue,
                    per_emotion_contribution={k: round(v, 8) for k, v in per_e.items() if v != 0.0},
                    explanation_rank_hint=round(rank_hint, 6),
                )
            )
            if tok not in matched_set:
                matched_set.add(tok)
                matched_in_order.append(tok)

        # Transparent valence sink: suppressed joy from negation is not re-mapped to anger.
        _apply_negation_joy_to_sadness_sink(raw_scores)

        unmatched = [t for t in proc_tokens if t not in matched_set]
        matched_count = len(matched_set)
        coverage_ratio = (matched_count / total_cov) if total_cov else 0.0
        unmatched_ratio = (len(unmatched) / total_cov) if total_cov else 0.0

        coverage = Coverage(
            matched_term_count=matched_count,
            total_tokens_considered=total_cov,
            coverage_ratio=round(coverage_ratio, 6),
            matched_terms=list(matched_in_order),
            unmatched_terms=sorted(set(unmatched))[:80],
        )

        mass_abs = sum(abs(raw_scores[e]) for e in self.labels)
        mass_pos = sum(max(0.0, raw_scores[e]) for e in self.labels)
        # Meaningful = lexicon-aligned affect activity (even if net polarity is negative pre-sink).
        has_meaningful_signal = len(term_contributions) > 0 and mass_abs > 1e-12
        # Low evidence when nothing matched or no positive mass to summarize as a distribution.
        has_low_evidence = len(term_contributions) == 0 or mass_pos <= 1e-12

        norm_pos = normalize_positive_l1(raw_scores, self.labels)
        norm_sm = softmax_positive_or_zeros(raw_scores, self.labels)
        top_terms = top_terms_by_emotion_from_contribs(
            term_contributions, self.labels, top_k=5, lexicon=self._lexicon
        )
        maxv, topk = per_emotion_max_and_topk(term_contributions, self.labels, k=3)
        ent = distribution_entropy(norm_sm, self.labels)
        sm_margin = dominant_margin(norm_sm, self.labels)

        can_dominant = has_meaningful_signal and mass_pos > 1e-12
        dominant, top1s, top2s, pos_margin = select_dominant_emotions(
            norm_pos,
            raw_scores,
            self.labels,
            has_meaningful_signal=can_dominant,
            single_dominant_margin=0.06,
        )

        support_summary = _build_support_summary(
            dominant,
            coverage_ratio,
            negation_hits,
            intensifier_hits,
            has_meaningful_signal,
            has_low_evidence,
        )

        feat_vec, feat_names = build_feature_vector(
            raw_scores,
            norm_sm,
            coverage_ratio,
            matched_count,
            unmatched_ratio,
            maxv,
            topk,
            len(negation_hits),
            len(intensifier_hits),
            ent,
            sm_margin,
            self.labels,
            meaningful_signal_01=1.0 if has_meaningful_signal else 0.0,
            total_positive_evidence=mass_pos,
            total_abs_evidence=mass_abs,
        )

        raw_rounded = {e: round(raw_scores[e], 8) for e in self.labels}
        norm_rounded = {e: round(norm_pos[e], 8) for e in self.labels}

        return AnalysisResult(
            raw_emotion_scores=raw_rounded,
            normalized_emotion_scores=norm_rounded,
            top_terms_by_emotion=top_terms,
            term_contributions=term_contributions,
            coverage=coverage,
            matched_terms=list(matched_in_order),
            unmatched_terms=coverage.unmatched_terms,
            negation_hits=negation_hits,
            intensifier_hits=intensifier_hits,
            dominant_emotions=dominant,
            support_summary=support_summary,
            feature_vector=[round(float(x), 8) for x in feat_vec],
            feature_names=feat_names,
            total_evidence=round(mass_abs, 8),
            total_positive_evidence=round(mass_pos, 8),
            top1_score=round(top1s, 8),
            top2_score=round(top2s, 8),
            dominance_margin=round(pos_margin, 8),
            has_meaningful_signal=has_meaningful_signal,
            has_low_evidence=has_low_evidence,
        )

    def analyze_batch(self, texts: Sequence[str]) -> List[Dict[str, Any]]:
        """Analyze many documents; returns JSON-ready dicts (stable ordering)."""
        return [self.analyze(t).to_dict() for t in texts]

    def get_feature_vector(self, text: str) -> Tuple[List[float], List[str]]:
        """Return ``(feature_vector, feature_names)`` for *text*."""
        r = self.analyze(text)
        return list(r.feature_vector), list(r.feature_names)

    def explain(self, text: str) -> Dict[str, Any]:
        """Human-readable explanation bundle (see :mod:`EmoTFIDF.evidence.explain`)."""
        from EmoTFIDF.evidence.explain import build_explanation

        return build_explanation(self.analyze(text)).to_dict()

    def verify_label(self, text: str, predicted_label: str) -> Dict[str, Any]:
        """Lexical support check for a proposed label (see :mod:`EmoTFIDF.evidence.verifier`)."""
        from EmoTFIDF.evidence.verifier import verify_label as verify

        return verify(self.analyze(text), predicted_label).to_dict()

    def to_prompt_features(self, text: str) -> Dict[str, Any]:
        """Compact prompt-side features (see :mod:`EmoTFIDF.evidence.prompt_features`)."""
        from EmoTFIDF.evidence.prompt_features import build_prompt_features

        return build_prompt_features(self.analyze(text))


def _apply_negation_joy_to_sadness_sink(raw_scores: Dict[str, float]) -> None:
    """
    If negation drove *joy* negative, move part of that magnitude into *sadness* (explicit rule).

    Without this, positive-mass normalization would show all-zero emotions even when the
    reader should see a conservative negative-valence hint—not unrelated anger from min-shift.
    """
    joy = raw_scores.get("joy", 0.0)
    if joy < 0.0:
        raw_scores["sadness"] = raw_scores.get("sadness", 0.0) + NEGATION_SUPPRESSED_JOY_TO_SADNESS_FRACTION * abs(
            joy
        )


def _build_support_summary(
    dominant: List[str],
    coverage_ratio: float,
    neg_hits: List[NegationHit],
    int_hits: List[IntensifierHit],
    has_meaningful_signal: bool,
    has_low_evidence: bool,
) -> str:
    if not has_meaningful_signal or has_low_evidence:
        if not has_meaningful_signal:
            return "No emotional evidence detected from lexicon or phrase rules."
        return "Lexicon hits present but no positive affect mass to summarize (treat as low evidence)."
    dom = ", ".join(dominant[:2]) if dominant else "none"
    strength = (
        "strong"
        if coverage_ratio >= 0.35
        else "moderate"
        if coverage_ratio >= 0.15
        else "weak"
    )
    bits = [f"Dominant signal: {dom} (lexical coverage {strength})."]
    if neg_hits:
        bits.append(f"Negation cues affecting {len(neg_hits)} affect-bearing term(s).")
    if int_hits:
        bits.append(f"Intensifier/downtoner cues near {len(int_hits)} affect-bearing term(s).")
    return " ".join(bits)
