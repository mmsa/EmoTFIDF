"""Aggregation helpers and deterministic richer feature vectors."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from EmoTFIDF.evidence.lexeme_prior import explanation_rank_multiplier
from EmoTFIDF.evidence.lexicon import DEFAULT_EMOTION_LABELS
from EmoTFIDF.evidence.schemas import TermContribution


def emotion_dict_zeros(labels: List[str]) -> Dict[str, float]:
    return {e: 0.0 for e in labels}


def relu_scores(raw: Dict[str, float], labels: List[str]) -> Dict[str, float]:
    """Positive parts only (used so negated dimensions do not leak mass via min-shift tricks)."""
    return {e: max(0.0, float(raw.get(e, 0.0))) for e in labels}


def normalize_positive_l1(raw: Dict[str, float], labels: List[str]) -> Dict[str, float]:
    """
    L1-normalize max(0, raw[e]) across labels.

    If there is **no positive mass**, return all zeros (no fake uniform distribution).
    This replaces older min-shift+L1 logic that could assign mass to unrelated emotions
    when one emotion was strongly negative (e.g. negated joy).
    """
    pos = [max(0.0, float(raw.get(e, 0.0))) for e in labels]
    s = sum(pos)
    if s <= 0.0:
        return {e: 0.0 for e in labels}
    return {e: pos[i] / s for i, e in enumerate(labels)}


def normalize_shifted_l1(raw: Dict[str, float], labels: List[str]) -> Dict[str, float]:
    """Deprecated for emotion display; prefer :func:`normalize_positive_l1`. Kept for tests if needed."""
    return normalize_positive_l1(raw, labels)


def softmax(raw: Dict[str, float], labels: List[str], temperature: float = 1.0) -> Dict[str, float]:
    """Numerically stable softmax over the label-ordered vector."""
    t = max(temperature, 1e-6)
    xs = [raw.get(e, 0.0) / t for e in labels]
    mx = max(xs)
    exps = [math.exp(x - mx) for x in xs]
    s = sum(exps) or 1.0
    return {e: exps[i] / s for i, e in enumerate(labels)}


def softmax_positive_or_zeros(
    raw: Dict[str, float], labels: List[str], temperature: float = 1.0
) -> Dict[str, float]:
    """
    Softmax on ReLU(raw). If there is no positive mass, return zeros (not a uniform).

    Uniform softmax on all-zero input was a source of fake balanced ``emotions`` on
    punctuation-only text in earlier builds.
    """
    pos = relu_scores(raw, labels)
    if sum(pos.values()) <= 0.0:
        return {e: 0.0 for e in labels}
    return softmax(pos, labels, temperature)


def distribution_entropy(p: Dict[str, float], labels: List[str]) -> float:
    """Shannon entropy of the label distribution (natural log)."""
    h = 0.0
    for e in labels:
        x = p.get(e, 0.0)
        if x > 0.0:
            h -= x * math.log(x + 1e-12)
    return h


def select_dominant_emotions(
    norm_positive: Dict[str, float],
    raw: Dict[str, float],
    labels: List[str],
    *,
    has_meaningful_signal: bool,
    single_dominant_margin: float = 0.06,
) -> Tuple[List[str], float, float, float]:
    """
    Choose dominant label(s) from positive-mass-normalized scores only.

    Returns (dominant_list, top1_score, top2_score, margin top1-top2).
    When *has_meaningful_signal* is False or all normalized scores are zero, returns ([], 0, 0, 0).
    Tie-break equal scores with label name for determinism (``anger`` before ``disgust``).
    """
    if not has_meaningful_signal or sum(norm_positive.get(e, 0.0) for e in labels) <= 0.0:
        return [], 0.0, 0.0, 0.0
    ranked = sorted(
        labels,
        key=lambda e: (-norm_positive.get(e, 0.0), -max(0.0, float(raw.get(e, 0.0))), e),
    )
    top1, top2 = ranked[0], ranked[1]
    s1 = float(norm_positive[top1])
    s2 = float(norm_positive[top2])
    margin = s1 - s2
    if margin >= single_dominant_margin:
        return [top1], s1, s2, margin
    topk = [e for e in ranked if norm_positive.get(e, 0.0) > 0.0][:3]
    return topk, s1, s2, margin


def dominant_margin(norm: Dict[str, float], labels: List[str]) -> float:
    ranked = sorted((norm.get(e, 0.0), e) for e in labels)
    if len(ranked) < 2:
        return 0.0
    top1 = ranked[-1][0]
    top2 = ranked[-2][0]
    return top1 - top2


def top_terms_by_emotion_from_contribs(
    contribs: List[TermContribution],
    labels: List[str],
    top_k: int = 5,
    lexicon: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Aggregate positive mass per (emotion, token).

    When *lexicon* is provided, terms are **re-ranked** for explanation quality using
    :func:`EmoTFIDF.evidence.lexeme_prior.explanation_rank_multiplier` so weak contextual
    lexemes (e.g. *yesterday* tagged sadness in NRC) do not float above clearer affect words.
    Reported ``weight`` remains the **actual summed positive contribution** (lexicon-grounded).
    """
    acc: Dict[str, Dict[str, float]] = {e: {} for e in labels}
    for c in contribs:
        for e, v in c.per_emotion_contribution.items():
            if e not in acc:
                continue
            acc[e][c.token] = acc[e].get(c.token, 0.0) + max(0.0, v)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for e in labels:
        items = list(acc[e].items())
        if not items:
            out[e] = []
            continue
        if lexicon is not None:
            items.sort(
                key=lambda kv: (
                    -kv[1] * explanation_rank_multiplier(kv[0], lexicon),
                    -kv[1],
                    kv[0],
                )
            )
        else:
            items.sort(key=lambda kv: (-kv[1], kv[0]))
        out[e] = [{"term": t, "weight": round(w, 6)} for t, w in items[:top_k] if w > 0.0]
    return out


def build_feature_vector(
    raw: Dict[str, float],
    norm_softmax: Dict[str, float],
    coverage_ratio: float,
    matched_count: int,
    unmatched_ratio: float,
    per_emotion_max: Dict[str, float],
    per_emotion_topk_sum: Dict[str, float],
    negation_count: int,
    intensifier_count: int,
    entropy: float,
    margin: float,
    labels: List[str],
    *,
    meaningful_signal_01: float,
    total_positive_evidence: float,
    total_abs_evidence: float,
) -> Tuple[List[float], List[str]]:
    """
    Flatten a deterministic feature vector (order documented in ``feature_names``).

    Layout:
        [0:7)   raw emotion scores (label order)
        [7:14)  softmax on positive mass (zeros when no positive evidence)
        [14)    coverage_ratio
        [15)    matched_lexicon_term_count (float)
        [16)    unmatched_token_ratio
        [17:24) per-emotion max positive term contribution
        [24:31) per-emotion top-3 positive term sum
        [31)    negation cue count (float)
        [32)    intensifier / downtoner cue count (float)
        [33)    entropy of softmax distribution (0 when no positive mass)
        [34)    dominant margin (top1 - top2 on positive-mass softmax; 0 if no mass)
        [35)    meaningful_signal (0/1)
        [36)    total_positive_evidence (sum ReLU raw)
        [37)    total_abs_evidence (sum |raw|)
    """
    names: List[str] = []
    vec: List[float] = []
    for e in labels:
        names.append(f"raw_{e}")
        vec.append(float(raw.get(e, 0.0)))
    for e in labels:
        names.append(f"softmax_norm_{e}")
        vec.append(float(norm_softmax.get(e, 0.0)))
    names.extend(
        [
            "coverage_ratio",
            "matched_lexicon_term_count",
            "unmatched_token_ratio",
        ]
    )
    vec.extend([float(coverage_ratio), float(matched_count), float(unmatched_ratio)])
    for e in labels:
        names.append(f"max_term_contrib_{e}")
        vec.append(float(per_emotion_max.get(e, 0.0)))
    for e in labels:
        names.append(f"top3_term_sum_{e}")
        vec.append(float(per_emotion_topk_sum.get(e, 0.0)))
    names.extend(
        [
            "negation_count",
            "intensifier_modifier_count",
            "softmax_entropy",
            "dominant_margin_top1_minus_top2",
        ]
    )
    vec.extend([float(negation_count), float(intensifier_count), float(entropy), float(margin)])
    names.extend(
        [
            "meaningful_signal_01",
            "total_positive_evidence",
            "total_abs_evidence",
        ]
    )
    vec.extend(
        [
            float(meaningful_signal_01),
            float(total_positive_evidence),
            float(total_abs_evidence),
        ]
    )
    return vec, names


def per_emotion_max_and_topk(
    contribs: List[TermContribution],
    labels: List[str],
    k: int = 3,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-emotion max single-token positive contribution and sum of top-k positive token masses."""
    pos_by_e: Dict[str, List[float]] = {e: [] for e in labels}
    for c in contribs:
        for e, v in c.per_emotion_contribution.items():
            if e not in pos_by_e:
                continue
            if v > 0.0:
                pos_by_e[e].append(v)
    maxv: Dict[str, float] = {}
    topk: Dict[str, float] = {}
    for e in labels:
        arr = sorted(pos_by_e[e], reverse=True)
        maxv[e] = arr[0] if arr else 0.0
        topk[e] = sum(arr[:k])
    return maxv, topk
