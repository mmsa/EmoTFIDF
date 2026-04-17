"""Aggregation helpers and deterministic richer feature vectors."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from EmoTFIDF.v2.lexicon import DEFAULT_EMOTION_LABELS
from EmoTFIDF.v2.schemas import TermContribution


def emotion_dict_zeros(labels: List[str]) -> Dict[str, float]:
    return {e: 0.0 for e in labels}


def normalize_shifted_l1(raw: Dict[str, float], labels: List[str]) -> Dict[str, float]:
    """Shift so the minimum is zero, then L1-normalize to a probability-like vector."""
    vals = [raw.get(e, 0.0) for e in labels]
    m = min(vals) if vals else 0.0
    shifted = [max(0.0, raw.get(e, 0.0) - m) for e in labels]
    s = sum(shifted)
    if s <= 0.0:
        u = 1.0 / len(labels)
        return {e: u for e in labels}
    return {e: shifted[i] / s for i, e in enumerate(labels)}


def softmax(raw: Dict[str, float], labels: List[str], temperature: float = 1.0) -> Dict[str, float]:
    """Numerically stable softmax over the label-ordered vector."""
    t = max(temperature, 1e-6)
    xs = [raw.get(e, 0.0) / t for e in labels]
    mx = max(xs)
    exps = [math.exp(x - mx) for x in xs]
    s = sum(exps) or 1.0
    return {e: exps[i] / s for i, e in enumerate(labels)}


def distribution_entropy(p: Dict[str, float], labels: List[str]) -> float:
    """Shannon entropy of the label distribution (natural log)."""
    h = 0.0
    for e in labels:
        x = p.get(e, 0.0)
        if x > 0.0:
            h -= x * math.log(x + 1e-12)
    return h


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
) -> Dict[str, List[Dict[str, Any]]]:
    """Aggregate max positive contribution per (emotion, token)."""
    acc: Dict[str, Dict[str, float]] = {e: {} for e in labels}
    for c in contribs:
        for e, v in c.per_emotion_contribution.items():
            if e not in acc:
                continue
            acc[e][c.token] = acc[e].get(c.token, 0.0) + max(0.0, v)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for e in labels:
        items = sorted(acc[e].items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        out[e] = [{"term": t, "weight": round(w, 6)} for t, w in items if w > 0.0]
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
) -> Tuple[List[float], List[str]]:
    """
    Flatten a deterministic feature vector (order documented in ``feature_names``).

    Layout:
        [0:7)   raw emotion scores (label order)
        [7:14)  softmax-normalized emotion scores
        [14)    coverage_ratio
        [15)    matched_lexicon_term_count (float)
        [16)    unmatched_token_ratio
        [17:24) per-emotion max positive term contribution
        [24:31) per-emotion top-3 positive term sum
        [31)    negation cue count (float)
        [32)    intensifier / downtoner cue count (float)
        [33)    entropy of softmax distribution
        [34)    dominant margin (top1 - top2 softmax)
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
