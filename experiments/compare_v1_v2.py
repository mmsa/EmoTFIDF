#!/usr/bin/env python3
"""
Compare legacy V1 EmoTFIDF (lexicon + TF-IDF emotfidf) vs V2 EmoTFIDFv2 (EmoTFIDF.evidence).

Run from repo root:

    python experiments/compare_v1_v2.py

Uses the packaged emotions_lex.json (no URL fetch for the V1 instance after set_lexicon_path).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LEX_PATH = REPO_ROOT / "EmoTFIDF" / "emotions_lex.json"

LABELS: List[str] = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]

CORPUS: List[str] = [
    "I am happy today and everything feels great and wonderful.",
    "I am not happy today and everything feels wrong and miserable.",
    "I feel sad and disappointed about the terrible news.",
    "This makes me angry and frustrated with the situation.",
    "I am surprised and curious about what happened next.",
    "The meeting was neutral and boring without much emotion.",
]

TEST_TEXTS: List[str] = [
    "I am very happy today!",
    "I am not happy today.",
    "I feel furious and angry about this.",
    "She was sad and crying yesterday.",
    "What a wonderful surprise party!",
    "!!! ...",
]


def _v1_em_tfidf(corpus: Sequence[str], text: str):
    from EmoTFIDF.EmoTFIDF import EmoTFIDF

    v1 = EmoTFIDF()
    v1.set_lexicon_path(str(LEX_PATH))
    v1.compute_tfidf(list(corpus))
    v1.set_text(text)
    v1.get_emotfidf()
    return {k: float(v1.em_tfidf.get(k, 0.0)) for k in LABELS}


def _v2_analysis(corpus: Sequence[str], text: str):
    from EmoTFIDF.evidence import EmoTFIDFv2

    v2 = EmoTFIDFv2(lexicon_path=str(LEX_PATH))
    v2.fit(list(corpus))
    return v2.analyze(text)


def _v2_normalized(corpus: Sequence[str], text: str) -> Dict[str, float]:
    r = _v2_analysis(corpus, text)
    return {k: float(r.normalized_emotion_scores.get(k, 0.0)) for k in LABELS}


def _argmax_v1(d: Dict[str, float]) -> str:
    m = max(d.get(k, 0.0) for k in LABELS)
    if m <= 0.0:
        return "(no signal)"
    top = max(LABELS, key=lambda k: d.get(k, 0.0))
    runners = [k for k in LABELS if abs(d.get(k, 0.0) - m) < 1e-12]
    if len(runners) > 1:
        return f"{top} (tie)"
    return top


def _v2_dominant_label(r) -> str:
    if not r.dominant_emotions:
        return "(no dominant)"
    if len(r.dominant_emotions) == 1:
        return r.dominant_emotions[0]
    return f"{r.dominant_emotions[0]} (+{len(r.dominant_emotions) - 1} close)"


def _l1(a: Dict[str, float], b: Dict[str, float]) -> float:
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in LABELS)


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    va = [a[k] for k in LABELS]
    vb = [b[k] for k in LABELS]
    na = math.sqrt(sum(x * x for x in va)) or 1e-9
    nb = math.sqrt(sum(x * x for x in vb)) or 1e-9
    return sum(x * y for x, y in zip(va, vb)) / (na * nb)


def run_rows(
    corpus: Sequence[str] = CORPUS,
    texts: Sequence[str] = TEST_TEXTS,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for text in texts:
        s1 = _v1_em_tfidf(corpus, text)
        r2 = _v2_analysis(corpus, text)
        s2 = {k: float(r2.normalized_emotion_scores.get(k, 0.0)) for k in LABELS}
        v1d, v2d = _argmax_v1(s1), _v2_dominant_label(r2)
        rows.append(
            {
                "text": text[:72] + ("…" if len(text) > 72 else ""),
                "v1_dominant": v1d,
                "v2_dominant": v2d,
                "l1_dist": round(_l1(s1, s2), 4),
                "cosine": round(_cosine(s1, s2), 4),
                "v1_scores": s1,
                "v2_scores": s2,
                "v2_has_meaningful_signal": r2.has_meaningful_signal,
                "v2_has_low_evidence": r2.has_low_evidence,
                "v2_dominance_margin": round(r2.dominance_margin, 4),
                "v2_top_terms_preview": [
                    {"emotion": e, "terms": r2.top_terms_by_emotion.get(e, [])[:3]}
                    for e in LABELS
                    if r2.top_terms_by_emotion.get(e)
                ][:4],
                "v2_negation_hits": len(r2.negation_hits),
                "v2_intensifier_hits": len(r2.intensifier_hits),
            }
        )
    return rows


def main() -> None:
    if not LEX_PATH.is_file():
        raise SystemExit(f"Missing lexicon at {LEX_PATH}")
    rows = run_rows()
    print(f"Lexicon: {LEX_PATH}")
    print(f"Corpus: {len(CORPUS)} docs; {len(TEST_TEXTS)} test strings")
    print()
    print(
        "Note: V1 get_emotfidf has no negation handling; V2 uses cue windows, positive-mass "
        "normalization, and no uniform fallback on empty input. L1/cosine are indicative only."
    )
    print()
    for r in rows:
        print(f"Text: {r['text']!r}")
        print(
            f"  V1 dominant: {r['v1_dominant']:14}  V2: {r['v2_dominant']:14}  "
            f"meaningful={r['v2_has_meaningful_signal']}  low_evidence={r['v2_has_low_evidence']}  "
            f"margin={r['v2_dominance_margin']}"
        )
        print(
            f"  neg_hits={r['v2_negation_hits']}  int_hits={r['v2_intensifier_hits']}  "
            f"L1={r['l1_dist']}  cosine={r['cosine']}"
        )
        if r["v2_top_terms_preview"]:
            print(f"  V2 top terms (sample): {json.dumps(r['v2_top_terms_preview'], indent=None)[:200]}…")
    print()
    print("Full vectors (JSON) for last row:")
    print(json.dumps({"v1": rows[-1]["v1_scores"], "v2": rows[-1]["v2_scores"]}, indent=2))


if __name__ == "__main__":
    main()
