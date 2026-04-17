#!/usr/bin/env python3
"""
Lightweight regression benchmark: V1 vs V2 on curated examples (not paper-scale evaluation).

Run from repo root:

    python experiments/benchmark_v1_v2_regression.py

Evaluates dominant agreement (heuristic), abstention, negation, explanation lists, and
verifier fields—intended to gate explanation quality before larger benchmarks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.regression_examples import (  # noqa: E402
    CURATED_CORPUS,
    CURATED_EXAMPLES,
)

LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]


def _v1_dominant(text: str, corpus, lex_path: Path) -> str:
    from EmoTFIDF.EmoTFIDF import EmoTFIDF

    v1 = EmoTFIDF()
    v1.set_lexicon_path(str(lex_path))
    v1.compute_tfidf(list(corpus))
    v1.set_text(text)
    v1.get_emotfidf()
    d = {k: float(v1.em_tfidf.get(k, 0.0)) for k in LABELS}
    m = max(d.values())
    if m <= 0:
        return "(no signal)"
    return max(LABELS, key=d.get)


def _v2_bundle(text: str, corpus, lex_path: Path):
    from EmoTFIDF.evidence import EmoTFIDFv2

    v2 = EmoTFIDFv2(lexicon_path=str(lex_path))
    v2.fit(list(corpus))
    r = v2.analyze(text)
    expl = v2.explain(text)
    ver = v2.verify_label(text, r.dominant_emotions[0] if r.dominant_emotions else "neutral")
    return r, expl, ver


def run_benchmark() -> Dict[str, Any]:
    lex_path = REPO_ROOT / "EmoTFIDF" / "emotions_lex.json"
    if not lex_path.is_file():
        raise SystemExit(f"Missing lexicon: {lex_path}")

    rows: List[Dict[str, Any]] = []
    passed = 0
    for ex in CURATED_EXAMPLES:
        text = ex["text"]
        v1d = _v1_dominant(text, CURATED_CORPUS, lex_path)
        r, expl, ver = _v2_bundle(text, CURATED_CORPUS, lex_path)
        v2d = r.dominant_emotions[0] if r.dominant_emotions else None

        checks: Dict[str, bool] = {}
        if ex.get("expect_abstain"):
            checks["abstain"] = not r.has_meaningful_signal and r.dominant_emotions == []
        if ex.get("expect_v2_primary"):
            checks["v2_primary"] = v2d == ex["expect_v2_primary"]
        if ex.get("expect_no_anger_dominant"):
            checks["no_anger_dom"] = v2d != "anger"
        if ex.get("expect_meaningful"):
            checks["meaningful"] = r.has_meaningful_signal
        if ex.get("expect_min_affect_terms"):
            n = ex["expect_min_affect_terms"]
            checks["min_affect_terms"] = (
                sum(
                    1
                    for c in r.term_contributions
                    if sum(max(0.0, float(v)) for v in c.per_emotion_contribution.values()) > 1e-12
                )
                >= n
            )
        if ex.get("expect_top_sadness_term_not_weak_contextual"):
            top = r.top_terms_by_emotion.get("sadness", [])
            checks["sadness_top_not_yesterday"] = bool(top) and top[0]["term"] != "yesterday"

        ok = all(checks.values()) if checks else True
        passed += int(ok)

        rows.append(
            {
                "id": ex["id"],
                "tags": ex["tags"],
                "text": text[:60],
                "v1_dominant": v1d,
                "v2_dominant": v2d,
                "checks": checks,
                "ok": ok,
                "v2_meaningful": r.has_meaningful_signal,
                "v2_low_evidence": r.has_low_evidence,
                "explain_top_tokens": [w["token"] for w in expl.get("top_contributing_words", [])[:5]],
                "verifier": {
                    "support_level": ver.get("support_level"),
                    "support_score": ver.get("support_score"),
                    "dominance_margin": ver.get("dominance_margin"),
                    "coverage_score": ver.get("coverage_score"),
                    "evidence_term_count": ver.get("evidence_term_count"),
                },
            }
        )

    return {
        "lexicon": str(lex_path),
        "cases_total": len(CURATED_EXAMPLES),
        "cases_passed_contracts": passed,
        "rows": rows,
    }


def main() -> None:
    report = run_benchmark()
    print(json.dumps(report, indent=2))
    if report["cases_passed_contracts"] < report["cases_total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
