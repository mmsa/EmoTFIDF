#!/usr/bin/env python3
"""
Verifier calibration on held-out GoEmotions: classifier predictions vs V2 lexical support.

For each test row, maps the **predicted** GoEmotions class name to a seven-way evidence label
(see :mod:`label_bridge`), runs :func:`EmoTFIDF.evidence.verifier.verify_label` on a single
:func:`EmoTFIDFv2.analyze` result, and compares support fields for **correct** vs **incorrect**
predictions.

Writes under ``experiments/eval_outputs/``:

  - ``verifier_per_row.csv``
  - ``verifier_aggregate.csv``
  - ``verifier_summary.md``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from eval_data import load_aligned_goemotions_eval_split  # noqa: E402
from hybrid_model import fused_predictions  # noqa: E402
from label_bridge import goemotion_class_to_evidence_label  # noqa: E402
from transformer_model import predict_distilbert  # noqa: E402
from utils import (  # noqa: E402
    ARTIFACTS_DIR,
    EVAL_OUTPUTS_DIR,
    ensure_repo_on_path,
    load_json,
    set_global_seed,
)
from v2_evidence_features import v2_normalized_emotion_matrix  # noqa: E402


def _weak_or_unsupported(level: str) -> bool:
    return level in ("weak", "unsupported")


def _strong_or_moderate(level: str) -> bool:
    return level in ("strong", "moderate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2 verifier support vs prediction correctness.")
    p.add_argument("--artifacts-dir", type=str, default=str(ARTIFACTS_DIR))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prediction-source",
        choices=("distilbert", "v1_fusion", "v2_fusion"),
        default="v2_fusion",
        help="Which classifier outputs to audit with the verifier.",
    )
    p.add_argument(
        "--fusion-weight-transformer",
        type=float,
        default=0.8,
        help="Fusion weight when prediction-source is v1_fusion or v2_fusion.",
    )
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument(
        "--include-text",
        action="store_true",
        help="Store full text in per-row CSV (large). Default: excerpt only.",
    )
    p.add_argument("--text-excerpt-chars", type=int, default=240)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    ensure_repo_on_path()

    from EmoTFIDF.evidence import EmoTFIDFv2
    from EmoTFIDF.evidence.verifier import verify_label as verify_label_fn

    artifacts = Path(args.artifacts_dir)
    meta_path = artifacts / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run: python experiments/train.py\n"
            "Then re-run this script."
        )
    meta = load_json(meta_path)
    emo_lr: LogisticRegression = joblib.load(artifacts / meta["artifacts"]["emotfidf_logreg"])
    tf_dir = str(artifacts / meta["artifacts"]["distilbert"])

    train_texts, train_y, test_texts, test_y, class_names = load_aligned_goemotions_eval_split(
        meta_path,
        max_test_samples=args.max_test_samples,
    )
    name_for = class_names.__getitem__

    print(
        f"Verifier analysis | test={len(test_texts)} | preds={args.prediction_source}",
        flush=True,
    )

    from emotfidf_wrapper import EmoTFIDFVectorizer

    emo_v1 = EmoTFIDFVectorizer()
    emo_v1.fit(train_texts)
    X_v1_test = emo_v1.transform(test_texts)

    X_v2_train = v2_normalized_emotion_matrix(train_texts, train_texts)
    X_v2_test = v2_normalized_emotion_matrix(test_texts, train_texts)
    lr_v2 = LogisticRegression(max_iter=4000, solver="lbfgs", random_state=args.seed)
    lr_v2.fit(X_v2_train, train_y)

    pred_bert, prob_bert = predict_distilbert(test_texts, tf_dir)
    prob_v1 = emo_lr.predict_proba(X_v1_test)
    prob_v2 = lr_v2.predict_proba(X_v2_test)
    w = float(args.fusion_weight_transformer)

    if args.prediction_source == "distilbert":
        y_pred = pred_bert
    elif args.prediction_source == "v1_fusion":
        y_pred = fused_predictions(prob_bert, prob_v1, w)
    else:
        y_pred = fused_predictions(prob_bert, prob_v2, w)

    v2 = EmoTFIDFv2()
    v2.fit([str(t) for t in train_texts])

    rows: List[Dict[str, Any]] = []
    for i, text in enumerate(test_texts):
        yt = int(test_y[i])
        yp = int(y_pred[i])
        pred_go = name_for(yp)
        evidence_label = goemotion_class_to_evidence_label(pred_go)
        analysis = v2.analyze(str(text))
        vr = verify_label_fn(analysis, evidence_label)
        d = vr.to_dict()
        correct = yt == yp
        excerpt = str(text) if args.include_text else str(text)[: max(0, int(args.text_excerpt_chars))]
        rows.append(
            {
                "row_index": i,
                "text_excerpt": excerpt,
                "y_true": yt,
                "y_pred": yp,
                "correct": correct,
                "true_class": name_for(yt),
                "pred_class": pred_go,
                "evidence_label_for_pred": evidence_label,
                "support_score": d["support_score"],
                "support_level": d["support_level"],
                "dominance_margin": d.get("dominance_margin", 0.0),
                "coverage_score": d.get("coverage_score", 0.0),
                "evidence_term_count": d.get("evidence_term_count", 0),
                "supporting_terms_json": json.dumps(d.get("supporting_terms", []), ensure_ascii=False),
                "conflicting_emotions_json": json.dumps(
                    d.get("conflicting_emotions", []), ensure_ascii=False
                ),
                "notes_json": json.dumps(d.get("notes", []), ensure_ascii=False),
            }
        )

    df = pd.DataFrame(rows)
    EVAL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    per_path = EVAL_OUTPUTS_DIR / "verifier_per_row.csv"
    df.to_csv(per_path, index=False)

    def _subset(mask: np.ndarray) -> pd.DataFrame:
        return df.loc[mask]

    m_correct = df["correct"].to_numpy(dtype=bool)
    m_incorrect = ~m_correct
    correct_df = _subset(m_correct)
    incorrect_df = _subset(m_incorrect)

    n_cor = int(correct_df.shape[0])
    n_inc = int(incorrect_df.shape[0])

    mean_sup_cor = float(correct_df["support_score"].mean()) if n_cor else float("nan")
    mean_sup_inc = float(incorrect_df["support_score"].mean()) if n_inc else float("nan")

    if n_inc:
        inc_weak = incorrect_df["support_level"].map(_weak_or_unsupported)
        p_inc_flagged = float(inc_weak.mean())
    else:
        p_inc_flagged = float("nan")

    if n_cor:
        cor_sm = correct_df["support_level"].map(_strong_or_moderate)
        p_cor_flagged = float(cor_sm.mean())
    else:
        p_cor_flagged = float("nan")

    agg = pd.DataFrame(
        [
            {
                "prediction_source": args.prediction_source,
                "fusion_weight_transformer": w,
                "n_test": len(df),
                "n_correct": n_cor,
                "n_incorrect": n_inc,
                "mean_support_score_correct": mean_sup_cor,
                "mean_support_score_incorrect": mean_sup_inc,
                "prop_incorrect_weak_or_unsupported": p_inc_flagged,
                "prop_correct_strong_or_moderate": p_cor_flagged,
            }
        ]
    )
    agg_path = EVAL_OUTPUTS_DIR / "verifier_aggregate.csv"
    agg.to_csv(agg_path, index=False)

    md_path = EVAL_OUTPUTS_DIR / "verifier_summary.md"
    level_counts = (
        df.groupby(["correct", "support_level"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .to_string()
    )
    md = "\n".join(
        [
            "# V2 verifier support vs classifier correctness",
            "",
            f"- Prediction source: **{args.prediction_source}**",
            f"- Fusion weight (DistilBERT): **{w}** (when applicable)",
            f"- Test rows: **{len(df)}**",
            "",
            "## Aggregate metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Mean support_score (correct) | {mean_sup_cor:.4f} |",
            f"| Mean support_score (incorrect) | {mean_sup_inc:.4f} |",
            f"| Share of **incorrect** with weak/unsupported verifier | {p_inc_flagged:.4f} |",
            f"| Share of **correct** with strong/moderate verifier | {p_cor_flagged:.4f} |",
            "",
            "## Counts by correctness × support_level",
            "",
            "```",
            level_counts,
            "```",
            "",
            "## Behavioral regression (separate evaluation)",
            "",
            "```bash",
            "python experiments/benchmark_v1_v2_regression.py",
            "pytest tests/test_benchmark_regression_smoke.py -q",
            "```",
            "",
        ]
    )
    md_path.write_text(md, encoding="utf-8")
    print(f"Wrote {per_path}\nWrote {agg_path}\nWrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
