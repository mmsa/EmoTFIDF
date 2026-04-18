#!/usr/bin/env python3
"""
DistilBERT vs probability fusion (V1 EmoTFIDF head vs V2 evidence head).

Reuses trained artifacts from ``experiments/train.py`` (DistilBERT + EmoTFIDF+LR).
Trains a fresh **seven-way logistic head on V2 normalized vectors** on the training split,
then reports::

  - DistilBERT alone (argmax on transformer probs)
  - 0.8 * DistilBERT + 0.2 * V1 EmoTFIDF+LR (existing fusion in ``hybrid_model``)
  - 0.8 * DistilBERT + 0.2 * V2 seven-way+LR (same fusion weighting for comparability)

Outputs under ``experiments/eval_outputs/``:

  - ``fusion_ablation_metrics.csv``
  - ``fusion_ablation_summary.md``

Does **not** run full paper benchmarks; intended as a controlled ablation stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from eval_data import load_aligned_goemotions_eval_split  # noqa: E402
from hybrid_model import fused_predictions  # noqa: E402
from transformer_model import predict_distilbert  # noqa: E402
from utils import (  # noqa: E402
    ARTIFACTS_DIR,
    EVAL_OUTPUTS_DIR,
    ensure_repo_on_path,
    load_json,
    set_global_seed,
)
from v2_evidence_features import v2_normalized_emotion_matrix  # noqa: E402


def _metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DistilBERT vs V1/V2 fusion ablation.")
    p.add_argument("--artifacts-dir", type=str, default=str(ARTIFACTS_DIR))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap on evaluation rows (in addition to train.py subsample).",
    )
    p.add_argument(
        "--fusion-weight-transformer",
        type=float,
        default=0.8,
        help="Weight on DistilBERT probabilities (same as hybrid_model default).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    ensure_repo_on_path()
    artifacts = Path(args.artifacts_dir)
    meta_path = artifacts / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run: python experiments/train.py\n"
            "Then re-run this ablation."
        )
    meta = load_json(meta_path)
    emo_lr: LogisticRegression = joblib.load(artifacts / meta["artifacts"]["emotfidf_logreg"])
    tf_dir = str(artifacts / meta["artifacts"]["distilbert"])

    train_texts, train_y, test_texts, test_y, _ = load_aligned_goemotions_eval_split(
        meta_path,
        max_test_samples=args.max_test_samples,
    )
    print(
        f"Fusion ablation | train={len(train_texts)} test={len(test_texts)} "
        f"| transformer_weight={args.fusion_weight_transformer}",
        flush=True,
    )

    from emotfidf_wrapper import EmoTFIDFVectorizer

    emo_v1 = EmoTFIDFVectorizer()
    print("V1 EmoTFIDF: fit + transform …", flush=True)
    emo_v1.fit(train_texts)
    X_v1_train = emo_v1.transform(train_texts)
    X_v1_test = emo_v1.transform(test_texts)

    print("V2 evidence: fit + seven-way matrix …", flush=True)
    X_v2_train = v2_normalized_emotion_matrix(train_texts, train_texts)
    X_v2_test = v2_normalized_emotion_matrix(test_texts, train_texts)

    lr_v2 = LogisticRegression(
        max_iter=4000,
        solver="lbfgs",
        random_state=args.seed,
    )
    lr_v2.fit(X_v2_train, train_y)

    print("DistilBERT inference …", flush=True)
    pred_bert, prob_bert = predict_distilbert(test_texts, tf_dir)

    prob_v1 = emo_lr.predict_proba(X_v1_test)
    prob_v2 = lr_v2.predict_proba(X_v2_test)

    w = float(args.fusion_weight_transformer)
    pred_fusion_v1 = fused_predictions(prob_bert, prob_v1, w)
    pred_fusion_v2 = fused_predictions(prob_bert, prob_v2, w)

    rows: List[Dict[str, Any]] = [
        _metrics("distilbert_alone", test_y, pred_bert),
        _metrics(f"distilbert_plus_v1_fusion_{w:.1f}", test_y, pred_fusion_v1),
        _metrics(f"distilbert_plus_v2_fusion_{w:.1f}", test_y, pred_fusion_v2),
    ]
    df = pd.DataFrame(rows)

    EVAL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EVAL_OUTPUTS_DIR / "fusion_ablation_metrics.csv"
    md_path = EVAL_OUTPUTS_DIR / "fusion_ablation_summary.md"
    df.to_csv(csv_path, index=False)

    md_lines: List[str] = [
        "# DistilBERT vs V1/V2 probability fusion (ablation)",
        "",
        "Interpretable emotional evidence layer — controlled comparison, not a full benchmark.",
        "",
        f"See also `experiments/EVAL_LLM_ERA_README.md` for verifier analysis and other stages.",
        "",
        "## Setup",
        "",
        f"- Artifacts: `{artifacts}`",
        f"- Test rows: **{len(test_y)}**",
        f"- Fusion: **{w}** × DistilBERT probs + **{1 - w:.1f}** × lexicon-side LR probs",
        "- V1 head: pre-trained `emotfidf_logreg` on legacy seven-way EmoTFIDF vectors.",
        "- V2 head: `LogisticRegression` fit on this run using V2 `normalized_emotion_scores` (seven-way).",
        "",
        "## Metrics",
        "",
        "| Model | Accuracy | Macro F1 |",
        "|---|---:|---:|",
    ]
    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} |"
        )
    md_lines.extend(
        [
            "",
            "## Behavioral regression (separate)",
            "",
            "Library correctness / explanation smoke tests (not GoEmotions accuracy):",
            "",
            "```bash",
            "pytest tests/test_benchmark_regression_smoke.py tests/test_v2_explanation_quality.py -q",
            "```",
            "",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {csv_path}\nWrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
