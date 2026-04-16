#!/usr/bin/env python3
"""
Evaluate trained checkpoints on the official test split and write reports.

Run from the repository root::

    python experiments/evaluate.py

Reads ``artifacts/meta.json`` produced by ``train.py`` and writes
``experiments/results.csv`` plus ``experiments/explanations.txt``.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from baselines import NRCLexiconBaseline  # noqa: E402
from data_loader import GoEmotionsConfig, load_goemotions_benchmark  # noqa: E402
from emotfidf_wrapper import EmoTFIDFVectorizer  # noqa: E402
from hybrid_model import fused_predictions, predict_hybrid_concat  # noqa: E402
from transformer_model import predict_distilbert  # noqa: E402
from utils import (  # noqa: E402
    ARTIFACTS_DIR,
    EXPLANATIONS_TXT,
    RESULTS_CSV,
    load_json,
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate benchmark models.")
    p.add_argument(
        "--artifacts-dir",
        type=str,
        default=str(ARTIFACTS_DIR),
        help="Directory that contains meta.json and checkpoints.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap on evaluation rows (must match training subsample).",
    )
    return p.parse_args()


def _aligned_dataset(meta: Dict[str, Any], max_test: int | None):
    cfg = GoEmotionsConfig(
        top_k=int(meta.get("top_k", 8)),
        dataset_name=str(meta.get("dataset_name", "go_emotions")),
        dataset_config=str(meta.get("dataset_config", "simplified")),
    )
    ds, class_names, _, _ = load_goemotions_benchmark(cfg)
    if class_names != meta["class_names"]:
        raise ValueError("Class name mismatch between meta.json and freshly loaded data.")
    test_texts = ds["test"]["text"]
    test_y = np.array(ds["test"]["y"], dtype=np.int64)
    train_texts = ds["train"]["text"]
    train_y = np.array(ds["train"]["y"], dtype=np.int64)

    sub = meta.get("subsample") or {}
    max_tr = sub.get("max_train_samples")
    max_te = sub.get("max_test_samples")
    if max_tr is not None:
        train_texts = train_texts[: int(max_tr)]
        train_y = train_y[: int(max_tr)]
    if max_te is not None:
        test_texts = test_texts[: int(max_te)]
        test_y = test_y[: int(max_te)]
    if max_test is not None:
        test_texts = test_texts[: int(max_test)]
        test_y = test_y[: int(max_test)]
    return train_texts, train_y, test_texts, test_y, class_names


def _metrics_row(model: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        "Model": model,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _write_explanations(
    path: Path,
    samples: List[str],
    rows: List[Dict[str, Any]],
    emotfidf_terms: List[List[Tuple[str, float]]],
) -> None:
    lines: List[str] = []
    for i, text in enumerate(samples):
        lines.append(f"===== Sample {i + 1} =====")
        lines.append(f"Text: {text}")
        lines.append("Predictions:")
        for k, v in rows[i].items():
            if k == "gold":
                lines.append(f"  gold_label: {v}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("Top EmoTFIDF lexicon terms (word, TF-IDF weight):")
        if emotfidf_terms[i]:
            for w, s in emotfidf_terms[i]:
                lines.append(f"  {w!r}: {s:.5f}")
        else:
            lines.append("  (none)")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    artifacts = Path(args.artifacts_dir)
    meta_path = artifacts / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run python experiments/train.py before evaluating."
        )
    meta = load_json(meta_path)
    train_texts, train_y, test_texts, test_y, class_names = _aligned_dataset(
        meta, args.max_test_samples
    )

    tfidf_clf = joblib.load(artifacts / meta["artifacts"]["tfidf_logreg"])
    emo_lr: LogisticRegression = joblib.load(artifacts / meta["artifacts"]["emotfidf_logreg"])
    tf_dir = str(artifacts / meta["artifacts"]["distilbert"])
    hybrid_path = str(artifacts / meta["artifacts"]["hybrid_concat_logreg"])

    emotfidf_vec = EmoTFIDFVectorizer()
    emotfidf_vec.fit(list(train_texts))
    X_emo_test = emotfidf_vec.transform(list(test_texts))

    nrc = NRCLexiconBaseline(class_names)
    pred_nrc = nrc.predict(test_texts)
    pred_tfidf = tfidf_clf.predict(test_texts).astype(np.int64)
    pred_emo_lr = emo_lr.predict(X_emo_test).astype(np.int64)
    pred_bert, prob_bert = predict_distilbert(list(test_texts), tf_dir)
    pred_hybrid = predict_hybrid_concat(
        test_texts, X_emo_test, tf_dir, hybrid_path
    )
    prob_emo = emo_lr.predict_proba(X_emo_test)
    pred_fusion = fused_predictions(prob_bert, prob_emo, transformer_weight=0.8)

    table = [
        _metrics_row("Lexicon (NRC)", test_y, pred_nrc),
        _metrics_row("EmoTFIDF + LR", test_y, pred_emo_lr),
        _metrics_row("TF-IDF + LR", test_y, pred_tfidf),
        _metrics_row("DistilBERT", test_y, pred_bert),
        _metrics_row("Hybrid (concat + LR)", test_y, pred_hybrid),
        _metrics_row("Hybrid (0.8 T + 0.2 E)", test_y, pred_fusion),
    ]
    df = pd.DataFrame(table)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)

    md_lines = [
        "| Model | Accuracy | Macro F1 | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Macro F1']:.4f} | "
            f"{r['Precision']:.4f} | {r['Recall']:.4f} |"
        )
    print("\n".join(md_lines))
    print(f"\nSaved CSV to {RESULTS_CSV}")

    rng = random.Random(args.seed)
    idxs = rng.sample(range(len(test_texts)), k=min(5, len(test_texts)))
    sample_texts = [test_texts[i] for i in idxs]
    sample_rows: List[Dict[str, Any]] = []
    sample_terms: List[List[Tuple[str, float]]] = []
    for t, idx in zip(sample_texts, idxs):
        xv = emotfidf_vec.transform([t])
        sample_terms.append(emotfidf_vec.top_emotion_lexicon_terms(t, top_k=12))
        pb, pr = predict_distilbert([t], tf_dir)
        sample_rows.append(
            {
                "Lexicon (NRC)": class_names[nrc.predict_one(t)],
                "EmoTFIDF + LR": class_names[int(emo_lr.predict(xv)[0])],
                "TF-IDF + LR": class_names[int(tfidf_clf.predict([t])[0])],
                "DistilBERT": class_names[int(pb[0])],
                "Hybrid (concat + LR)": class_names[
                    int(predict_hybrid_concat([t], xv, tf_dir, hybrid_path)[0])
                ],
                "Hybrid (0.8 T + 0.2 E)": class_names[
                    int(
                        fused_predictions(
                            pr,
                            emo_lr.predict_proba(xv),
                            0.8,
                        )[0]
                    )
                ],
                "gold": class_names[int(test_y[idx])],
            }
        )

    _write_explanations(EXPLANATIONS_TXT, sample_texts, sample_rows, sample_terms)
    print(f"Wrote qualitative explanations to {EXPLANATIONS_TXT}")


if __name__ == "__main__":
    main()
