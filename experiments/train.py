#!/usr/bin/env python3
"""
Train benchmark models on the GoEmotions slice and persist artifacts.

Run from the repository root (use ``python3`` if ``python`` is Anaconda 3.8)::

    python3 experiments/train.py

Requires network access the first time datasets / models are downloaded.

By default the hybrid (concat+LR) step fits on a stratified subset of training rows
(see ``--hybrid-fit-samples``) so laptop runs finish sooner; use ``0`` for the full set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from baselines import build_tfidf_logistic  # noqa: E402
from data_loader import GoEmotionsConfig, load_goemotions_benchmark  # noqa: E402
from emotfidf_wrapper import EmoTFIDFVectorizer  # noqa: E402
from hybrid_model import train_hybrid_concat_classifier  # noqa: E402
from transformer_model import TransformerTrainConfig, train_distilbert_classifier  # noqa: E402
from utils import ARTIFACTS_DIR, set_global_seed  # noqa: E402


def _maybe_subsample_texts_labels(
    texts: List[str], labels: List[int], max_n: int | None
) -> Tuple[List[str], np.ndarray]:
    if max_n is None or max_n >= len(texts):
        return texts, np.array(labels, dtype=np.int64)
    return texts[:max_n], np.array(labels[:max_n], dtype=np.int64)


def _hybrid_fit_slice(
    texts: Sequence[str],
    x_emo: np.ndarray,
    y: np.ndarray,
    max_rows: int,
    seed: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Stratified subset for hybrid concat+LR so the DistilBERT CLS pass stays short.

    ``max_rows <= 0`` or ``>= len(texts)`` keeps the full training set.
    """
    n = len(texts)
    if max_rows <= 0 or max_rows >= n:
        return list(texts), x_emo, y
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=max_rows, random_state=seed
    )
    idx, _ = next(splitter.split(np.zeros(n), y))
    idx = np.sort(idx)
    t_sub = [texts[i] for i in idx]
    return t_sub, x_emo[idx], y[idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EmoTFIDF benchmark baselines.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for debugging (uses the first N training rows).",
    )
    p.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap for debugging (uses the first N test rows).",
    )
    p.add_argument("--transformer-epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--hybrid-cls-batch-size",
        type=int,
        default=64,
        help="Batch size for DistilBERT CLS extraction in the hybrid step (raise on GPU/MPS if memory allows).",
    )
    p.add_argument(
        "--hybrid-fit-samples",
        type=int,
        default=10_000,
        help=(
            "Fit hybrid concat+LR on a stratified subset of this many training rows "
            "(skips most DistilBERT forwards during training). Use 0 for the full training set."
        ),
    )
    p.add_argument(
        "--artifacts-dir",
        type=str,
        default=str(ARTIFACTS_DIR),
        help="Directory for checkpoints (defaults to experiments/artifacts).",
    )
    p.add_argument(
        "--force-redownload-dataset",
        action="store_true",
        help="Pass download_mode=force_redownload to HuggingFace (fixes bad local cache).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    cfg = GoEmotionsConfig(
        top_k=args.top_k,
        force_redownload=args.force_redownload_dataset,
    )
    print("Loading GoEmotions (HuggingFace) and building the top-k slice …", flush=True)
    ds, class_names, new_to_old, meta = load_goemotions_benchmark(cfg)

    train_texts = ds["train"]["text"]
    train_y_list = ds["train"]["y"]
    test_texts = ds["test"]["text"]
    test_y_list = ds["test"]["y"]

    train_texts, train_y = _maybe_subsample_texts_labels(
        train_texts, train_y_list, args.max_train_samples
    )
    test_texts, test_y = _maybe_subsample_texts_labels(
        test_texts, test_y_list, args.max_test_samples
    )

    num_labels = len(class_names)
    id2label = {i: class_names[i] for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}

    meta_out: Dict[str, Any] = {
        **meta,
        "seed": args.seed,
        "num_labels": num_labels,
        "new_to_old_label_id": {str(k): v for k, v in new_to_old.items()},
        "artifacts": {},
        "subsample": {
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
        },
    }

    print(
        f"Train rows: {len(train_texts)} | Test rows (held out of training): {len(test_texts)} "
        f"| Classes: {num_labels}",
        flush=True,
    )

    # --- TF-IDF + Logistic Regression ---
    print("[1/4] TF-IDF + logistic regression …", flush=True)
    tfidf_clf = build_tfidf_logistic()
    tfidf_clf.fit(train_texts, train_y)
    tfidf_path = artifacts / "tfidf_logreg.joblib"
    joblib.dump(tfidf_clf, tfidf_path)
    meta_out["artifacts"]["tfidf_logreg"] = tfidf_path.name

    # --- EmoTFIDF features + Logistic Regression head ---
    print(
        "[2/4] EmoTFIDF corpus + document vectors + emotion logistic head "
        "(first run may download NLTK / HF assets for the library) …",
        flush=True,
    )
    emotfidf_vec = EmoTFIDFVectorizer()
    emotfidf_vec.fit(train_texts)
    X_emo_train = emotfidf_vec.transform(train_texts)
    emo_lr = LogisticRegression(
        max_iter=4000,
        solver="lbfgs",
        random_state=args.seed,
    )
    emo_lr.fit(X_emo_train, train_y)
    emo_lr_path = artifacts / "emotfidf_logreg.joblib"
    joblib.dump(emo_lr, emo_lr_path)
    meta_out["artifacts"]["emotfidf_logreg"] = emo_lr_path.name

    # --- DistilBERT fine-tuning ---
    print("[3/4] Fine-tuning DistilBERT with HuggingFace Trainer …", flush=True)
    raw_train = ds["train"]
    if args.max_train_samples is not None:
        raw_train = raw_train.select(range(min(args.max_train_samples, len(raw_train))))
    try:
        tv_split = raw_train.train_test_split(
            test_size=0.05, seed=args.seed, stratify_by_column="y"
        )
    except Exception:
        tv_split = raw_train.train_test_split(test_size=0.05, seed=args.seed)
    tr_hf = tv_split["train"]
    va_hf = tv_split["test"]

    tf_dir = str(artifacts / "distilbert_goemotions")
    tf_cfg = TransformerTrainConfig(
        output_dir=tf_dir,
        num_train_epochs=args.transformer_epochs,
        per_device_train_batch_size=args.batch_size,
        seed=args.seed,
    )
    train_distilbert_classifier(
        tr_hf,
        va_hf,
        num_labels=num_labels,
        cfg=tf_cfg,
        id2label=id2label,
        label2id=label2id,
    )
    meta_out["artifacts"]["distilbert"] = "distilbert_goemotions"

    # --- Hybrid (concat + LR) ---
    print()  # newline after HuggingFace Trainer tqdm (avoids merged lines in the log)
    print("[4/4] Hybrid classifier (CLS + EmoTFIDF features) …", flush=True)
    hybrid_cap = args.hybrid_fit_samples
    h_texts, h_X_emo, h_y = _hybrid_fit_slice(
        train_texts, X_emo_train, train_y, hybrid_cap, args.seed
    )
    if len(h_texts) < len(train_texts):
        print(
            f"  Hybrid LR fit uses {len(h_texts)} stratified rows "
            f"(cap={hybrid_cap}; full set: --hybrid-fit-samples 0).",
            flush=True,
        )
    hybrid_path = artifacts / "hybrid_concat_logreg.joblib"
    train_hybrid_concat_classifier(
        h_texts,
        h_X_emo,
        h_y,
        tf_dir,
        str(hybrid_path),
        seed=args.seed,
        cls_batch_size=args.hybrid_cls_batch_size,
    )
    meta_out["artifacts"]["hybrid_concat_logreg"] = hybrid_path.name
    meta_out["hybrid_concat"] = {
        "fit_row_count": len(h_texts),
        "hybrid_fit_samples_cap": hybrid_cap if hybrid_cap > 0 else None,
    }

    meta_out["transformer_train"] = {
        "model_name": "distilbert-base-uncased",
        "epochs": args.transformer_epochs,
        "batch_size": args.batch_size,
    }

    meta_path = artifacts / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
