"""
Shared GoEmotions loading aligned with ``artifacts/meta.json`` for evaluation scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from data_loader import GoEmotionsConfig, load_goemotions_benchmark
from utils import load_json


def load_aligned_goemotions_eval_split(
    meta_path: Path,
    *,
    max_test_samples: int | None = None,
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, List[str]]:
    """
    Return train/test texts and integer labels using the same ordering and optional
    subsampling recorded in ``meta.json`` from ``experiments/train.py``.
    """
    meta: Dict[str, Any] = load_json(meta_path)
    cfg = GoEmotionsConfig(
        top_k=int(meta.get("top_k", 8)),
        dataset_name=str(meta.get("dataset_name", "go_emotions")),
        dataset_config=str(meta.get("dataset_config", "simplified")),
    )
    ds, class_names, _, _ = load_goemotions_benchmark(cfg)
    if list(class_names) != list(meta["class_names"]):
        raise ValueError("class_names mismatch vs meta.json")

    train_texts = list(ds["train"]["text"])
    train_y = np.array(ds["train"]["y"], dtype=np.int64)
    test_texts = list(ds["test"]["text"])
    test_y = np.array(ds["test"]["y"], dtype=np.int64)

    sub = meta.get("subsample") or {}
    if sub.get("max_train_samples"):
        n = int(sub["max_train_samples"])
        train_texts, train_y = train_texts[:n], train_y[:n]
    if sub.get("max_test_samples"):
        n = int(sub["max_test_samples"])
        test_texts, test_y = test_texts[:n], test_y[:n]
    if max_test_samples is not None:
        test_texts = test_texts[: int(max_test_samples)]
        test_y = test_y[: int(max_test_samples)]

    return train_texts, train_y, test_texts, test_y, list(class_names)
