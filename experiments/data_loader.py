"""
GoEmotions loading and preprocessing for single-label, top-k emotion benchmarks.

The HuggingFace ``go_emotions`` simplified configuration stores ``labels`` as
a **list of class ids** (multi-label). We collapse to a single label with the
first id in that list. If a length-``|names|`` multihot vector is encountered
instead, we use the smallest index with value ``1``. Empty vectors fall back to
``neutral`` when that class exists.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from datasets import Dataset, DatasetDict, load_dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The Hugging Face `datasets` package is required. From the repo root run:\n"
        "  python -m pip install -r requirements.txt\n"
        "or: python -m pip install datasets"
    ) from exc


@dataclass
class GoEmotionsConfig:
    """Hyper-parameters for building the benchmark slice."""

    top_k: int = 8
    dataset_name: str = "go_emotions"
    dataset_config: str = "simplified"
    #: If True, always ``force_redownload`` (slow; fixes corrupt / stale cache).
    force_redownload: bool = False


def _load_raw_goemotions(cfg: GoEmotionsConfig) -> DatasetDict:
    """
    Load the raw HuggingFace ``DatasetDict``, working around stale cache issues.

    Some ``datasets`` 3.x installs on Python 3.8 fail while parsing cached
    ``dataset_info.json`` (``TypeError: must be called with a dataclass type``).
    We retry once with ``force_redownload`` when that happens.
    """

    def _call(download_mode: Optional[str]) -> DatasetDict:
        if download_mode is None:
            return load_dataset(cfg.dataset_name, cfg.dataset_config)
        return load_dataset(
            cfg.dataset_name, cfg.dataset_config, download_mode=download_mode
        )

    if cfg.force_redownload:
        return _call("force_redownload")

    try:
        return _call(None)
    except TypeError as err:
        msg = str(err).lower()
        if "dataclass" in msg or "datasetinfo" in msg or "features" in msg:
            warnings.warn(
                "GoEmotions load failed while reading the local HuggingFace cache "
                "(often a ``datasets`` major-version mismatch). Retrying with "
                "download_mode='force_redownload'. For a lasting fix on Python "
                "3.8, use: python -m pip install 'datasets>=2.14,<3'",
                UserWarning,
                stacklevel=2,
            )
            return _call("force_redownload")
        raise


def _label_names(ds_split) -> List[str]:
    """Return the ordered list of emotion names for the simplified config."""
    feature = ds_split.features["labels"]
    return list(feature.feature.names)


def multihot_to_first_label(
    multihot: Sequence[int], names: List[str], neutral_name: str = "neutral"
) -> int:
    """
    Convert a multi-label **binary vector** (length ``len(names)``) to one id.

    The winning index is the smallest ``i`` with value ``1``, matching the
    canonical emotion order in ``names``.
    """
    positives = [i for i, v in enumerate(multihot) if int(v) == 1]
    if positives:
        return positives[0]
    if neutral_name in names:
        return names.index(neutral_name)
    raise ValueError("No positive labels and no neutral class available.")


def goemotions_labels_to_single(
    labels: Sequence[int], names: List[str], neutral_name: str = "neutral"
) -> int:
    """
    Collapse GoEmotions ``labels`` to a single class index.

    The HuggingFace ``simplified`` config stores **lists of class ids** (multi
    label). Some community exports use a length-``|names|`` multihot instead;
    both are handled here. The user-requested rule is: use the **first** list
    entry for the index form, or the first active bit for multihot form.
    """
    if labels is None or len(labels) == 0:
        return names.index(neutral_name)
    is_multihot = len(labels) == len(names) and all(int(v) in (0, 1) for v in labels)
    if is_multihot:
        return multihot_to_first_label(labels, names, neutral_name)
    return int(labels[0])


def _add_single_label_column(batch: Dict, names: List[str]) -> Dict:
    """``datasets`` map function: add ``single_label`` column."""
    singles: List[int] = []
    for row in batch["labels"]:
        singles.append(goemotions_labels_to_single(row, names))
    return {"single_label": singles}


def select_top_k_classes(
    train_ds: Dataset, label_column: str, names: List[str], k: int
) -> Tuple[List[int], List[str], List[int]]:
    """
    Choose the ``k`` most frequent class indices on the training split.

    Returns:
        - ``top_ids_sorted``: sorted class indices (for filtering / stable ids)
        - ``freq_order_names``: class names ordered by descending frequency
        - ``freq_order_ids``: class indices in the same frequency order
    """
    counts = Counter(int(x) for x in train_ds[label_column])
    most_common = counts.most_common(k)
    ordered_by_freq = [idx for idx, _ in most_common]
    top_ids_sorted = sorted(ordered_by_freq)
    freq_order_names = [names[i] for i in ordered_by_freq]
    return top_ids_sorted, freq_order_names, ordered_by_freq


def filter_to_classes(ds: Dataset, label_column: str, allowed: List[int]) -> Dataset:
    """Keep only rows whose label is in ``allowed``."""
    allowed_set = set(allowed)

    def _keep(batch: Dict) -> Dict:
        return {"keep": [lbl in allowed_set for lbl in batch[label_column]]}

    mapped = ds.map(_keep, batched=True)
    filtered = mapped.filter(lambda x: x["keep"])
    return filtered.remove_columns(["keep"])


def relabel_to_contiguous(
    ds: Dataset, label_column: str, old_ids: List[int]
) -> Tuple[Dataset, Dict[int, int]]:
    """
    Remap original class indices to ``0 .. num_classes-1``.

    Returns:
        Remapped dataset and mapping ``new_id -> original_id``.
    """
    old_sorted = sorted(old_ids)
    old_to_new = {old: i for i, old in enumerate(old_sorted)}
    new_to_old = {i: old for old, i in old_to_new.items()}

    def _map_labels(batch: Dict) -> Dict:
        return {
            "y": [old_to_new[int(lbl)] for lbl in batch[label_column]],
        }

    out = ds.map(_map_labels, batched=True)
    return out, new_to_old


def load_goemotions_benchmark(
    cfg: GoEmotionsConfig | None = None,
) -> Tuple[DatasetDict, List[str], Dict[int, int], Dict[str, Any]]:
    """
    Build train/test ``DatasetDict`` with single labels and top-``k`` classes.

    Returns:
        - ``ds``: ``DatasetDict`` with ``train`` / ``test`` splits, columns
          ``text`` and contiguous ``y``.
        - ``class_names``: names aligned with ``y`` (0..k-1).
        - ``new_to_old``: maps contiguous id -> original GoEmotions id.
        - ``meta``: small JSON-serializable description for artifacts.
    """
    cfg = cfg or GoEmotionsConfig()
    raw: DatasetDict = _load_raw_goemotions(cfg)

    names = _label_names(raw["train"])
    train_single = raw["train"].map(
        lambda b: _add_single_label_column(b, names),
        batched=True,
    )
    test_single = raw["test"].map(
        lambda b: _add_single_label_column(b, names),
        batched=True,
    )

    top_ids, freq_names, freq_ids = select_top_k_classes(
        train_single, "single_label", names, cfg.top_k
    )
    train_f = filter_to_classes(train_single, "single_label", top_ids)
    test_f = filter_to_classes(test_single, "single_label", top_ids)

    train_y, new_to_old = relabel_to_contiguous(train_f, "single_label", top_ids)
    test_y, _ = relabel_to_contiguous(test_f, "single_label", top_ids)

    # Resolve contiguous class names (sorted by old id for stable 0..k-1 order)
    old_sorted = sorted(top_ids)
    class_names = [names[i] for i in old_sorted]

    drop_cols = [
        c
        for c in train_y.column_names
        if c not in ("text", "y")
    ]
    train_clean = train_y.remove_columns(drop_cols)
    test_clean = test_y.remove_columns(
        [c for c in test_y.column_names if c not in ("text", "y")]
    )

    meta = {
        "dataset_name": cfg.dataset_name,
        "dataset_config": cfg.dataset_config,
        "top_k": cfg.top_k,
        "class_names": class_names,
        "top_class_ids_by_frequency": freq_ids,
        "top_class_names_by_frequency": freq_names,
        "labeling": "first_class_id_in_label_list_or_first_active_multihot_bit",
    }

    ds_out = DatasetDict({"train": train_clean, "test": test_clean})
    return ds_out, class_names, new_to_old, meta
