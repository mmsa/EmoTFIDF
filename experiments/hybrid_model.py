"""
Hybrid models combining DistilBERT with EmoTFIDF features.

Two variants are supported:

1. **Concatenation**: CLS embedding (768-d) + EmoTFIDF vector (7-d) fed to
   ``StandardScaler`` + ``LogisticRegression``.
2. **Probability fusion**: ``0.8 * p_transformer + 0.2 * p_emotfidf`` with
   ``p_emotfidf`` coming from a logistic head fit on EmoTFIDF features alone.
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from transformer_model import load_distilbert


def _inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def extract_cls_embeddings(
    texts: Sequence[str],
    model_dir: str,
    batch_size: int = 64,
    max_length: int = 128,
) -> np.ndarray:
    """
    Extract the final hidden CLS vector from DistilBERT for each document.

    Uses the fine-tuned sequence classification checkpoint so that the
    backbone weights match the emotion task.
    """
    tokenizer, model = load_distilbert(model_dir)
    device = _inference_device()
    model.to(device)
    model.eval()
    base = getattr(model, "distilbert", None) or getattr(model, "bert", None)
    if base is None:
        raise RuntimeError("Expected a DistilBERT-backed checkpoint.")

    n_doc = len(texts)
    n_batch = (n_doc + batch_size - 1) // batch_size
    if n_doc >= 1500:
        msg = (
            f"  Hybrid: extracting CLS vectors for {n_doc} texts in {n_batch} batches "
            f"(batch_size={batch_size}, device={device.type})."
        )
        if device.type == "cpu":
            msg += " This step is slowest on CPU."
        print(msg, flush=True)

    out_rows: List[np.ndarray] = []
    log_every = max(1, n_batch // 12)
    for bi, i in enumerate(range(0, n_doc, batch_size)):
        if n_doc >= 1500 and bi > 0 and bi % log_every == 0:
            print(
                f"    … CLS batches {bi}/{n_batch} (~{100 * bi // n_batch}%)",
                flush=True,
            )
        batch = [str(t) for t in texts[i : i + batch_size]]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        enc_in = {k: enc[k] for k in ("input_ids", "attention_mask") if k in enc}
        hidden = base(**enc_in)[0]  # (b, seq, h)
        cls_vec = hidden[:, 0, :].detach().cpu().numpy()
        out_rows.append(cls_vec)
    return np.vstack(out_rows)


def train_hybrid_concat_classifier(
    texts: Sequence[str],
    emotfidf_vecs: np.ndarray,
    y: np.ndarray,
    transformer_dir: str,
    save_path: str,
    seed: int = 42,
    cls_batch_size: int = 64,
) -> Pipeline:
    """
    Fit scaler + logistic regression on ``[CLS] || EmoTFIDF`` features.
    """
    cls_mat = extract_cls_embeddings(
        texts, transformer_dir, batch_size=cls_batch_size
    )
    if cls_mat.shape[0] != emotfidf_vecs.shape[0]:
        raise ValueError("CLS matrix and EmoTFIDF matrix row counts differ.")
    X = np.hstack([cls_mat, emotfidf_vecs])
    clf: Pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    clf.fit(X, y)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    joblib.dump(clf, save_path)
    return clf


def predict_hybrid_concat(
    texts: Sequence[str],
    emotfidf_vecs: np.ndarray,
    transformer_dir: str,
    clf_path: str,
    cls_batch_size: int = 64,
) -> np.ndarray:
    """Predict class ids for the concatenated hybrid."""
    clf: Pipeline = joblib.load(clf_path)
    cls_mat = extract_cls_embeddings(
        texts, transformer_dir, batch_size=cls_batch_size
    )
    X = np.hstack([cls_mat, emotfidf_vecs])
    return clf.predict(X).astype(np.int64)


def fused_probabilities(
    probs_transformer: np.ndarray,
    probs_emotfidf_lr: np.ndarray,
    transformer_weight: float = 0.8,
) -> np.ndarray:
    """Convex combination of predicted probability tables."""
    w_t = float(transformer_weight)
    w_e = 1.0 - w_t
    if probs_transformer.shape != probs_emotfidf_lr.shape:
        raise ValueError("Probability tensors must share shape.")
    return w_t * probs_transformer + w_e * probs_emotfidf_lr


def fused_predictions(
    probs_transformer: np.ndarray,
    probs_emotfidf_lr: np.ndarray,
    transformer_weight: float = 0.8,
) -> np.ndarray:
    """Argmax over fused probabilities."""
    fused = fused_probabilities(
        probs_transformer, probs_emotfidf_lr, transformer_weight
    )
    return np.argmax(fused, axis=-1).astype(np.int64)
