"""
Fine-tuning ``distilbert-base-uncased`` with the HuggingFace ``Trainer`` API.

The head is a linear layer sized to the number of GoEmotions classes in the
benchmark slice (typically eight after frequency filtering).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "distilbert-base-uncased"


def _inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TransformerTrainConfig:
    """Training knobs for DistilBERT fine-tuning."""

    output_dir: str
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42
    max_length: int = 128


def _tokenize_batch(
    examples: Dict[str, List], tokenizer: Any, max_length: int
) -> Dict[str, List]:
    """Tokenize ``text`` and set ``labels`` from ``y`` for ``Trainer`` loss."""
    encoded = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    # Plain dict so ``datasets`` keeps a ``labels`` column (BatchEncoding can confuse caching).
    return {**dict(encoded), "labels": examples["y"]}


def _build_metrics_fn():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        }

    return compute_metrics


def train_distilbert_classifier(
    train_ds: Dataset,
    eval_ds: Dataset,
    num_labels: int,
    cfg: TransformerTrainConfig,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
) -> str:
    """
    Fine-tune DistilBERT and persist weights under ``cfg.output_dir``.

    Returns:
        Path to the saved model directory (same as ``cfg.output_dir``).
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, clean_up_tokenization_spaces=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label or {i: str(i) for i in range(num_labels)},
        label2id=label2id or {str(i): i for i in range(num_labels)},
    )

    remove_cols = train_ds.column_names
    train_tok = train_ds.map(
        lambda batch: _tokenize_batch(batch, tokenizer, cfg.max_length),
        batched=True,
        remove_columns=remove_cols,
        # Stale map caches from older code (no ``labels``) otherwise reuse broken arrow files.
        load_from_cache_file=False,
    )
    eval_remove = eval_ds.column_names
    eval_tok = eval_ds.map(
        lambda batch: _tokenize_batch(batch, tokenizer, cfg.max_length),
        batched=True,
        remove_columns=eval_remove,
        load_from_cache_file=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        save_total_limit=1,
        seed=cfg.seed,
        fp16=use_cuda,
        report_to=[],
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_build_metrics_fn(),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return cfg.output_dir


def load_distilbert(model_dir: str) -> Tuple[Any, Any]:
    """Load tokenizer + model from a saved training directory."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, clean_up_tokenization_spaces=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model


@torch.no_grad()
def predict_distilbert(
    texts: List[str],
    model_dir: str,
    batch_size: int = 64,
    max_length: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run batched inference.

    Returns:
        - ``preds`` int array ``(n,)``
        - ``probs`` float array ``(n, num_labels)``
    """
    tokenizer, model = load_distilbert(model_dir)
    device = _inference_device()
    model.to(device)
    model.eval()
    n = len(texts)
    n_batch = (n + batch_size - 1) // batch_size
    if n >= 800:
        print(
            f"  DistilBERT inference: {n} texts, {n_batch} batches, device={device.type}",
            flush=True,
        )
    preds: List[int] = []
    prob_chunks: List[np.ndarray] = []
    log_every = max(1, n_batch // 10)
    for bi, i in enumerate(range(0, n, batch_size)):
        if n >= 800 and bi > 0 and bi % log_every == 0:
            print(
                f"    … batches {bi}/{n_batch} (~{100 * bi // n_batch}%)",
                flush=True,
            )
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred = np.argmax(probs, axis=-1)
        preds.extend(pred.tolist())
        prob_chunks.append(probs)
    return np.array(preds, dtype=np.int64), np.vstack(prob_chunks)
