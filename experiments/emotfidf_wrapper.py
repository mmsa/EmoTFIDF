"""
Thin wrapper around the repository ``EmoTFIDF`` class for batch feature extraction.

The upstream library fits a ``TfidfVectorizer`` on a corpus, then for each
document computes lexicon-based emotions re-weighted by TF-IDF (``em_tfidf``).

Imports of ``EmoTFIDF.EmoTFIDF`` are **deferred** until ``fit()`` runs so that
dataset preparation and TF-IDF baselines can proceed while HuggingFace / NLTK
assets for the legacy library finish downloading in the background.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple, Type

import numpy as np

from utils import ensure_repo_on_path

# Populated on first backend load (same order as the library ``labels`` list).
_emotion_label_order: List[str] | None = None


def _ensure_nltk_punkt_for_word_tokenize() -> None:
    """
    NLTK 3.8+ ``word_tokenize`` expects ``punkt_tab`` (not only legacy ``punkt``).

    The core ``EmoTFIDF`` module calls ``nltk.word_tokenize``; fetch data here
    so benchmark runs work on fresh environments without manual downloader steps.

    ``nltk.data.find`` can raise ``OSError`` (broken on-disk tree) as well as
    ``LookupError``; treat both as missing and re-download.
    """
    import nltk

    def _punkt_tab_ok() -> bool:
        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
            return True
        except (LookupError, OSError):
            return False

    if _punkt_tab_ok():
        return
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:  # pragma: no cover — NLTK too old for ``punkt_tab`` package
        pass
    if not _punkt_tab_ok():
        try:
            nltk.download("punkt_tab", quiet=True, force=True)
        except Exception:
            pass
    if _punkt_tab_ok():
        return
    # Older NLTK can resolve ``punkt_tab`` through a ``punkt/...`` path; a corrupt ``punkt``
    # install then raises ``OSError`` on find. Re-fetch ``punkt`` before the final fallback.
    try:
        nltk.download("punkt", quiet=True, force=True)
    except Exception:
        pass
    if _punkt_tab_ok():
        return
    nltk.download("punkt", quiet=True)


def get_emotion_label_order() -> List[str]:
    """Return the library's canonical emotion keys (length seven)."""
    global _emotion_label_order
    if _emotion_label_order is None:
        ensure_repo_on_path()
        import EmoTFIDF.EmoTFIDF as emolib  # noqa: WPS433 — runtime import

        _emotion_label_order = list(emolib.labels)
    return _emotion_label_order


def _load_emotfidf_class() -> "Type[EmoTFIDFType]":
    ensure_repo_on_path()
    from EmoTFIDF.EmoTFIDF import EmoTFIDF  # noqa: WPS433

    return EmoTFIDF


class EmoTFIDFVectorizer:
    """
    Fit TF-IDF on a reference corpus, then emit EmoTFIDF-weighted emotion vectors.

    Parameters
    ----------
    emo_tf_idf:
        Optional pre-constructed ``EmoTFIDF`` instance (useful for tests). If
        ``None``, the heavy library import is deferred until ``fit()`` is called.
    """

    def __init__(self, emo_tf_idf: Any = None) -> None:
        self._model = emo_tf_idf
        self._corpus_fitted = False

    def _ensure_model(self) -> None:
        if self._model is None:
            print(
                "  Initializing EmoTFIDF backend (first import may download "
                "the library's DistilRoBERTa emotion checkpoint; be patient) …",
                flush=True,
            )
            _ensure_nltk_punkt_for_word_tokenize()
            cls = _load_emotfidf_class()
            self._model = cls()

    def fit(self, corpus: Sequence[str]) -> "EmoTFIDFVectorizer":
        """Fit the internal TF-IDF model on ``corpus`` (list of raw documents)."""
        self._ensure_model()
        docs = [str(x) for x in corpus]
        self._model.compute_tfidf(docs)
        self._corpus_fitted = True
        return self

    def _require_fit(self) -> None:
        if not self._corpus_fitted:
            raise RuntimeError("Call fit(corpus) before scoring documents.")

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """
        Return an ``(n_samples, 7)`` matrix of EmoTFIDF scores (library order).
        """
        self._require_fit()
        rows: List[np.ndarray] = []
        for t in texts:
            rows.append(self.score_document(str(t)))
        return np.vstack(rows)

    def fit_transform(self, corpus: Sequence[str]) -> np.ndarray:
        """Fit on ``corpus`` and score the same documents."""
        self.fit(corpus)
        return self.transform(corpus)

    def score_document(self, text: str) -> np.ndarray:
        """
        Compute the EmoTFIDF vector for a single document.

        Missing lexicon hits or degenerate statistics yield a zero vector rather
        than raising, so batch pipelines remain robust.
        """
        self._require_fit()
        keys = get_emotion_label_order()
        self._model.set_text(text)
        try:
            self._model.get_emotfidf()
            raw = getattr(self._model, "em_tfidf", None)
            if not isinstance(raw, dict):
                raise ValueError("em_tfidf not populated")
        except (ZeroDivisionError, ValueError, ArithmeticError, KeyError):
            raw = {k: 0.0 for k in keys}
        vec = np.array([float(raw.get(k, 0.0)) for k in keys], dtype=np.float64)
        s = float(vec.sum())
        if s > 0:
            vec = vec / s
        return vec

    def top_emotion_lexicon_terms(
        self, text: str, top_k: int = 12
    ) -> List[Tuple[str, float]]:
        """
        Return up to ``top_k`` lexicon words with the largest TF-IDF weights.

        Words are drawn from ``em_dict`` keys that received a non-zero TF-IDF
        score for this document after ``get_emotfidf`` runs internally.
        """
        self._require_fit()
        self._model.set_text(text)
        try:
            self._model.get_emotfidf()
        except (ZeroDivisionError, ValueError, ArithmeticError, KeyError):
            return []
        scores = getattr(self._model, "ifidf_for_words", None)
        em_dict = getattr(self._model, "em_dict", None)
        if not isinstance(scores, dict) or not isinstance(em_dict, dict):
            return []
        pairs: List[Tuple[str, float]] = []
        for w in em_dict.keys():
            if w in scores:
                pairs.append((w, float(scores[w])))
        pairs.sort(key=lambda x: -x[1])
        return pairs[:top_k]
