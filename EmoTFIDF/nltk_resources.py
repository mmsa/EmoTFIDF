"""Ensure NLTK tokenizer data exists (NLTK 3.8+ prefers ``punkt_tab`` over legacy ``punkt``)."""

from __future__ import annotations

import nltk


def ensure_nltk_word_tokenize_deps() -> None:
    """
    ``nltk.word_tokenize`` may require ``punkt_tab``; download quietly when missing.

    Safe to call repeatedly (e.g. CI, fresh venvs).
    """

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
    except Exception:
        pass
    if _punkt_tab_ok():
        return
    try:
        nltk.download("punkt", quiet=True, force=True)
    except Exception:
        pass
