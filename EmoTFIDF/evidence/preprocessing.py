"""Tokenization and preprocessing for V2 (independent of V1 EmoTFIDF.py import side effects)."""

from __future__ import annotations

import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords

from ..nltk_resources import ensure_nltk_word_tokenize_deps

# NLTK 3.8+ ``word_tokenize`` uses ``punkt_tab``; CI runners often lack it.
ensure_nltk_word_tokenize_deps()
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

_WORD_EDGE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def strip_edges(token: str) -> str:
    """Remove leading/trailing non-word characters (punctuation)."""
    return _WORD_EDGE.sub("", token or "")


def process_message_for_tfidf(message: str) -> str:
    """
    Match V1 ``process_message`` behavior: tokenize, drop stopwords, short tokens, etc.

    Used as the document string for TF-IDF fitting and lookup (same as legacy EmoTFIDF).
    """
    words = nltk.word_tokenize(message.lower())
    words = [w for w in words if len(w) > 3]
    sw = stopwords.words("english")
    words = [word for word in words if not word.isnumeric()]
    words = [word for word in words if word not in sw]
    words = [word for word in words if word not in string.punctuation]
    return " ".join(words)


def tokenize_raw_sequence(text: str) -> List[str]:
    """Lowercased NLTK tokens for cue windows and alignment."""
    return nltk.word_tokenize(text.lower())


def content_tokens_for_coverage(processed_doc: str) -> List[str]:
    """Tokens considered for coverage denominator (same filtering as TF-IDF doc)."""
    if not processed_doc.strip():
        return []
    return [t for t in nltk.word_tokenize(processed_doc) if len(t) > 1]
