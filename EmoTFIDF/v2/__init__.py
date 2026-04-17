"""
EmoTFIDF V2 — interpretable lexical + TF-IDF emotional evidence layer.

Use :class:`EmoTFIDF.v2.analyzer.EmoTFIDFv2` for the main API. V1 remains in
``EmoTFIDF.EmoTFIDF.EmoTFIDF`` unchanged.
"""

from EmoTFIDF.v2.analyzer import EmoTFIDFv2
from EmoTFIDF.v2.schemas import (
    AnalysisResult,
    ExplanationBundle,
    VerificationResult,
)

__all__ = [
    "EmoTFIDFv2",
    "AnalysisResult",
    "VerificationResult",
    "ExplanationBundle",
]
