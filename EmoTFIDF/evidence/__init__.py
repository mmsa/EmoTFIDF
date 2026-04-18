"""
EmoTFIDF V2 — interpretable lexical + TF-IDF emotional evidence layer.

Use :class:`EmoTFIDF.evidence.analyzer.EmoTFIDFv2` for the main API. V1 remains in
``EmoTFIDF.EmoTFIDF.EmoTFIDF`` unchanged.
"""

from EmoTFIDF.evidence.analyzer import EmoTFIDFv2
from EmoTFIDF.evidence.schemas import (
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
