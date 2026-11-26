"""Public APIs for lexical evidence and whole-sentence emotion classification."""

__all__ = ["EmoTFIDFv2"]


def __getattr__(name):
    if name == "EmoTFIDFv2":
        from EmoTFIDF.evidence import EmoTFIDFv2
        return EmoTFIDFv2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
