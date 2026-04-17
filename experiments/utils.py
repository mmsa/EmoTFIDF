"""
Shared paths, random seeds, and import helpers for the benchmark pipeline.

All paths are derived from this file's location so the repo can be moved
without editing configuration.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT: Path = Path(__file__).resolve().parent
ARTIFACTS_DIR: Path = EXPERIMENTS_ROOT / "artifacts"
RESULTS_CSV: Path = EXPERIMENTS_ROOT / "results.csv"
EXPLANATIONS_TXT: Path = EXPERIMENTS_ROOT / "explanations.txt"


def ensure_repo_on_path() -> None:
    """Insert the repository root on ``sys.path`` so ``EmoTFIDF`` imports work."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def set_global_seed(seed: int) -> None:
    """Fix seeds for NumPy and the standard ``random`` module."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    """Create artifact and experiment directories if needed."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON file with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def class_names_from_meta(meta: Dict[str, Any]) -> List[str]:
    """Return ordered human-readable class names stored in training metadata."""
    return list(meta["class_names"])
