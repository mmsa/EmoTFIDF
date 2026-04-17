# EmoTFIDF V2 — development notes

**Import path:** use `EmoTFIDF.evidence` (the Python package was renamed from `EmoTFIDF.v2` so the module name matches the “evidence layer” role).

## What changed

- Added a **parallel V2 API** under `EmoTFIDF.evidence` that does not replace or rewrite the legacy `EmoTFIDF.EmoTFIDF.EmoTFIDF` class.
- V2 focuses on an **interpretable emotional evidence layer**: structured outputs, per-term contributions, coverage, lightweight negation/intensifier windows, a richer deterministic feature vector, LLM-oriented prompt exports, and a lexical **verifier** for proposed labels.
- The default lexicon path resolves to the **packaged** `EmoTFIDF/emotions_lex.json` (no network required for V2 tests).

## New public API

- **`EmoTFIDF.evidence.EmoTFIDFv2`**
  - `fit(corpus_texts)` — fit `TfidfVectorizer` on V1-style preprocessed documents.
  - `analyze(text)` → `AnalysisResult` (dataclass with `to_dict()`).
  - `analyze_batch(texts)` → list of dicts.
  - `get_feature_vector(text)` → `(vector, names)` (38 floats; see `feature_names` on analysis).
  - `explain(text)` → explanation bundle dict.
  - `verify_label(text, predicted_label)` → verifier dict.
  - `to_prompt_features(text)` → compact JSON-friendly dict.

- **Module layout**: `EmoTFIDF/evidence/` — `analyzer`, `preprocessing`, `lexicon`, `weighting`, `rules`, `explain`, `verifier`, `prompt_features`, `schemas`.

## Behavior notes (transparent heuristics)

- **Negation**: fixed cue list; if a cue appears within `negation_window` tokens *before* an affect-bearing lexicon hit, emotion mass for that token is multiplied by `negation_factor` (default **-0.55**, partial inversion). Suppressed **joy** mass is partially attributed to **sadness** via `NEGATION_SUPPRESSED_JOY_TO_SADNESS_FRACTION` in `rules.py` (transparent valence sink—not mapped to anger). Not full dependency parsing.
- **Positive-mass normalization**: scores are **L1-normalized on ReLU(raw)** only; all-zero when there is no positive evidence (no uniform pseudo-distribution on punctuation-only input).
- **Intensifier tokens as cues only**: tokens in `INTENSIFIERS_UP` / `INTENSIFIERS_DOWN` / `NEGATION_CUES` do **not** emit lexicon emotions (so e.g. *extremely* does not inject spurious joy from the NRC entry).
- **Lexicon duplicate tags**: per-token mass is split with **inverse duplicate count** weighting (`inverse_count_emotion_shares`) so repeated disgust tags do not drown a single anger tag.
- **Weak contextual lexemes** (`lexeme_prior.py`): temporal/deictic words that NRC still tags (e.g. *yesterday* → sadness) get a **lower affect multiplier** on scored mass and are **de-prioritized** in explanation ordering so clearer affect terms surface first.
- **Intensifiers / downtoners (neighbors)**: closest cue within `intensifier_window` before the affect token applies a multiplier (**×1.35** up, **×0.72** down). Tokens that are both negation and downtoner candidates are modeled as **negation** in `rules.py` (e.g. `barely`, `hardly` are negation cues, not downtoners).
- **Verifier**: combines normalized label share with raw lexical mass; levels are **strong / moderate / weak / unsupported**. This is **evidence-only**, not a correctness oracle.

## What to benchmark next (not implemented here)

- Agreement with human or model labels when V2 features are fed to a lightweight classifier.
- Ablations: negation window size, `negation_factor`, TF-IDF `max_features`, and corpus size for `fit`.
- Hybrid pipelines: V2 prompt features prepended to LLM prompts vs. post-hoc verification only.
- Calibration of verifier thresholds on a held-out set (avoid tuning on the same demos used for development).
