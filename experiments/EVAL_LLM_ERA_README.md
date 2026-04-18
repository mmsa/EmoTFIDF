# LLM-era evaluation (interpretable evidence layer)

This folder contains **reproducible evaluation scripts** for EmoTFIDF V2 as an interpretable emotional evidence component in hybrid / transformer pipelines. These are **not** full paper-scale benchmarks; they are controlled stages you can run after `experiments/train.py` has produced `experiments/artifacts/meta.json` and the saved DistilBERT + EmoTFIDF+LR heads.

## Prerequisites

1. Train or restore artifacts (from repo root):

   ```bash
   python experiments/train.py
   ```

2. Ensure GoEmotions can load (HF datasets / cache as in `data_loader.py`).

## 1. Fusion ablation (DistilBERT vs V1 vs V2 probability fusion)

Compares:

- DistilBERT argmax alone
- `0.8` DistilBERT + `0.2` legacy EmoTFIDF+LR (V1 head from artifacts)
- `0.8` DistilBERT + `0.2` fresh seven-way logistic head on V2 normalized emotion vectors (same fusion weighting)

```bash
python experiments/run_fusion_ablation.py
python experiments/run_fusion_ablation.py --max-test-samples 500
```

Outputs: `experiments/eval_outputs/fusion_ablation_metrics.csv`, `fusion_ablation_summary.md`.

## 2. Verifier analysis (held-out predictions vs V2 support)

Runs the chosen classifier on the held-out split, maps each **predicted** GoEmotions class to a seven-way evidence label (`label_bridge.py`), and runs `EmoTFIDF.evidence.verifier.verify_label` on one V2 analysis per row.

```bash
python experiments/run_verifier_analysis.py --prediction-source v2_fusion
python experiments/run_verifier_analysis.py --prediction-source distilbert --max-test-samples 300
```

Outputs: `experiments/eval_outputs/verifier_per_row.csv`, `verifier_aggregate.csv`, `verifier_summary.md`.

## 3. Behavioral regression suite (unchanged, separate section)

Curated V1 vs V2 contracts (explanation / verifier smoke, not GoEmotions accuracy):

```bash
python experiments/benchmark_v1_v2_regression.py
pytest tests/test_benchmark_regression_smoke.py tests/test_v2_explanation_quality.py -q
```

Use this as a **gate** before expanding to larger held-out runs.
