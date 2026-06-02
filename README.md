# AMP Discovery

Antimicrobial peptide discovery via protein language model fine-tuning.

## Quickstart

```bash
bash scripts/bootstrap_remote.sh --train
uv run python scripts/doctor.py
```

## Local Smoke Test (RTX 4060 8GB)

```bash
uv run python -m amp_discovery.eval configs/eval_local_4060.yaml
uv run python -m amp_discovery.train configs/local_4060_smoke.yaml
```

## Cloud Training (RunPod A100 80GB)

```bash
uv run python -m amp_discovery.train configs/cloud_a100.yaml
uv run python -m amp_discovery.eval configs/eval_cloud_a100.yaml
```

## Datasets

- **ESCAPE**: 80K peptides, NeurIPS 2025, multilabel classification
- **GenPept-Curated-2025**: 11K sequences, leakage-free benchmark
- **PepBenchmark**: Multi-task peptide ML

## Plan

1. Smoke test on local RTX 4060 with ESM-2 8M
2. Train ESM-2 650M on RunPod A100 with ESCAPE
3. Validate on GenPept (cross-benchmark, leakage-free)
4. Screen NCBI NR for novel AMP candidates
5. Contact wet-lab for validation, co-author on publication

## Current Results

Baseline (ESM-2 8M, untrained) vs Trained (ESM-2 650M + LoRA, 10 epochs, RunPod A6000):

- Accuracy: 0.0% -> 97.0%
- F1 Micro: 3.3% -> 68.9% (21x)
- F1 Antibacterial: 6.8% -> 81.0%
- F1 Antifungal: 3.7% -> 40.5%
- F1 Antiviral: 0.7% -> 31.0%
- F1 Antiparasitic: 0.0% -> 0.0% (class imbalance)
