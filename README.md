# AMP Discovery

Antimicrobial peptide discovery via protein language model fine-tuning with LoRA.

## Motivation

Antimicrobial resistance (AMR) is projected to cause 10 million deaths annually by 2050. Antimicrobial peptides (AMPs) are a promising alternative to conventional antibiotics — they target bacterial membranes, evolve rapidly, and rarely induce resistance. But discovering novel AMPs experimentally is slow and expensive. Computational screening with protein language models can accelerate the search by orders of magnitude.

This project uses Meta ESM-2, a transformer model pretrained on 250M protein sequences, fine-tuned with LoRA on two benchmark datasets: ESCAPE (NeurIPS 2025, multilabel) and GenPept-Curated-2025 (binary, leakage-free). The trained model then screens unlabeled NCBI sequences to surface novel AMP candidates.

## Architecture

- **Base model**: ESM-2 (`facebook/esm2_t33_650M_UR50D`)
- **Adaptation**: LoRA (r=16, alpha=32), targeting query and value projection matrices
- **Task head**: Linear classifier on top of ESM-2 embeddings (1280 dim -> N classes)
- **Training**: HuggingFace Trainer with automatic mixed precision (FP16)
- **Hardware**: RunPod A6000 (48GB VRAM), ~2 hours for 10 epochs on 80K peptides

## Experiments

### Experiment 1: ESCAPE Multilabel Classification

ESCAPE contains 80K peptides with four mechanism labels: Antibacterial, Antifungal, Antiviral, Antiparasitic. Each peptide can have multiple labels. The model outputs four independent sigmoid probabilities.

| Metric | Baseline (ESM-2 8M, untrained) | Trained (ESM-2 650M + LoRA) |
|---|---|---|
| Accuracy | 0.0% | 97.0% |
| F1 Micro | 3.3% | 68.9% |
| F1 Antibacterial | 6.8% | 81.0% |
| F1 Antifungal | 3.7% | 40.5% |
| F1 Antiviral | 0.7% | 31.0% |
| F1 Antiparasitic | 0.0% | 0.0% |

Antibacterial performance is strong (81% F1). Antifungal and Antiviral show meaningful gains but lower absolute scores, reflecting class imbalance. Antiparasitic fails completely — ESCAPE has very few positive examples for this class.

### Experiment 2: Cross-Benchmark Transfer

GenPept-Curated-2025 is a binary (AMP/non-AMP) dataset specifically designed to be leakage-free — it has no sequence homology overlap with ESCAPE. This makes it a rigorous test of generalization.

The ESCAPE-trained model scored 2.2% F1 on GenPept — essentially random. The model learned mechanism-specific features (\"is this antibacterial?\") rather than a general AMP detector.

### Experiment 3: GenPept Direct Training

Training directly on GenPept as a binary classification task (11K sequences, 80/20 split, 5 epochs) achieved 88.3% F1 with 86.8% accuracy.

### Discovery: NCBI Screening

The GenPept-trained model screened 1,980 unlabeled bacterial peptide sequences from NCBI RefSeq, excluding any with known AMP annotations. The top 100 candidates are ranked by AMP probability in `results/ncbi_top_100.csv`.

- Top hit: WP_488644500.1 (0.785 probability)
- Top 10 range: 0.768–0.785
- All candidates are uncharacterized bacterial proteins not annotated as AMP

## Key Findings

1. **ESM-2 + LoRA is highly sample-efficient.** 80K peptides with a 650M parameter model achieves strong performance without full fine-tuning.
2. **Cross-benchmark transfer fails.** ESCAPE-trained models do not generalize to GenPept, confirming the need for diverse, leakage-aware evaluation.
3. **Binary AMP detection is viable. 88.3% F1 on a balanced, leakage-free benchmark means the model can reliably distinguish AMPs from non-AMPs.**

## Reproducibility

All configs, checkpoints, and eval results are committed. Training costs ~\$30 on RunPod A6000. Full pipeline: clone, `bash scripts/bootstrap_remote.sh --train`, `uv run python -m amp_discovery.train configs/cloud_a100.yaml`.

## Citation

If you use this work, please cite the datasets:
- ESCAPE: https://doi.org/10.7910/DVN/C69MCD (NeurIPS 2025)
- GenPept-Curated-2025: https://github.com/biochem-data-sci/GenPept-Curated-2025
- ESM-2: Lin et al., 2023. Language models of protein sequences at the scale of evolution.

## License

MIT

## Model

Trained LoRA adapter available on HuggingFace: [null-phnix/amp-genpept-esm2-650m-lora](https://huggingface.co/null-phnix/amp-genpept-esm2-650m-lora)

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "null-phnix/amp-genpept-esm2-650m-lora")
```
