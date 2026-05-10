# Enhancing LLM Translation Performance for Spanish–Valencian through Supervised Fine-Tuning and Reinforcement Learning

> **EAMT 2026** · University of the Basque Country (UPV/EHU)  
> Paula Guerrero Castelló · `pguerrero005@ikasle.ehu.eus`

This repository contains the code, models, and evaluation scripts for our EAMT 2026 paper on adapting a translation-specialized LLM to the low-resource Spanish-Valencian (ES→VLCA) direction using supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO).

---

## Abstract

Valencian lacks a dedicated language code in most multilingual MT systems and is systematically rendered as standard (Eastern) Catalan. We adapt **TranslateGemma-4B-IT** through three post-training strategies—SFT, GRPO with a learned naturalness classifier (GRPOv1), and GRPO with a composite automatic-metric reward (GRPOv2)—using only publicly available corpora and QLoRA-based training on commodity hardware. Our results show that SFT delivers the largest single-step gain, while GRPOv2 surpasses SFT on all five standard MT metrics. A classifier-based naturalness reward (GRPOv1) causes dialect suppression, underscoring that reward-function alignment with the target variety is critical for dialectal MT.

---

## Repository Structure

| Path | Description |
|---|---|
| `scripts/sft.py` | SFT — QLoRA fine-tuning on 50k ES–VA pairs |
| `scripts/classifier.py` | HT/MT translationese classifier (RoBERTa-ca) |
| `scripts/grpov1.py` | GRPOv1 — chrF + naturalness classifier reward |
| `scripts/grpov2.py` | GRPOv2 — composite reward ★ |
| `scripts/evaluate.py` | Full evaluation + dialectal analysis |
| `results/summary_metrics.xlsx` | Aggregated metrics for all systems |
| `results/eval_results_1k.xlsx` | Per-sentence metrics (1,000 sentences) |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## Training Pipeline

| Model | Initialization | Training Data | Objective |
|---|---|---|---|
| Baseline | `TranslateGemma-4B-IT` | — | Zero-shot inference |
| SFT | Baseline | 50k ES–VA pairs | QLoRA supervised fine-tuning |
| GRPOv1 | SFT checkpoint | 5k ES–VA pairs | `chrF + naturalness classifier reward` |
| GRPOv2 ★ | SFT checkpoint | 10k ES–VA pairs | `chrF + COMET + TTR − copy penalty` |

---

## Reward Functions

**GRPOv1**
```python
r = (1 - alpha) * chrF(hyp, ref) / 100 + alpha * P(HT | hyp)
# alpha annealed from 0 to 0.3 over first 50 steps
```

**GRPOv2** ★
```python
r = 0.5 * chrF(hyp, ref) + 0.3 * COMET(src, hyp, ref) \
  + 0.2 * TTR(hyp) - copy_penalty(src, hyp)
# copy_penalty = 1.0 if hyp == src else 0.0
```

---



## Installation

Install PyTorch with CUDA 12.1 support:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install remaining dependencies:

```bash
pip install -r requirements.txt
```

Install BLEURT separately:

```bash
pip install git+https://github.com/google-research/bleurt.git

wget https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip
unzip bleurt-base-128.zip
```

---

## Models and Datasets

### HuggingFace Models

| System | Model Hub |
|---|---|
| SFT | [`guerreropaula/translategemma4b-sft-es-va`](https://huggingface.co/guerreropaula/translategemma4b-sft-es-va) |
| GRPOv1 | [`guerreropaula/translategemma4b-grpov1-es-va`](https://huggingface.co/guerreropaula/translategemma4b-grpov1-es-va) |
| GRPOv2 ★ | [`guerreropaula/translategemma4b-grpov2-es-va`](https://huggingface.co/guerreropaula/translategemma4b-grpov2-es-va) |
| HT/MT Classifier | [`guerreropaula/ht_mt_classifier_best`](https://huggingface.co/guerreropaula/ht_mt_classifier_best) |

### Datasets

| Dataset | Usage |
|---|---|
| [`gplsi/amic_parallel`](https://huggingface.co/datasets/gplsi/amic_parallel) | SFT and GRPO training (ES–VLCA parallel) |
| [`gplsi/ES-VA_translation_test`](https://huggingface.co/datasets/gplsi/ES-VA_translation_test) | Evaluation test set|
| TildeMODEL, DOGC, Europarl | HT/MT classifier training (standard Catalan) |

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{guerrero-2026-enhancing,
  title     = {Enhancing {LLM} Translation Performance for {Spanish}-{Valencian}
               through Supervised Fine-tuning and Reinforcement Learning},
  author    = {Guerrero Castell{\'o}, Paula},
  booktitle = {Proceedings of the 25th Annual Conference of the
               European Association for Machine Translation (EAMT 2026)},
  year      = {2026}
}
```

---

## License

This work is licensed under a [Creative Commons Attribution-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nd/4.0/) (CC BY-ND 4.0).

© 2026 The authors. No derivative works permitted.
