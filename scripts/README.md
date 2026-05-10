# Scripts

This folder contains the training and evaluation scripts used in the project.

## Recommended order

1. `sft.py`
   Trains the SFT adapter for Spanish -> Valencian with QLoRA on `gplsi/amic_parallel`.

2. `classifier.py`
   Trains the HT/MT classifier used later as part of the `GRPOv1` reward.

3. `grpov1.py`
   Continues training from the SFT model with a reward based on `chrF` plus the HT/MT classifier.

4. `grpov2.py`
   Continues training from the SFT model with a composite reward based on `chrF`, `COMET`, `TTR`, and copy penalty.

5. `evaluation.py`
   Runs full evaluation on the test set and produces aggregate metrics plus dialectal analysis.

## What each script saves

- `sft.py`
  Saves checkpoints to `outputs/sft/checkpoints` and the best validation-BLEU model to `outputs/sft/best_model`.

- `classifier.py`
  Saves training checkpoints to `outputs/classifier/checkpoints` and the final inference-ready classifier to `outputs/classifier/best_model`.

- `grpov1.py`
  Saves checkpoints to `outputs/grpov1/checkpoints` and the best validation-reward model to `outputs/grpov1/best_model`.

- `grpov2.py`
  Saves checkpoints to `outputs/grpov2/checkpoints` and the best validation-reward model to `outputs/grpov2/best_model`.

- `evaluation.py`
  Writes JSON summaries and figures in the current working directory unless you change the paths in the script.

## Validation

- `sft.py`, `grpov1.py`, and `grpov2.py` now hold out a small validation split from `gplsi/amic_parallel`.
- `sft.py` selects the best model by validation `BLEU`.
- `grpov1.py` and `grpov2.py` select the best model by validation reward.
- The classifier still uses its own train/validation split through the Hugging Face `Trainer`.

## Notes

- Most scripts expect GPU access.
- If `HF_TOKEN` is left empty, the scripts skip Hugging Face login and still work for public models/datasets.
- Some scripts download large models and metrics resources on first run.
