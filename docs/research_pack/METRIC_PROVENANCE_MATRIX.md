# Metric Provenance Matrix

All metrics in this repository are preserved from archived experiment artefacts.
No metric values were newly generated in this documentation pass.
The trained checkpoint used to produce these values is not committed to
this repository.

---

## Provenance Matrix

| Metric | Value | Source File | Source Context | Evidence Confidence | Allowed Wording | Forbidden Wording |
|--------|-------|-------------|----------------|--------------------|-----------------|--------------------|
| Training samples | 300 | `results/metrics.json` → `dataset_sizes.train` | Archived notebook experiment | Verified from `results/metrics.json` | "training split used in the archived experiment" | "newly evaluated", "current split", "reproduced" |
| Validation samples | 96 | `results/metrics.json` → `dataset_sizes.valid` | Archived notebook experiment | Verified from `results/metrics.json` | "validation split used in the archived experiment" | "newly evaluated", "current split" |
| Test samples | 50 | `results/metrics.json` → `dataset_sizes.test` | Archived notebook experiment | Verified from `results/metrics.json` | "test split used in the archived experiment" | "newly evaluated", "current test set" |
| Training accuracy — epoch 1 | 0.5367 | `results/metrics.json` → `accuracy_epoch_history[0]` | Archived notebook experiment | Verified from `results/metrics.json` | "preserved training accuracy at epoch 1" | "current training accuracy", "reproduced" |
| Training accuracy — epoch 2 | 0.7233 | `results/metrics.json` → `accuracy_epoch_history[1]` | Archived notebook experiment | Verified from `results/metrics.json` | "preserved training accuracy at epoch 2" | "current training accuracy", "reproduced" |
| Training accuracy — epoch 3 | 0.7767 | `results/metrics.json` → `accuracy_epoch_history[2]` | Archived notebook experiment | Verified from `results/metrics.json` | "preserved training accuracy at epoch 3" | "current training accuracy", "reproduced" |
| Training accuracy — epoch 4 | 0.8133 | `results/metrics.json` → `accuracy_epoch_history[3]` | Archived notebook experiment | Verified from `results/metrics.json` | "preserved training accuracy at epoch 4" | "current training accuracy", "reproduced" |
| Training accuracy — epoch 5 (final) | 0.8800 | `results/metrics.json` → `accuracy_epoch_history[4]` | Archived notebook experiment | Verified from `results/metrics.json` | "final preserved training accuracy after 5 epochs" | "current training accuracy", "reproduced", overclaims about performance standing |
| Training loss — epoch 1 | 0.7201 | `results/metrics.json` → `loss_epoch_history[0]` | Archived notebook experiment | Verified from `results/metrics.json` | "preserved training loss at epoch 1" | "current training loss", "reproduced" |
| Training loss — epoch 5 (final) | 0.3204 | `results/metrics.json` → `loss_epoch_history[4]` | Archived notebook experiment | Verified from `results/metrics.json` | "preserved final training loss" | "current training loss", "reproduced" |
| Validation accuracy — epoch 1 | 0.6042 | `results/metrics.json` → `validation_accuracy_epoch_history[0]` | Archived notebook experiment | Verified from `results/metrics.json` | "preserved validation accuracy at epoch 1" | "current validation accuracy", "reproduced" |
| Validation accuracy — epoch 5 (final) | 0.7083 | `results/metrics.json` → `validation_accuracy_epoch_history[4]` | Archived notebook experiment | Verified from `results/metrics.json` | "final preserved validation accuracy after 5 epochs" | "current validation accuracy", "reproduced" |
| Validation loss — epoch 1 | 0.6338 | `results/metrics.json` → `validation_loss_epoch_history[0]` | Archived notebook experiment | Verified from `results/metrics.json` | "preserved validation loss at epoch 1" | "current validation loss", "reproduced" |
| Validation loss — epoch 5 (final) | 0.5098 | `results/metrics.json` → `validation_loss_epoch_history[4]` | Archived notebook experiment | Verified from `results/metrics.json` | "final preserved validation loss" | "current validation loss", "reproduced" |
| Test accuracy | 0.6875 | `results/metrics.json` → `test_accuracy` | Archived notebook experiment | Verified from `results/metrics.json` | "preserved test accuracy from archived experiment" | "current test accuracy", "reproduced", "production accuracy" |
| Test loss | 0.7326 | `results/metrics.json` → `test_loss` | Archived notebook experiment | Verified from `results/metrics.json` | "preserved test loss from archived experiment" | "current test loss", "reproduced" |
| Per-class precision / recall / F1 | N/A | `results/classification_report.txt` | Placeholder file — not saved in archived experiment | **Placeholder — do not use as performance evidence** | "not available from archived experiment artefacts" | Any numeric claim |
| Confusion matrix | N/A | `results/confusion_matrix.png` | Placeholder image — not saved in archived experiment | **Placeholder — do not use as performance evidence** | "not available from archived experiment artefacts" | Any numeric claim |

---

## Metric Provenance Statement

The `results/metrics.json` file contains the following provenance field
(verified by inspection):

> "Preserved from notebook outputs in `notebooks/aircraft_damage_vgg16_blip.ipynb`.
> The original checkpoint is not committed to this repository."

---

## Red Lines

The following claims are prohibited when referencing the above metrics:

- **Do not** call any preserved metric "newly reproduced" unless the evaluation
  commands (`scripts/run_evaluate.py`) were actually executed with a local
  checkpoint and dataset.
- **Do not** imply production readiness based on a 50-sample test split.
- **Do not** imply certification or safety-critical validation.
- **Do not** imply generalisation to aircraft types, damage categories, or
  imaging conditions outside the archived experiment.
- **Do not** imply spatial localisation of damage; the model classifies whole
  images only.
- **Do not** present `results/classification_report.txt` or
  `results/confusion_matrix.png` as actual evaluation evidence; both are
  placeholders.
- **Do not** add metric values not already present in committed artefacts.
- **Do not** invent confidence intervals, AUC, or F1 values not present in
  `results/metrics.json`.

---

## How to Generate Fresh Metrics

Once a local checkpoint and dataset are available:

```bash
python scripts/run_evaluate.py --config configs/inference.yaml
```

This will overwrite `results/classification_report.txt` and
`results/confusion_matrix.png` with real evaluation outputs.
