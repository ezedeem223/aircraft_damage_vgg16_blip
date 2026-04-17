# Results Directory

This directory contains two kinds of artifacts:

1. preserved outputs from the original notebook
2. generated outputs from the refactored scripts

## Preserved Baseline Artifacts

- `training_loss.png`
- `validation_loss.png`
- `accuracy_curve.png`
- `sample_predictions/notebook_sample_prediction.png`
- `sample_reports/notebook_blip_example_image.png`
- `sample_reports/notebook_blip_outputs.txt`
- `metrics.json`

The baseline metrics and plots above were copied from the committed notebook outputs. They are included for transparency because the repository does not contain the original trained checkpoint.

## Placeholders

- `classification_report.txt`
- `confusion_matrix.png`

These files are placeholders until you run evaluation with a local checkpoint:

```bash
python scripts/run_evaluate.py --config configs/inference.yaml
```
