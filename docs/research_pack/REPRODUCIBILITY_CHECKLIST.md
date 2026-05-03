# Reproducibility Checklist

---

> Some pipeline stages require a local checkpoint and dataset that are not
> bundled with this repository. Steps that can be completed without them
> are marked ✅ (no assets needed). Steps that require assets are marked
> ⚠️ (requires checkpoint / dataset).

---

## 1. Installation

✅ Can be completed without checkpoint or dataset.

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# or: .\.venv\Scripts\Activate.ps1  # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For development tooling (linting, testing):

```bash
python -m pip install -r requirements-dev.txt
```

**Verification**:
```bash
python -c "import aircraft_damage; print(aircraft_damage.__version__)"
```

---

## 2. Dataset Setup

⚠️ Required for training and evaluation.

**Option A — automatic download via training script**:
```bash
python scripts/run_train.py --config configs/train.yaml --download-data
```
This fetches the public tarball and extracts it to `data/aircraft_damage_dataset_v1/`.

**Option B — manual extraction**:
Download from:
```
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar
```
Extract so that `train/`, `valid/`, and `test/` exist under
`data/aircraft_damage_dataset_v1/`, each containing `crack/` and `dent/`
subdirectories.

**Verification**:
```bash
ls data/aircraft_damage_dataset_v1/train/crack/
ls data/aircraft_damage_dataset_v1/train/dent/
```

---

## 3. Training

⚠️ Requires dataset. Produces checkpoint at `models/vgg16_aircraft_damage.keras`.

```bash
python scripts/run_train.py --config configs/train.yaml
```

If dataset is not yet present, add `--download-data` to fetch it automatically.

**Expected outputs**:
- `models/vgg16_aircraft_damage.keras` — trained checkpoint
- `results/accuracy_curve.png`
- `results/training_loss.png`
- `results/validation_loss.png`
- `results/metrics.json` (updated with fresh evaluation)

---

## 4. Evaluation

⚠️ Requires checkpoint and test split.

```bash
python scripts/run_evaluate.py --config configs/inference.yaml
```

**Expected outputs**:
- `results/classification_report.txt` — per-class precision, recall, F1
- `results/confusion_matrix.png` — confusion matrix plot
- Console: test accuracy and test loss

---

## 5. Single-Image Prediction

⚠️ Requires checkpoint. BLIP download needed on first run (internet).

```bash
python scripts/run_predict.py \
  --image path/to/image.jpg \
  --config configs/inference.yaml \
  --report-config configs/report_generation.yaml
```

**Expected output**: printed inspection-style report string.

---

## 6. Gradio Demo

⚠️ Requires checkpoint. BLIP download needed on first run (internet).

```bash
python scripts/run_demo.py \
  --config configs/inference.yaml \
  --report-config configs/report_generation.yaml
```

Opens a Gradio interface for image upload and live inference.

---

## 7. Tests

✅ The test suite can run without checkpoint or dataset. Tests that require
inference are skipped when the checkpoint is absent.

```bash
python -m pytest
```

To run only tests that do not require assets:
```bash
python -m pytest -m "not requires_checkpoint"
```

---

## 8. Evidence-Pack Validation Tool

✅ Runs without checkpoint, dataset, BLIP download, or internet.

```bash
python tools/evidence/validate_research_pack.py
```

---

## 9. Linting

✅ Runs without checkpoint or dataset.

```bash
python -m ruff check .
```

---

## Required Local Assets Summary

| Asset | Required For | How to Obtain |
|-------|-------------|---------------|
| Dataset at `data/aircraft_damage_dataset_v1/` | Training, evaluation | `--download-data` flag or manual extraction |
| Checkpoint at `models/vgg16_aircraft_damage.keras` | Inference, evaluation, demo | Run training script |
| BLIP assets (Hugging Face cache) | Report generation, demo | Downloaded automatically on first use |

---

## What Can Be Verified Without Checkpoint or Dataset

| Step | Verifiable Without Assets |
|------|--------------------------|
| Package installation | ✅ |
| Module imports | ✅ |
| Config parsing | ✅ |
| Test suite (asset-independent tests) | ✅ |
| Evidence-pack validation tool | ✅ |
| Linting | ✅ |
| Preserved metrics in `results/metrics.json` | ✅ |
| Archived plot artefacts | ✅ (visual inspection only) |

---

## What Cannot Be Verified Without Checkpoint or Dataset

| Step | Blocker |
|------|---------|
| Fresh training | Dataset |
| Fresh evaluation metrics | Checkpoint + dataset |
| Single-image inference | Checkpoint |
| Gradio demo with active classifier | Checkpoint |
| BLIP report generation | BLIP assets (network or cache) |
| Confusion matrix (fresh) | Checkpoint + dataset |
| Per-class classification report (fresh) | Checkpoint + dataset |

---

## Expected Results Reference

Preserved baseline metrics (from `results/metrics.json`, not newly reproduced):

| Metric | Preserved value |
|--------|----------------|
| Test accuracy | 0.6875 |
| Test loss | 0.7326 |
| Final training accuracy (epoch 5) | 0.8800 |
| Final validation accuracy (epoch 5) | 0.7083 |

Fresh training results may differ depending on random seed, framework version,
and dataset download state.
