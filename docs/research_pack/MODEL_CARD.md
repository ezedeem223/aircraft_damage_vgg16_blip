# Model Card

## Aircraft Damage VGG16 + BLIP Pipeline

---

## Model Overview

This repository implements two cooperating model components:

1. **VGG16 Classifier** — binary image classification of aircraft surface damage
2. **BLIP Report Generator** — image-conditional natural language captioning
   and inspection-style text generation

---

## Component 1: VGG16 Classifier

### Model Family

- Architecture: VGG16 convolutional neural network
- Pretrained weights: ImageNet (`include_top=False`)
- Fine-tuning approach: frozen VGG16 backbone with a trainable dense head
- Source: TensorFlow / Keras

### Classifier Head

Verified from README and source:

```
VGG16 backbone (frozen, ImageNet weights)
  └─▶ Flatten
       └─▶ Dense(512, ReLU)
            └─▶ Dropout(0.3)
                 └─▶ Dense(512, ReLU)
                      └─▶ Dropout(0.3)
                           └─▶ Dense(1, Sigmoid)
```

### Input Assumptions

- Image size: 224 × 224 pixels (verified from `configs/inference.yaml`)
- Channels: RGB
- Preprocessing: standard Keras/PIL pipeline (see `src/aircraft_damage/preprocessing.py`)

### Class Labels

- `crack` — index 0 (verified from `configs/train.yaml` and `configs/inference.yaml`)
- `dent` — index 1 (verified from `configs/train.yaml` and `configs/inference.yaml`)
- Task type: binary classification with sigmoid output and threshold 0.5

### Intended Research Use

- Demonstrating a VGG16 transfer-learning pipeline for aircraft surface damage
  classification
- Providing a fast categorical signal as input to a vision-language
  report-generation component
- Serving as a research artefact and decision-support prototype

### Out-of-Scope Uses

- Certified aircraft inspection — this model is not approved for airworthiness
  determinations
- Maintenance approval decisions — generated outputs require human review
- Damage severity estimation — the model outputs a binary label only
- Damage localisation — the model classifies the whole image; no region
  detection is implemented
- Aircraft types or image conditions outside the training distribution

### Checkpoint Status

- **No trained checkpoint is committed to this repository. The checkpoint is not bundled.**
- The checkpoint used to produce the preserved baseline metrics is not included.
- To obtain a checkpoint: run `python scripts/run_train.py --config configs/train.yaml --download-data`
  or supply a locally trained `.keras` file at `models/vgg16_aircraft_damage.keras`.
- All inference, evaluation, and demo workflows will fail gracefully if the
  checkpoint is missing.

### Dataset Status

- **The dataset is not bundled with this repository.**
- Expected local layout: `data/aircraft_damage_dataset_v1/{train,valid,test}/{crack,dent}/`
- Dataset reference: Roboflow Aircraft Damage Dataset, CC BY 4.0
- Training script can download the public tarball automatically with `--download-data`.

### Preserved Metrics

All values below are preserved from `results/metrics.json` and were collected
from archived notebook experiment artefacts. They were not regenerated in this
documentation pass. The checkpoint that produced them is not committed.

| Metric | Value | Source |
|--------|-------|--------|
| Training samples | 300 | `results/metrics.json` |
| Validation samples | 96 | `results/metrics.json` |
| Test samples | 50 | `results/metrics.json` |
| Final training accuracy (epoch 5) | 0.8800 | `results/metrics.json` |
| Final validation accuracy (epoch 5) | 0.7083 | `results/metrics.json` |
| Test accuracy | 0.6875 | `results/metrics.json` |
| Test loss | 0.7326 | `results/metrics.json` |

Epoch-by-epoch training accuracy history (verified from `results/metrics.json`):
`[0.5367, 0.7233, 0.7767, 0.8133, 0.8800]`

Epoch-by-epoch validation accuracy history (verified from `results/metrics.json`):
`[0.6042, 0.6979, 0.7188, 0.6562, 0.7083]`

**Important notes on these metrics:**
- They are preserved historical evaluation artefacts, not newly reproduced values.
- The dataset split sizes are small; metrics may not generalise.
- A full per-class classification report (precision, recall, F1) was not saved
  in the archived artefacts; `results/classification_report.txt` is a placeholder.
- The confusion matrix (`results/confusion_matrix.png`) is also a placeholder.

### Known Technical Limitations

- Binary label space only (`crack` / `dent`)
- Whole-image classification with no damage localisation
- Small training dataset (300 samples); generalisation is not validated
- No data-augmentation is documented in the archived experiment configuration
- ImageNet-pretrained features may not fully transfer to aviation surface textures

---

## Component 2: BLIP Report Generator

### Model Family

- Architecture: BLIP (Bootstrapped Language-Image Pre-training) for conditional
  image captioning
- Default model identifier: `Salesforce/blip-image-captioning-base`
  (verified from `configs/report_generation.yaml` and
  `src/aircraft_damage/report_generator.py`)
- Source: Hugging Face Transformers

### BLIP Asset Status

- **BLIP weights are not stored in this repository.**
- They are downloaded by Hugging Face Transformers on first use unless already
  cached locally.
- See `HF_HOME` / `TRANSFORMERS_CACHE` in `.env.example` for cache configuration.

### Intended Research Use

- Generating descriptive natural-language text conditioned on aircraft images
  and classifier output
- Demonstrating a vision-language reporting pipeline as a decision-support
  prototype
- Providing human-readable context to supplement the binary classifier signal

### Out-of-Scope Uses

- BLIP is a general-purpose image captioning model and has not been fine-tuned
  on aviation imagery
- Generated text is **not** a certified maintenance document
- Generated text is **not** an airworthiness determination
- Generated text should not be used as a sole basis for maintenance decisions

### Ethical and Safety Limitations

- Generated reports are descriptive support artefacts and **require human review**
- BLIP may hallucinate details that are not present in the image
- Model outputs may misrepresent damage type, severity, or location
- Reports are not maintenance approvals or airworthiness determinations
- Do not use generated text as a substitute for qualified inspector judgment

---

## Human Review Requirement

All outputs from this pipeline — classifier labels, confidence scores, captions,
summaries, and consolidated reports — are decision-support artefacts. They must
be reviewed by a qualified human inspector before any maintenance or safety
decision is made. The codebase itself includes this notice in the report output:

> "Note: This output is decision support only and should be reviewed by a human
> inspector."

---

*Metric values on this card are preserved from committed experiment artefacts.
No new evaluation was run for this documentation pass.*
