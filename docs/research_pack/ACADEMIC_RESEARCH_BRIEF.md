# Academic Research Brief

## Aircraft Damage Classification and Vision-Language Reporting Workflow

---

## 1. Problem Definition

Aircraft surface inspection generates large volumes of photographic evidence
that must be reviewed, categorised, and documented. Manual review is
time-consuming and subject to reviewer fatigue and inconsistency. Automated
decision-support tools that combine image classification with descriptive
natural-language output can reduce the cognitive load on reviewers and
provide a consistent, structured first-pass signal.

This repository implements an inspection-support workflow that addresses two
complementary sub-problems:

1. **Classification** — given a surface image, assign a binary damage label
   (`crack` or `dent`) using a fine-tuned VGG16 convolutional network.
2. **Report generation** — given the same image and the classifier output,
   produce a short descriptive text using BLIP image captioning.

---

## 2. Why Aircraft Visual Inspection Support Matters

- Aircraft surface damage, if undetected, can affect structural integrity.
- Manual photographic review is a routine but labour-intensive part of
  maintenance documentation workflows.
- Binary classification provides a rapid categorical signal that can be used
  to triage images for human follow-up.
- Descriptive language output adds context that a label alone cannot convey,
  supporting documentation and reporting tasks.
- Combining both signals in a single lightweight pipeline makes the workflow
  practical for research demonstration and exploration.

The scope of this project is a **decision-support prototype**. It is not a
certified inspection tool, and all outputs require human review.

---

## 3. Task Scope

- **Input**: a single RGB aircraft surface image
- **Classification task**: binary — `crack` vs. `dent`
  (verified from `configs/train.yaml` and `configs/inference.yaml`)
- **Report-generation task**: image-conditional natural language caption and
  summary using BLIP
- **Out of scope**: damage severity estimation, spatial localisation of damage
  regions, multi-class or multi-label classification, maintenance approval

---

## 4. Technical Pipeline

```
Aircraft Image
     │
     ▼
VGG16 Classifier (binary)
     │
     ├─▶ Predicted class label (crack / dent)
     └─▶ Confidence score
             │
             ▼
     BLIP Caption + Summary Generator
             │
             ▼
     Structured Inspection-Style Report
```

The full end-to-end pipeline is implemented as an installable Python package
(`src/aircraft_damage/`), with config-driven scripts and a Gradio demo
interface. The notebook artefacts in `notebooks/` are preserved for provenance
but do not define the maintained runtime surface.

---

## 5. VGG16 Classifier Role

- **Backbone**: VGG16 pretrained on ImageNet with `include_top=False`
  (verified from README and source code in `src/aircraft_damage/train.py`)
- **Head** (verified from README):
  - Flatten → Dense(512, ReLU) → Dropout(0.3)
  - → Dense(512, ReLU) → Dropout(0.3)
  - → Dense(1, Sigmoid)
- **Training configuration used for archived experiment artefacts**
  (verified from README and `configs/train.yaml`):
  - Image size: 224 × 224
  - Batch size: 32
  - Epochs: 5
  - Optimiser: Adam, learning rate 1 × 10⁻⁴
  - Loss: binary cross-entropy
- The trained checkpoint is **not committed** to this repository.
  Preserved baseline metrics are stored in `results/metrics.json`.

---

## 6. BLIP Report-Generation Role

- **Model family**: BLIP image captioning
  (verified from `src/aircraft_damage/report_generator.py`)
- **Default model identifier**: `Salesforce/blip-image-captioning-base`
  (verified from `configs/report_generation.yaml` and source code)
- **Output**: a short caption, a longer descriptive summary, and a
  consolidated inspection-style report string
- **Fallback behaviour**: if BLIP assets are unavailable, the generator
  returns a fallback message and the classifier output is still surfaced
- BLIP is a general-purpose image captioning model not fine-tuned on
  aviation imagery; its output is **descriptive support text** and requires
  human review

---

## 7. Current Evidence Artefacts

The following artefacts are committed to this repository:

| Artefact | Location | Status |
|----------|----------|--------|
| Preserved baseline metrics | `results/metrics.json` | Verified |
| Training accuracy curve | `results/accuracy_curve.png` | Preserved |
| Training loss curve | `results/training_loss.png` | Preserved |
| Validation loss curve | `results/validation_loss.png` | Preserved |
| Sample prediction image | `results/sample_predictions/notebook_sample_prediction.png` | Preserved |
| BLIP example image | `results/sample_reports/notebook_blip_example_image.png` | Preserved |
| BLIP text outputs | `results/sample_reports/notebook_blip_outputs.txt` | Preserved |
| Classification report | `results/classification_report.txt` | Placeholder — requires checkpoint |
| Confusion matrix | `results/confusion_matrix.png` | Placeholder — requires checkpoint |

Metric values reported here and in the model card are sourced from
`results/metrics.json` and are **preserved historical evaluation artefacts**.
They were not regenerated in this documentation pass.

---

## 8. Limitations

- The classifier is binary and cannot distinguish damage types outside the
  `crack` / `dent` label space.
- No trained checkpoint is included; all inference and evaluation workflows
  require a locally trained or user-supplied checkpoint.
- BLIP is a general-purpose model and may produce captions that do not
  accurately reflect aviation-specific damage terminology.
- The pipeline classifies whole images; damage region localisation is not
  implemented.
- Preserved metrics come from a small dataset (300 training, 96 validation,
  50 test samples) and may not generalise to other aircraft types, image
  conditions, or damage categories.
- Generated reports are **descriptive support artefacts** and are not
  maintenance approvals or airworthiness determinations.

---

## 9. Future Academic Research Directions

- Fine-tuning BLIP or a domain-specific vision-language model on labelled
  aviation imagery to improve caption accuracy
- Extending binary classification to multi-class or multi-label damage
  taxonomy
- Adding spatial localisation (e.g. segmentation or object detection) for
  damage region identification
- Systematic evaluation across multiple aircraft types and imaging conditions
- Comparing VGG16 against modern backbone architectures (e.g. ViT, EfficientNet)
- Curated human evaluation studies of the report-generation output quality
- Severity estimation beyond categorical label assignment
- Calibration and uncertainty quantification for the classifier output

---

*This brief is institution-neutral. All metric values are preserved from
committed experiment artefacts and are clearly labelled as such. No new
model performance numbers were generated for this documentation pass.*
