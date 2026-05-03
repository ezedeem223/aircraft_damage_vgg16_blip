# Vision-Language Limitations

## BLIP Report Generator — Known Limitations

---

> All generated reports from this pipeline are descriptive support artefacts.
> They require human review and are not maintenance approvals or airworthiness
> determinations.

---

## 1. BLIP Is a General-Purpose Captioning Model

`Salesforce/blip-image-captioning-base` was trained on broad web-sourced
image-text pairs. It has not been fine-tuned on aviation imagery or
aircraft surface damage data in this repository.

Consequences:
- The model does not have specialised knowledge of aircraft components,
  damage nomenclature, or maintenance terminology.
- Output vocabulary reflects general image captioning training data, not
  aviation inspection practice.
- References to specific aircraft models, parts, or structural components
  in the generated text may be inaccurate.

---

## 2. Caption Hallucination Risk

BLIP can generate plausible-sounding text that does not accurately describe
the content of the input image. This is a known property of generative
vision-language models trained on broad data.

Examples of hallucination risk in this context:
- Naming an aircraft model or part that is not visible in the image
- Describing a damage type that contradicts the classifier label
- Adding contextual detail (e.g. location, weather, environment) that is
  not present in the image

The archived experiment artefacts in `results/sample_reports/notebook_blip_outputs.txt`
include an example where the summary references "the engine of a boeing 747"
for an image whose subject and damage class are not independently verified
to match that description.

---

## 3. Missing Damage Localisation

The BLIP component does not receive spatial information about where damage
is located in the image. It operates on the full image with a text prompt.

Consequences:
- Generated text may describe a region of the aircraft that is unrelated
  to the actual damage site.
- There is no guarantee that the "damage" referenced in the generated
  summary corresponds to the predicted class or to any visible damage.

The classifier component also classifies whole images; localisation is out
of scope for the current pipeline.

---

## 4. Image Framing and Cropping Sensitivity

BLIP output is sensitive to image framing, cropping, and composition.

- An image that frames background aircraft components prominently may produce
  captions focused on those components rather than the damage of interest.
- Close-up crops of damaged surfaces may produce less coherent captions than
  full-frame aircraft images.
- Aspect ratio and resolution changes can affect output quality.

---

## 5. Classifier-Report Mismatch Risk

The report consolidates two independent model outputs: the VGG16 classifier
label and the BLIP-generated text. These two components do not share a
common reasoning process.

Potential mismatches:
- The classifier predicts `crack`; the BLIP summary describes a dent or a
  different damage type.
- The classifier predicts `dent`; the BLIP caption makes no reference to
  any damage.
- High classifier confidence does not imply high caption accuracy, and
  vice versa.

A human reviewer must assess both components independently and not assume
that the report text confirms the classifier label.

---

## 6. Confidence Misinterpretation Risk

The confidence score reported by the classifier is a sigmoid output
interpreted relative to a 0.5 threshold. It is not a calibrated probability.

- A high confidence score does not mean the prediction is correct.
- Confidence values should not be compared across different checkpoint
  versions without recalibration.
- The confidence value in the report is a runtime artefact; it was not
  logged in the archived experiment artefacts.

---

## 7. Domain Vocabulary Limitations

BLIP may use general English vocabulary that does not align with aviation
inspection terminology. Generated text should not be expected to use terms
such as "stress fracture", "fatigue crack", "impact dent", or other
aviation-specific damage descriptors unless they appear in the training data
by coincidence.

---

## 8. Human Inspection Requirement

All pipeline outputs — classifier labels, confidence scores, captions,
summaries, and consolidated reports — must be reviewed by a qualified human
inspector before any maintenance or safety-relevant decision is made.

The pipeline is designed as a **decision-support prototype**. It is not a
replacement for qualified inspector judgment.

---

## 9. Curated Evaluation Required Before Any Operational Claims

Before making any claim about the suitability of this pipeline for real-world
inspection support, a curated human evaluation study is required that:

- Tests on images representative of the target operational environment
- Involves qualified aviation maintenance personnel as reviewers
- Measures both classifier accuracy and report text quality independently
- Documents failure modes observed in the target domain

No such evaluation has been conducted or is claimed by this repository.

---

## Summary Table

| Limitation | Affected Component | Severity |
|------------|--------------------|----------|
| General-purpose model, not aviation-specialised | BLIP | High |
| Hallucination risk | BLIP | High |
| No damage localisation | BLIP + Classifier | High |
| Image framing sensitivity | BLIP | Medium |
| Classifier-report mismatch | Pipeline | High |
| Confidence miscalibration risk | Classifier | Medium |
| Domain vocabulary gap | BLIP | Medium |
| No curated aviation evaluation | Pipeline | High |
