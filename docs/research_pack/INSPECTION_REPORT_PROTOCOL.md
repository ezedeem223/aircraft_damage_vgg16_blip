# Inspection Report Protocol

## Aircraft Damage Assessment — Report Generation Workflow

---

> **Important**: Generated reports are descriptive support artefacts and
> require human review.
> Reports are not maintenance approvals or airworthiness determinations.

---

## 1. Overview

The inspection-style report is the final output of the end-to-end pipeline.
It consolidates the classifier result and BLIP-generated text into a single
structured string. The full implementation is in
`src/aircraft_damage/report_generator.py`.

---

## 2. Input

- A single RGB image file path
- The predicted damage class from the VGG16 classifier
- The classifier confidence score (0.0 – 1.0, computed at runtime)
- Optional metadata dictionary (key-value pairs)

---

## 3. Classifier Output Stage

The VGG16 classifier produces:

- **Predicted class**: `crack` or `dent`
  (verified from `configs/inference.yaml` → `classifier.class_names`)
- **Raw sigmoid score**: a float between 0.0 and 1.0
- **Confidence**: derived from the raw score relative to the 0.5 threshold
  - If predicted index = 1 (dent): confidence = raw score
  - If predicted index = 0 (crack): confidence = 1.0 − raw score
- **Threshold**: 0.5 (verified from `configs/inference.yaml`)

Confidence values at runtime depend entirely on the local checkpoint and
input image. No confidence values are hardcoded or invented in this document.

---

## 4. BLIP Caption Stage

Using `src/aircraft_damage/report_generator.py` → `generate_caption()`:

- Model: `Salesforce/blip-image-captioning-base`
  (verified from `configs/report_generation.yaml`)
- Prompt: `"This is a picture of"` (verified from `configs/report_generation.yaml`)
- `max_new_tokens`: 40 (verified from `configs/report_generation.yaml`)
- Output: a short descriptive sentence conditioned on the image and prompt

---

## 5. BLIP Summary Stage

Using `src/aircraft_damage/report_generator.py` → `generate_summary()`:

- Same model and configuration as the caption stage
- Prompt: `"This is a detailed photo showing"` (verified from `configs/report_generation.yaml`)
- Output: a longer descriptive sentence

---

## 6. Consolidated Report Structure

The final report string is assembled by
`src/aircraft_damage/report_generator.py` → `generate_report_bundle()`.
The structure below is verified from the source code:

```
Aircraft Damage Assessment
Image: <image_file_name>
Predicted damage class: <class_name>
Confidence: <confidence_as_percentage>
Caption: <blip_caption_or_fallback>
Summary: <blip_summary_or_fallback>
Metadata:
- <key>: <value>  (or "No additional metadata supplied.")
Generation status: <blip_status_message>
Note: This output is decision support only and should be reviewed by a human inspector.
```

The note on the final line is embedded in the source code and is always present.

---

## 7. Metadata Handling

- The `metadata` parameter is optional (defaults to an empty dict).
- Keys are title-cased and underscores replaced with spaces.
- If no metadata is supplied, the block reads: `- No additional metadata supplied.`
- Metadata is never automatically inferred from the image; it must be passed
  explicitly by the caller.

---

## 8. Fallback Behaviour When BLIP Is Unavailable

If the BLIP model cannot be loaded (e.g. missing cache, network error, import
failure) and `use_template_fallback` is `true` (verified as the default in
`configs/report_generation.yaml`):

- `caption` is set to: `"BLIP caption unavailable."`
- `summary` is set to: `"BLIP summary unavailable."`
- `model_status` records the failure reason
- The classifier output (predicted class, confidence) is still included in
  the report
- No exception is raised to the caller

If `use_template_fallback` is `false`, the exception propagates.

---

## 9. Archived Example Outputs

The following BLIP outputs are preserved in
`results/sample_reports/notebook_blip_outputs.txt` and are quoted here for
reference. They are **archived experiment artefacts** from the original notebook,
not newly generated outputs:

```
Caption: this is a picture of a plane
Summary: this is a detailed photo showing the engine of a boeing 747

Caption: this is a picture of a plane that was sitting on the ground in a field
Summary: this is a detailed photo showing the damage to the fuselage of the aircraft
```

These examples illustrate the general character of BLIP output on aircraft
images. The summaries reference aircraft components (`engine`, `fuselage`) that
may not correspond to the actual damage region in the image, illustrating the
hallucination and domain-vocabulary risks documented in
[VISION_LANGUAGE_LIMITATIONS.md](VISION_LANGUAGE_LIMITATIONS.md).

---

## 10. Labelling Generated Reports

Reports produced by this pipeline should always be labelled as:

- **"AI-assisted descriptive support output"**
- **"Requires human review before any maintenance or safety decision"**
- **Not a maintenance approval**
- **Not an airworthiness determination**

The source code already includes the following footer in every report:

> "Note: This output is decision support only and should be reviewed by a human inspector."

---

## 11. How to Generate a Report at Runtime

```bash
python scripts/run_predict.py \
  --image path/to/image.jpg \
  --config configs/inference.yaml \
  --report-config configs/report_generation.yaml
```

Requires:
- A local checkpoint at `models/vgg16_aircraft_damage.keras`
- BLIP assets cached locally or network access for first download

---

*Generated reports are descriptive support artefacts and require human review.
Reports are not maintenance approvals or airworthiness determinations.*
