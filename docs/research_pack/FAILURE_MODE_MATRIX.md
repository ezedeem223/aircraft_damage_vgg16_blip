# Failure Mode Matrix

## Aircraft Damage Classification and Report-Generation Pipeline

---

| # | Failure Mode | Affected Component | Why It Matters | Reviewer Risk | Detection / Mitigation | Safe Wording |
|---|-------------|-------------------|----------------|---------------|------------------------|--------------|
| 1 | Low-resolution image | Classifier + BLIP | Features required for damage discrimination are lost during resize to 224 × 224; BLIP captioning quality degrades with low-information input | Misclassification reported with high confidence | Inspect input image quality before submission; add resolution check to preprocessing | "Image quality may be insufficient for reliable classification" |
| 2 | Blur or motion artefacts | Classifier + BLIP | Blurred edges obscure crack boundaries and surface texture needed for classification | False label; uninformative caption | Flag blurred inputs; request a sharper image | "Image sharpness may affect classification and report quality" |
| 3 | Bad lighting / low contrast | Classifier + BLIP | Dark or overexposed images reduce texture visibility; damage may not be distinguishable from surface variation | Missed damage; confident wrong label | Pre-screen images for exposure; surface damage should be visible under inspection-quality lighting | "Lighting conditions may reduce classification reliability" |
| 4 | Unusual aircraft part or viewpoint | Classifier + BLIP | Training distribution is limited to the archived dataset; unusual viewpoints or structural components may be out of distribution | Out-of-distribution prediction with no warning | Document that the model was trained on specific image types; do not deploy on other aircraft types without retraining | "Image subject may be outside the training distribution" |
| 5 | Occlusion | Classifier + BLIP | Damage region partially hidden by equipment, shadow, or other structures | Damage missed or misclassified | Human review is mandatory; flag images with visible occlusion | "Occluded regions cannot be assessed by the classifier" |
| 6 | Background clutter | Classifier + BLIP | Complex backgrounds (ground vehicles, tarmac markings, other aircraft) may dominate image features | Classifier attends to background; BLIP describes non-damage content | Crop to damage region where possible; whole-image classification is a known limitation | "Background elements may influence classification output" |
| 7 | Damage outside the classifier label set | Classifier | The model predicts only `crack` or `dent`; corrosion, delamination, missing fasteners, or paint damage will be forced into one of the two classes | Incorrect label applied to damage type outside training scope | Document label space explicitly; human reviewer must confirm the label is applicable | "This classifier is binary; damage types other than crack and dent are outside its label space" |
| 8 | Crack / dent ambiguity | Classifier | Some surface damage features are ambiguous between crack and dent categories; the model assigns a binary label regardless | Borderline cases reported with apparent confidence | Confidence score near 0.5 indicates low certainty; flag for mandatory human review | "Low-confidence predictions indicate ambiguous or borderline input" |
| 9 | No visible damage | Classifier + BLIP | Undamaged surfaces may still be classified as one of the two damage categories, as the model has no "no damage" class | False positive damage report | The current binary label set has no negative/undamaged class; this is a known scope limitation | "The classifier does not have a 'no damage' label; undamaged images will be assigned a damage class" |
| 10 | Classifier-report mismatch | Pipeline | Classifier and BLIP are independent; the classifier may predict `crack` while BLIP describes a dent or unrelated surface | Human reviewer may over-rely on descriptive text that contradicts the label | Human reviewer must cross-check classifier label and report text independently | "Classifier label and report text are produced by independent models and must be reviewed separately" |
| 11 | BLIP hallucinated detail | BLIP | BLIP may generate plausible-sounding descriptions of damage or aircraft components that are not visible in the image | Reviewer relies on invented detail | Treat all BLIP text as descriptive support only; do not assume accuracy without independent verification | "Generated text may contain inaccurate or hallucinated detail; human review is required" |
| 12 | Missing classifier checkpoint | Classifier | No trained checkpoint is committed to the repository; all inference workflows fail gracefully with a descriptive error | Classifier output unavailable; report shows "Unavailable" | Train or supply a checkpoint; see `models/README.md` | "Classifier checkpoint not found; train or supply a checkpoint before running inference" |
| 13 | Missing or uncached BLIP assets | BLIP | BLIP weights must be downloaded or cached; if unavailable and fallback is enabled, caption and summary are replaced with fallback strings | Report text is a fallback placeholder, not model output | Pre-cache BLIP assets; check network access; see `.env.example` for cache configuration | "BLIP assets unavailable; report text shows fallback message, not model output" |
| 14 | Dataset shift | Classifier | The model was trained on a specific dataset (Roboflow Aircraft Damage Dataset, CC BY 4.0); images from other aircraft types, sensors, or inspection conditions may differ significantly from the training distribution | Unpredictable accuracy on out-of-distribution data | Retraining or fine-tuning on target-domain data is required before applying to new contexts | "Model performance on images outside the training distribution is not validated" |
| 15 | Confidence threshold miscalibration | Classifier | The default threshold (0.5) is applied to a sigmoid output that is not calibrated; confidence values are not true probabilities | Overconfident or underconfident predictions | Do not interpret sigmoid outputs as calibrated probabilities; treat confidence as a relative ranking signal only | "Confidence scores are not calibrated probabilities; interpret as relative ranking only" |

---

## General Mitigation Principles

1. **Human review is mandatory** for all pipeline outputs before any
   maintenance or safety-relevant decision.
2. **Confidence near 0.5** should trigger mandatory human escalation.
3. **Classifier label and report text are independent** and must be
   cross-checked by the reviewer.
4. **The binary label space** (`crack`, `dent`) cannot represent all damage
   types; reviewers must confirm applicability.
5. **Generated reports are descriptive support artefacts**, not maintenance
   approvals or airworthiness determinations.
