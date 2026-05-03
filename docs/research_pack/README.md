# Aircraft Damage Research Evidence Pack

This directory contains institution-neutral research documentation for the
aircraft damage classification and vision-language report-generation workflow
implemented in `aircraft_damage_vgg16_blip`.

## Contents

| File | Purpose |
|------|---------|
| [ACADEMIC_RESEARCH_BRIEF.md](ACADEMIC_RESEARCH_BRIEF.md) | Problem definition, pipeline overview, evidence artifacts, and academic framing |
| [MODEL_CARD.md](MODEL_CARD.md) | Classifier and report-generator model card |
| [METRIC_PROVENANCE_MATRIX.md](METRIC_PROVENANCE_MATRIX.md) | Traceability matrix for every preserved metric |
| [INSPECTION_REPORT_PROTOCOL.md](INSPECTION_REPORT_PROTOCOL.md) | Documentation of the report-generation workflow and output format |
| [VISION_LANGUAGE_LIMITATIONS.md](VISION_LANGUAGE_LIMITATIONS.md) | Known limitations of the BLIP-based report-generation component |
| [FAILURE_MODE_MATRIX.md](FAILURE_MODE_MATRIX.md) | Structured failure mode catalogue for the full pipeline |
| [DATASET_AND_TASK_CARD.md](DATASET_AND_TASK_CARD.md) | Task definition, label space, and dataset provenance |
| [REPRODUCIBILITY_CHECKLIST.md](REPRODUCIBILITY_CHECKLIST.md) | Step-by-step checklist for reproducing all pipeline stages |

## Validation

Run the evidence-pack validation tool from the repository root:

```bash
python tools/evidence/validate_research_pack.py
```

The tool checks that all required documentation files are present, that
preserved metrics are accessible, and that forbidden overclaims are absent from
the documentation.

## Scope

This evidence pack is a documentation and reproducibility artefact. It does
not contain a trained checkpoint, dataset images, or newly generated model
outputs. All metric values referenced here are preserved from committed
experiment artefacts and are clearly labelled as such.
