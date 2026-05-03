# Dataset and Task Card

---

## Task

| Field | Value |
|-------|-------|
| Task name | Binary aircraft surface damage classification |
| Task type | Binary image classification |
| Label space | `crack`, `dent` (two classes) |
| Input | Single RGB aircraft surface image |
| Output | Predicted class label + confidence score |
| Secondary task | Vision-language report generation (BLIP captioning) |

**Verified from**: `configs/train.yaml`, `configs/inference.yaml`,
`src/aircraft_damage/predict.py`

---

## Label Space

- **`crack`** — class index 0 (verified from `configs/train.yaml` →
  `classifier.class_names[0]`)
- **`dent`** — class index 1 (verified from `configs/train.yaml` →
  `classifier.class_names[1]`)
- Binary threshold: 0.5 (verified from `configs/inference.yaml`)

### Task Boundaries

| In scope | Out of scope |
|----------|-------------|
| Binary crack / dent classification | Multi-class or multi-label damage taxonomy |
| Whole-image classification | Damage region localisation or segmentation |
| Categorical label output | Damage severity estimation |
| Research prototype | Maintenance certification |
| Decision-support artefact | Airworthiness determination |

---

## Dataset Status

**The dataset is not bundled with this repository.**

Images must be obtained separately. The training script includes a
`--download-data` option that fetches the public tarball referenced below.

### Expected Local Layout

```text
data/
└── aircraft_damage_dataset_v1/
    ├── train/
    │   ├── crack/
    │   └── dent/
    ├── valid/
    │   ├── crack/
    │   └── dent/
    └── test/
        ├── crack/
        └── dent/
```

Verified from `data/README.md` and `configs/train.yaml` →
`paths.dataset_root`.

### Dataset Source Reference

| Field | Value |
|-------|-------|
| Name | Roboflow Aircraft Damage Dataset |
| Author | Youssef Donia |
| License | CC BY 4.0 |
| Public tarball URL | `https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar` |

Verified from `data/README.md` and README.md.

---

## Split Sizes (Preserved from Archived Experiment)

| Split | Samples |
|-------|---------|
| Train | 300 |
| Validation | 96 |
| Test | 50 |

These values are preserved from `results/metrics.json` → `dataset_sizes`.
They reflect the archived experiment and are not newly confirmed in this
documentation pass.

---

## Checkpoint Status

**The model checkpoint is not bundled with this repository.**

- Default checkpoint path: `models/vgg16_aircraft_damage.keras`
  (verified from `configs/inference.yaml` → `paths.model_checkpoint`)
- To obtain a checkpoint: train from scratch using
  `python scripts/run_train.py --config configs/train.yaml --download-data`
  or supply a locally trained `.keras` file at the path above.
- All inference, evaluation, and demo workflows fail gracefully if the
  checkpoint is absent.

---

## Licensing and Provenance Notes

- **Code**: MIT License (see `LICENSE`)
- **Dataset**: CC BY 4.0 (Roboflow Aircraft Damage Dataset) — do not commit
  dataset images to the repository
- **BLIP weights**: subject to Hugging Face model licensing terms for
  `Salesforce/blip-image-captioning-base`
- **Archived notebooks**: preserved for provenance; marked `linguist-vendored`
  in `.gitattributes`
- **Metrics**: preserved from archived notebook outputs; checkpoint not included

---

## Reproducibility Notes

- Running inference or evaluation requires a locally trained or user-supplied
  checkpoint.
- Running training requires downloading the dataset (can be automated with
  `--download-data`).
- Package installation and dataset/checkpoint setup steps are documented in
  [REPRODUCIBILITY_CHECKLIST.md](REPRODUCIBILITY_CHECKLIST.md).
- The validation tool (`tools/evidence/validate_research_pack.py`) can run
  without a checkpoint, dataset, BLIP download, or internet access.
- Tests in `tests/` can run without a checkpoint or dataset; tests that
  require inference are skipped when the checkpoint is absent.
