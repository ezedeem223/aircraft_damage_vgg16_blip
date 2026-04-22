# Data Directory

This repository does not track the aircraft image dataset in Git.

## Expected Layout

```text
data/
`-- aircraft_damage_dataset_v1/
    |-- train/
    |   |-- crack/
    |   `-- dent/
    |-- valid/
    |   |-- crack/
    |   `-- dent/
    `-- test/
        |-- crack/
        `-- dent/
```

## Source

- Public dataset tarball reference used in earlier experiments:
  `https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar`
- Original source reference: Roboflow Aircraft Damage Dataset

## How to Prepare the Data

1. Download and extract the dataset so that `train/`, `valid/`, and `test/` sit under `data/aircraft_damage_dataset_v1/`.
2. Keep the class folder names consistent with the current project configuration: `crack` and `dent`.
3. Do not commit the raw images to Git.

## Script Support

- `python scripts/run_train.py --config configs/train.yaml --download-data` downloads the public tarball and extracts it locally.
- `python scripts/run_evaluate.py --config configs/inference.yaml` expects the `test/` split to already exist.
