"""Dataset access, validation, and download helpers."""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .utils import AssetNotFoundError, ensure_directory, resolve_path

DEFAULT_DATASET_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar"
)


@dataclass(frozen=True)
class DatasetPaths:
    """Canonical split locations for the aircraft damage dataset."""

    root: Path
    train: Path
    valid: Path
    test: Path


def get_dataset_paths(dataset_root: str | Path) -> DatasetPaths:
    """Construct the standard dataset split paths."""

    root = resolve_path(dataset_root)
    return DatasetPaths(
        root=root,
        train=root / "train",
        valid=root / "valid",
        test=root / "test",
    )


def infer_class_names(split_dir: str | Path) -> list[str]:
    """Infer class names from the directory structure."""

    directory = resolve_path(split_dir)
    if not directory.exists():
        return []
    return sorted(path.name for path in directory.iterdir() if path.is_dir())


def validate_dataset_layout(dataset_root: str | Path) -> tuple[DatasetPaths, list[str]]:
    """Validate that the expected split directories exist."""

    paths = get_dataset_paths(dataset_root)
    missing = [path.name for path in (paths.train, paths.valid, paths.test) if not path.exists()]
    if missing:
        raise AssetNotFoundError(
            f"Dataset not found or incomplete under: {paths.root}",
            next_steps=[
                "Place the extracted dataset under the configured dataset_root.",
                "Expected subdirectories: train/, valid/, test/.",
                "Use scripts/run_train.py --download-data to fetch the public dataset archive.",
            ],
        )

    class_names = infer_class_names(paths.train)
    if not class_names:
        raise AssetNotFoundError(
            f"No class folders were found inside the training split: {paths.train}",
            next_steps=[
                "Verify the dataset extraction completed successfully.",
                "Expected class folders similar to crack/ and dent/.",
            ],
        )
    return paths, class_names


def _safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    for member in tar.getmembers():
        member_path = (destination / member.name).resolve()
        if not str(member_path).startswith(str(destination.resolve())):
            raise RuntimeError(f"Unsafe path detected in tar archive: {member.name}")
    tar.extractall(destination)


def download_dataset(
    dataset_root: str | Path,
    archive_path: str | Path,
    dataset_url: str = DEFAULT_DATASET_URL,
    force: bool = False,
) -> DatasetPaths:
    """Download and extract the dataset archive referenced by the project config."""

    paths = get_dataset_paths(dataset_root)
    archive = resolve_path(archive_path)
    ensure_directory(paths.root.parent)
    ensure_directory(archive.parent)

    if paths.root.exists() and not force:
        try:
            return validate_dataset_layout(paths.root)[0]
        except AssetNotFoundError:
            pass

    if paths.root.exists() and force:
        shutil.rmtree(paths.root)

    urllib.request.urlretrieve(dataset_url, archive)
    with tarfile.open(archive, "r") as tar_file:
        _safe_extract(tar_file, paths.root.parent)

    return validate_dataset_layout(paths.root)[0]
