"""Image preprocessing and generator creation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .dataset import validate_dataset_layout
from .utils import AssetNotFoundError, resolve_path


def _image_generators() -> tuple[Any, Any, Any]:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    return train_datagen, valid_datagen, test_datagen


def create_generators(
    dataset_root: str | Path,
    image_size: tuple[int, int],
    batch_size: int,
    seed: int,
) -> tuple[Any, Any, Any]:
    """Create training, validation, and test generators from a directory dataset."""

    paths, _ = validate_dataset_layout(dataset_root)
    train_datagen, valid_datagen, test_datagen = _image_generators()

    train_generator = train_datagen.flow_from_directory(
        str(paths.train),
        target_size=image_size,
        batch_size=batch_size,
        seed=seed,
        class_mode="binary",
        shuffle=True,
    )
    valid_generator = valid_datagen.flow_from_directory(
        str(paths.valid),
        target_size=image_size,
        batch_size=batch_size,
        seed=seed,
        class_mode="binary",
        shuffle=False,
    )
    test_generator = test_datagen.flow_from_directory(
        str(paths.test),
        target_size=image_size,
        batch_size=batch_size,
        seed=seed,
        class_mode="binary",
        shuffle=False,
    )
    return train_generator, valid_generator, test_generator


def create_directory_generator(
    directory: str | Path,
    image_size: tuple[int, int],
    batch_size: int,
    seed: int,
    shuffle: bool = False,
) -> Any:
    """Create an evaluation generator from a single split directory."""

    split_dir = resolve_path(directory)
    if not split_dir.exists():
        raise AssetNotFoundError(
            f"Directory not found: {split_dir}",
            next_steps=["Verify the configured test_dir path in configs/inference.yaml."],
        )

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    return datagen.flow_from_directory(
        str(split_dir),
        target_size=image_size,
        batch_size=batch_size,
        seed=seed,
        class_mode="binary",
        shuffle=shuffle,
    )


def load_image_array(image_path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    """Load and resize an image for inference."""

    path = resolve_path(image_path)
    if not path.exists():
        raise AssetNotFoundError(
            f"Image not found: {path}",
            next_steps=["Provide a valid image path to scripts/run_predict.py --image <path>."],
        )

    with Image.open(path) as image:
        rgb_image = image.convert("RGB").resize(image_size)
        array = np.asarray(rgb_image, dtype=np.float32) / 255.0
    return array
