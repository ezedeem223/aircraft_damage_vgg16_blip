"""Prediction helpers for single-image inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .preprocessing import load_image_array
from .utils import AssetNotFoundError, ConfigurationError, resolve_path


@dataclass
class PredictionResult:
    """Structured classifier output for an input image."""

    image_path: str
    predicted_class: str
    confidence: float
    raw_score: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_classifier(checkpoint_path: str | Path) -> Any:
    """Load a saved Keras classifier checkpoint."""

    checkpoint = resolve_path(checkpoint_path)
    if not checkpoint.exists():
        raise AssetNotFoundError(
            f"Classifier checkpoint not found: {checkpoint}",
            next_steps=[
                "Train the model with python scripts/run_train.py --config configs/train.yaml.",
                "Or place an existing .keras/.h5 checkpoint under models/ and update the config.",
            ],
        )

    from tensorflow import keras

    return keras.models.load_model(checkpoint)


def predict_image(
    image_path: str | Path,
    class_names: list[str],
    image_size: tuple[int, int],
    threshold: float = 0.5,
    model: Any | None = None,
    checkpoint_path: str | Path | None = None,
) -> PredictionResult:
    """Predict aircraft damage class for a single image."""

    if len(class_names) != 2:
        raise ConfigurationError(
            "This project currently expects exactly two class names for binary classification."
        )

    classifier = model or load_classifier(checkpoint_path or "")
    image_array = load_image_array(image_path=image_path, image_size=image_size)
    batch = np.expand_dims(image_array, axis=0)

    prediction = classifier.predict(batch, verbose=0).reshape(-1)
    raw_score = float(prediction[0])
    predicted_index = int(raw_score >= threshold)
    confidence = raw_score if predicted_index == 1 else 1.0 - raw_score

    return PredictionResult(
        image_path=str(resolve_path(image_path)),
        predicted_class=class_names[predicted_index],
        confidence=confidence,
        raw_score=raw_score,
        threshold=threshold,
    )
