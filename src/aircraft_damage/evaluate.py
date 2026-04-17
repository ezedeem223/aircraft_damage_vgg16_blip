"""Evaluation helpers for directory-based test data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .predict import load_classifier
from .preprocessing import create_directory_generator
from .utils import ensure_directory, write_json, write_text
from .visualization import save_confusion_matrix


@dataclass
class EvaluationResult:
    """Evaluation outputs ready for persistence."""

    loss: float
    accuracy: float
    confusion_matrix: np.ndarray
    classification_report_text: str
    metrics: dict[str, Any]


def evaluate_classifier(
    checkpoint_path: str | Path,
    test_dir: str | Path,
    class_names: list[str],
    image_size: tuple[int, int],
    batch_size: int,
    seed: int,
    threshold: float = 0.5,
) -> EvaluationResult:
    """Evaluate the classifier on a directory-based test split."""

    model = load_classifier(checkpoint_path)
    generator = create_directory_generator(
        directory=test_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
    )

    loss, accuracy = model.evaluate(generator, verbose=0)
    generator.reset()
    probabilities = model.predict(generator, verbose=0).reshape(-1)
    predicted_labels = (probabilities >= threshold).astype(int)
    true_labels = generator.classes
    matrix = confusion_matrix(true_labels, predicted_labels)
    report_text = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        zero_division=0,
    )

    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "threshold": threshold,
        "num_samples": int(generator.samples),
        "class_names": class_names,
    }
    return EvaluationResult(
        loss=float(loss),
        accuracy=float(accuracy),
        confusion_matrix=matrix,
        classification_report_text=report_text,
        metrics=metrics,
    )


def save_evaluation_results(
    result: EvaluationResult,
    output_dir: str | Path,
    class_names: list[str],
) -> dict[str, str]:
    """Persist evaluation outputs to the results directory."""

    destination = ensure_directory(output_dir)
    metrics_path = write_json(destination / "metrics.json", result.metrics)
    report_path = write_text(
        destination / "classification_report.txt", result.classification_report_text
    )
    confusion_matrix_path = save_confusion_matrix(
        matrix=result.confusion_matrix,
        class_names=class_names,
        output_path=destination / "confusion_matrix.png",
    )
    return {
        "metrics": str(metrics_path),
        "classification_report": str(report_path),
        "confusion_matrix": str(confusion_matrix_path),
    }
