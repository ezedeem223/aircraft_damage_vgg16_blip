"""Plotting and figure export utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from .utils import ensure_directory

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_history_plots(history: dict[str, list[float]], output_dir: str | Path) -> None:
    """Save the notebook-style training plots."""

    destination = ensure_directory(output_dir)

    plt.figure()
    plt.title("Training Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(history.get("loss", []))
    plt.tight_layout()
    plt.savefig(destination / "training_loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.title("Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(history.get("val_loss", []))
    plt.tight_layout()
    plt.savefig(destination / "validation_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(history.get("accuracy", []), label="Training Accuracy")
    plt.plot(history.get("val_accuracy", []), label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination / "accuracy_curve.png", dpi=150)
    plt.close()


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> Path:
    """Render and save a confusion matrix figure."""

    destination = Path(output_path)
    ensure_directory(destination.parent)

    figure, axis = plt.subplots(figsize=(5, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(ax=axis, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def save_placeholder_figure(
    output_path: str | Path,
    title: str,
    body: str,
) -> Path:
    """Create a placeholder figure when an artifact has not been generated yet."""

    destination = Path(output_path)
    ensure_directory(destination.parent)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, body, ha="center", va="center", wrap=True)
    plt.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
