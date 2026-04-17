"""VGG16-based classifier training utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .preprocessing import create_generators
from .utils import ensure_directory, set_random_seed
from .visualization import save_history_plots


def build_vgg16_classifier(
    image_size: tuple[int, int] = (224, 224),
    dense_units: int = 512,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-4,
) -> Any:
    """Recreate the notebook's VGG16 transfer-learning classifier."""

    from tensorflow import keras
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.optimizers import Adam

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size[0], image_size[1], 3),
    )
    output = base_model.layers[-1].output
    output = keras.layers.Flatten()(output)
    feature_extractor = keras.Model(base_model.input, output, name="vgg16_feature_extractor")

    for layer in feature_extractor.layers:
        layer.trainable = False

    model = keras.Sequential(
        [
            feature_extractor,
            keras.layers.Dense(dense_units, activation="relu"),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(dense_units, activation="relu"),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="aircraft_damage_vgg16_classifier",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_classifier(config: dict[str, Any]) -> dict[str, Any]:
    """Train the classifier from a config dictionary."""

    classifier_cfg = config["classifier"]
    paths_cfg = config["paths"]

    image_size = tuple(classifier_cfg["image_size"])
    batch_size = int(classifier_cfg["batch_size"])
    seed = int(classifier_cfg["seed"])
    epochs = int(classifier_cfg["epochs"])
    set_random_seed(seed)

    train_generator, valid_generator, test_generator = create_generators(
        dataset_root=paths_cfg["dataset_root"],
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )

    model = build_vgg16_classifier(
        image_size=image_size,
        dense_units=int(classifier_cfg["dense_units"]),
        dropout_rate=float(classifier_cfg["dropout_rate"]),
        learning_rate=float(classifier_cfg["learning_rate"]),
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=valid_generator,
    )

    checkpoint_path = Path(paths_cfg["model_checkpoint"])
    ensure_directory(checkpoint_path.parent)
    model.save(checkpoint_path)

    save_history_plots(history.history, paths_cfg["results_dir"])

    return {
        "model": model,
        "history": history.history,
        "test_generator": test_generator,
        "class_indices": train_generator.class_indices,
        "checkpoint_path": str(checkpoint_path),
    }
