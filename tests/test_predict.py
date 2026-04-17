from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aircraft_damage.predict import load_classifier, predict_image
from aircraft_damage.utils import AssetNotFoundError


class DummyModel:
    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        assert batch.shape == (1, 224, 224, 3)
        return np.array([[0.9]], dtype=float)


def test_predict_image_returns_structured_result(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_path)

    result = predict_image(
        image_path=image_path,
        model=DummyModel(),
        class_names=["crack", "dent"],
        image_size=(224, 224),
        threshold=0.5,
    )

    assert result.predicted_class == "dent"
    assert result.confidence == pytest.approx(0.9)
    assert result.raw_score == pytest.approx(0.9)


def test_load_classifier_raises_for_missing_checkpoint() -> None:
    with pytest.raises(AssetNotFoundError):
        load_classifier("models/does_not_exist.keras")
