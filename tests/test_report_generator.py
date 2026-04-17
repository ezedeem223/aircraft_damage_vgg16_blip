from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aircraft_damage.report_generator import BLIPReportGenerator, generate_damage_report
from aircraft_damage.utils import AssetNotFoundError


def test_generate_damage_report_falls_back_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_path)

    generator = BLIPReportGenerator(use_template_fallback=True)

    def _raise(_: str | Path) -> str:
        raise RuntimeError("mock BLIP failure")

    monkeypatch.setattr(generator, "generate_caption", _raise)
    monkeypatch.setattr(generator, "generate_summary", _raise)

    report = generate_damage_report(
        image_path=image_path,
        predicted_class="crack",
        confidence=0.73,
        report_generator=generator,
    )

    assert "Predicted damage class: crack" in report
    assert "BLIP unavailable" in report


def test_generate_damage_report_raises_for_missing_image() -> None:
    generator = BLIPReportGenerator(use_template_fallback=True)

    with pytest.raises(AssetNotFoundError):
        generate_damage_report(
            image_path="missing.jpg",
            predicted_class="dent",
            confidence=0.5,
            report_generator=generator,
        )
