"""BLIP-powered captioning and report generation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .utils import AssetNotFoundError, resolve_path


@dataclass
class ReportBundle:
    """Structured output for generated captions and inspection text."""

    caption: str
    summary: str
    report: str
    model_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BLIPReportGenerator:
    """Wrap BLIP image captioning with graceful fallback behavior."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        caption_prompt: str = "This is a picture of",
        summary_prompt: str = "This is a detailed photo showing",
        max_new_tokens: int = 40,
        device: str = "auto",
        use_template_fallback: bool = True,
    ) -> None:
        self.model_name = model_name
        self.caption_prompt = caption_prompt
        self.summary_prompt = summary_prompt
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.use_template_fallback = use_template_fallback
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._runtime_device: Any | None = None
        self._failure_reason: str | None = None

    def _select_device(self, torch_module: Any) -> Any:
        if self.device == "cpu":
            return torch_module.device("cpu")
        if self.device == "cuda":
            return torch_module.device("cuda")
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        return torch_module.device("cpu")

    def _load(self) -> bool:
        if self._processor is not None and self._model is not None:
            return True
        if self._failure_reason is not None:
            return False

        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self._torch = torch
            self._runtime_device = self._select_device(torch)
            self._processor = BlipProcessor.from_pretrained(self.model_name)
            self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)
            self._model.to(self._runtime_device)
            return True
        except Exception as exc:
            self._failure_reason = str(exc)
            return False

    def _generate_with_prompt(self, image_path: Path, prompt: str) -> str:
        if not self._load():
            raise RuntimeError(self._failure_reason or "BLIP model could not be loaded.")

        from PIL import Image

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            inputs = self._processor(images=rgb_image, text=prompt, return_tensors="pt")
            inputs = {
                key: value.to(self._runtime_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            return self._processor.decode(output[0], skip_special_tokens=True)

    def generate_caption(self, image_path: str | Path) -> str:
        """Generate a short caption for an aircraft image."""

        path = resolve_path(image_path)
        return self._generate_with_prompt(path, self.caption_prompt)

    def generate_summary(self, image_path: str | Path) -> str:
        """Generate a longer descriptive sentence for an aircraft image."""

        path = resolve_path(image_path)
        return self._generate_with_prompt(path, self.summary_prompt)

    def generate_report_bundle(
        self,
        image_path: str | Path,
        predicted_class: str,
        confidence: float,
        metadata: dict[str, Any] | None = None,
    ) -> ReportBundle:
        """Generate caption, summary, and a consolidated inspection report."""

        path = resolve_path(image_path)
        if not path.exists():
            raise AssetNotFoundError(
                f"Image not found: {path}",
                next_steps=["Provide a valid image path before requesting report generation."],
            )

        metadata = metadata or {}
        try:
            caption = self.generate_caption(path)
            summary = self.generate_summary(path)
            model_status = f"BLIP loaded from {self.model_name}"
        except Exception as exc:
            if not self.use_template_fallback:
                raise
            reason = self._failure_reason or str(exc)
            caption = "BLIP caption unavailable."
            summary = "BLIP summary unavailable."
            model_status = f"BLIP unavailable: {reason}"

        metadata_lines = "\n".join(
            f"- {key.replace('_', ' ').title()}: {value}" for key, value in metadata.items()
        )
        metadata_block = metadata_lines or "- No additional metadata supplied."

        report = (
            "Aircraft Damage Assessment\n"
            f"Image: {path.name}\n"
            f"Predicted damage class: {predicted_class}\n"
            f"Confidence: {confidence:.2%}\n"
            f"Caption: {caption}\n"
            f"Summary: {summary}\n"
            "Metadata:\n"
            f"{metadata_block}\n"
            f"Generation status: {model_status}\n"
            "Note: This output is decision support only and should be reviewed by a human inspector."
        )

        return ReportBundle(
            caption=caption,
            summary=summary,
            report=report,
            model_status=model_status,
        )


def generate_damage_report(
    image_path: str | Path,
    predicted_class: str,
    confidence: float,
    metadata: dict[str, Any] | None = None,
    report_generator: BLIPReportGenerator | None = None,
) -> str:
    """Generate a human-readable aircraft damage report string."""

    generator = report_generator or BLIPReportGenerator()
    bundle = generator.generate_report_bundle(
        image_path=image_path,
        predicted_class=predicted_class,
        confidence=confidence,
        metadata=metadata,
    )
    return bundle.report
