"""Lightweight Gradio demo for aircraft damage inference."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import gradio as gr

from aircraft_damage.config import load_config
from aircraft_damage.predict import load_classifier, predict_image
from aircraft_damage.report_generator import BLIPReportGenerator
from aircraft_damage.utils import ProjectError


@lru_cache(maxsize=2)
def _load_configs(
    inference_config_path: str = "configs/inference.yaml",
    report_config_path: str = "configs/report_generation.yaml",
) -> tuple[dict[str, Any], dict[str, Any]]:
    return load_config(inference_config_path), load_config(report_config_path)


@lru_cache(maxsize=1)
def _load_classifier_for_demo(checkpoint_path: str) -> tuple[Any | None, str | None]:
    try:
        return load_classifier(checkpoint_path), None
    except Exception as exc:
        return None, str(exc)


@lru_cache(maxsize=1)
def _load_report_generator(report_config_path: str) -> BLIPReportGenerator:
    report_config = load_config(report_config_path)
    report_cfg = report_config["report_generation"]
    return BLIPReportGenerator(
        model_name=report_cfg["model_name"],
        caption_prompt=report_cfg["caption_prompt"],
        summary_prompt=report_cfg["summary_prompt"],
        max_new_tokens=int(report_cfg["max_new_tokens"]),
        device=report_cfg["device"],
        use_template_fallback=bool(report_cfg["use_template_fallback"]),
    )


def _analyze_image(
    image_path: str | None,
    inference_config_path: str,
    report_config_path: str,
) -> tuple[str, str, str]:
    if not image_path:
        return "", "", "Upload an aircraft image to run classification and report generation."

    inference_config, _ = _load_configs(
        inference_config_path=inference_config_path,
        report_config_path=report_config_path,
    )
    classifier_cfg = inference_config["classifier"]
    paths_cfg = inference_config["paths"]

    model, model_error = _load_classifier_for_demo(paths_cfg["model_checkpoint"])
    if model is None:
        return (
            "Unavailable",
            "Unavailable",
            "Classifier checkpoint is not available.\n"
            f"{model_error}\n"
            "Train the model first or place a checkpoint under models/.",
        )

    try:
        prediction = predict_image(
            image_path=image_path,
            model=model,
            class_names=classifier_cfg["class_names"],
            image_size=tuple(classifier_cfg["image_size"]),
            threshold=float(classifier_cfg["threshold"]),
        )
        generator = _load_report_generator(report_config_path)
        bundle = generator.generate_report_bundle(
            image_path=image_path,
            predicted_class=prediction.predicted_class,
            confidence=prediction.confidence,
        )
        return prediction.predicted_class, f"{prediction.confidence:.2%}", bundle.report
    except ProjectError as exc:
        return "Error", "Error", str(exc)


def build_demo(
    inference_config_path: str = "configs/inference.yaml",
    report_config_path: str = "configs/report_generation.yaml",
) -> gr.Blocks:
    """Build the Gradio demo app."""

    with gr.Blocks(title="Aircraft Damage VGG16 + BLIP Demo") as demo:
        gr.Markdown(
            """
            # Aircraft Damage Classification and Report Generation
            Upload an aircraft image to run the VGG16 classifier and generate a BLIP-based inspection report.
            """
        )

        with gr.Row():
            image_input = gr.Image(
                label="Aircraft image",
                type="filepath",
                sources=["upload", "clipboard"],
            )

        analyze_button = gr.Button("Analyze image", variant="primary")

        with gr.Row():
            predicted_class = gr.Textbox(label="Predicted class")
            confidence = gr.Textbox(label="Confidence")

        report_output = gr.Textbox(label="Generated report", lines=12)

        analyze_button.click(
            fn=lambda image: _analyze_image(
                image,
                inference_config_path=inference_config_path,
                report_config_path=report_config_path,
            ),
            inputs=image_input,
            outputs=[predicted_class, confidence, report_output],
        )

    return demo
