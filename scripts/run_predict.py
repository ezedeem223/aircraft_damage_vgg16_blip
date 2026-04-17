"""Predict aircraft damage for a single image and optionally generate a report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to the image to analyze.")
    parser.add_argument(
        "--config",
        default="configs/inference.yaml",
        help="Path to inference config.",
    )
    parser.add_argument(
        "--report-config",
        default="configs/report_generation.yaml",
        help="Path to report generation config.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip BLIP-backed report generation and only run classification.",
    )
    return parser.parse_args()


def main() -> int:
    from aircraft_damage.config import load_config
    from aircraft_damage.predict import predict_image
    from aircraft_damage.report_generator import BLIPReportGenerator
    from aircraft_damage.utils import ProjectError, write_json, write_text

    args = parse_args()

    try:
        config = load_config(args.config)
        classifier_cfg = config["classifier"]
        paths_cfg = config["paths"]

        prediction = predict_image(
            image_path=args.image,
            checkpoint_path=paths_cfg["model_checkpoint"],
            class_names=classifier_cfg["class_names"],
            image_size=tuple(classifier_cfg["image_size"]),
            threshold=float(classifier_cfg["threshold"]),
        )
        print(f"Predicted class: {prediction.predicted_class}")
        print(f"Confidence: {prediction.confidence:.2%}")

        image_stem = Path(args.image).stem
        prediction_path = write_json(
            Path(paths_cfg["sample_predictions_dir"]) / f"{image_stem}_prediction.json",
            prediction.to_dict(),
        )
        print(f"Prediction artifact saved to: {prediction_path}")

        if not args.skip_report:
            report_config = load_config(args.report_config)
            report_cfg = report_config["report_generation"]
            generator = BLIPReportGenerator(
                model_name=report_cfg["model_name"],
                caption_prompt=report_cfg["caption_prompt"],
                summary_prompt=report_cfg["summary_prompt"],
                max_new_tokens=int(report_cfg["max_new_tokens"]),
                device=report_cfg["device"],
                use_template_fallback=bool(report_cfg["use_template_fallback"]),
            )
            bundle = generator.generate_report_bundle(
                image_path=args.image,
                predicted_class=prediction.predicted_class,
                confidence=prediction.confidence,
            )
            report_path = write_text(
                Path(paths_cfg["sample_reports_dir"]) / f"{image_stem}_report.txt",
                bundle.report,
            )
            print(f"Report artifact saved to: {report_path}")
            print("Report preview:")
            print(bundle.report)

        return 0
    except ProjectError as exc:
        print(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
