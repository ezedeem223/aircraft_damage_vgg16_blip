"""Evaluate a trained classifier and save structured results."""

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
    parser.add_argument(
        "--config",
        default="configs/inference.yaml",
        help="Path to inference/evaluation config.",
    )
    return parser.parse_args()


def main() -> int:
    from aircraft_damage.config import load_config
    from aircraft_damage.evaluate import evaluate_classifier, save_evaluation_results
    from aircraft_damage.utils import ProjectError

    args = parse_args()

    try:
        config = load_config(args.config)
        classifier_cfg = config["classifier"]
        paths_cfg = config["paths"]

        result = evaluate_classifier(
            checkpoint_path=paths_cfg["model_checkpoint"],
            test_dir=paths_cfg["test_dir"],
            class_names=classifier_cfg["class_names"],
            image_size=tuple(classifier_cfg["image_size"]),
            batch_size=int(classifier_cfg["batch_size"]),
            seed=int(classifier_cfg["seed"]),
            threshold=float(classifier_cfg["threshold"]),
        )
        saved = save_evaluation_results(
            result=result,
            output_dir=paths_cfg["results_dir"],
            class_names=classifier_cfg["class_names"],
        )
        print("Evaluation completed successfully.")
        print(f"Metrics JSON: {saved['metrics']}")
        print(f"Classification report: {saved['classification_report']}")
        print(f"Confusion matrix: {saved['confusion_matrix']}")
        return 0
    except ProjectError as exc:
        print(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
