"""Train the aircraft damage classifier."""

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
    parser.add_argument("--config", default="configs/train.yaml", help="Path to train config.")
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download the public dataset tarball before training.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download and re-extract the dataset even if the folder already exists.",
    )
    return parser.parse_args()


def main() -> int:
    from aircraft_damage.config import load_config
    from aircraft_damage.dataset import download_dataset
    from aircraft_damage.train import train_classifier
    from aircraft_damage.utils import ProjectError

    args = parse_args()

    try:
        config = load_config(args.config)
        if args.download_data:
            dataset_cfg = config["dataset"]
            paths_cfg = config["paths"]
            dataset_paths = download_dataset(
                dataset_root=paths_cfg["dataset_root"],
                archive_path=paths_cfg["archive_path"],
                dataset_url=dataset_cfg["source_url"],
                force=args.force_download,
            )
            print(f"Dataset ready under: {dataset_paths.root}")

        artifacts = train_classifier(config)
        print("Training completed successfully.")
        print(f"Checkpoint saved to: {artifacts['checkpoint_path']}")
        print(f"Training plots saved under: {config['paths']['results_dir']}")
        print("Next step: run python scripts/run_evaluate.py --config configs/inference.yaml")
        return 0
    except ProjectError as exc:
        print(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
