"""Launch the Gradio demo app."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the demo server.")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind the demo server.")
    parser.add_argument(
        "--config",
        default="configs/inference.yaml",
        help="Path to the inference config.",
    )
    parser.add_argument(
        "--report-config",
        default="configs/report_generation.yaml",
        help="Path to the report generation config.",
    )
    return parser.parse_args()


def main() -> int:
    from app.gradio_app import build_demo

    args = parse_args()
    demo = build_demo(
        inference_config_path=args.config,
        report_config_path=args.report_config,
    )
    demo.launch(server_name=args.host, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
