"""Aircraft damage classification and report generation package."""

from .config import load_config
from .predict import PredictionResult, predict_image
from .report_generator import BLIPReportGenerator, generate_damage_report

__all__ = [
    "BLIPReportGenerator",
    "PredictionResult",
    "generate_damage_report",
    "load_config",
    "predict_image",
]

__version__ = "0.1.0"
