"""Tests verifying metric provenance from results/metrics.json.

These tests do not require a trained checkpoint, dataset, BLIP download,
or internet access.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_FILE = ROOT / "results" / "metrics.json"
PROVENANCE_MATRIX = ROOT / "docs" / "research_pack" / "METRIC_PROVENANCE_MATRIX.md"


def _load_metrics() -> dict:
    assert METRICS_FILE.exists(), f"Missing: {METRICS_FILE}"
    return json.loads(METRICS_FILE.read_text(encoding="utf-8"))


def test_metrics_json_exists() -> None:
    assert METRICS_FILE.exists(), "results/metrics.json is missing"


def test_metrics_json_is_valid_json() -> None:
    text = METRICS_FILE.read_text(encoding="utf-8")
    data = json.loads(text)
    assert isinstance(data, dict)


def test_metrics_json_has_required_keys() -> None:
    data = _load_metrics()
    for key in ("test_accuracy", "test_loss", "dataset_sizes"):
        assert key in data, f"metrics.json: missing key '{key}'"


def test_metrics_test_accuracy_value() -> None:
    data = _load_metrics()
    assert data["test_accuracy"] == 0.6875, (
        f"Preserved test accuracy mismatch: got {data['test_accuracy']}"
    )


def test_metrics_test_loss_value() -> None:
    data = _load_metrics()
    assert data["test_loss"] == 0.7326, (
        f"Preserved test loss mismatch: got {data['test_loss']}"
    )


def test_metrics_dataset_sizes() -> None:
    data = _load_metrics()
    sizes = data["dataset_sizes"]
    assert sizes["train"] == 300
    assert sizes["valid"] == 96
    assert sizes["test"] == 50


def test_metrics_epoch_histories_length() -> None:
    data = _load_metrics()
    assert len(data["accuracy_epoch_history"]) == 5
    assert len(data["loss_epoch_history"]) == 5
    assert len(data["validation_accuracy_epoch_history"]) == 5
    assert len(data["validation_loss_epoch_history"]) == 5


def test_metrics_final_training_accuracy() -> None:
    data = _load_metrics()
    final = data["accuracy_epoch_history"][-1]
    assert final == 0.88, f"Final training accuracy mismatch: got {final}"


def test_metrics_final_validation_accuracy() -> None:
    data = _load_metrics()
    final = data["validation_accuracy_epoch_history"][-1]
    assert abs(final - 0.7083) < 1e-4, f"Final validation accuracy mismatch: got {final}"


def test_metrics_provenance_field_present() -> None:
    data = _load_metrics()
    assert "provenance" in data, "metrics.json should contain a 'provenance' field"
    assert "checkpoint" in data["provenance"].lower(), (
        "provenance field should mention checkpoint status"
    )


def test_provenance_matrix_exists() -> None:
    assert PROVENANCE_MATRIX.exists(), "METRIC_PROVENANCE_MATRIX.md is missing"


def test_provenance_matrix_references_metrics_json() -> None:
    content = PROVENANCE_MATRIX.read_text(encoding="utf-8")
    assert "metrics.json" in content, (
        "METRIC_PROVENANCE_MATRIX.md should reference results/metrics.json"
    )


def test_provenance_matrix_contains_test_accuracy() -> None:
    content = PROVENANCE_MATRIX.read_text(encoding="utf-8")
    assert "0.6875" in content, (
        "METRIC_PROVENANCE_MATRIX.md should contain the preserved test accuracy value"
    )


def test_provenance_matrix_contains_test_loss() -> None:
    content = PROVENANCE_MATRIX.read_text(encoding="utf-8")
    assert "0.7326" in content, (
        "METRIC_PROVENANCE_MATRIX.md should contain the preserved test loss value"
    )


def test_provenance_matrix_mentions_placeholder() -> None:
    content = PROVENANCE_MATRIX.read_text(encoding="utf-8").lower()
    assert "placeholder" in content, (
        "METRIC_PROVENANCE_MATRIX.md should flag placeholder artefacts"
    )


def test_provenance_matrix_forbidden_claims_absent() -> None:
    """Check that the provenance matrix does not make forbidden positive claims.

    Note: terms like 'newly reproduced' and 'production' legitimately appear
    in the matrix's red-lines / forbidden-wording columns to document what NOT
    to say.  Only check for positive overclaim phrases that should never appear
    in any context.
    """
    content = PROVENANCE_MATRIX.read_text(encoding="utf-8").lower()
    # These phrases should be entirely absent — they cannot appear even in a
    # 'do not use' column without risk of being misread as positive claims.
    strictly_absent = [
        "state-" + "of-" + "the-" + "art",
        "certified aircraft inspection",
        "maintenance approval",
        "airworthiness decision",
    ]
    found = [p for p in strictly_absent if p in content]
    assert not found, (
        f"Forbidden overclaim phrases found in METRIC_PROVENANCE_MATRIX.md: {found}"
    )
