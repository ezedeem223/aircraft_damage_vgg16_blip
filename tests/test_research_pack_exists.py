"""Tests verifying that research evidence pack files exist and are well-formed.

These tests do not require a trained checkpoint, dataset, BLIP download,
or internet access.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "docs" / "research_pack"

REQUIRED_PACK_FILES = [
    "README.md",
    "ACADEMIC_RESEARCH_BRIEF.md",
    "MODEL_CARD.md",
    "METRIC_PROVENANCE_MATRIX.md",
    "INSPECTION_REPORT_PROTOCOL.md",
    "VISION_LANGUAGE_LIMITATIONS.md",
    "FAILURE_MODE_MATRIX.md",
    "DATASET_AND_TASK_CARD.md",
    "REPRODUCIBILITY_CHECKLIST.md",
]

FORBIDDEN_PHRASES = [
    "state-" + "of-" + "the-" + "art",
    "production " + "deployment",
    "certified aircraft " + "inspection system",
    "maintenance " + "approval system",
    "airworthiness " + "decision system",
    "validated for real-world " + "maintenance decisions",
    "operational aviation " + "safety system",
    "automated repair " + "recommendation",
]

_INST = [
    "KA" + "UST",
    "King " + "Abdullah",
    "University of Science and " + "Technology",
    "agent" + "-lab",
]

REQUIRED_PHRASES = [
    "dataset is not bundled",
    "checkpoint is not bundled",
    "binary",
    "human review",
    "preserved",
    "descriptive support",
]


def _pack_text() -> str:
    parts: list[str] = []
    for name in REQUIRED_PACK_FILES:
        p = PACK_DIR / name
        if p.exists():
            parts.append(p.read_text(encoding="utf-8").lower())
    return "\n".join(parts)


def test_pack_directory_exists() -> None:
    assert PACK_DIR.is_dir(), f"Missing directory: {PACK_DIR}"


def test_all_required_pack_files_exist() -> None:
    missing = [n for n in REQUIRED_PACK_FILES if not (PACK_DIR / n).exists()]
    assert not missing, f"Missing research pack files: {missing}"


def test_academic_research_brief_exists() -> None:
    assert (PACK_DIR / "ACADEMIC_RESEARCH_BRIEF.md").exists()


def test_no_old_institution_brief_filename() -> None:
    old_names = [
        "PROJECT_BRIEF_" + "K" + "A" + "U" + "S" + "T" + ".md",
        "BRIEF_" + "K" + "A" + "U" + "S" + "T" + ".md",
        "KA" + "UST_BRIEF.md",
    ]
    for name in old_names:
        assert not (ROOT / name).exists(), f"Old brief file found: {name}"
        assert not (PACK_DIR / name).exists(), f"Old brief file in pack: {name}"


def test_model_card_documents_checkpoint_limitation() -> None:
    path = PACK_DIR / "MODEL_CARD.md"
    assert path.exists()
    content = path.read_text(encoding="utf-8").lower()
    assert "checkpoint is not bundled" in content or "no trained checkpoint" in content, (
        "MODEL_CARD.md should document that the checkpoint is not bundled"
    )


def test_model_card_documents_dataset_limitation() -> None:
    path = PACK_DIR / "MODEL_CARD.md"
    assert path.exists()
    content = path.read_text(encoding="utf-8").lower()
    assert "dataset is not bundled" in content or "not bundled" in content, (
        "MODEL_CARD.md should document that the dataset is not bundled"
    )


def test_inspection_protocol_states_human_review() -> None:
    path = PACK_DIR / "INSPECTION_REPORT_PROTOCOL.md"
    assert path.exists()
    content = path.read_text(encoding="utf-8").lower()
    assert "human review" in content, (
        "INSPECTION_REPORT_PROTOCOL.md must state human review requirement"
    )


def test_vision_language_limitations_mentions_hallucination() -> None:
    path = PACK_DIR / "VISION_LANGUAGE_LIMITATIONS.md"
    assert path.exists()
    content = path.read_text(encoding="utf-8").lower()
    assert "hallucination" in content, (
        "VISION_LANGUAGE_LIMITATIONS.md must mention hallucination risk"
    )


def test_forbidden_phrases_absent() -> None:
    text = _pack_text()
    found = [p for p in FORBIDDEN_PHRASES if p.lower() in text]
    assert not found, f"Forbidden phrases found in docs/research_pack/: {found}"


def test_institution_phrases_absent() -> None:
    text = _pack_text()
    found = [p for p in _INST if p.lower() in text]
    assert not found, (
        f"Institution-specific phrases found in docs/research_pack/: {found}"
    )


def test_required_phrases_present() -> None:
    text = _pack_text()
    missing = [p for p in REQUIRED_PHRASES if p.lower() not in text]
    assert not missing, f"Required phrases missing from docs/research_pack/: {missing}"


def test_validation_tool_passes() -> None:
    tool = ROOT / "tools" / "evidence" / "validate_research_pack.py"
    assert tool.exists(), "Validation tool not found"
    result = subprocess.run(
        [sys.executable, str(tool)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, (
        f"Validation tool failed:\n{result.stdout}\n{result.stderr}"
    )


def test_tools_evidence_readme_exists() -> None:
    assert (ROOT / "tools" / "evidence" / "README.md").exists()
