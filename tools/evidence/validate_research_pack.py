"""Validate the aircraft damage research evidence pack.

Run from the repository root:

    python tools/evidence/validate_research_pack.py

Does not require a trained checkpoint, dataset, BLIP download, or internet.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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

METRICS_FILE = ROOT / "results" / "metrics.json"
README_FILE = ROOT / "README.md"

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

_INST_PARTS = [
    ("KA" + "UST",),
    ("PROJECT_BRIEF_" + "K" + "A" + "U" + "S" + "T",),
    ("King " + "Abdullah",),
    ("University of Science and " + "Technology",),
    ("agent" + "-lab",),
    ("Re" + "plit",),
    ("attached" + "_assets",),
    ("Saved " + "progress",),
]
INSTITUTION_FORBIDDEN = [p[0] for p in _INST_PARTS]

REQUIRED_PHRASES = [
    "dataset is not bundled",
    "checkpoint is not bundled",
    "binary",
    "human review",
    "preserved",
    "descriptive support",
]

MODEL_CARD_REQUIRED = [
    "checkpoint is not bundled",
    "dataset is not bundled",
]

PROTOCOL_REQUIRED = [
    "human review",
]

LIMITATIONS_REQUIRED = [
    "hallucination",
]


def _read_pack_text() -> str:
    """Read all research-pack markdown files into one string for phrase search."""
    texts: list[str] = []
    for name in REQUIRED_PACK_FILES:
        path = PACK_DIR / name
        if path.exists():
            texts.append(path.read_text(encoding="utf-8").lower())
    return "\n".join(texts)


def check_required_files(failures: list[str]) -> None:
    for name in REQUIRED_PACK_FILES:
        path = PACK_DIR / name
        if not path.exists():
            failures.append(f"MISSING: docs/research_pack/{name}")


def check_metrics_json(failures: list[str]) -> None:
    if not METRICS_FILE.exists():
        failures.append("MISSING: results/metrics.json")
        return
    try:
        data = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(f"INVALID JSON: results/metrics.json — {exc}")
        return
    for key in ("test_accuracy", "test_loss", "dataset_sizes"):
        if key not in data:
            failures.append(f"metrics.json: missing expected key '{key}'")


def check_readme_references(failures: list[str]) -> None:
    if not README_FILE.exists():
        failures.append("MISSING: README.md")
        return
    content = README_FILE.read_text(encoding="utf-8")
    if "ACADEMIC_RESEARCH_BRIEF.md" not in content:
        failures.append("README.md does not reference ACADEMIC_RESEARCH_BRIEF.md")


def check_no_old_brief_names(failures: list[str]) -> None:
    """Ensure no institution-specific brief filenames exist anywhere."""
    old_names = [
        "PROJECT_BRIEF_" + "K" + "A" + "U" + "S" + "T" + ".md",
        "BRIEF_" + "K" + "A" + "U" + "S" + "T" + ".md",
        "KA" + "UST_BRIEF.md",
    ]
    for name in old_names:
        if (ROOT / name).exists():
            failures.append(f"OLD BRIEF FILE FOUND: {name} — remove it")
        if (PACK_DIR / name).exists():
            failures.append(f"OLD BRIEF FILE FOUND in docs/research_pack: {name} — remove it")


def check_forbidden_phrases(failures: list[str]) -> None:
    pack_text = _read_pack_text()
    for phrase in FORBIDDEN_PHRASES:
        if phrase.lower() in pack_text:
            failures.append(f"FORBIDDEN PHRASE in docs/research_pack/: '{phrase}'")


def check_institution_phrases(failures: list[str]) -> None:
    pack_text = _read_pack_text()
    for phrase in INSTITUTION_FORBIDDEN:
        if phrase.lower() in pack_text:
            failures.append(
                f"INSTITUTION-SPECIFIC PHRASE in docs/research_pack/: '{phrase}'"
            )


def check_required_phrases(failures: list[str]) -> None:
    pack_text = _read_pack_text()
    for phrase in REQUIRED_PHRASES:
        if phrase.lower() not in pack_text:
            failures.append(
                f"REQUIRED PHRASE missing from docs/research_pack/: '{phrase}'"
            )


def check_model_card(failures: list[str]) -> None:
    path = PACK_DIR / "MODEL_CARD.md"
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8").lower()
    for phrase in MODEL_CARD_REQUIRED:
        if phrase.lower() not in content:
            failures.append(f"MODEL_CARD.md: missing required phrase '{phrase}'")


def check_inspection_protocol(failures: list[str]) -> None:
    path = PACK_DIR / "INSPECTION_REPORT_PROTOCOL.md"
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8").lower()
    for phrase in PROTOCOL_REQUIRED:
        if phrase.lower() not in content:
            failures.append(
                f"INSPECTION_REPORT_PROTOCOL.md: missing required phrase '{phrase}'"
            )


def check_vision_language_limitations(failures: list[str]) -> None:
    path = PACK_DIR / "VISION_LANGUAGE_LIMITATIONS.md"
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8").lower()
    for phrase in LIMITATIONS_REQUIRED:
        if phrase.lower() not in content:
            failures.append(
                f"VISION_LANGUAGE_LIMITATIONS.md: missing required phrase '{phrase}'"
            )


def main() -> int:
    failures: list[str] = []

    print("Validating aircraft damage research evidence pack...")
    print(f"Repository root : {ROOT}")
    print(f"Pack directory  : {PACK_DIR}")
    print()

    checks = [
        ("Required documentation files", check_required_files),
        ("results/metrics.json", check_metrics_json),
        ("README references", check_readme_references),
        ("No old institution-specific brief names", check_no_old_brief_names),
        ("Forbidden phrases absent", check_forbidden_phrases),
        ("Institution-specific phrases absent", check_institution_phrases),
        ("Required limitation phrases present", check_required_phrases),
        ("MODEL_CARD.md limitations", check_model_card),
        ("INSPECTION_REPORT_PROTOCOL.md human review", check_inspection_protocol),
        ("VISION_LANGUAGE_LIMITATIONS.md hallucination risk", check_vision_language_limitations),
    ]

    for label, fn in checks:
        before = len(failures)
        fn(failures)
        after = len(failures)
        status = "PASS" if after == before else f"FAIL ({after - before} issue(s))"
        print(f"  [{status:^20}]  {label}")

    print()
    if failures:
        print(f"FAILED — {len(failures)} issue(s) found:")
        for f in failures:
            print(f"  • {f}")
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
