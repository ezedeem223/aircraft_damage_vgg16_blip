# Evidence Validation Tools

This directory contains tooling for validating the research evidence pack
in `docs/research_pack/`.

## Tools

### validate_research_pack.py

Validates the research evidence pack without requiring a trained checkpoint,
dataset, BLIP download, or internet access.

**Usage** (from repository root):

```bash
python tools/evidence/validate_research_pack.py
```

**Checks performed**:

- All required documentation files in `docs/research_pack/` are present
- `results/metrics.json` exists and is valid JSON
- `README.md` references `ACADEMIC_RESEARCH_BRIEF.md`
- Forbidden overclaim phrases are absent from `docs/research_pack/`
- Required limitation phrases are present somewhere in `docs/research_pack/`
- `MODEL_CARD.md` documents checkpoint and dataset limitations
- `INSPECTION_REPORT_PROTOCOL.md` states the human review requirement
- `VISION_LANGUAGE_LIMITATIONS.md` mentions hallucination risk

**Exit codes**:

- `0` — all checks passed
- `1` — one or more checks failed (details printed to stdout)
