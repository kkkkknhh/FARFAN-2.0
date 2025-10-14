# AGENTS.md

## Commands

### Initial Setup
```bash
# No virtual environment needed (Python scripts only)
pip install pymupdf networkx pandas spacy pyyaml fuzzywuzzy python-Levenshtein pydot
python -m spacy download es_core_news_lg
```

### Build
No build step required (Python scripts).

### Lint
No linter configured.

### Tests
```bash
python test_canonical_notation.py
```

### Dev Server
No dev server (batch processing framework).

## Tech Stack
- **Language**: Python 3.11+
- **Core Libraries**: PyMuPDF (PDF), spaCy (NLP), NetworkX (graphs), pandas (data)
- **Domain**: Colombian territorial development plan auditing and DNP standards compliance

## Architecture
- **Orchestrator** (`orchestrator.py`): Main entry point, 300-question evaluation system
- **DNP Modules**: Municipal competencies, MGA indicators, PDET guidelines, validation
- **Canonical Notation** (`canonical_notation.py`): P#-D#-Q# ID system (10 policies × 6 dimensions)
- **Reports**: Micro (300 answers), Meso (clusters × dimensions), Macro (global alignment)

## Code Style
- Python 3.11+ with type hints and dataclasses
- Docstrings in English for modules, Spanish for domain-specific content
- Canonical IDs format: P1-P10 (policies), D1-D6 (dimensions), Q1+ (questions)
- Enums for categorical data, dataclasses for structured data
