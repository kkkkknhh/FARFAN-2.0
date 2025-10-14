# AGENTS.md - FARFAN-2.0 Development Guide

## Setup
```bash
pip install pymupdf networkx pandas spacy pyyaml fuzzywuzzy python-Levenshtein pydot
python -m spacy download es_core_news_lg
```

## Commands
- **Build**: N/A (Python project, no compilation)
- **Lint**: N/A (no linter configured)
- **Test**: `python test_canonical_notation.py` or `python -m unittest test_canonical_notation.py`
- **Demo**: `python demo_orchestrator.py --simple` or `python orchestrator.py <pdf_file> --policy-code <code> --output-dir <dir>`

## Tech Stack
- **Language**: Python 3.11+
- **Core**: PyMuPDF (PDF parsing), spaCy (NLP), NetworkX (causal graphs), Pandas (data)
- **Domain**: Colombian Municipal Development Plan (PDM) evaluation framework with DNP compliance validation

## Architecture
- **Orchestrator**: `orchestrator.py` - main 9-stage canonical evaluation pipeline
- **DNP Modules**: `dnp_integration.py`, `competencias_municipales.py`, `mga_indicadores.py`, `pdet_lineamientos.py`
- **Canonical Notation**: `canonical_notation.py` - P#-D#-Q# system (10 policies × 6 dimensions × N questions = 300 questions)
- **Reporting**: `report_generator.py` - micro (300 answers), meso (clusters), macro (global alignment)

## Code Style
- Python 3 with type hints, dataclasses, and enums
- Docstrings: triple-quoted with module/function purpose
- Imports: standard lib, third-party, local (grouped and sorted)
- Spanish for domain terms (e.g., "Teoría de Cambio"), English for code
