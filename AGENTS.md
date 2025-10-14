# FARFAN 2.0 - Agent Guide

## Commands

**Setup:**  
```bash
pip install -r requirements.txt
python -m spacy download es_core_news_lg
```

**Build:** N/A (Python project)

**Lint:** N/A (no linter configured)

**Tests:**  
```bash
python -m unittest test_canonical_notation.py
python -m unittest test_circuit_breaker.py
python test_risk_mitigation.py
```

**Dev/Run:**  
```bash
python orchestrator.py <pdf_file> --policy-code <code> --output-dir <dir> [--pdet]
python demo_orchestrator.py --simple
python dereck_beach <pdf_file> --output-dir <dir> --policy-code <code> [--pdet]
```

## Tech Stack

- **Language:** Python 3.11+
- **Core:** PyMuPDF, spaCy (es_core_news_lg), networkx, pandas
- **Processing:** scipy, numpy, fuzzywuzzy
- **Visualization:** pydot

## Architecture

FARFAN 2.0 is a framework for auditing Colombian Municipal Development Plans (PDM) using causal analysis and DNP standards validation. Main modules:
- `orchestrator.py`: 9-stage pipeline coordinating all modules to answer 300 evaluation questions
- `question_answering_engine.py`, `report_generator.py`, `module_choreographer.py`: Question answering and reporting
- `canonical_notation.py`: Canonical notation system (P#-D#-Q# format)
- `dnp_integration.py`, `competencias_municipales.py`, `mga_indicadores.py`, `pdet_lineamientos.py`: DNP compliance validation
- `circuit_breaker.py`: Circuit breaker pattern for distributed pipeline resilience
- `risk_mitigation_layer.py`: Pre-execution risk assessment with severity-based escalation and comprehensive logging
- `dereck_beach`: CDAF (Causal Deconstruction and Audit Framework) standalone processor

## Code Style

- Python 3 shebang (`#!/usr/bin/env python3`) for executables
- Docstrings in Spanish for domain logic, English for technical infrastructure
- Type hints and dataclasses preferred
- Logging via `logging` module (INFO level default)
- JSON for structured data output
