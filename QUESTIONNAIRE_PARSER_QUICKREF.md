# Canonical Questionnaire Integration - Quick Reference

## Overview
The canonical questionnaire parser (`questionnaire_parser.py`) is now the single source of truth for all 300 questions in FARFAN 2.0, ensuring deterministic and auditable access across all orchestration components.

## Key Files

### Core Implementation
- **`questionnaire_parser.py`**: Canonical parser for cuestionario_canonico
- **`cuestionario_canonico`**: Single source of truth (UTF-8 text file)
- **`test_questionnaire_integration.py`**: Integration test suite

### Integrated Orchestrators
- **`orchestrator.py`**: Main analytical orchestrator
- **`orchestration/unified_orchestrator.py`**: Unified 9-stage pipeline
- **`orchestration/pdm_orchestrator.py`**: PDM state machine orchestrator

## Questionnaire Structure

```
10 Policies (P1-P10)
├── P1: Derechos de las mujeres e igualdad de género
├── P2: Prevención de la violencia y protección frente al conflicto
├── P3: Ambiente sano, cambio climático y prevención de desastres
├── P4: Derechos económicos, sociales y culturales (DESC)
├── P5: Derechos de las víctimas y construcción de paz
├── P6: Derecho al buen futuro de la niñez, adolescencia y juventud
├── P7: Tierras y territorios
├── P8: Líderes y defensores de derechos humanos
├── P9: Crisis de derechos de personas privadas de la libertad
└── P10: Migración transfronteriza

Each Policy has:
  6 Dimensions (D1-D6)
    ├── D1: INSUMOS / Diagnóstico y Recursos
    ├── D2: ACTIVIDADES / Diseño de Intervención
    ├── D3: PRODUCTOS / Productos y Outputs
    ├── D4: RESULTADOS / Resultados y Outcomes
    ├── D5: IMPACTOS / Impactos y Largo Plazo
    └── D6: CAUSALIDAD / Teoría de Cambio y Coherencia

  Each Dimension has:
    5 Questions (Q1-Q5)
      Total: 10 × 6 × 5 = 300 questions
```

## Usage Examples

### Basic Parser Usage

```python
from questionnaire_parser import create_questionnaire_parser

# Initialize parser
parser = create_questionnaire_parser()

# Get all policies
policies = parser.get_all_policies()  # Returns list of 10 Policy objects

# Get specific policy
policy = parser.get_policy("P1")  # Returns Policy object

# Get specific dimension
dimension = parser.get_dimension("P1", "D1")  # Returns Dimension object

# Get specific question
question = parser.get_question("P1-D1-Q1")  # Returns Question object
print(question.text)  # Full question text

# Get all questions for a dimension
questions = parser.get_questions_by_dimension("P1", "D1")  # Returns list of Questions

# Get dimension names
dim_names = parser.get_dimension_names()  # Returns {"D1": "INSUMOS (D1)", ...}

# Get policy names
policy_names = parser.get_policy_names()  # Returns {"P1": "Derechos de las mujeres...", ...}

# Validate structure
validation = parser.validate_structure()
print(validation["valid"])  # True if structure is correct
```

### Orchestrator Integration

```python
from orchestrator import create_orchestrator

# Create orchestrator (parser auto-initialized)
orchestrator = create_orchestrator()

# Access dimension description
dim_desc = orchestrator.get_dimension_description("D1")
print(dim_desc)  # "INSUMOS (D1)"

# Access policy description
policy_desc = orchestrator.get_policy_description("P1")
print(policy_desc)  # "Derechos de las mujeres e igualdad de género"

# Access question
question = orchestrator.get_question("P1-D1-Q1")
print(question.text)

# Check canonical path in metadata
metadata = orchestrator._global_report["orchestration_metadata"]
print(metadata["canonical_questionnaire_path"])
```

### Question Object Structure

```python
question = parser.get_question("P1-D1-Q1")

# Available attributes:
question.policy_id        # "P1"
question.policy_name      # "Derechos de las mujeres e igualdad de género"
question.dimension_id     # "D1"
question.dimension_name   # "INSUMOS (D1)"
question.question_id      # "Q1"
question.full_id          # "P1-D1-Q1"
question.text            # Full question text
question.rubric          # Optional rubric (currently None)
question.weight          # Question weight (default 1.0)
question.guide           # Optional guide (currently None)
```

## SIN_CARRETA Compliance

### Single Source of Truth
- ✅ Only `cuestionario_canonico` is parsed
- ✅ No aliases or legacy versions exist
- ✅ Canonical path tracked in orchestration metadata
- ✅ Explicit validation prevents data drift

### Deterministic Access
- ✅ Same parsing produces identical results
- ✅ Indexed lookups provide O(1) access
- ✅ Reproducible across runs
- ✅ Test suite validates determinism

### Complete Auditability
- ✅ Canonical path in orchestration metadata
- ✅ Full question metadata preserved
- ✅ Validation results auditable
- ✅ 7-year retention compatible

### Contract Enforcement
- ✅ Structured dataclasses with type hints
- ✅ Explicit validation of questionnaire structure
- ✅ Factory function ensures correct initialization
- ✅ No silent failures

## Testing

Run integration tests:
```bash
cd /home/runner/work/FARFAN-2.0/FARFAN-2.0
python3 test_questionnaire_integration.py
```

Quick verification:
```bash
python3 -c "
from orchestrator import create_orchestrator
orch = create_orchestrator()
print(f'Questions: {len(orch.questionnaire_parser.get_all_questions())}')
print(f'Validation: {orch.questionnaire_parser.validate_structure()[\"valid\"]}')
"
```

## Migration Notes

**No Breaking Changes:**
- Existing orchestration logic preserved
- Backward compatible accessor methods added
- Optional integration with existing code

**Future Enhancements:**
- Connect question IDs to analytical module routing
- Use question metadata in report generation
- Leverage rubric levels in scoring framework
- Add question weights to scoring calculations

## Traceability

All questionnaire data originates from:
```
/home/runner/work/FARFAN-2.0/FARFAN-2.0/cuestionario_canonico
```

This path is:
- Hardcoded as default in `questionnaire_parser.py`
- Tracked in orchestrator metadata
- Validated at parser initialization
- Immutable (no runtime changes)

## Error Handling

```python
from questionnaire_parser import create_questionnaire_parser

try:
    parser = create_questionnaire_parser()
except FileNotFoundError as e:
    # Canonical file not found
    print(f"Error: {e}")

# Validate structure
validation = parser.validate_structure()
if not validation["valid"]:
    print("Validation errors:", validation["errors"])
```

## Documentation References

- **CODE_FIX_REPORT.md**: Detailed implementation notes and SIN_CARRETA compliance
- **README.md**: Module listing and capabilities overview
- **ORCHESTRATOR_README.md**: Orchestrator usage and configuration
- This file: Quick reference for developers

---

**Last Updated**: 2025-10-16  
**Status**: ✅ Production Ready  
**Version**: 2.0.0
