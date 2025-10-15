# Convergence Verification System - FARFAN 2.0

## Overview

The convergence verification system ensures that all 300 evaluation questions across `questions_config.json`, `guia_cuestionario.json`, and `cuestionario_canonico` are properly aligned and use correct canonical notation.

## Quick Start

### Running the Verification

```bash
python verify_convergence.py
```

This will:
1. Verify all 300 questions use canonical notation (P#-D#-Q# format)
2. Check scoring rubrics consistency
3. Validate dimension mappings and weights
4. Ensure no legacy file contributor mappings exist
5. Verify module references
6. Generate `convergence_report.json`

### Running Tests

```bash
python -m unittest test_convergence.py -v
```

## Verification Components

### 1. Canonical Notation Validation

**Format:** `P#-D#-Q#`
- **P (Policy):** P1-P10 (10 policy areas from Decálogo)
- **D (Dimension):** D1-D6 (6 causal dimensions)
- **Q (Question):** Q1-Q5 (5 questions per dimension)

**Examples:**
- ✓ Valid: `P1-D1-Q1`, `P10-D6-Q5`, `P4-D2-Q3`
- ✗ Invalid: `P11-D1-Q1`, `P1-D7-Q1`, `P1-D1-Q0`

**Verification:** Ensures all 300 questions (10 × 6 × 5) follow this format.

### 2. Scoring Consistency

**Rubric Levels:**
- `excelente`: min_score ≥ 0.85
- `bueno`: min_score ≥ 0.70
- `aceptable`: min_score ≥ 0.55
- `insuficiente`: min_score ≥ 0.0

**Verification:** Confirms all questions have complete scoring rubrics with these four levels.

### 3. Dimension Mapping Validation

**Requirements:**
- All 10 policies (P1-P10) must have dimension mappings
- Dimension weights must sum to 1.0 (±0.01 tolerance)
- Critical dimensions must be specified per policy

**Example:**
```json
"P1": {
  "D1_weight": 0.20,
  "D2_weight": 0.20,
  "D3_weight": 0.15,
  "D4_weight": 0.20,
  "D5_weight": 0.15,
  "D6_weight": 0.10,
  "critical_dimensions": ["D1", "D2", "D4"]
}
```

### 4. Legacy Mapping Detection

**Prohibited Patterns:**
- `file_contributors`
- `archivo_contribuyente`
- `source_files`
- `contributing_files`

**Verification:** Ensures these legacy patterns are not present in any configuration file.

### 5. Module Reference Validation

**Approved Modules:**
- `dnp_integration`
- `dereck_beach`
- `competencias_municipales`
- `mga_indicadores`
- `pdet_lineamientos`
- `initial_processor_causal_policy`

**Verification:** Confirms all module references in questions are from this approved list.

## Configuration Files

### questions_config.json

**Structure:**
- `metadata`: Version, total questions (300), policy areas (10), dimensions (6)
- `dimensiones`: D1-D6 definitions with weights per policy
- `puntos_decalogo`: P1-P10 definitions with indicators
- `preguntas_base`: 30 base questions (6 dimensions × 5 questions)

**Note:** File contains multiple JSON objects (one per policy area). The verifier handles this automatically.

### guia_cuestionario.json

**Structure:**
- `metadata`: Version and compatibility info
- `decalogo_dimension_mapping`: Weights and critical dimensions per policy
- `causal_verification_templates`: Validation patterns per dimension
- `question_verification_checklist`: D#-Q# specific verification rules
- `scoring_system`: Response scale and aggregation formulas
- `causal_glossary`: Technical term definitions

### cuestionario_canonico

**Format:** Markdown file with 300 questions organized by:
- Policy (P1-P10)
- Dimension (D1-D6)
- Question (Q1-Q5)

Each question uses the format: `**P#-D#-Q#:** [Question text]`

## Convergence Report

### Report Structure

```json
{
  "convergence_issues": [
    {
      "question_id": "P1-D1-Q1",
      "issue_type": "scoring_mismatch",
      "description": "...",
      "suggested_fix": "...",
      "severity": "HIGH"
    }
  ],
  "recommendations": [
    "Alinear scoring y mapping de todas las preguntas...",
    "Verificar que todas las funciones usadas..."
  ],
  "verification_summary": {
    "percent_questions_converged": 100.0,
    "issues_detected": 0,
    "critical_issues": 0,
    "high_priority_issues": 0,
    "medium_priority_issues": 0,
    "low_priority_issues": 0,
    "total_questions_expected": 300,
    "verification_timestamp": "2025-10-14T23:57:42Z"
  }
}
```

### Severity Levels

- **CRITICAL**: Blocks convergence, must be fixed immediately
  - Invalid canonical notation
  - Missing questions
  - Invalid policy/dimension IDs

- **HIGH**: Significant issues requiring attention
  - Invalid weight sums
  - Missing score ranges
  - Legacy mapping patterns

- **MEDIUM**: Minor issues that should be addressed
  - Incomplete scoring rubrics
  - Unknown module references

- **LOW**: Cosmetic issues or suggestions
  - Style inconsistencies
  - Optional improvements

## Success Criteria

✓ **Complete Convergence:**
- All 300 questions present
- All questions use correct canonical notation
- All scoring rubrics complete
- All dimension weights sum to 1.0
- No legacy mappings
- All module references valid

✓ **Report Output:**
- `percent_questions_converged`: 100.0%
- `critical_issues`: 0
- Recommendations list includes "✓ Sistema completamente convergente"

## Integration with FARFAN 2.0

### Canonical Notation Usage

The verification system uses `canonical_notation.py` module for validation:

```python
from canonical_notation import (
    CanonicalNotationValidator,
    CanonicalID,
    QUESTION_UNIQUE_ID_PATTERN,
    RUBRIC_KEY_PATTERN
)

validator = CanonicalNotationValidator()

# Validate question ID
is_valid = validator.validate_question_unique_id("P1-D1-Q1")

# Parse canonical ID
canonical_id = CanonicalID.from_string("P1-D1-Q1")
print(canonical_id.policy)     # "P1"
print(canonical_id.dimension)  # "D1"
print(canonical_id.question)   # 1
```

### Question Answering Engine Integration

The question answering engine should use canonical IDs when processing questions:

```python
from verify_convergence import ConvergenceVerifier
from canonical_notation import CanonicalID

# Verify convergence before processing
verifier = ConvergenceVerifier()
report = verifier.run_full_verification()

if report['verification_summary']['critical_issues'] > 0:
    raise ValueError("Critical convergence issues detected")

# Process questions using canonical IDs
question_id = "P1-D1-Q1"
canonical = CanonicalID.from_string(question_id)
# ... process question
```

## Troubleshooting

### Common Issues

**Issue:** `JSONDecodeError: Extra data`
- **Cause:** Multiple JSON objects in file
- **Solution:** Verifier automatically handles this for `questions_config.json` and `guia_cuestionario`

**Issue:** Invalid canonical notation detected
- **Cause:** Question ID doesn't match P#-D#-Q# format
- **Solution:** Check regex patterns and fix IDs in source files

**Issue:** Dimension weights don't sum to 1.0
- **Cause:** Rounding errors or incorrect weight values
- **Solution:** Adjust weights in `guia_cuestionario.json` to sum exactly to 1.0

**Issue:** Legacy mapping patterns detected
- **Cause:** Old file contributor references remain in configs
- **Solution:** Remove all `file_contributors`, `source_files`, etc. from JSON files

## Development

### Adding New Verification Checks

1. Add method to `ConvergenceVerifier` class:
```python
def verify_new_check(self) -> None:
    """Verify new aspect"""
    # Implementation
    if issue_found:
        self.issues.append(ConvergenceIssue(...))
```

2. Call from `run_full_verification()`:
```python
def run_full_verification(self) -> Dict[str, Any]:
    # ... existing checks
    self.verify_new_check()
    # ...
```

3. Add tests in `test_convergence.py`:
```python
def test_new_check(self):
    """Test new verification check"""
    self.verifier.verify_new_check()
    # Assertions
```

### Extending Severity Levels

Modify `ConvergenceIssue.severity` validation if needed:
```python
# Current: LOW, MEDIUM, HIGH, CRITICAL
# Add new level: BLOCKER
```

## References

- [Canonical Notation Documentation](CANONICAL_NOTATION_DOCS.md)
- [Question Answering Engine](question_answering_engine.py)
- [Guía Cuestionario](guia_cuestionario)
- [DNP Integration](DNP_INTEGRATION_DOCS.md)

## Version History

- **1.0.0** (2025-10-14): Initial convergence verification system
  - 300 questions validated
  - Complete canonical notation compliance
  - Comprehensive scoring verification
  - Legacy mapping detection
  - Module reference validation
