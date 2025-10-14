# Canonical Notation Quick Reference

## Quick Start

```python
from canonical_notation import CanonicalID, RubricKey, EvidenceEntry

# Create a canonical ID
canonical_id = CanonicalID(policy="P4", dimension="D2", question=3)
print(canonical_id)  # Output: P4-D2-Q3

# Parse from string
canonical_id = CanonicalID.from_string("P7-D3-Q5")

# Create evidence entry
evidence = EvidenceEntry.create(
    policy="P7", dimension="D3", question=5,
    score=0.82, confidence=0.82,
    stage="teoria_cambio", evidence_id_prefix="toc_"
)
```

## Format Specifications

### Question Unique ID (P#-D#-Q#)
- **Pattern:** `^P(10|[1-9])-D[1-6]-Q[1-9][0-9]*$`
- **Examples:** `P4-D2-Q3`, `P1-D1-Q1`, `P10-D6-Q30`
- **Components:**
  - P# = Policy (P1-P10)
  - D# = Dimension (D1-D6)
  - Q# = Question (positive integer)

### Rubric Key (D#-Q#)
- **Pattern:** `^D[1-6]-Q[1-9][0-9]*$`
- **Examples:** `D2-Q3`, `D1-Q1`, `D6-Q30`
- **Derivation:** Extract D# and Q# from question unique ID

## Policy Areas (P1-P10)

| ID | Area |
|----|------|
| P1 | Derechos de las mujeres e igualdad de género |
| P2 | Prevención de la violencia y protección frente al conflicto |
| P3 | Ambiente sano, cambio climático, prevención y atención a desastres |
| P4 | Derechos económicos, sociales y culturales |
| P5 | Derechos de las víctimas y construcción de paz |
| P6 | Derecho al buen futuro de la niñez, adolescencia, juventud |
| P7 | Tierras y territorios |
| P8 | Líderes y defensores de derechos humanos |
| P9 | Crisis de derechos de personas privadas de la libertad |
| P10 | Migración transfronteriza |

## Analytical Dimensions (D1-D6)

| ID | Dimension |
|----|-----------|
| D1 | Diagnóstico y Recursos |
| D2 | Diseño de Intervención |
| D3 | Productos y Outputs |
| D4 | Resultados y Outcomes |
| D5 | Impactos y Efectos de Largo Plazo |
| D6 | Teoría de Cambio y Coherencia Causal |

## Common Operations

### Validation
```python
from canonical_notation import CanonicalNotationValidator

validator = CanonicalNotationValidator()

# Validate formats
validator.validate_question_unique_id("P4-D2-Q3")  # True
validator.validate_rubric_key("D2-Q3")  # True

# Extract rubric key
rubric_key = validator.extract_rubric_key_from_question_id("P4-D2-Q3")
# Returns: "D2-Q3"
```

### Legacy Migration
```python
# Migrate D#-Q# format to P#-D#-Q#
canonical_id = validator.migrate_legacy_id("D2-Q3", inferred_policy="P4")
# Returns: "P4-D2-Q3"
```

### Evidence Entry Structure
```json
{
  "evidence_id": "toc_P7-D3-Q5",
  "question_unique_id": "P7-D3-Q5",
  "content": {
    "policy": "P7",
    "dimension": "D3",
    "question": 5,
    "score": 0.82,
    "rubric_key": "D3-Q5"
  },
  "confidence": 0.82,
  "stage": "teoria_cambio",
  "metadata": {}
}
```

## System Structure

- **Total Policies:** 10 (P1-P10)
- **Total Dimensions:** 6 (D1-D6)
- **Default Questions/Dimension:** 5
- **Total Default Questions:** 300 (10 × 6 × 5)

## Integration with Guia Cuestionario

```python
from canonical_integration import GuiaCuestionarioIntegration

integration = GuiaCuestionarioIntegration()

# Get dimension weight
weight = integration.get_dimension_weight("P4", "D2")  # 0.15

# Check if critical
is_critical = integration.is_critical_dimension("P4", "D3")  # True

# Create enriched evidence
evidence = integration.create_canonical_evidence(
    policy="P4", dimension="D2", question=3,
    score=0.85, confidence=0.90,
    stage="questionnaire_validation"
)
```

## Error Messages

All validation errors provide clear messages:

```python
# Invalid policy
CanonicalID(policy="P11", dimension="D1", question=1)
# ValueError: Invalid policy format: P11. Must match P(10|[1-9])

# Invalid dimension
CanonicalID(policy="P1", dimension="D7", question=1)
# ValueError: Invalid dimension format: D7. Must match D[1-6]

# Invalid question
CanonicalID(policy="P1", dimension="D1", question=0)
# ValueError: Invalid question number: 0. Must be positive integer
```

## Module Files

- **canonical_notation.py** - Core module with CanonicalID, RubricKey, EvidenceEntry
- **canonical_integration.py** - Integration with guia_cuestionario
- **test_canonical_notation.py** - 46 unit tests
- **ejemplo_canonical_notation.py** - Interactive examples
- **CANONICAL_NOTATION_DOCS.md** - Full documentation

## Testing

```bash
# Run all tests
python test_canonical_notation.py

# Run examples
python ejemplo_canonical_notation.py

# Run integration demo
python canonical_integration.py
```

## Common Patterns

### Creating Evidence for All Questions
```python
from canonical_notation import generate_default_questions

questions = generate_default_questions(max_questions_per_dimension=5)

for canonical_id in questions:
    evidence = EvidenceEntry.create(
        policy=canonical_id.policy,
        dimension=canonical_id.dimension,
        question=canonical_id.question,
        score=0.75,
        confidence=0.8,
        stage="evaluation"
    )
    # Process evidence...
```

### Filtering by Policy or Dimension
```python
questions = generate_default_questions()

# Filter by policy
p4_questions = [q for q in questions if q.policy == "P4"]

# Filter by dimension
d2_questions = [q for q in questions if q.dimension == "D2"]

# Filter by critical dimensions
integration = GuiaCuestionarioIntegration()
critical = [q for q in questions 
            if integration.is_critical_dimension(q.policy, q.dimension)]
```

## Best Practices

1. **Always validate external IDs** before parsing
2. **Use CanonicalID.from_string()** for string parsing
3. **Use EvidenceEntry.create()** for evidence creation
4. **Store using question_unique_id** as primary key
5. **Derive rubric_key** with to_rubric_key() method

## Version

**2.0.0** - Compatible with FARFAN-2.0 framework

For detailed documentation, see `CANONICAL_NOTATION_DOCS.md`
