# Canonical Notation System Documentation

## Overview

The Canonical Notation System provides a standardized format for all question identifiers and rubric keys across the FARFAN-2.0 PDM (Plan de Desarrollo Municipal) evaluation framework. This ensures consistency, traceability, and deterministic evaluation.

## System Components

### Policy Areas (P1-P10)

The system defines 10 thematic policy areas representing the core pillars of Colombian Municipal Development Plans:

| ID  | Title                                                              |
|-----|--------------------------------------------------------------------|
| P1  | Derechos de las mujeres e igualdad de género                      |
| P2  | Prevención de la violencia y protección frente al conflicto       |
| P3  | Ambiente sano, cambio climático, prevención y atención a desastres |
| P4  | Derechos económicos, sociales y culturales                         |
| P5  | Derechos de las víctimas y construcción de paz                     |
| P6  | Derecho al buen futuro de la niñez, adolescencia, juventud         |
| P7  | Tierras y territorios                                              |
| P8  | Líderes y defensores de derechos humanos                           |
| P9  | Crisis de derechos de personas privadas de la libertad             |
| P10 | Migración transfronteriza                                          |

### Analytical Dimensions (D1-D6)

Six analytical dimensions are used to evaluate each policy area:

| ID | Name                                   | Focus Areas                                                                       |
|----|----------------------------------------|-----------------------------------------------------------------------------------|
| D1 | Diagnóstico y Recursos                 | Baseline, problem magnitude, resources, institutional capacity                    |
| D2 | Diseño de Intervención                 | Activities, target population, intervention design                                |
| D3 | Productos y Outputs                    | Technical standards, proportionality, quantification, accountability              |
| D4 | Resultados y Outcomes                  | Result indicators, differentiation, magnitude of change, attribution              |
| D5 | Impactos y Efectos de Largo Plazo      | Impact indicators, temporal horizons, systemic effects, sustainability            |
| D6 | Teoría de Cambio y Coherencia Causal   | Theory of change, assumptions, logical framework, monitoring                      |

## Canonical Formats

### Question Unique ID (P#-D#-Q#)

**Format:** `P#-D#-Q#`

**Components:**
- `P#`: Policy area (P1 through P10)
- `D#`: Analytical dimension (D1 through D6)
- `Q#`: Question number (Q1 and up)

**Regex Pattern:** `^P(10|[1-9])-D[1-6]-Q[1-9][0-9]*$`

**Examples:**
- `P4-D2-Q3` - Policy 4, Dimension 2, Question 3
- `P1-D1-Q1` - Policy 1, Dimension 1, Question 1
- `P10-D6-Q30` - Policy 10, Dimension 6, Question 30

### Rubric Key (D#-Q#)

**Format:** `D#-Q#`

**Components:**
- `D#`: Analytical dimension (D1 through D6)
- `Q#`: Question number (Q1 and up)

**Regex Pattern:** `^D[1-6]-Q[1-9][0-9]*$`

**Examples:**
- `D2-Q3` - Dimension 2, Question 3
- `D1-Q1` - Dimension 1, Question 1
- `D6-Q30` - Dimension 6, Question 30

**Derivation from Question Unique ID:**
```python
# Example: "P4-D2-Q3" → "D2-Q3"
canonical_id = CanonicalID.from_string("P4-D2-Q3")
rubric_key = canonical_id.to_rubric_key()
print(str(rubric_key))  # Output: "D2-Q3"
```

## System Structure

### Default Configuration

By default, the system supports:
- **10 policies** × **6 dimensions** × **5 questions** = **300 total questions**

This can be customized using the `generate_default_questions()` function:

```python
from canonical_notation import generate_default_questions

# Generate 300 default questions (5 per dimension)
questions = generate_default_questions(max_questions_per_dimension=5)

# Generate custom structure (e.g., 10 questions per dimension = 600 total)
questions = generate_default_questions(max_questions_per_dimension=10)
```

## Usage Examples

### Basic Usage

```python
from canonical_notation import CanonicalID, RubricKey, EvidenceEntry

# 1. Create a canonical ID
canonical_id = CanonicalID(policy="P4", dimension="D2", question=3)
print(canonical_id)  # Output: P4-D2-Q3

# 2. Parse from string
canonical_id = CanonicalID.from_string("P7-D3-Q5")
print(canonical_id.get_policy_title())  # Output: Tierras y territorios
print(canonical_id.get_dimension_name())  # Output: Productos y Outputs

# 3. Derive rubric key
rubric_key = canonical_id.to_rubric_key()
print(rubric_key)  # Output: D3-Q5

# 4. Create rubric key directly
rubric_key = RubricKey(dimension="D2", question=3)
print(rubric_key)  # Output: D2-Q3
```

### Creating Evidence Entries

Evidence entries follow a standardized structure with canonical notation:

```python
from canonical_notation import EvidenceEntry

# Create evidence entry
evidence = EvidenceEntry.create(
    policy="P7",
    dimension="D3",
    question=5,
    score=0.82,
    confidence=0.82,
    stage="teoria_cambio",
    evidence_id_prefix="toc_"
)

# Output as JSON
print(evidence.to_json())
```

**Output:**
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

### Validation

```python
from canonical_notation import CanonicalNotationValidator

validator = CanonicalNotationValidator()

# Validate question unique ID
is_valid = validator.validate_question_unique_id("P4-D2-Q3")
print(is_valid)  # Output: True

# Validate rubric key
is_valid = validator.validate_rubric_key("D2-Q3")
print(is_valid)  # Output: True

# Extract rubric key from question ID
rubric_key = validator.extract_rubric_key_from_question_id("P4-D2-Q3")
print(rubric_key)  # Output: D2-Q3
```

### Legacy Format Migration

The system supports migration from legacy formats:

```python
from canonical_notation import CanonicalNotationValidator

validator = CanonicalNotationValidator()

# Migrate legacy D#-Q# format (requires policy inference)
legacy_id = "D2-Q3"
canonical_id = validator.migrate_legacy_id(
    legacy_id, 
    inferred_policy="P4"
)
print(canonical_id)  # Output: P4-D2-Q3

# Already canonical format - no change
canonical_id = validator.migrate_legacy_id("P4-D2-Q3")
print(canonical_id)  # Output: P4-D2-Q3
```

### Getting System Information

```python
from canonical_notation import get_system_structure_summary, PolicyArea, AnalyticalDimension

# Get full system structure
structure = get_system_structure_summary()
print(f"Total questions: {structure['default_total_questions']}")
print(f"Policies: {structure['total_policies']}")
print(f"Dimensions: {structure['total_dimensions']}")

# Get policy information
title = PolicyArea.get_title("P1")
print(title)  # Output: Derechos de las mujeres e igualdad de género

# Get dimension information
name = AnalyticalDimension.get_name("D2")
print(name)  # Output: Diseño de Intervención

focus = AnalyticalDimension.get_focus("D2")
print(focus)  # Output: Activities, target population, intervention design
```

## Evidence Entry Requirements

All evidence entries **must** follow this canonical structure:

### Required Fields

1. **evidence_id**: Unique identifier for the evidence entry
   - Format: `{prefix}{question_unique_id}` (e.g., `toc_P7-D3-Q5`)

2. **question_unique_id**: Must match pattern `P#-D#-Q#`
   - Example: `P7-D3-Q5`

3. **content**: Dictionary with required fields
   - `policy`: Must match pattern `P#`
   - `dimension`: Must match pattern `D#`
   - `question`: Positive integer
   - `score`: Float between 0.0 and 1.0
   - `rubric_key`: Must match pattern `D#-Q#`

4. **confidence**: Float between 0.0 and 1.0

5. **stage**: String identifying the processing stage

6. **metadata**: Optional dictionary with additional information

### Validation Rules

- The `question_unique_id` must be consistent with content fields
- The `rubric_key` in content must match the derived rubric key
- Score and confidence must be in range [0.0, 1.0]
- All components must follow canonical format patterns

## Error Handling

The system provides comprehensive error messages:

```python
from canonical_notation import CanonicalID

# Invalid policy
try:
    CanonicalID(policy="P11", dimension="D1", question=1)
except ValueError as e:
    print(e)  # Output: Invalid policy format: P11. Must match P(10|[1-9])

# Invalid dimension
try:
    CanonicalID(policy="P1", dimension="D7", question=1)
except ValueError as e:
    print(e)  # Output: Invalid dimension format: D7. Must match D[1-6]

# Invalid question
try:
    CanonicalID(policy="P1", dimension="D1", question=0)
except ValueError as e:
    print(e)  # Output: Invalid question number: 0. Must be positive integer
```

## Integration with FARFAN-2.0

The canonical notation system is designed to integrate seamlessly with the FARFAN-2.0 framework:

### In DNP Integration

```python
from canonical_notation import CanonicalID, EvidenceEntry

# Create evidence from DNP validation results
for policy_area in ["P1", "P2", "P3"]:
    for dimension in ["D1", "D2", "D3"]:
        for question_num in range(1, 6):
            canonical_id = CanonicalID(
                policy=policy_area,
                dimension=dimension,
                question=question_num
            )
            # Use canonical_id for consistent tracking
            evidence = EvidenceEntry.create(
                policy=policy_area,
                dimension=dimension,
                question=question_num,
                score=0.75,
                confidence=0.8,
                stage="dnp_validation"
            )
```

### In Causal Analysis

```python
# Track evidence with canonical IDs
evidence_tracker = {}

for evidence_entry in evidence_list:
    question_id = evidence_entry.question_unique_id
    evidence_tracker[question_id] = evidence_entry
    
# Later, query by canonical ID
evidence = evidence_tracker.get("P4-D2-Q3")
```

## Best Practices

1. **Always use canonical format**: Create IDs using `CanonicalID` class to ensure validity
2. **Validate external IDs**: Use `CanonicalNotationValidator` to check string formats
3. **Use evidence entry creation**: Use `EvidenceEntry.create()` to ensure consistency
4. **Track with question_unique_id**: Use the full canonical ID as the primary key
5. **Derive rubric keys**: Use `to_rubric_key()` method instead of manual string manipulation

## Future Extensions

### Cluster Support (Planned)

The problem statement mentions extending the notation for clusters. A potential format could be:

```
C#-P#-D#-Q#
```

Where:
- `C#`: Cluster identifier (e.g., C1, C2, etc.)
- Rest: Standard P#-D#-Q# format

This would allow grouping multiple policies into clusters while maintaining backward compatibility with the existing notation.

## API Reference

### Classes

- **CanonicalID**: Represents a P#-D#-Q# identifier
- **RubricKey**: Represents a D#-Q# identifier
- **EvidenceEntry**: Represents a standardized evidence entry
- **CanonicalNotationValidator**: Provides validation utilities
- **PolicyArea**: Enum of 10 policy areas
- **AnalyticalDimension**: Enum of 6 analytical dimensions

### Functions

- **generate_default_questions()**: Generate default question structure
- **get_system_structure_summary()**: Get system configuration summary

For detailed API documentation, see the module docstrings in `canonical_notation.py`.

## Testing

Run the comprehensive test suite:

```bash
python test_canonical_notation.py
```

The test suite includes 46 tests covering:
- ID creation and validation
- String parsing
- Evidence entry creation
- Legacy format migration
- Regex pattern matching
- Error handling

## Version

Current version: **2.0.0**

Compatible with FARFAN-2.0 framework.
