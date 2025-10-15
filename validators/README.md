# Axiomatic Validator - Unified Validation System

## Overview

The **Axiomatic Validator** is a unified validation system for Phase III-B and III-C of the PDM evaluation framework. It provides a single point of validation with explicit execution order and automatic governance triggers.

## Purpose

This validator consolidates three previously dispersed validation systems:
1. **TeoriaCambio** - Structural/Causal validation
2. **PolicyContradictionDetectorV2** - Semantic contradiction detection
3. **ValidadorDNP** - DNP regulatory compliance

## Benefits

- **Single Point of Validation**: One unified interface instead of managing three separate validators
- **Explicit Execution Order**: Validation runs in a defined sequence (Structural → Semantic → Regulatory)
- **Automatic Governance Triggers**: Built-in rules for when manual review is required
- **Consistent Results Format**: Standardized output structure across all validation types

## Architecture

### Data Structures

#### ValidationConfig
Configuration for the axiomatic validator:
```python
config = ValidationConfig(
    dnp_lexicon_version="2025",
    es_municipio_pdet=False,
    contradiction_threshold=0.05,
    enable_structural_penalty=True,
    enable_human_gating=True
)
```

#### PDMOntology
Encapsulates PDM ontology data including canonical chain:
```python
ontology = PDMOntology()
# Default canonical chain: INSUMOS → PROCESOS → PRODUCTOS → RESULTADOS → CAUSALIDAD
```

#### SemanticChunk
Represents a semantic chunk of text from the PDM:
```python
chunk = SemanticChunk(
    text="El municipio invertirá en educación...",
    dimension="ESTRATEGICO",
    position=(10, 20),
    metadata={'section': 'vision'}
)
```

#### ExtractedTable
Represents financial/data tables:
```python
table = ExtractedTable(
    title="Presupuesto 2025",
    headers=["Rubro", "Monto"],
    rows=[["Educación", 1000000]]
)
```

#### AxiomaticValidationResult
Comprehensive validation results:
```python
result = AxiomaticValidationResult(
    is_valid=True,
    structural_valid=True,
    contradiction_density=0.02,
    regulatory_score=75.0
)
```

## Usage

### Basic Usage

```python
from validators import AxiomaticValidator, ValidationConfig, PDMOntology, SemanticChunk
import networkx as nx

# 1. Create configuration
config = ValidationConfig()
ontology = PDMOntology()

# 2. Initialize validator
validator = AxiomaticValidator(config, ontology)

# 3. Prepare input data
causal_graph = nx.DiGraph()
causal_graph.add_node("Recursos", categoria="INSUMOS")
causal_graph.add_node("Capacitación", categoria="PROCESOS")
causal_graph.add_edge("Recursos", "Capacitación")

semantic_chunks = [
    SemanticChunk(
        text="El municipio destinará recursos para capacitación.",
        dimension="ESTRATEGICO"
    )
]

# 4. Run validation
result = validator.validate_complete(causal_graph, semantic_chunks)

# 5. Check results
if result.is_valid:
    print("Validation passed!")
else:
    print(f"Validation failed. Failures: {len(result.failures)}")
    for failure in result.failures:
        print(f"  - {failure.dimension}-{failure.question}: {failure.impact}")
```

### Getting Validation Summary

```python
summary = result.get_summary()
print(f"Valid: {summary['is_valid']}")
print(f"Contradiction Density: {summary['contradiction_density']}")
print(f"Regulatory Score: {summary['regulatory_score']}/100")
print(f"Requires Manual Review: {summary['requires_manual_review']}")
```

## Validation Sequence

The validator executes validation in the following order:

### 1. Structural Validation (D6-Q1/Q2)
Uses `TeoriaCambio` to validate:
- Causal order compliance
- Presence of all required categories (INSUMOS → PROCESOS → PRODUCTOS → RESULTADOS → CAUSALIDAD)
- Complete causal pathways

**Impact**: If structural violations are found:
- Critical failures are added to results
- Structural penalty can be applied (if enabled)
- Graph edges may be marked with reduced confidence

### 2. Semantic Validation (D2-Q5, D6-Q3)
Uses `PolicyContradictionDetectorV2` to detect:
- Semantic contradictions
- Logical inconsistencies
- Temporal conflicts

**Impact**: If contradiction density exceeds threshold (default 0.05):
- Triggers manual review requirement
- Sets hold reason to 'HIGH_CONTRADICTION_DENSITY'
- Governance gate activated

### 3. Regulatory Validation (D1-Q5, D4-Q5)
Uses `ValidadorDNP` to validate:
- Municipal competencies
- MGA indicator compliance
- PDET guidelines (if applicable)

**Impact**: If regulatory score < 60:
- Overall validation marked as invalid
- Recommendations provided for improvement

## Governance Triggers

The validator includes built-in governance triggers:

1. **High Contradiction Density**
   - Threshold: > 5% (configurable)
   - Action: `requires_manual_review = True`
   - Reason: 'HIGH_CONTRADICTION_DENSITY'

2. **Structural Violations**
   - Trigger: Any order violations in causal chain
   - Action: Critical failure added, penalty applied
   - Impact: Reduced edge confidence scores

3. **Low Regulatory Compliance**
   - Threshold: < 60/100
   - Action: Overall validation fails
   - Impact: Cannot proceed without remediation

## Integration with Existing Systems

The Axiomatic Validator is designed to work seamlessly with existing components:

### TeoriaCambio Integration
```python
# The validator uses TeoriaCambio.validacion_completa()
structural_result = validator._validate_structural(causal_graph)
# Returns: ValidacionResultado with violations, missing categories, etc.
```

### PolicyContradictionDetectorV2 Integration
```python
# The validator uses detect() method for each semantic chunk
contradictions = validator._validate_semantic(causal_graph, semantic_chunks)
# Returns: List of detected contradictions
```

### ValidadorDNP Integration
```python
# The validator aggregates DNP validation
dnp_result = validator._validate_regulatory(semantic_chunks, financial_data)
# Returns: ResultadoValidacionDNP with compliance scores
```

## Canonical Notation Mapping

The validator maps to the canonical notation system:

- **D1-Q5**: Regulatory constraints (ValidadorDNP)
- **D2-Q5**: Semantic coherence (PolicyContradictionDetectorV2)
- **D4-Q5**: DNP compliance (ValidadorDNP)
- **D6-Q1**: Theory of Change structure (TeoriaCambio)
- **D6-Q2**: Causal order (TeoriaCambio)
- **D6-Q3**: Contradiction detection (PolicyContradictionDetectorV2)

## Testing

Run the structure tests:
```bash
python3 -m unittest test_validator_structure -v
```

All tests should pass, validating:
- Module structure
- Data class creation
- Configuration options
- Result aggregation
- Failure tracking

## Future Enhancements

Potential improvements:
1. Weighted scoring across validation dimensions
2. Configurable governance thresholds per municipality type
3. Integration with ML-based quality prediction
4. Real-time validation streaming for large documents
5. Automated remediation suggestions

## References

- **Front C.1**: Structural penalty application for order violations
- **Phase III-B**: Structural validation implementation
- **Phase III-C**: Semantic validation implementation
- **Phase III-D**: Regulatory validation implementation
- **Governance Standard**: Manual review triggers and hold mechanisms
