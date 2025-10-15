# Scoring Framework for FARFAN 2.0

## Overview

The Scoring Framework implements deterministic, auditable scoring for Colombian Municipal Development Plans (PDM) using the canonical P1-P10 × D1-D6 × Q1-Q5 evaluation structure (300 total questions).

## Key Features

### 1. Canonical Question Validation
- **Coverage**: All 300 questions (10 policies × 6 dimensions × 5 questions)
- **Format**: P#-D#-Q# (e.g., P4-D2-Q3)
- **Validation**: Complete regex-based validation using `CanonicalNotationValidator`

### 2. Dimension Weight Enforcement
- **Constraint**: Weights for each policy's 6 dimensions sum to exactly 1.0
- **Precision**: Validated to 9 decimal places (< 1e-9 tolerance)
- **Policy-Specific**: Different policies have adjusted weights (e.g., P7 "Tierras" emphasizes D1 baseline data)

```python
DIMENSION_WEIGHTS = {
    "P1": {"D1": 0.15, "D2": 0.20, "D3": 0.15, "D4": 0.20, "D5": 0.15, "D6": 0.15},
    "P7": {"D1": 0.20, "D2": 0.18, "D3": 0.15, "D4": 0.18, "D5": 0.14, "D6": 0.15},
    # ... P2-P10
}
```

### 3. Rubric Threshold Mappings
- **Categories**: excelente (85-100%), bueno (70-85%), aceptable (55-70%), insuficiente (0-55%)
- **Consistency**: All thresholds validated for coverage of [0.0, 1.0] range
- **Bidirectional**: Score → category and category → score range mappings

```python
RUBRIC_THRESHOLDS = {
    "excelente": (0.85, 1.00),
    "bueno": (0.70, 0.85),
    "aceptable": (0.55, 0.70),
    "insuficiente": (0.00, 0.55)
}
```

### 4. MICRO → MESO → MACRO Aggregation

#### MICRO Level: Questions (Q1-Q5)
- Individual question scores: 0.0 - 1.0
- Confidence tracking
- DNP compliance metadata

#### MESO Level: Dimensions & Policies
**Dimension Score** (simple average):
```
dimension_score = avg(Q1, Q2, Q3, Q4, Q5)
```

**Policy Score** (weighted average):
```
policy_score = Σ(dimension_score × dimension_weight) / Σ(dimension_weight)
              = Σ(dimension_score × dimension_weight)  [since Σweights = 1.0]
```

#### MACRO Level: Clusters & Overall
**Cluster Score** (simple average):
```
cluster_score = avg(policy_scores for policies in cluster)
```

**Overall Decálogo Score** (weighted average):
```
overall_score = Σ(cluster_score × cluster_weight) / Σ(cluster_weight)
              = Σ(cluster_score × cluster_weight)  [since Σweights = 1.0]
```

### 5. D6 Manual Review Threshold
- **Trigger**: D6 (Theory of Change) score < 0.55
- **Action**: Automatic flag for manual review
- **Logging**: WARNING level alert with policy ID and score
- **Report**: Flags included in MacroScore.manual_review_flags

```python
D6_MANUAL_REVIEW_THRESHOLD = 0.55

# Example flag:
"P1-D6: score=0.500 < 0.55 (Theory of Change weak)"
```

### 6. DNP Regulatory Compliance Integration

#### D1-Q5: Regulatory Framework Compliance
- **Question**: "¿El plan justifica su alcance mencionando el marco legal o reconociendo restricciones explícitas?"
- **Integration**: Evaluates compliance with Colombian regulatory framework (e.g., Ley 1257, Ley 1448, Ley 1523)
- **Adjustment**: Modifies score based on compliance level (full/partial/minimal/none)

#### D4-Q5: Plan Nacional de Desarrollo (PND) Alignment
- **Question**: "¿El plan declara la alineación de sus resultados con marcos superiores como el PND o los ODS?"
- **Integration**: Checks alignment with national development priorities
- **Adjustment**: Modifies score based on alignment strength (strong/moderate/weak/none)

### 7. Complete Provenance Chain

The framework documents every step of score aggregation:

```python
provenance = {
    "aggregation_method": "weighted_average",
    "levels": {
        "MICRO": "300 questions (P1-P10 × D1-D6 × Q1-Q5)",
        "MESO_dimension": "6 dimensions per policy (simple average of 5 questions each)",
        "MESO_policy": "10 policies (weighted average of 6 dimensions using DIMENSION_WEIGHTS)",
        "MACRO_cluster": "4 clusters (simple average of policies in cluster)",
        "MACRO_overall": "1 overall score (weighted average of 4 clusters using CLUSTER_WEIGHTS)"
    },
    "dimension_weights": {...},
    "cluster_weights": {...},
    "rubric_thresholds": {...},
    "manual_review_threshold_d6": 0.55,
    "cluster_breakdown": {...}
}
```

## Policy Clusters

4 thematic clusters group the 10 policies:

```python
POLICY_CLUSTERS = {
    "derechos_humanos": ["P1", "P2", "P8"],          # 30% weight
    "sostenibilidad": ["P3", "P7"],                  # 20% weight
    "desarrollo_social": ["P4", "P6"],               # 30% weight
    "paz_y_reconciliacion": ["P5", "P9", "P10"]     # 20% weight
}
```

## Usage Examples

### Basic Validation
```python
from scoring_framework import validate_scoring_framework

report = validate_scoring_framework()
print(f"All valid: {report['all_valid']}")
print(f"Total questions validated: {report['question_validation']['total_questions']}")
```

### Calculating Scores
```python
from scoring_framework import ScoringEngine, QuestionScore

engine = ScoringEngine()

# Create question scores for P1-D1
question_scores = [
    QuestionScore(f"P1-D1-Q{i}", 0.75, "bueno") 
    for i in range(1, 6)
]

# Calculate dimension score
dim_score = engine.calculate_dimension_score("P1", "D1", question_scores)
print(f"P1-D1 score: {dim_score.score:.3f}, weight: {dim_score.weight}")
```

### DNP Integration
```python
from scoring_framework import ScoringEngine, QuestionScore

engine = ScoringEngine()
qs = QuestionScore("P3-D1-Q5", 0.75, "bueno")

# Integrate DNP compliance (requires dnp_validator instance)
qs_with_compliance = engine.integrate_dnp_compliance(qs, dnp_validator)
print(f"DNP compliance: {qs_with_compliance.dnp_compliance}")
```

### Complete End-to-End Scoring
```python
from scoring_framework import ScoringEngine

engine = ScoringEngine()

# Build complete score hierarchy
# ... (populate question_scores for all P1-P10 × D1-D6 × Q1-Q5)

# Calculate macro score
macro_score = engine.calculate_macro_score(cluster_scores)

# Generate report
report = engine.generate_scoring_report(macro_score)

print(f"Overall Score: {report['overall_score']:.3f}")
print(f"Category: {report['rubric_category']}")
print(f"Manual Review Required: {report['manual_review_required']}")

if report['manual_review_flags']:
    print("Manual Review Flags:")
    for flag in report['manual_review_flags']:
        print(f"  - {flag}")
```

## SIN_CARRETA Compliance

### Determinism
- All weights and thresholds are mathematical constants
- No randomness or time-dependent calculations
- Reproducible results for identical inputs

### Contract Enforcement
- Assertions validate all constraints (weights sum to 1.0, scores in [0,1])
- Type hints for all functions
- Dataclass validation in `__post_init__`

### Observability
- Structured logging at INFO/DEBUG/WARNING levels
- Complete provenance chain documentation
- Audit trail for all aggregation steps

### Error Handling
- Graceful degradation when DNP validator unavailable
- Clear error messages with context
- Validation before computation

## Testing

Run the complete test suite:

```bash
python3 -m unittest test_scoring_framework -v
```

Test coverage:
- ✓ Dimension weights sum to 1.0 (all P1-P10)
- ✓ Rubric thresholds consistency
- ✓ All 300 canonical questions validated
- ✓ MICRO→MESO→MACRO aggregation logic
- ✓ D6 < 0.55 manual review trigger
- ✓ DNP integration at D1-Q5 and D4-Q5
- ✓ Complete provenance chain
- ✓ Framework validation

All tests passing: 21/21 ✓

## Integration with Orchestrators

### PDMOrchestrator
```python
from orchestration.pdm_orchestrator import PDMOrchestrator
from scoring_framework import ScoringEngine

orchestrator = PDMOrchestrator(config)
orchestrator.scorer = ScoringEngine()

# Orchestrator will use ScoringEngine for quality score calculation
result = await orchestrator.analyze_plan(pdf_path)
```

### AnalyticalOrchestrator
```python
from orchestrator import AnalyticalOrchestrator
from scoring_framework import ScoringEngine

orchestrator = AnalyticalOrchestrator()
engine = ScoringEngine()

# Use scoring engine for rubric mapping
category = engine.score_to_rubric_category(coherence_score)
```

## File Structure

```
scoring_framework.py              # Main framework implementation
test_scoring_framework.py         # Comprehensive test suite
SCORING_FRAMEWORK_README.md       # This file
canonical_notation.py             # Canonical ID validation (dependency)
```

## Mathematical Guarantees

1. **Weight Conservation**: Σ(dimension_weights) = 1.0 for all policies
2. **Cluster Weight Conservation**: Σ(cluster_weights) = 1.0
3. **Score Bounds**: All scores ∈ [0.0, 1.0]
4. **Completeness**: All 300 questions validated
5. **Consistency**: Bidirectional score ↔ category mapping

## Future Enhancements

1. **Bayesian Integration**: Connect to Bayesian inference engine for uncertainty quantification
2. **Historical Tracking**: Score evolution across PDM versions
3. **Comparative Analysis**: Benchmark against similar municipalities
4. **Interactive Visualization**: Generate provenance diagrams
5. **Machine Learning**: Learn optimal dimension weights from evaluation data

## References

- Canonical Notation System: `canonical_notation.py`
- PDM Orchestrator: `orchestration/pdm_orchestrator.py`
- Analytical Orchestrator: `orchestrator.py`
- DNP Integration: `dnp_integration.py`
- AGENTS.md: Build/test commands
