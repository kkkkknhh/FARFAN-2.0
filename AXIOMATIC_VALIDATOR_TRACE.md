# AxiomaticValidator Execution Flow Trace

## Executive Summary

This document traces the complete execution flow of the AxiomaticValidator system, verifying the validation sequence, output chaining, governance triggers, canonical notation mapping, and identifying gaps in the implementation.

---

## 1. Validation Sequence (CONFIRMED ✅)

The AxiomaticValidator implements a **strict three-phase validation sequence**:

### Phase 1: Structural Validation (Lines 279-299)
**Module**: `TeoriaCambio`  
**Location**: `validators/axiomatic_validator.py:346-368`  
**Canonical Mapping**: D6-Q1 (Theory of Change structure), D6-Q2 (Causal order)

```python
structural = self._validate_structural(causal_graph)
results.structural_valid = structural.es_valida
results.violaciones_orden = structural.violaciones_orden
results.categorias_faltantes = [cat.name for cat in structural.categorias_faltantes]
results.caminos_completos = structural.caminos_completos
```

**Verification**: ✅ Structural validation precedes semantic validation

### Phase 2: Semantic Validation (Lines 301-322)
**Module**: `PolicyContradictionDetectorV2`  
**Location**: `validators/axiomatic_validator.py:369-414`  
**Canonical Mapping**: D2-Q5 (Semantic coherence), D6-Q3 (Contradiction detection)

```python
contradictions = self._validate_semantic(causal_graph, semantic_chunks)
results.contradictions = contradictions
results.contradiction_density = len(contradictions) / results.total_edges
```

**Verification**: ✅ Semantic validation precedes regulatory validation

### Phase 3: Regulatory Validation (Lines 324-330)
**Module**: `ValidadorDNP`  
**Location**: `validators/axiomatic_validator.py:416-464`  
**Canonical Mapping**: D1-Q5 (Regulatory constraints), D4-Q5 (DNP compliance)

```python
dnp_results = self._validate_regulatory(semantic_chunks, financial_data)
results.dnp_compliance = dnp_results
results.regulatory_score = dnp_results.score_total if dnp_results else 0.0
```

**Verification**: ✅ Regulatory validation executes last

---

## 2. Output Chaining (CONFIRMED ✅)

### Sequential Output Flow

1. **TeoriaCambio Output** → `ValidacionResultado` (Line 280-285)
   - `es_valida` → `results.structural_valid`
   - `violaciones_orden` → `results.violaciones_orden`
   - `categorias_faltantes` → `results.categorias_faltantes`
   - `caminos_completos` → `results.caminos_completos`

2. **PolicyContradictionDetectorV2 Output** → List of contradictions (Line 303-310)
   - Raw contradictions → `results.contradictions`
   - Computed metric → `results.contradiction_density`

3. **ValidadorDNP Output** → `ResultadoValidacionDNP` (Line 326-328)
   - Full result → `results.dnp_compliance`
   - Score extraction → `results.regulatory_score`

### Aggregation into Unified Structure (Line 268-273)
```python
results = AxiomaticValidationResult()
results.validation_timestamp = datetime.now().isoformat()
results.total_edges = causal_graph.number_of_edges()
results.total_nodes = causal_graph.number_of_nodes()
```

**Verification**: ✅ Outputs are properly chained and aggregated into `AxiomaticValidationResult`

---

## 3. Governance Triggers (PARTIAL ✅⚠️)

### 3.1 Contradiction Density Trigger (CONFIRMED ✅)
**Threshold**: 0.05 (5%)  
**Location**: `validators/axiomatic_validator.py:93, 317-322`

```python
contradiction_threshold: float = 0.05

if results.contradiction_density > self.config.contradiction_threshold:
    if self.config.enable_human_gating:
        results.requires_manual_review = True
        results.hold_reason = 'HIGH_CONTRADICTION_DENSITY'
        logger.warning("High contradiction density detected - manual review required")
```

**Tracked Failures**: ✅ Sets `requires_manual_review=True` and `hold_reason`  
**Remediation**: ⚠️ No explicit remediation recommendations generated at this trigger point

### 3.2 Dimension D6 Score < 0.55 Trigger (CONFIRMED ✅)
**Threshold**: 0.55  
**Location**: `orchestration/pdm_orchestrator.py:372-375`

```python
d6_score = final_score.dimension_scores.get("D6", 0.7)
self.metrics.record("dimension.avg_score_D6", d6_score)
if d6_score < 0.55:
    self.metrics.alert("CRITICAL", "D6_SCORE_BELOW_THRESHOLD")
```

**Additional Monitoring**: `infrastructure/observability.py:344-345`
```python
if dimension == "D6" and score < 0.55:
    self.alert("CRITICAL", f"D6 score below threshold: {score:.2f}")
```

**Tracked Failures**: ✅ Alert generated and logged  
**Remediation**: ⚠️ Remediation occurs in `orchestration/pdm_orchestrator.py:507-509` but not directly linked to D6 trigger

### 3.3 Structural Violations Trigger (CONFIRMED ✅)
**Location**: `validators/axiomatic_validator.py:287-295`

```python
if structural.violaciones_orden:
    logger.warning("Found %d structural violations", len(structural.violaciones_orden))
    results.add_critical_failure(
        dimension='D6',
        question='Q2',
        evidence=structural.violaciones_orden,
        impact='Saltos lógicos detectados',
        recommendations=structural.sugerencias  # ✅ Remediation from TeoriaCambio
    )
```

**Tracked Failures**: ✅ Creates `ValidationFailure` with severity=CRITICAL  
**Remediation**: ✅ Uses `structural.sugerencias` from TeoriaCambio

### 3.4 Low Regulatory Score Trigger (CONFIRMED ✅)
**Threshold**: < 60.0  
**Location**: `validators/axiomatic_validator.py:333-337`

```python
results.is_valid = (
    results.structural_valid and
    not results.requires_manual_review and
    results.regulatory_score >= 60.0  # Minimum acceptable threshold
)
```

**Tracked Failures**: ✅ Sets `is_valid=False`  
**Remediation**: ⚠️ No explicit remediation recommendations for low regulatory score

---

## 4. Canonical Notation Mapping (PARTIAL ✅⚠️❌)

### 4.1 Documented Mappings (Lines 209-211, 255-257)

| Question ID | Validator | Purpose | Implementation Status |
|------------|-----------|---------|----------------------|
| D1-Q5 | ValidadorDNP | Regulatory constraints | ⚠️ IMPLICIT |
| D2-Q5 | PolicyContradictionDetectorV2 | Semantic coherence | ⚠️ IMPLICIT |
| D4-Q5 | ValidadorDNP | DNP compliance | ⚠️ IMPLICIT |
| D6-Q1 | TeoriaCambio | Theory of Change structure | ⚠️ IMPLICIT |
| D6-Q2 | TeoriaCambio | Causal order | ✅ EXPLICIT |
| D6-Q3 | PolicyContradictionDetectorV2 | Contradiction detection | ⚠️ IMPLICIT |

### 4.2 Explicit Mapping Implementation

**D6-Q2 Only** (Line 290-291):
```python
results.add_critical_failure(
    dimension='D6',
    question='Q2',  # ✅ EXPLICIT
    evidence=structural.violaciones_orden,
    impact='Saltos lógicos detectados',
    recommendations=structural.sugerencias
)
```

### 4.3 Missing Explicit Mappings ❌

**D6-Q1** (Theory of Change structure): 
- **Status**: Validated via `TeoriaCambio.validacion_completa()`
- **Gap**: No explicit `add_critical_failure(dimension='D6', question='Q1', ...)` for missing categories
- **Evidence**: `results.categorias_faltantes` is captured but not mapped to D6-Q1 failure

**D2-Q5** (Semantic coherence):
- **Status**: Validated via `PolicyContradictionDetectorV2.detect()`
- **Gap**: No explicit failure tracking for D2-Q5
- **Evidence**: `results.contradiction_density` captured but not linked to D2-Q5 canonical notation

**D6-Q3** (Contradiction detection):
- **Status**: Contradictions detected and counted
- **Gap**: No explicit failure for D6-Q3 when contradictions exist
- **Current**: Only triggers manual review, not a D6-Q3 failure record

**D1-Q5** (Regulatory constraints):
- **Status**: Validated via `ValidadorDNP`
- **Gap**: No explicit failure tracking for D1-Q5
- **Evidence**: `results.dnp_compliance` captured but not decomposed to D1-Q5

**D4-Q5** (DNP compliance):
- **Status**: Validated via `ValidadorDNP`
- **Gap**: No explicit failure tracking for D4-Q5
- **Evidence**: `results.regulatory_score` captured but not linked to D4-Q5

---

## 5. Results Aggregation Analysis

### 5.1 Aggregation Structure (CONFIRMED ✅)

The `AxiomaticValidationResult` dataclass aggregates all validation results:

```python
@dataclass
class AxiomaticValidationResult:
    # Overall validation status
    is_valid: bool = True
    
    # Structural validation (Phase III-B)
    structural_valid: bool = True
    violaciones_orden: List[Tuple[str, str]] = field(default_factory=list)
    categorias_faltantes: List[str] = field(default_factory=list)
    caminos_completos: List[List[str]] = field(default_factory=list)
    
    # Semantic validation (Phase III-C)
    contradiction_density: float = 0.0
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Regulatory validation (Phase III-D)
    regulatory_score: float = 0.0
    dnp_compliance: Optional[Any] = None
    
    # Failures and recommendations
    failures: List[ValidationFailure] = field(default_factory=list)
    
    # Governance triggers
    requires_manual_review: bool = False
    hold_reason: Optional[str] = None
```

### 5.2 Summary Method (Lines 186-197)
```python
def get_summary(self) -> Dict[str, Any]:
    return {
        'is_valid': self.is_valid,
        'structural_valid': self.structural_valid,
        'contradiction_density': self.contradiction_density,
        'regulatory_score': self.regulatory_score,
        'critical_failures': len([f for f in self.failures 
                                  if f.severity == ValidationSeverity.CRITICAL]),
        'requires_manual_review': self.requires_manual_review,
        'hold_reason': self.hold_reason
    }
```

**Verification**: ✅ Results properly aggregated with summary method

---

## 6. Identified Gaps

### 6.1 Validation Results Aggregation Gaps ⚠️

1. **Missing D6-Q1 Failure Mapping**: 
   - `categorias_faltantes` is captured but not converted to a `ValidationFailure` for D6-Q1
   - **Impact**: Missing categories don't generate tracked failures with remediation

2. **Missing D2-Q5 Failure Mapping**:
   - Semantic coherence issues are not explicitly tracked as D2-Q5 failures
   - **Impact**: No direct link between contradiction density and D2-Q5 canonical question

3. **Missing D6-Q3 Failure Mapping**:
   - Individual contradictions are tracked but not aggregated as D6-Q3 failure
   - **Impact**: Contradiction count > 0 doesn't create a D6-Q3 failure record

4. **Missing D1-Q5/D4-Q5 Failure Mappings**:
   - DNP validation results are not decomposed to question-level failures
   - **Impact**: Cannot determine if D1-Q5 or D4-Q5 specifically failed

### 6.2 Governance Threshold Enforcement Gaps ⚠️

1. **D6 Score < 0.55**:
   - **Enforcement**: ✅ Alert generated in observability system
   - **Gap**: ⚠️ Does not automatically add failure to `results.failures`
   - **Gap**: ⚠️ No explicit remediation recommendations generated

2. **Contradiction Density > 0.05**:
   - **Enforcement**: ✅ Sets `requires_manual_review=True`
   - **Gap**: ⚠️ Does not create a failure record with recommendations
   - **Gap**: ⚠️ No specific guidance on which contradictions to resolve

3. **Regulatory Score < 60.0**:
   - **Enforcement**: ✅ Sets `is_valid=False`
   - **Gap**: ❌ No failure record created
   - **Gap**: ❌ No remediation recommendations

### 6.3 Canonical Notation Mapping Gaps ❌

1. **Only D6-Q2 is explicitly mapped** to canonical notation in failure tracking
2. **D1-Q5, D2-Q5, D4-Q5, D6-Q1, D6-Q3** are documented in comments but not implemented in code
3. **No programmatic way** to query results by canonical question ID
4. **No reverse mapping** from validator output to canonical question

### 6.4 Remediation Recommendation Gaps ⚠️

1. **Structural violations**: ✅ Uses `structural.sugerencias` from TeoriaCambio
2. **High contradiction density**: ❌ No recommendations generated
3. **Low regulatory score**: ❌ No recommendations generated
4. **Missing categories**: ❌ Not tracked as failure, no recommendations
5. **D6 score threshold**: ❌ No recommendations generated

---

## 7. Recommendations for Improvement

### 7.1 Complete Canonical Mapping Implementation

```python
# In _validate_structural
if structural.categorias_faltantes:
    results.add_critical_failure(
        dimension='D6',
        question='Q1',  # Add explicit Q1 mapping
        evidence=structural.categorias_faltantes,
        impact='Categorías de teoría de cambio faltantes',
        recommendations=[f"Incluir nodo de categoría {cat}" 
                        for cat in structural.categorias_faltantes]
    )

# In _validate_semantic  
if results.contradiction_density > 0:
    results.add_critical_failure(
        dimension='D6',
        question='Q3',  # Add explicit Q3 mapping
        evidence=contradictions[:5],  # Top 5 contradictions
        impact=f'Densidad de contradicción: {results.contradiction_density:.2%}',
        recommendations=['Resolver contradicciones lógicas en texto']
    )

# In _validate_regulatory
if results.regulatory_score < 60.0:
    results.add_critical_failure(
        dimension='D4',
        question='Q5',  # Add explicit D4-Q5 mapping
        evidence={'score': results.regulatory_score},
        impact='Bajo cumplimiento DNP',
        recommendations=['Revisar alineación con MGA', 
                        'Verificar competencias municipales']
    )
```

### 7.2 Add Query Methods for Canonical Questions

```python
def get_failure_by_question(self, dimension: str, question: str) -> Optional[ValidationFailure]:
    """Get failure for specific canonical question"""
    for failure in self.failures:
        if failure.dimension == dimension and failure.question == question:
            return failure
    return None

def get_all_question_failures(self) -> Dict[str, ValidationFailure]:
    """Get all failures mapped by canonical ID"""
    return {f"{f.dimension}-{f.question}": f for f in self.failures}
```

### 7.3 Enhance Governance Trigger Enforcement

```python
# Add to validate_complete after D6 score check
if d6_score < 0.55:
    results.add_critical_failure(
        dimension='D6',
        question='Q1',  # Theory of Change quality
        evidence={'score': d6_score},
        impact='Calidad de teoría de cambio por debajo del umbral',
        recommendations=[
            'Revisar completitud de categorías causales',
            'Verificar orden lógico de conexiones',
            'Validar caminos completos de insumos a resultados'
        ]
    )
```

### 7.4 Add Recommendation Generation Layer

```python
def _generate_remediation_recommendations(
    self, 
    validation_type: str,
    evidence: Any
) -> List[str]:
    """Generate context-specific remediation recommendations"""
    
    if validation_type == "high_contradiction_density":
        return [
            "Revisar segmentos con contradicciones lógicas",
            "Verificar coherencia temporal de objetivos",
            "Eliminar inconsistencias en metas cuantitativas"
        ]
    elif validation_type == "low_regulatory_score":
        return [
            "Alinear proyectos con competencias municipales (Ley 152/1994)",
            "Validar indicadores contra catálogo MGA",
            "Revisar requisitos PDET si aplica"
        ]
    # ... more cases
```

---

## 8. Test Coverage Analysis

### Existing Tests ✅
- `test_validator_structure.py`: 9 tests, all passing - basic structure validation
- `test_axiomatic_validator.py`: 9 tests - integration tests with component validators

### Missing Tests ❌
1. Test for D6-Q1 failure creation when categories missing
2. Test for D2-Q5 failure creation for semantic issues
3. Test for D6-Q3 failure creation when contradictions detected
4. Test for D1-Q5/D4-Q5 failure creation for regulatory failures
5. Test for canonical notation query methods
6. Test for remediation recommendation generation
7. Test for all governance thresholds creating tracked failures

---

## 9. Conclusion

### ✅ Confirmed Working:
1. **Validation Sequence**: Structural → Semantic → Regulatory order is strictly enforced
2. **Output Chaining**: All validator outputs properly flow into unified result structure
3. **Basic Governance Triggers**: Contradiction density > 0.05, D6 score < 0.55, structural violations all trigger
4. **Result Aggregation**: Unified `AxiomaticValidationResult` properly aggregates all validation phases

### ⚠️ Partially Implemented:
1. **Governance Enforcement**: Triggers fire but don't always create tracked failures with remediation
2. **Canonical Mapping**: Documented but only D6-Q2 explicitly implemented in code
3. **Remediation**: Only structural violations get recommendations from TeoriaCambio

### ❌ Missing/Gaps:
1. **Complete Canonical Notation Mapping**: D1-Q5, D2-Q5, D4-Q5, D6-Q1, D6-Q3 not explicitly tracked
2. **Query Interface**: No methods to retrieve failures by canonical question ID
3. **Comprehensive Remediation**: Most governance triggers lack actionable recommendations
4. **Regulatory Decomposition**: DNP results not broken down to D1-Q5 vs D4-Q5
5. **Semantic Granularity**: Contradiction results not linked to specific D2-Q5 or D6-Q3 failures

### Risk Assessment:
- **HIGH**: Canonical notation mappings are incomplete, limiting traceability
- **MEDIUM**: Remediation recommendations are sparse, reducing actionability
- **LOW**: Core validation sequence is correct, results aggregate properly

The system has a solid foundation but requires explicit canonical question mapping and remediation layer to meet full requirements.
