# Phase Wiring Integration Documentation
# ========================================

## Contract Boundaries and Dataflow

This document describes the explicit contract boundaries between orchestrator phases and the live analytical modules.

## SIN_CARRETA Compliance

All integrations follow the SIN_CARRETA doctrine:
- **NO** silent fallbacks or magic
- **NO** implicit conversions of data types or schema
- **ALL** transitions validated with runtime assertions
- **ALL** errors trigger explicit failures
- **ALL** decision points emit telemetry events

## Phase Execution Flow

```
Input: text, plan_name, dimension
  ↓
Phase 1: Extract Statements (PolicyContradictionDetectorV2)
  ↓ (outputs: statements[PolicyStatement])
Phase 2: Detect Contradictions (PolicyContradictionDetectorV2)
  ↓ (outputs: contradictions[], temporal_conflicts[])
Phase 3: Analyze Regulatory Constraints (ValidadorDNP)
  ↓ (outputs: d1_q5_regulatory_analysis{}, dnp_validation_result)
Phase 4: Validate Regulatory (TeoriaCambio)
  ↓ (outputs: teoria_cambio_validation{})
Phase 5: Calculate Coherence Metrics (derived from contradictions)
  ↓ (outputs: coherence_metrics{})
Phase 6: Generate Audit Summary (derived metrics)
  ↓ (outputs: harmonic_front_4_audit{})
Phase 7: Generate Recommendations (SMARTRecommendation)
  ↓ (outputs: smart_recommendations[])
Phase 8: Compile Final Report (aggregation)
  ↓
Output: Unified structured report + Audit log
```

## Immutable Dataclasses

### PhaseInput (frozen=True)

Immutable input contract for all phases:

```python
@dataclass(frozen=True)
class PhaseInput:
    text: str                # Full policy document text (non-empty)
    plan_name: str           # Policy plan identifier (non-empty)
    dimension: str           # Analytical dimension (non-empty)
    trace_id: str            # Unique trace identifier (non-empty)
    
    def __post_init__(self):
        # Runtime validation enforces:
        - text is non-empty string
        - plan_name is non-empty string
        - dimension is non-empty string
        - trace_id is non-empty string
```

### PhaseResult

Standardized output contract for all phases:

```python
@dataclass
class PhaseResult:
    phase_name: str          # Phase identifier
    inputs: Dict[str, Any]   # All inputs to this phase
    outputs: Dict[str, Any]  # All outputs from this phase
    metrics: Dict[str, Any]  # Quantitative metrics
    timestamp: str           # ISO 8601 timestamp
    status: str              # "success" or "error"
    error: Optional[str]     # Error message if status is "error"
    
    def __post_init__(self):
        # Runtime validation enforces:
        - status in ["success", "error"]
        - if status == "error", error field is not None
        - outputs is a dictionary
        - metrics is a dictionary
```

## Phase Contracts

### Phase 1: Extract Statements

**Module**: `PolicyContradictionDetectorV2._extract_policy_statements`

**Input Contract**:
```python
PhaseInput(
    text: str,              # Full policy document
    plan_name: str,         # Plan identifier
    dimension: str,         # "estratégico", "diagnóstico", etc.
    trace_id: str           # Unique trace ID
)
```

**Output Contract**:
```python
{
    "statements": List[PolicyStatement]  # Extracted policy statements
}

# Each PolicyStatement has:
- text: str
- dimension: PolicyDimension
- position: Tuple[int, int]
- entities: List[Dict]
- temporal_markers: List[Dict]
- quantitative_claims: List[Dict]
- regulatory_references: List[str]
```

**Metrics**:
- statements_count: int
- avg_statement_length: float

**Raises**:
- RuntimeError if PolicyContradictionDetectorV2 not available
- RuntimeError if extraction fails

---

### Phase 2: Detect Contradictions

**Module**: `PolicyContradictionDetectorV2.detect`

**Input Contract**:
```python
PhaseInput + statements: List[PolicyStatement]
```

**Output Contract**:
```python
{
    "contradictions": List[Dict],      # Detected contradictions
    "temporal_conflicts": List[Dict],  # Temporal conflicts subset
    "full_detection_result": Dict      # Complete detection output
}

# Each contradiction dict contains:
- contradiction_type: str (e.g., "CAUSAL_INCOHERENCE")
- severity: float (0.0-1.0)
- confidence: float
- statement_a: Dict
- statement_b: Dict
```

**Metrics**:
- total_contradictions: int
- critical_severity_count: int (severity > 0.85)
- high_severity_count: int (0.70 < severity <= 0.85)
- medium_severity_count: int (0.50 < severity <= 0.70)
- temporal_conflicts_count: int

**Raises**:
- RuntimeError if PolicyContradictionDetectorV2 not available
- RuntimeError if detection fails

---

### Phase 3: Analyze Regulatory Constraints

**Module**: `ValidadorDNP.validar_proyecto_integral`

**Input Contract**:
```python
PhaseInput + statements + temporal_conflicts
```

**Output Contract**:
```python
{
    "d1_q5_regulatory_analysis": {
        "regulatory_references_count": int,
        "constraint_types_mentioned": int,
        "is_consistent": bool,
        "dnp_compliance_level": str,  # "excelente", "bueno", etc.
        "dnp_score": float,
        "cumple_competencias": bool,
        "cumple_mga": bool,
        "alertas_criticas": List[str],
        "recomendaciones": List[str],
        "d1_q5_quality": str
    },
    "dnp_validation_result": ResultadoValidacionDNP
}
```

**Metrics**:
- regulatory_references: int
- constraint_types: int
- dnp_score: float (0-100)
- critical_alerts: int

**Raises**:
- RuntimeError if ValidadorDNP not available
- RuntimeError if validation fails

---

### Phase 4: Validate Regulatory

**Module**: `TeoriaCambio.validar_estructura`

**Input Contract**:
```python
PhaseInput + statements
```

**Output Contract**:
```python
{
    "teoria_cambio_validation": {
        "es_valida": bool,
        "violaciones_orden": List[Tuple[str, str]],
        "categorias_faltantes": List[str],  # Category names
        "sugerencias": List[str]
    }
}
```

**Metrics**:
- is_valid: float (1.0 if valid else 0.0)
- violations_count: int
- missing_categories: int

**Raises**:
- RuntimeError if TeoriaCambio not available
- RuntimeError if validation fails

---

### Phase 5: Calculate Coherence Metrics

**Module**: Derived from contradiction detection results

**Input Contract**:
```python
PhaseInput + contradictions + statements
```

**Output Contract**:
```python
{
    "coherence_metrics": {
        "overall_coherence_score": float,  # 0.0-1.0
        "temporal_consistency": float,     # 0.0-1.0
        "causal_coherence": float,         # 0.0-1.0
        "quality_grade": str,              # "Excelente", "Bueno", "Insuficiente"
        "meets_threshold": bool            # >= COHERENCE_THRESHOLD
    }
}
```

**Metrics**:
- overall_score: float
- meets_threshold: float (1.0 or 0.0)
- temporal_consistency: float
- causal_coherence: float

**Calibration Constants Used**:
- COHERENCE_THRESHOLD (0.7)

---

### Phase 6: Generate Audit Summary

**Module**: Derived metrics with calibration constants

**Input Contract**:
```python
PhaseInput + contradictions
```

**Output Contract**:
```python
{
    "harmonic_front_4_audit": {
        "total_contradictions": int,
        "causal_incoherence_flags": int,
        "structural_failures": int,
        "quality_grade": str,  # "Excelente", "Bueno", "Regular"
        "meets_causal_limit": bool
    }
}
```

**Metrics**:
- quality_grade: str
- causal_flags: int
- structural_failures: int
- meets_causal_limit: float (1.0 or 0.0)

**Calibration Constants Used**:
- EXCELLENT_CONTRADICTION_LIMIT (5)
- GOOD_CONTRADICTION_LIMIT (10)
- CAUSAL_INCOHERENCE_LIMIT (5)

---

### Phase 7: Generate Recommendations

**Module**: `SMARTRecommendation` framework

**Input Contract**:
```python
PhaseInput + contradictions + regulatory_outputs
```

**Output Contract**:
```python
{
    "smart_recommendations": List[Dict]  # SMART recommendation dicts
}

# Each recommendation dict contains:
- id: str
- title: str
- smart_criteria: {specific, measurable, achievable, relevant, time_bound}
- scoring: {impact, cost, urgency, viability, ahp_total}
- priority: str (CRITICAL, HIGH, MEDIUM, LOW)
- impact_level: str
- estimated_duration_days: int
- responsible_entity: str
- ods_alignment: List[str]
```

**Metrics**:
- recommendations_count: int
- avg_ahp_score: float
- critical_count: int
- high_count: int

**Validation**:
- All recommendations validated with `rec.validate()` before output
- Any invalid recommendation raises ValueError

---

### Phase 8: Compile Final Report

**Module**: Aggregation (no external module)

**Input Contract**:
```python
All previous phase results via self._audit_log
```

**Output Contract**:
```python
{
    "orchestration_metadata": {
        "version": "2.0.0",
        "calibration": {...},
        "execution_start": str,
        "execution_end": str,
        "trace_id": str
    },
    "plan_name": str,
    "dimension": str,
    "analysis_timestamp": str,
    "trace_id": str,
    "total_statements": int,
    "total_contradictions": int,
    
    # Individual phase outputs (no overwrites)
    "extract_statements": {inputs, outputs, metrics, timestamp, status},
    "detect_contradictions": {inputs, outputs, metrics, timestamp, status},
    "analyze_regulatory_constraints": {inputs, outputs, metrics, timestamp, status},
    "validate_regulatory": {inputs, outputs, metrics, timestamp, status},
    "calculate_coherence_metrics": {inputs, outputs, metrics, timestamp, status},
    "generate_audit_summary": {inputs, outputs, metrics, timestamp, status},
    "generate_recommendations": {inputs, outputs, metrics, timestamp, status}
}
```

---

## Telemetry Events

Telemetry events are emitted at every phase boundary:

```python
# Phase start
metrics.record("phase.{phase_name}.start", 1.0)

# Phase completion
metrics.record("phase.{phase_name}.complete", 1.0)
metrics.record("phase.{phase_name}.status.{success|error}", 1.0)

# Structured log event
logger.info(f"[{trace_id}] Phase {phase_name} completed: status={status}")
```

## Error Handling

All errors are explicit with structured exceptions:

```python
# NO silent failures
try:
    result = module.call(...)
except Exception as e:
    logger.error(f"Phase {phase_name} failed: {e}", exc_info=True)
    raise RuntimeError(f"Phase {phase_name} failed: {e}") from e
```

Error messages always include:
- Phase name
- trace_id for traceability
- Original exception context

## Audit Trail

Every phase appends to immutable audit log:

```python
# In-memory audit log
self._audit_log.append(phase_result)

# Persistent audit log (JSONL format)
self.audit_logger.append_record(
    run_id=run_id,
    orchestrator="AnalyticalOrchestrator",
    sha256_source=sha256_source,
    event="phase_completed",
    trace_id=trace_id,
    ...
)
```

Audit logs are:
- Immutable (append-only)
- SHA-256 hashed for integrity
- ISO 8601 timestamped
- Include full calibration constants
- Persisted to `logs/orchestrator/audit_logs.jsonl`

## Calibration Constants

All phases use centralized calibration from `CALIBRATION` singleton:

```python
from infrastructure.calibration_constants import CALIBRATION

# Used in phases:
CALIBRATION.COHERENCE_THRESHOLD           # 0.7
CALIBRATION.CAUSAL_INCOHERENCE_LIMIT      # 5
CALIBRATION.REGULATORY_DEPTH_FACTOR       # 1.3
CALIBRATION.CRITICAL_SEVERITY_THRESHOLD   # 0.85
CALIBRATION.HIGH_SEVERITY_THRESHOLD       # 0.70
CALIBRATION.MEDIUM_SEVERITY_THRESHOLD     # 0.50
CALIBRATION.EXCELLENT_CONTRADICTION_LIMIT # 5
CALIBRATION.GOOD_CONTRADICTION_LIMIT      # 10
```

**NO** hardcoded thresholds in phase methods - all reference `self.calibration`.

## Module Availability

The orchestrator gracefully handles missing modules:

```python
# Import with error handling
try:
    from contradiction_deteccion import PolicyContradictionDetectorV2
    CONTRADICTION_DETECTOR_AVAILABLE = True
except ImportError:
    CONTRADICTION_DETECTOR_AVAILABLE = False

# Runtime check in phases
if not self.contradiction_detector:
    raise RuntimeError("PolicyContradictionDetectorV2 not available")
```

Module availability is logged during initialization:
- ✓ Module initialized successfully
- ⚠ Module not available - degraded mode

## Testing

All contracts are validated with comprehensive tests:

```bash
pytest test_phase_wiring_integration.py -v
```

Test coverage includes:
- Contract validation (PhaseInput, PhaseResult)
- Module availability checks
- Phase dependency validation
- Telemetry emission
- Explicit error handling
- Calibration constant usage
- Audit trail generation
- Deterministic behavior

---

**Last Updated**: 2025-10-16
**Version**: 2.0.0
**SIN_CARRETA Compliant**: ✓
