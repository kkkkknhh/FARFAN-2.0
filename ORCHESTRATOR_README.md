# Analytical Orchestrator Documentation

## Overview

The Analytical Orchestrator is a unified pipeline manager for the FARFAN 2.0 analytical framework. It orchestrates the execution of all analytical modules (regulatory analysis, contradiction detection, coherence metrics, and audit generation) with deterministic behavior, complete data flow integrity, and auditable metrics.

**SIN_CARRETA Compliance**: Maximum auditability and determinism with structured telemetry, explicit contract checks, and immutable audit trails per 7-year retention policy.

## Key Features

### 1. Deterministic Execution
- All analytical phases execute in a strict, predefined order
- Mathematical calibration constants remain stable across runs
- Same input always produces same output (reproducible results)
- **NEW**: Input/output hashing for reproducibility verification

### 2. Complete Audit Trail
- Every phase logs its execution to an immutable audit log
- Logs persist to `logs/orchestrator/` with full traceability
- Includes timestamps, inputs, outputs, and metrics for every phase
- **NEW**: 7-year retention policy enforced (SIN_CARRETA compliance)
- **NEW**: JSONL append-only format for immutability

### 3. Structured Telemetry (NEW)
- Every phase boundary emits structured telemetry events
- Distributed tracing with trace_id and audit_id
- Phase start, decision, and completion events
- Input/output hashing for determinism validation
- Telemetry completeness verification in CI

### 4. Contract Enforcement (NEW)
- PhaseResult contract validation for all phases
- Structured exceptions for contract violations
- Explicit type checking and format validation
- NO silent failures - all violations are logged and raised

### 5. Data Flow Integrity
- Structured data contracts ensure consistent phase outputs
- No merge conflicts - each phase has its own namespace
- Explicit dependency validation prevents circular references

### 6. Error Resilience
- Graceful error handling with fallback mechanisms
- Pipeline continues even if non-critical phases fail
- All errors logged with full context and trace information

## Installation

The orchestrator is a standalone module with minimal dependencies:

```bash
# No additional installation needed - uses Python standard library
cd /path/to/FARFAN-2.0
python orchestrator.py  # Run validation
```

## Usage

### Basic Usage

```python
from orchestrator import create_orchestrator

# Create orchestrator with default calibration
orchestrator = create_orchestrator()

# Execute analysis pipeline
result = orchestrator.orchestrate_analysis(
    text="Plan de desarrollo municipal...",
    plan_name="PDM_Municipio_2024",
    dimension="estratégico"
)

# Access results
print(f"Total contradictions: {result['total_contradictions']}")
print(f"Coherence score: {result['calculate_coherence_metrics']['metrics']['overall_score']}")
print(f"Quality grade: {result['generate_audit_summary']['outputs']['harmonic_front_4_audit']['quality_grade']}")
```

### Custom Calibration

```python
from orchestrator import create_orchestrator

# Override calibration constants
orchestrator = create_orchestrator(
    coherence_threshold=0.8,           # Increase from default 0.7
    causal_incoherence_limit=3,        # Decrease from default 5
    regulatory_depth_factor=1.5        # Increase from default 1.3
)

result = orchestrator.orchestrate_analysis(text, plan_name, dimension)
```

### Custom Log Directory

```python
from pathlib import Path
from orchestrator import create_orchestrator

# Store audit logs in custom directory
orchestrator = create_orchestrator(
    log_dir=Path("/custom/path/to/logs")
)

result = orchestrator.orchestrate_analysis(text, plan_name, dimension)
```

## Pipeline Phases

The orchestrator executes 6 sequential phases:

### Phase 1: Extract Statements
- **Input**: Raw policy document text
- **Output**: Structured policy statements with metadata
- **Key**: `extract_statements`

### Phase 2: Detect Contradictions
- **Input**: Policy statements, full text
- **Output**: List of contradictions with severity scores
- **Key**: `detect_contradictions`

### Phase 3: Analyze Regulatory Constraints
- **Input**: Statements, text, temporal conflicts
- **Output**: Regulatory compliance analysis
- **Key**: `analyze_regulatory_constraints`
- **Special Output**: `d1_q5_regulatory_analysis`

### Phase 4: Calculate Coherence Metrics
- **Input**: Contradictions, statements, text
- **Output**: Coherence scores and quality metrics
- **Key**: `calculate_coherence_metrics`

### Phase 5: Generate Audit Summary
- **Input**: Contradictions
- **Output**: Audit summary with quality grade
- **Key**: `generate_audit_summary`
- **Special Output**: `harmonic_front_4_audit`

### Phase 6: Compile Final Report
- **Input**: All previous phase results
- **Output**: Unified structured report
- **Result**: Complete analysis with all phase data

## Calibration Constants

Default values (can be overridden):

| Constant | Default | Description |
|----------|---------|-------------|
| `COHERENCE_THRESHOLD` | 0.7 | Minimum acceptable coherence score |
| `CAUSAL_INCOHERENCE_LIMIT` | 5 | Maximum allowed causal incoherence flags |
| `REGULATORY_DEPTH_FACTOR` | 1.3 | Regulatory analysis depth multiplier |
| `CRITICAL_SEVERITY_THRESHOLD` | 0.85 | Threshold for critical contradictions |
| `HIGH_SEVERITY_THRESHOLD` | 0.70 | Threshold for high-severity contradictions |
| `MEDIUM_SEVERITY_THRESHOLD` | 0.50 | Threshold for medium-severity contradictions |
| `EXCELLENT_CONTRADICTION_LIMIT` | 5 | Maximum contradictions for "Excelente" grade |
| `GOOD_CONTRADICTION_LIMIT` | 10 | Maximum contradictions for "Bueno" grade |

## Output Structure

The orchestrator returns a unified dictionary with this structure:

```python
{
    "orchestration_metadata": {
        "version": "2.0.0",
        "calibration": {...},
        "execution_start": "2025-10-15T04:42:53.674000",
        "execution_end": "2025-10-15T04:42:53.678000"
    },
    "plan_name": "PDM_Test",
    "dimension": "estratégico",
    "analysis_timestamp": "2025-10-15T04:42:53.678000",
    "total_statements": 15,
    "total_contradictions": 3,
    
    "extract_statements": {
        "inputs": {...},
        "outputs": {"statements": [...]},
        "metrics": {...},
        "timestamp": "...",
        "status": "success"
    },
    
    "detect_contradictions": {
        "inputs": {...},
        "outputs": {"contradictions": [...], "temporal_conflicts": [...]},
        "metrics": {...},
        "timestamp": "...",
        "status": "success"
    },
    
    "analyze_regulatory_constraints": {
        "inputs": {...},
        "outputs": {"d1_q5_regulatory_analysis": {...}},
        "metrics": {...},
        "timestamp": "...",
        "status": "success"
    },
    
    "calculate_coherence_metrics": {
        "inputs": {...},
        "outputs": {"coherence_metrics": {...}},
        "metrics": {...},
        "timestamp": "...",
        "status": "success"
    },
    
    "generate_audit_summary": {
        "inputs": {...},
        "outputs": {"harmonic_front_4_audit": {...}},
        "metrics": {...},
        "timestamp": "...",
        "status": "success"
    }
}
```

## Audit Logs

Audit logs are persisted as JSON files in `logs/orchestrator/` with this format:

```
audit_log_{plan_name}_{timestamp}.json
```

Example: `audit_log_PDM_Municipio_2024_20251015_044253.json`

Each log contains:
- Plan name and timestamp
- Complete calibration constants used
- All phase executions with inputs, outputs, metrics, and status
- Timestamps for each phase (ISO 8601 format)

## Validation

The orchestrator includes built-in validation:

```python
from orchestrator import create_orchestrator

orchestrator = create_orchestrator()

# Verify no dependency cycles
validation = orchestrator.verify_phase_dependencies()
print(validation)

# Expected output:
# {
#   "has_cycles": false,
#   "dependencies": {...},
#   "validation_status": "PASS"
# }
```

## Testing

Run the test suite to validate orchestrator functionality:

```bash
python test_orchestrator.py
```

Tests verify:
- Orchestrator creation with default and custom calibration
- Phase dependency validation (no cycles)
- Complete pipeline execution
- Deterministic behavior (same input → same output)
- Error handling with graceful fallbacks
- Calibration constant preservation
- Audit log generation and immutability

## GitHub Copilot Integration

Detailed instructions for GitHub Copilot are in `.github/copilot-instructions.md`.

Key points:
- Always use calibration constants from orchestrator
- Never reorder phases
- Follow PhaseResult data contract
- Log all phase executions
- Handle errors gracefully with fallbacks

## Error Handling

When a phase fails:

1. Error is logged with full context
2. Fallback values are generated (empty lists, zero metrics)
3. Pipeline continues unless critical dependency is missing
4. Final report includes partial results and error information

Example error result:

```python
{
    "status": "error",
    "error_message": "Division by zero in coherence calculation",
    "timestamp": "2025-10-15T04:42:53.678000",
    "calibration": {...},
    "partial_results": {
        "extract_statements": {...},
        "detect_contradictions": {...}
    }
}
```

## Best Practices

### DO:
✅ Use calibration constants from orchestrator  
✅ Follow strict phase ordering  
✅ Return PhaseResult objects  
✅ Log all phase transitions  
✅ Handle errors gracefully  
✅ Validate phase dependencies  
✅ Use ISO 8601 timestamps  

### DON'T:
❌ Hardcode thresholds in modules  
❌ Skip phase dependency validation  
❌ Reorder phases  
❌ Modify audit logs after creation  
❌ Return None on errors  
❌ Use different timestamp formats  

## Troubleshooting

### Issue: Orchestrator validation fails
**Solution**: Check phase dependencies for cycles using `verify_phase_dependencies()`

### Issue: Audit log not created
**Solution**: Ensure log directory exists and is writable. Check `log_dir` parameter.

### Issue: Non-deterministic results
**Solution**: Verify all modules use fixed random seeds and avoid timestamp-based logic.

### Issue: Phase fails with error
**Solution**: Check audit log for detailed error information. Fallback values should be used.

## Contributing

When modifying the orchestrator:

1. Preserve function signatures
2. Update tests for new functionality
3. Run validation: `python orchestrator.py`
4. Run tests: `python test_orchestrator.py`
5. Update this documentation
6. Update `.github/copilot-instructions.md` if rules change

## Version History

### Version 2.0.0 (Current)
- Initial orchestrator implementation
- Deterministic phase execution
- Complete audit trail
- Calibration constant management
- Error resilience with fallbacks
- Dependency validation
- Comprehensive test suite

## Telemetry and Auditability (NEW)

### Structured Telemetry System

The orchestrator now includes a comprehensive telemetry system for maximum auditability:

#### Telemetry Events

Every phase boundary emits structured events:

1. **PHASE_START**: Emitted when a phase begins execution
   - Includes input hash for reproducibility
   - Captures trace context for distributed tracing
   
2. **PHASE_COMPLETION**: Emitted when a phase completes successfully
   - Includes output hash for verification
   - Contains metrics for performance analysis
   
3. **PHASE_DECISION**: Emitted when a phase makes a decision
   - Documents decision rationale
   - Links decision to inputs
   
4. **CONTRACT_VIOLATION**: Emitted when contract checks fail
   - Structured exception with full context
   - Prevents silent failures
   
5. **VALIDATION_CHECK**: Emitted during runtime validations
   - Documents pass/fail status
   - Includes check details

#### Trace Context

All events include distributed tracing context:

```python
{
    "trace_id": "unique-trace-id",      # Same across all events in a run
    "span_id": "unique-span-id",        # Unique per phase
    "parent_span_id": "parent-span",    # Links to parent phase
    "audit_id": "audit_run_001_xyz",    # Compliance audit identifier
    "timestamp": "2025-10-15T12:00:00"  # ISO 8601 timestamp
}
```

### Contract Enforcement

PhaseResult objects enforce strict contracts:

```python
@dataclass
class PhaseResult:
    phase_name: str          # Must be non-empty
    inputs: Dict[str, Any]   # Must be dict
    outputs: Dict[str, Any]  # Must be dict
    metrics: Dict[str, Any]  # Must be dict
    timestamp: str           # Must be ISO 8601 format
    status: str              # Must be "success" or "error"
    input_hash: str          # SHA-256 of inputs
    output_hash: str         # SHA-256 of outputs
    trace_context: TraceContext  # Distributed tracing
    
    def validate_contract(self) -> None:
        """Raises ContractViolationError if invalid"""
```

### Deterministic Hashing

All inputs and outputs are hashed for reproducibility:

```python
# Same inputs always produce same hash
hash1 = TelemetryCollector.hash_data({"key": "value"})
hash2 = TelemetryCollector.hash_data({"key": "value"})
assert hash1 == hash2  # Guaranteed determinism
```

### Audit Log Structure

Audit logs are persisted in immutable JSONL format:

```jsonl
{"run_id":"run_001","orchestrator":"AnalyticalOrchestrator","timestamp":"...","sha256_source":"abc123","event":"orchestrate_analysis_complete","data":{...}}
{"run_id":"run_002","orchestrator":"AnalyticalOrchestrator","timestamp":"...","sha256_source":"def456","event":"orchestrate_analysis_complete","data":{...}}
```

Each line is a complete JSON object, enabling:
- Append-only writes (immutability)
- Streaming analysis
- 7-year retention compliance

### Telemetry Files

Telemetry events are persisted separately:

```
logs/orchestrator/
├── audit_logs.jsonl                                    # Immutable audit trail
├── telemetry_analytical_PDM_20251015_120000.jsonl     # Telemetry events
└── audit_log_PDM_20251015_120000.json                 # Legacy format
```

### CI Validation

Use `ci_telemetry_validation.py` to enforce auditability in CI/CD:

```bash
python ci_telemetry_validation.py
```

This script validates:
- ✅ Telemetry completeness (all phases have start/completion events)
- ✅ Phase boundary events (no missing events)
- ✅ Trace context consistency (single trace_id per run)
- ✅ Deterministic hashing (same input = same hash)
- ✅ Audit log immutability (JSONL append-only format)
- ✅ Telemetry persistence (events saved to disk)
- ✅ Contract enforcement (all PhaseResults valid)

Exit codes:
- `0`: All checks passed
- `1`: Validation failed
- `2`: Contract violations detected

### Testing Auditability

Run comprehensive auditability tests:

```bash
# All tests (19 tests)
pytest test_orchestrator_auditability.py -v

# Telemetry module tests
pytest test_orchestrator_auditability.py::TestTelemetryModule -v

# Contract validation tests
pytest test_orchestrator_auditability.py::TestPhaseResultContract -v

# Orchestrator auditability tests
pytest test_orchestrator_auditability.py::TestOrchestratorAuditability -v

# Retention policy tests
pytest test_orchestrator_auditability.py::TestAuditRetention -v
```

### Example: Accessing Telemetry

```python
from orchestrator import create_orchestrator

orch = create_orchestrator()
result = orch.orchestrate_analysis(text, plan_name, dimension)

# Get all telemetry events
events = orch.telemetry.get_events()
print(f"Total events: {len(events)}")

# Get events for specific phase
phase_events = orch.telemetry.get_events(phase_name="detect_contradictions")
print(f"Contradiction detection events: {len(phase_events)}")

# Verify telemetry completeness
verification = orch.telemetry.verify_all_phases([
    "extract_statements",
    "detect_contradictions",
    "analyze_regulatory_constraints",
    "calculate_coherence_metrics",
    "generate_audit_summary"
])
print(f"All phases complete: {verification['all_complete']}")

# Get telemetry statistics
stats = orch.telemetry.get_statistics()
print(f"Retention years: {stats['retention_years']}")  # 7
print(f"Total events: {stats['total_events']}")
```

### SIN_CARRETA Compliance Checklist

Before merging orchestrator changes, verify:

- [ ] All phases emit PHASE_START and PHASE_COMPLETION events
- [ ] PhaseResult contracts are validated (validate_contract() called)
- [ ] Input/output hashes are generated deterministically
- [ ] Trace context propagates correctly across phases
- [ ] Audit logs are immutable (JSONL append-only)
- [ ] Telemetry files are persisted to disk
- [ ] 7-year retention policy is configured
- [ ] CI validation script passes
- [ ] All auditability tests pass (19/19)
- [ ] No silent failures or missing traces

## License

Part of the FARFAN 2.0 framework for Colombian municipal development plan analysis.

## Support

For questions or issues:
1. Review this documentation
2. Check `.github/copilot-instructions.md` for detailed rules
3. Examine audit logs for execution traces
4. Run validation and tests to identify issues
5. **NEW**: Run `ci_telemetry_validation.py` to verify auditability
