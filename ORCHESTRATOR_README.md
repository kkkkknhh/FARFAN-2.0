# Analytical Orchestrator Documentation

## Overview

The Analytical Orchestrator is a unified pipeline manager for the FARFAN 2.0 analytical framework. It orchestrates the execution of all analytical modules (regulatory analysis, contradiction detection, coherence metrics, and audit generation) with deterministic behavior, complete data flow integrity, and auditable metrics.

## Key Features

### 1. Deterministic Execution
- All analytical phases execute in a strict, predefined order
- Mathematical calibration constants remain stable across runs
- Same input always produces same output (reproducible results)

### 2. Complete Audit Trail
- Every phase logs its execution to an immutable audit log
- Logs persist to `logs/orchestrator/` with full traceability
- Includes timestamps, inputs, outputs, and metrics for every phase

### 3. Data Flow Integrity
- Structured data contracts ensure consistent phase outputs
- No merge conflicts - each phase has its own namespace
- Explicit dependency validation prevents circular references

### 4. Error Resilience
- Graceful error handling with fallback mechanisms
- Pipeline continues even if non-critical phases fail
- All errors logged with full context

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

## License

Part of the FARFAN 2.0 framework for Colombian municipal development plan analysis.

## Support

For questions or issues:
1. Review this documentation
2. Check `.github/copilot-instructions.md` for detailed rules
3. Examine audit logs for execution traces
4. Run validation and tests to identify issues
