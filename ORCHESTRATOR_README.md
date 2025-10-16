# Unified Orchestrator Documentation

## Overview

⚠️  **IMPORTANT: This documentation describes the NEW unified orchestrator.**

The **Unified Orchestrator** (`orchestration.unified_orchestrator.UnifiedOrchestrator`) is the single source of truth for all pipeline orchestration in FARFAN 2.0. It consolidates three previously separate orchestrators (PDMOrchestrator, AnalyticalOrchestrator, and CDAFFramework) into a single 9-stage deterministic pipeline with explicit contract enforcement.

### Migration Notice

**DEPRECATED orchestrators** (maintained for backward compatibility only):
- ❌ `orchestrator.py` (AnalyticalOrchestrator) - **DEPRECATED**
- ❌ `orchestration/pdm_orchestrator.py` (PDMOrchestrator) - **DEPRECATED**

**NEW orchestrator** (all new code must use):
- ✅ `orchestration/unified_orchestrator.py` (UnifiedOrchestrator) - **REQUIRED**

For migration guide, see: `IMPLEMENTATION_SUMMARY_ORCHESTRATOR_UNIFICATION.md`

## Key Features

### 1. Explicit Contract Enforcement
- **NO fallback behavior** - All components MUST be injected before execution
- **ComponentNotInjectedError** raised if components missing
- **ContractViolationError** raised if component violates interface contract
- All failures are explicit exceptions with structured payloads

### 2. Deterministic Execution
- Immutable prior snapshots break circular dependencies
- Fixed execution order (STAGE_0 through STAGE_8)
- Same input always produces same output (reproducible results)
- Mathematical calibration constants remain stable across runs

### 3. Structured Telemetry
- [TELEMETRY] markers at all phase transitions
- START, DECISION, COMPLETE events with context
- Structured event payloads via EventBus
- Complete audit trail with append-only logs

### 4. 9-Stage Unified Pipeline
- **STAGE_0**: PDF Ingestion
- **STAGE_1**: Semantic Extraction
- **STAGE_2**: Causal Graph Construction
- **STAGE_3**: Bayesian Inference (3 AGUJAS)
- **STAGE_4**: Contradiction Detection
- **STAGE_5**: Axiomatic Validation
- **STAGE_6**: Scoring Aggregation (MICRO→MESO→MACRO)
- **STAGE_7**: Report Generation
- **STAGE_8**: Adaptive Learning Loop (penalty factors)

## Installation

The unified orchestrator requires the full FARFAN-2.0 dependency stack:

```bash
cd /path/to/FARFAN-2.0
pip install -r requirements.txt
```

## Usage

### Basic Usage (Production)

```python
from orchestration.unified_orchestrator import UnifiedOrchestrator

# Create configuration
from dataclasses import dataclass, field

@dataclass
class SelfReflection:
    enable_prior_learning: bool = True
    prior_history_path: str = "logs/prior_history.json"
    feedback_weight: float = 0.1
    min_documents_for_learning: int = 3

@dataclass
class Config:
    self_reflection: SelfReflection = field(default_factory=SelfReflection)
    prior_decay_factor: float = 0.9
    queue_size: int = 100
    max_inflight_jobs: int = 10
    worker_timeout_secs: int = 300
    min_quality_threshold: float = 0.6

config = Config()

# Create orchestrator
orchestrator = UnifiedOrchestrator(config)

# Inject ALL required components (production mode)
# NOTE: All components MUST be provided - no fallback behavior
orchestrator.inject_components(
    extraction_pipeline=extraction_pipeline,  # Your extraction pipeline instance
    causal_builder=causal_builder,            # Your causal graph builder instance
    bayesian_engine=bayesian_engine,          # Your Bayesian engine instance
    contradiction_detector=contradiction_detector,  # Your detector instance
    validator=validator,                      # Your AxiomaticValidator instance
    scorer=scorer,                            # Your scoring framework instance
    report_generator=report_generator         # Your report generator instance
)

# Execute pipeline
import asyncio

result = await orchestrator.execute_pipeline("path/to/pdm_document.pdf")

# Access results
print(f"Success: {result.success}")
print(f"Macro score: {result.macro_score}")
print(f"Total duration: {result.total_duration:.2f}s")
print(f"Report: {result.report_path}")
```

### Testing Mode

```python
from orchestration.unified_orchestrator import UnifiedOrchestrator
from unittest.mock import AsyncMock, MagicMock

# Create orchestrator
orchestrator = UnifiedOrchestrator(config)

# Inject mock components for testing
orchestrator.inject_components(
    extraction_pipeline=AsyncMock(),
    causal_builder=AsyncMock(),
    bayesian_engine=AsyncMock(),
    contradiction_detector=MagicMock(),
    validator=MagicMock(),
    scorer=MagicMock(),
    report_generator=AsyncMock()
)

# Execute with test data
result = await orchestrator.execute_pipeline("test.pdf")
```

### Contract Enforcement

The orchestrator enforces explicit contracts at all integration points:

```python
from orchestration.unified_orchestrator import (
    ComponentNotInjectedError,
    ContractViolationError
)

# Example: Missing component
orchestrator = UnifiedOrchestrator(config)
# Don't inject components

try:
    result = await orchestrator.execute_pipeline("test.pdf")
except ComponentNotInjectedError as e:
    # Explicit error: "extraction_pipeline not injected"
    print(f"Contract violation: {e}")

# Example: Invalid component result
bad_extraction = AsyncMock()
bad_extraction.extract_complete = AsyncMock(return_value={
    "invalid_key": "missing semantic_chunks"  # Contract violation
})

orchestrator.inject_components(extraction_pipeline=bad_extraction, ...)

try:
    result = await orchestrator.execute_pipeline("test.pdf")
except ContractViolationError as e:
    # Explicit error: "extraction_pipeline.extract_complete() must return 
    # object with 'semantic_chunks' attribute"
    print(f"Contract violation: {e}")
```

## Pipeline Stages

The unified orchestrator executes 9 sequential stages:

### STAGE_0: PDF Ingestion
- **Input**: Path to PDF document
- **Output**: PDF data dict with 'path' and 'loaded' keys
- **Component**: None (internal)
- **Contract**: PDF file MUST exist
- **Telemetry**: stage.ingestion.start, stage.ingestion.complete

### STAGE_1: Semantic Extraction
- **Input**: PDF data
- **Output**: Dict with 'chunks' (list) and 'tables' (list) keys
- **Component**: `extraction_pipeline.extract_complete(pdf_path)`
- **Contract**: Must return object with 'semantic_chunks' attribute/key
- **Telemetry**: stage.extraction.complete

### STAGE_2: Causal Graph Construction
- **Input**: Semantic chunks, tables
- **Output**: NetworkX DiGraph
- **Component**: `causal_builder.build_graph(chunks, tables)`
- **Contract**: Must return `nx.DiGraph` instance
- **Telemetry**: stage.graph.complete

### STAGE_3: Bayesian Inference (3 AGUJAS)
- **Input**: Causal graph, chunks, immutable prior snapshot
- **Output**: Dict with 'mechanisms' (list) and 'posteriors' (dict) keys
- **Component**: `bayesian_engine.infer_all_mechanisms(graph, chunks)`
- **Contract**: Must return list of mechanism results
- **Telemetry**: stage.bayesian.complete
- **Note**: Uses immutable prior snapshot (breaks circular dependency)

### STAGE_4: Contradiction Detection
- **Input**: Semantic chunks
- **Output**: List of contradiction dicts
- **Component**: `contradiction_detector.detect(text, plan_name, dimension)`
- **Contract**: Must return dict with 'contradictions' key
- **Telemetry**: stage.contradiction.complete

### STAGE_5: Axiomatic Validation
- **Input**: Causal graph, semantic chunks, tables
- **Output**: ValidationResult with 'passed' attribute
- **Component**: `validator.validate_complete(graph, chunks, tables)`
- **Contract**: Result must have 'passed' attribute
- **Telemetry**: stage.validation.complete
- **Decision**: Logs validation outcome (passed/requires_review)

### STAGE_6: Scoring Aggregation (MICRO→MESO→MACRO)
- **Input**: Graph, mechanisms, validation result, contradictions
- **Output**: Dict with 'micro', 'meso', 'macro' keys
- **Component**: `scorer.calculate_all_levels(graph, mechanisms, validation, contradictions)`
- **Contract**: Must return dict with all three keys (micro, meso, macro)
- **Telemetry**: stage.scoring.complete
- **Decision**: Logs scoring breakdown

### STAGE_7: Report Generation
- **Input**: UnifiedResult, PDF path, run ID
- **Output**: Path to generated report
- **Component**: `report_generator.generate(result, pdf_path, run_id)`
- **Contract**: Must return Path or str
- **Telemetry**: stage.report.complete

### STAGE_8: Adaptive Learning Loop
- **Input**: UnifiedResult (with mechanism results)
- **Output**: Dict of penalty factors by mechanism type
- **Component**: Internal (uses AdaptiveLearningLoop)
- **Contract**: Penalty factors for NEXT run (current run unaffected)
- **Telemetry**: stage.learning.complete
- **Note**: Applies penalties to prior store for next execution

## Component Interface Requirements

All components injected via `inject_components()` MUST implement these interfaces:

### extraction_pipeline
```python
async def extract_complete(pdf_path: str) -> ExtractionResult:
    """
    Extract semantic chunks and tables from PDF.
    
    Returns:
        Object with 'semantic_chunks' (list) and 'tables' (list) attributes/keys
    """
```

### causal_builder
```python
async def build_graph(chunks: List[Any], tables: List[Any]) -> nx.DiGraph:
    """
    Build causal graph from extracted chunks and tables.
    
    Returns:
        nx.DiGraph instance (NetworkX directed graph)
    """
```

### bayesian_engine
```python
async def infer_all_mechanisms(graph: nx.DiGraph, chunks: List[Any]) -> List[MechanismResult]:
    """
    Perform Bayesian inference on causal mechanisms.
    
    Returns:
        List of mechanism results
    """
```

### contradiction_detector
```python
def detect(text: str, plan_name: str, dimension: str) -> Dict[str, Any]:
    """
    Detect contradictions in policy text.
    
    Returns:
        Dict with 'contradictions' key containing list of contradictions
    """
```

### validator
```python
def validate_complete(
    graph: nx.DiGraph,
    chunks: List[SemanticChunk],
    tables: List[Any]
) -> ValidationResult:
    """
    Validate graph structure, semantics, and regulatory compliance.
    
    Returns:
        ValidationResult with 'passed' (bool) attribute
    """
```

### scorer
```python
def calculate_all_levels(
    graph: nx.DiGraph,
    mechanisms: List[Any],
    validation: ValidationResult,
    contradictions: List[Dict]
) -> Dict[str, Any]:
    """
    Calculate MICRO→MESO→MACRO scores.
    
    Returns:
        Dict with 'micro' (dict), 'meso' (dict), 'macro' (float) keys
    """
```

### report_generator
```python
async def generate(
    result: UnifiedResult,
    pdf_path: str,
    run_id: str
) -> Path:
    """
    Generate final report from pipeline results.
    
    Returns:
        Path to generated report file
    """
```

## Testing

### Running Contract Enforcement Tests

```bash
# Run comprehensive contract tests
python -m pytest test_orchestrator_contracts.py -v

# Run specific test
python -m pytest test_orchestrator_contracts.py::test_component_not_injected_error_raised -v
```

### Test Coverage

The test suite proves:
1. **Contract Enforcement**: ComponentNotInjectedError and ContractViolationError
2. **Determinism**: Same inputs produce same outputs
3. **Telemetry**: Structured logging at all phase boundaries

### Creating Custom Tests

```python
import pytest
from orchestration.unified_orchestrator import UnifiedOrchestrator

@pytest.mark.asyncio
async def test_custom_pipeline(config):
    orchestrator = UnifiedOrchestrator(config)
    
    # Inject your components
    orchestrator.inject_components(...)
    
    # Execute and verify
    result = await orchestrator.execute_pipeline("test.pdf")
    assert result.success
```

## SIN_CARRETA Compliance

The unified orchestrator adheres to SIN_CARRETA doctrine:

- ✅ **No fallback behavior** - All failures are explicit exceptions
- ✅ **Explicit contracts** - All component interfaces documented and enforced
- ✅ **Runtime assertions** - Contract violations caught at execution time
- ✅ **Structured telemetry** - All phase transitions logged with context
- ✅ **Deterministic behavior** - Immutable prior snapshots, fixed execution order
- ✅ **Comprehensive testing** - Contract enforcement and determinism proven by tests

## Migration from Legacy Orchestrators

### From AnalyticalOrchestrator (orchestrator.py)

```python
# OLD (DEPRECATED):
from orchestrator import AnalyticalOrchestrator, create_orchestrator
orch = create_orchestrator()
result = orch.orchestrate_analysis(text, plan_name, dimension)

# NEW (REQUIRED):
from orchestration.unified_orchestrator import UnifiedOrchestrator

config = create_config()  # Your config creation
orch = UnifiedOrchestrator(config)
orch.inject_components(...)  # Inject all required components

# Convert text to PDF or use existing PDF
result = await orch.execute_pipeline(pdf_path)
```

### From PDMOrchestrator (orchestration/pdm_orchestrator.py)

```python
# OLD (DEPRECATED):
from orchestration.pdm_orchestrator import PDMOrchestrator
orch = PDMOrchestrator(config)
result = await orch.analyze_pdm(pdf_path)

# NEW (REQUIRED):
from orchestration.unified_orchestrator import UnifiedOrchestrator
orch = UnifiedOrchestrator(config)
orch.inject_components(...)  # Inject all required components
result = await orch.execute_pipeline(pdf_path)
```

## Troubleshooting
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
