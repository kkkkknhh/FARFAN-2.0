# Data Integrity Validation and Resource Management

## Overview

This document describes the data integrity validation and resource management improvements implemented in FARFAN 2.0.

## 1. Data Integrity Validation

### 1.1 Pydantic Schema Validation

All pipeline stage data is now validated using Pydantic models, ensuring type safety and data integrity.

#### Key Validators

**Pipeline Validators Module** (`pipeline_validators.py`)

- `DocumentProcessingData`: Stage 1-2 validation
- `SemanticAnalysisData`: Stage 3 validation
- `CausalExtractionData`: Stage 4 validation (critical invariant)
- `MechanismInferenceData`: Stage 5 validation
- `FinancialAuditData`: Stage 6 validation
- `DNPValidationData`: Stage 7 validation (compliance score)
- `QuestionAnsweringData`: Stage 8 validation (300 questions)
- `ReportGenerationData`: Stage 9 validation

#### Example Usage

```python
from pipeline_validators import CausalExtractionData, validate_stage_transition

# Validate stage 4 data
stage_data = CausalExtractionData(
    causal_graph=graph,
    nodes=extracted_nodes,
    causal_chains=chains
)
validate_stage_transition("4", stage_data)
```

### 1.2 Critical Invariants

The following invariants are enforced at stage transitions:

#### Post-Stage 4: Causal Extraction
```python
# INVARIANT: len(nodes) > 0 or abort
# If no nodes are extracted, raises ValueError and aborts pipeline
```

**Implementation:**
```python
@field_validator('nodes')
@classmethod
def validate_nodes_not_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
    if len(v) == 0:
        raise ValueError("CRITICAL: Stage 4 must extract at least one node.")
    return v
```

#### Post-Stage 7: DNP Validation
```python
# INVARIANT: 0 <= compliance_score <= 100 with warning
# Out-of-range scores are clamped and logged as warnings
```

**Implementation:**
```python
@field_validator('compliance_score')
@classmethod
def validate_compliance_score_range(cls, v: float) -> float:
    if not (0.0 <= v <= 100.0):
        logger.warning(f"⚠️  WARNING: Compliance score out of range [0, 100]: {v}")
        return max(0.0, min(100.0, v))
    return v
```

#### Post-Stage 8: Question Answering
```python
# INVARIANT: len(question_responses) == 300 or abort
# Exactly 300 questions must be answered
```

**Implementation:**
```python
@field_validator('question_responses')
@classmethod
def validate_question_count(cls, v: Dict[str, Dict]) -> Dict[str, Dict]:
    expected_count = 300
    actual_count = len(v)
    if actual_count != expected_count:
        raise ValueError(f"CRITICAL: Expected {expected_count} questions, got {actual_count}")
    return v
```

### 1.3 Strategic Assertions

Critical assumptions are documented with assertions:

```python
# In _stage_causal_extraction
if ctx.causal_graph is not None:
    assert ctx.causal_graph.number_of_nodes() > 0, \
        "Causal graph must have at least one node"
```

### 1.4 Type Hints and mypy

Strict type checking is enabled via `mypy.ini`:

```ini
[mypy]
python_version = 3.12
warn_return_any = True
disallow_untyped_defs = True
check_untyped_defs = True
strict_equality = True
```

Run type checking:
```bash
python3 -m mypy orchestrator.py --config-file mypy.ini
```

## 2. Resource Management

### 2.1 Context Managers for Stage Execution

Each pipeline stage is wrapped in a resource management context manager:

```python
with managed_stage_execution("STAGE 4"):
    ctx = self._stage_causal_extraction(ctx)
    memory_monitor.check("After Stage 4")
```

**Features:**
- Automatic garbage collection after each stage
- Memory usage logging (before/after)
- Graceful error handling

### 2.2 Memory Monitoring

The `MemoryMonitor` class tracks memory usage throughout pipeline execution:

```python
# Initialize monitor
memory_monitor = MemoryMonitor(log_interval_mb=100.0)

# Check memory after critical operations
memory_monitor.check("After Stage 4")

# Generate final report
memory_report = memory_monitor.report()
```

**Output:**
```
[MONITOR] Memory monitoring started at 245.32 MB
[MONITOR] After Stage 4 - Memory: 678.45 MB (Δ +433.13 MB)
[MONITOR] Memory Report - Initial: 245.32 MB, Final: 512.67 MB, Peak: 678.45 MB
```

### 2.3 Garbage Collection

Explicit garbage collection is triggered after each stage:

```python
with managed_stage_execution("Stage Name", force_gc=True):
    # ... stage code ...
# Automatic gc.collect() called here
```

Manual cleanup for intermediate data:

```python
from resource_management import cleanup_intermediate_data

# Process heavy data
large_dataset = load_large_data()
process_data(large_dataset)

# Explicitly clean up
cleanup_intermediate_data(large_dataset)
```

### 2.4 Memory Profiling Decorator

Profile memory usage of individual functions:

```python
from resource_management import memory_profiling_decorator

@memory_profiling_decorator
def heavy_computation():
    # ... expensive operation ...
    pass
```

**Output:**
```
[MEMORY] heavy_computation - Before: 245.32 MB
[MEMORY] heavy_computation - After: 567.89 MB (Δ +322.57 MB)
```

### 2.5 Heavy Document Loader

Context manager for PDF document loading:

```python
from resource_management import heavy_document_loader

with heavy_document_loader(pdf_path) as doc:
    # ... process document ...
# Document automatically closed and memory freed
```

## 3. Testing

### 3.1 Run Validator Tests

```bash
python3 -m pytest test_pipeline_validators.py -v
```

Tests cover:
- Valid data scenarios
- Invalid data rejection (critical invariants)
- Edge cases
- Boundary conditions

### 3.2 Run Resource Management Tests

```bash
python3 -m pytest test_resource_management.py -v
```

Tests cover:
- Memory monitoring accuracy
- Context manager behavior
- Garbage collection
- Error handling

### 3.3 Run All Tests

```bash
python3 -m pytest test_*.py -v
```

## 4. Integration Example

Complete pipeline with validation and resource management:

```python
from orchestrator import FARFANOrchestrator
from pathlib import Path

# Create orchestrator
orchestrator = FARFANOrchestrator(
    output_dir=Path("./results"),
    log_level="INFO"
)

# Process plan (with automatic validation and resource management)
context = orchestrator.process_plan(
    pdf_path=Path("plan.pdf"),
    policy_code="PDM-2024-001",
    es_municipio_pdet=False
)

# All stages automatically:
# 1. Validated with Pydantic models
# 2. Monitored for memory usage
# 3. Garbage collected after completion
# 4. Type checked (if using mypy)
```

## 5. Performance Considerations

### Memory Usage Guidelines

- **Stage 1-2 (Document Extraction)**: ~100-200 MB per document
- **Stage 4 (Causal Extraction)**: ~200-400 MB for large graphs
- **Stage 8 (Question Answering)**: ~50-100 MB
- **Total Peak**: Expect 500-800 MB for typical 170-page PDF

### Optimization Tips

1. **Lazy Loading**: Use generators for large datasets
2. **Batch Processing**: Process PDFs in batches if handling multiple files
3. **Memory Monitoring**: Set appropriate `log_interval_mb` to detect leaks early
4. **Explicit Cleanup**: Use `cleanup_intermediate_data()` for heavy objects

## 6. Error Handling

### Critical Errors (Pipeline Aborts)

1. **Stage 4**: No nodes extracted → `ValueError`
2. **Stage 8**: Not exactly 300 questions → `ValueError`

### Warnings (Pipeline Continues)

1. **Stage 1-2**: Empty text extraction → Logged warning
2. **Stage 7**: Compliance score out of range → Clamped and logged
3. **Stage 6**: Negative financial allocation → Logged warning

## 7. Maintenance

### Adding New Validators

1. Create Pydantic model in `pipeline_validators.py`
2. Add field validators for invariants
3. Update stage method to use new validator
4. Add tests in `test_pipeline_validators.py`

Example:

```python
class NewStageData(BaseModel):
    """Validated data from new stage"""
    field_name: Type = Field(...)
    
    @field_validator('field_name')
    @classmethod
    def validate_field(cls, v: Type) -> Type:
        # Add validation logic
        if condition_not_met:
            raise ValueError("Validation failed")
        return v
```

### Updating Memory Monitoring

Adjust monitoring parameters in `process_plan()`:

```python
# Initialize memory monitor with custom interval
memory_monitor = MemoryMonitor(log_interval_mb=50.0)  # Log every 50 MB change
```

## 8. Best Practices

1. **Always use context managers** for stage execution
2. **Monitor memory** in development to establish baselines
3. **Write tests** for new validators before deploying
4. **Run mypy** before committing code changes
5. **Review logs** for validation warnings regularly
6. **Profile memory** when processing large documents

## 9. References

- Pydantic Documentation: https://docs.pydantic.dev/
- mypy Documentation: https://mypy.readthedocs.io/
- psutil Documentation: https://psutil.readthedocs.io/
