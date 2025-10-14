# Data Integrity Validation and Resource Management - Quick Start

## What's New

This implementation adds:

1. **Pydantic Schema Validation** - All pipeline stages have validated data models
2. **Critical Invariants** - Automatic enforcement of business rules
3. **Memory Management** - Context managers and garbage collection
4. **Type Checking** - mypy strict mode support

## Quick Usage

### Run Tests

```bash
# All tests
python3 -m pytest test_pipeline_validators.py test_resource_management.py -v

# Specific test
python3 -m pytest test_pipeline_validators.py::TestCausalExtractionData -v
```

### Run Demo

```bash
python3 demo_validation_and_resources.py
```

### Type Checking

```bash
python3 -m mypy orchestrator.py pipeline_validators.py resource_management.py --config-file mypy.ini
```

## Key Features

### Critical Invariants Enforced

1. **Stage 4** - Must extract at least 1 node (or abort)
2. **Stage 7** - Compliance score must be 0-100 (or clamp with warning)  
3. **Stage 8** - Must answer exactly 300 questions (or abort)

### Example: Using Validators

```python
from pipeline_validators import CausalExtractionData

# This will FAIL (no nodes)
try:
    data = CausalExtractionData(nodes={}, causal_graph=None, causal_chains=[])
except ValidationError:
    print("Validation failed as expected")

# This will PASS
data = CausalExtractionData(
    nodes={"node1": {"type": "outcome"}},
    causal_graph=None,
    causal_chains=[]
)
```

### Example: Using Resource Management

```python
from resource_management import managed_stage_execution, MemoryMonitor

monitor = MemoryMonitor()

with managed_stage_execution("Stage 4"):
    # Your stage code here
    # Automatic memory cleanup after this block
    pass

report = monitor.report()
print(f"Peak memory: {report['peak_mb']:.2f} MB")
```

## Integration with Orchestrator

The orchestrator automatically:

1. Validates data at each stage transition
2. Monitors memory usage
3. Performs garbage collection after each stage
4. Logs validation results

No changes needed to use it - just run:

```python
from orchestrator import FARFANOrchestrator

orchestrator = FARFANOrchestrator(output_dir="./results")
context = orchestrator.process_plan(
    pdf_path="plan.pdf",
    policy_code="PDM-2024-001"
)
```

## Files

- `pipeline_validators.py` - Pydantic models for all 9 stages
- `resource_management.py` - Memory management utilities
- `test_pipeline_validators.py` - Validator tests (18 tests)
- `test_resource_management.py` - Resource management tests (12 tests)
- `demo_validation_and_resources.py` - Interactive demo
- `mypy.ini` - Type checking configuration
- `DATA_INTEGRITY_AND_RESOURCE_MANAGEMENT.md` - Full documentation

## Test Coverage

✅ 30 tests total, all passing
- 18 validator tests
- 12 resource management tests

## Dependencies

Install with:
```bash
pip install pydantic mypy pytest psutil
```

## Documentation

See `DATA_INTEGRITY_AND_RESOURCE_MANAGEMENT.md` for complete documentation.
