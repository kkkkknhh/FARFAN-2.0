# Orchestrator Package

Pure orchestration logic and dependency injection for FARFAN 2.0 policy analysis pipeline.

## Components

### `pipeline.py`
Pure orchestration layer that coordinates the complete FARFAN 2.0 analytical pipeline:

1. **Statement Extraction** - Extract policy statements from text
2. **Contradiction Detection** - Detect contradictions between statements
3. **Coherence Metrics** - Calculate coherence scores
4. **Regulatory Analysis** - Analyze regulatory compliance
5. **Audit Summary** - Generate comprehensive audit summary

The pipeline accepts `PipelineInput` contracts and returns `PipelineOutput` contracts. All I/O is done through ports - the pipeline itself contains no I/O logic.

### `factory.py`
Dependency injection container that creates production and test dependencies:

- **Production dependencies** - Real adapters for production use
- **Test dependencies** - Mock/in-memory adapters for testing

## Usage

### Basic Usage

```python
from orchestrator import create_pipeline, create_production_dependencies
from core_contracts import PipelineInput, CURRENT_VERSIONS

# Create dependencies
deps = create_production_dependencies()

# Create pipeline
pipeline = create_pipeline(
    log_port=deps["log_port"],
    file_port=deps["file_port"],
    clock_port=deps["clock_port"],
)

# Prepare input
pipeline_input: PipelineInput = {
    "text": "Your PDM text here...",
    "plan_name": "PDM Ejemplo 2024-2027",
    "dimension": "estratégico",
    "config": {},
    "schema_version": CURRENT_VERSIONS["pipeline"],
}

# Execute pipeline
result = pipeline.orchestrate(pipeline_input)

# Access results
print(f"Coherence Score: {result['coherence_metrics']['coherence_score']}")
print(f"Overall Grade: {result['audit_summary']['overall_grade']}")
```

### Testing

```python
from orchestrator import create_pipeline, create_test_dependencies

# Create test dependencies (in-memory, deterministic)
deps = create_test_dependencies()

# Create pipeline with test dependencies
pipeline = create_pipeline(
    log_port=deps["log_port"],
    file_port=deps["file_port"],
    clock_port=deps["clock_port"],
)

# Use pipeline in tests...
```

## Design Principles

### Pure Orchestration
The pipeline is a pure orchestration layer - it coordinates different analytical modules but does not perform I/O directly. All I/O is done through ports.

### Dependency Injection
All dependencies are injected through constructor parameters. This makes the pipeline:
- **Testable** - Easy to test with mock dependencies
- **Composable** - Easy to wire up different implementations
- **Maintainable** - Clear dependency graph

### Contract-Driven
The pipeline works entirely with TypedDict contracts from `core_contracts.py`:
- **Input:** `PipelineInput`
- **Output:** `PipelineOutput`
- All intermediate results also use contracts

### Fail-Safe
The pipeline includes meaningful error messages and graceful degradation. If a phase fails, it logs the error and continues with fallback values.

## Calibration Constants

The pipeline uses calibration constants for quality thresholds:

```python
COHERENCE_THRESHOLD = 0.7
CAUSAL_INCOHERENCE_LIMIT = 5
EXCELLENT_CONTRADICTION_LIMIT = 5
GOOD_CONTRADICTION_LIMIT = 10
CRITICAL_SEVERITY_THRESHOLD = 0.85
HIGH_SEVERITY_THRESHOLD = 0.70
MEDIUM_SEVERITY_THRESHOLD = 0.50
```

These constants define quality grades and can be overridden through the pipeline's `calibration` dictionary.

## Architecture

```
PipelineInput
     ↓
Extract Statements → StatementExtractionOutput
     ↓
Detect Contradictions → ContradictionDetectionOutput
     ↓
Calculate Coherence → CoherenceMetricsOutput
     ↓
Analyze Regulatory → RegulatoryAnalysisOutput
     ↓
Generate Audit Summary → AuditSummaryOutput
     ↓
PipelineOutput
```

Each arrow represents a data dependency. Each phase appends to structured output contracts.

## Future Work

The current implementation uses placeholder logic for each analytical phase. Production implementation will:

1. Integrate actual statement extraction module
2. Integrate contradiction detection module
3. Integrate coherence calculation module
4. Integrate regulatory analysis module
5. Integrate audit summary generation module

The orchestrator structure is already in place and ready to compose these modules.
