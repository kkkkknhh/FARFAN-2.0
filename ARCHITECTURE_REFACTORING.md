# FARFAN 2.0 - Architecture Refactoring Guide

**Version**: 1.0  
**Last Updated**: 2024-10-31  
**Target**: Hexagonal Architecture + Ports & Adapters

## Executive Summary

FARFAN 2.0 is undergoing an architectural refactoring to achieve:

- **Zero side effects** in core modules
- **Pure, testable functions** operating on typed contracts
- **Dependency injection** for all I/O operations
- **Hexagonal architecture** with ports and adapters
- **100% boundary enforcement** via CI

This guide explains the architecture, migration strategy, and design principles.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     CLI Layer                            │
│  (examples/, cli/) - User-facing commands                │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│              Orchestrator Layer                          │
│  (orchestrator/) - Pipeline composition & DI             │
│  ▪ Composes pure steps                                   │
│  ▪ Injects adapters                                      │
│  ▪ Validates contracts                                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                  Core Domain                             │
│  Pure business logic - NO I/O, NO side effects          │
│                                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Semantic Analysis    │  Contradiction Detection │   │
│  │  Statement Extraction │  Coherence Metrics       │   │
│  │  Regulatory Analysis  │  Audit Generation        │   │
│  └──────────────────────────────────────────────────┘   │
│                                                           │
│  Depends ONLY on:                                        │
│  ▪ TypedDict contracts (core_contracts.py)              │
│  ▪ Port protocols (ports.py)                            │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│               Infrastructure Layer                       │
│  (infrastructure/) - Adapters implement ports            │
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Filesystem  │  │    HTTP     │  │ Environment │     │
│  │  Adapters   │  │  Adapters   │  │  Adapters   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │    Clock    │  │     Log     │  │    Cache    │     │
│  │  Adapters   │  │  Adapters   │  │  Adapters   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Hexagonal Architecture (Ports & Adapters)

**Concept**: Core domain is independent of external concerns.

- **Core** defines what it needs (ports)
- **Infrastructure** provides implementations (adapters)
- **Orchestrator** wires them together

**Benefits**:
- Easy to test (swap real adapters for mocks)
- Easy to change infrastructure (swap implementations)
- Pure business logic (no I/O coupling)

### 2. Dependency Injection

**Rule**: Core never creates its dependencies.

**Before** (tightly coupled):
```python
def analyze_document(text: str):
    # Direct I/O coupling
    with open('config.json') as f:
        config = json.load(f)
    
    # Direct network coupling
    response = requests.get('http://api.example.com/data')
    
    # Analysis logic
    return analyze(text, config, response.json())
```

**After** (injected dependencies):
```python
def analyze_document(
    text: str,
    file_port: FilePort,
    http_port: HttpPort
) -> dict:
    # Use injected ports
    config = file_port.read_json('config.json')
    response = http_port.get('http://api.example.com/data')
    
    # Pure business logic
    return analyze(text, config, response['body'])
```

**Benefits**:
- Testable (inject mocks)
- Flexible (inject different implementations)
- Explicit dependencies (clear from signature)

### 3. Contract-Based Boundaries

**Rule**: All public APIs use TypedDict contracts.

**Benefits**:
- Type-safe interfaces
- Versioned compatibility
- Runtime validation
- Self-documenting

**Example**:
```python
from core_contracts import SemanticAnalyzerInput, SemanticAnalyzerOutput
from contracts_runtime import SemanticAnalyzerInputModel

def analyze_semantics(
    input: SemanticAnalyzerInput,
    file_port: FilePort
) -> SemanticAnalyzerOutput:
    # Validate at boundary
    validated = SemanticAnalyzerInputModel(**input)
    
    # Pure transformation
    result = perform_analysis(validated.text, validated.segments)
    
    # Return typed contract
    return {
        'chunks': result.chunks,
        'coherence_score': result.score,
        'quality_metrics': result.metrics,
        'schema_version': input['schema_version']
    }
```

### 4. Pure Functions

**Rule**: Core functions have no side effects.

**Characteristics of pure functions**:
- Same input → same output (deterministic)
- No side effects (no I/O, no mutation)
- Composable (easy to chain)
- Testable (no mocking needed)

**Before** (impure):
```python
def extract_statements(text: str) -> list:
    # Side effect: writes to log file
    with open('extraction.log', 'a') as f:
        f.write(f"Extracting from {len(text)} chars\n")
    
    # Side effect: modifies global state
    global extraction_count
    extraction_count += 1
    
    # Actual logic
    return extract(text)
```

**After** (pure):
```python
def extract_statements(
    text: str,
    log_port: LogPort
) -> StatementExtractionOutput:
    # Explicit I/O through port (no hidden side effects)
    log_port.info("Extracting statements", chars=len(text))
    
    # Pure transformation
    statements = extract(text)
    
    # Return contract
    return {
        'statements': statements,
        'total_count': len(statements),
        'schema_version': 'stmt-1.3'
    }
```

### 5. Fail Fast Validation

**Rule**: Validate at architectural boundaries.

**Where to validate**:
- ✅ Public API entry points
- ✅ Orchestrator inputs/outputs
- ✅ Adapter boundaries
- ❌ Internal functions (trust contracts)

**Example**:
```python
from contracts_runtime import PipelineInputModel, PipelineOutputModel

def orchestrate_analysis(
    input_dict: dict,
    file_port: FilePort
) -> dict:
    # Validate at boundary
    try:
        input = PipelineInputModel(**input_dict)
    except ValidationError as e:
        raise ValueError(f"Invalid input: {e}")
    
    # Core logic (assumes valid input)
    result = run_pipeline(input, file_port)
    
    # Validate output
    output = PipelineOutputModel(**result)
    return output.model_dump()
```

## Migration Strategy

### Phase 1: I/O Extraction

**Goal**: Remove all I/O from core modules.

**Steps**:
1. Identify I/O operations (use boundary scanner)
2. Define port interface for that I/O type
3. Replace direct I/O with port calls
4. Inject port via function parameter
5. Update tests to use mock adapters

**Example Migration**:

**Before**:
```python
def load_document(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
```

**After**:
```python
def load_document(path: str, file_port: FilePort) -> dict:
    return file_port.read_json(path)
```

### Phase 2: Contract Adoption

**Goal**: Replace dict/Any types with TypedDict contracts.

**Steps**:
1. Identify function that returns `dict` or `Any`
2. Define contract in `core_contracts.py`
3. Define validator in `contracts_runtime.py`
4. Update function signature
5. Add validation at boundaries

**Example Migration**:

**Before**:
```python
def detect_contradictions(statements: list, text: str) -> dict:
    # Returns untyped dict
    return {
        'contradictions': [...],
        'total': 5,
        'grade': 'Bueno'
    }
```

**After**:
```python
def detect_contradictions(
    input: ContradictionDetectionInput,
    file_port: FilePort
) -> ContradictionDetectionOutput:
    # Returns typed contract
    return {
        'contradictions': [...],
        'total_count': 5,
        'quality_grade': 'Bueno',
        'schema_version': input['schema_version']
    }
```

### Phase 3: Orchestrator Composition

**Goal**: Move orchestration logic out of monolithic executors.

**Steps**:
1. Identify orchestration flow in executor
2. Break into pure steps
3. Create pipeline in `orchestrator/pipeline.py`
4. Create factory in `orchestrator/factory.py` for DI
5. Move CLI logic to `cli/` directory

**Example**:

**Before** (monolithic executor):
```python
def execute_analysis(text: str, plan_name: str):
    # Everything in one function
    statements = extract_statements(text)
    contradictions = detect_contradictions(statements, text)
    coherence = calculate_coherence(contradictions, statements)
    # ... more steps ...
    return compile_report(...)
```

**After** (orchestrated pipeline):
```python
# orchestrator/pipeline.py
def run_analysis_pipeline(
    input: PipelineInput,
    file_port: FilePort,
    clock_port: ClockPort
) -> PipelineOutput:
    # Pure, composable steps
    stmt_result = extract_statements_step(input, file_port)
    contra_result = detect_contradictions_step(stmt_result, input, file_port)
    coher_result = calculate_coherence_step(contra_result, stmt_result)
    # ... more steps ...
    
    return compile_pipeline_output(
        stmt_result, contra_result, coher_result,
        schema_version=input['schema_version']
    )
```

## Directory Structure

```
FARFAN-2.0/
├── core_contracts.py          # TypedDict definitions (single source of truth)
├── contracts_runtime.py        # Pydantic validators
├── ports.py                    # Port protocols
│
├── infrastructure/             # Adapters (concrete implementations)
│   ├── filesystem.py          # LocalFileAdapter, InMemoryFileAdapter
│   ├── http.py                # RequestsHttpAdapter, MockHttpAdapter
│   ├── environment.py         # OsEnvAdapter, DictEnvAdapter
│   └── ...
│
├── orchestrator/              # Orchestration layer
│   ├── pipeline.py           # Pure orchestration steps
│   └── factory.py            # DI container, composition root
│
├── cli/                       # Command-line interface
│   ├── analyze.py            # Analysis command
│   ├── report.py             # Reporting command
│   └── ...
│
├── core/                      # Pure business logic (future)
│   ├── semantic/             # Semantic analysis
│   ├── contradiction/        # Contradiction detection
│   └── ...
│
├── tools/                     # Development tools
│   ├── scan_boundaries.py    # Boundary scanner
│   └── ...
│
├── tests/                     # Test suite
│   ├── test_contracts.py     # Contract validation tests
│   ├── test_adapters.py      # Adapter tests
│   └── ...
│
└── examples/                  # Runnable examples
    ├── basic_analysis.py     # Basic usage
    └── custom_adapter.py     # Custom adapter example
```

## Testing Strategy

### Unit Tests

**Goal**: Test pure logic in isolation.

**Approach**: Use in-memory/mock adapters.

```python
def test_extract_statements():
    # Use in-memory adapter
    file_port = InMemoryFileAdapter()
    file_port.write_json('/config.json', {'threshold': 0.7})
    
    # Test pure function
    input: StatementExtractionInput = {
        'text': 'Sample document...',
        'plan_name': 'PDM-2024',
        'extract_all': False,
        'schema_version': 'stmt-1.3'
    }
    
    result = extract_statements(input, file_port)
    
    assert result['total_count'] >= 0
    assert result['schema_version'] == 'stmt-1.3'
```

### Integration Tests

**Goal**: Test component interactions.

**Approach**: Use real adapters with temp directories.

```python
def test_full_pipeline_integration():
    # Use real filesystem adapter with temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        file_port = LocalFileAdapter(base_path=temp_dir)
        clock_port = SystemClockAdapter()
        
        # Run full pipeline
        input: PipelineInput = {...}
        result = run_analysis_pipeline(input, file_port, clock_port)
        
        assert result['schema_version'] == 'pipe-1.3'
        assert file_port.exists('output/report.json')
```

### Contract Tests

**Goal**: Ensure contract compliance.

**Approach**: Property-based testing with hypothesis.

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1))
def test_semantic_analyzer_input_accepts_any_text(text: str):
    data = {
        'text': text,
        'segments': [],
        'ontology_params': {},
        'schema_version': 'sem-1.3'
    }
    
    # Should not raise
    model = SemanticAnalyzerInputModel(**data)
    assert model.text == text
```

## CI Enforcement

### Boundary Scanner

Runs on every PR to enforce:
- No I/O operations in core modules
- No `__main__` blocks outside allowed directories
- Generates SARIF report for code scanning

### Contract Validation

Runs on every PR to ensure:
- All contracts are importable
- Pydantic models match TypedDict definitions
- Schema versions are valid

### Coverage Gates

Per-package coverage requirements:
- `core/`: 90% line, 80% branch
- `orchestrator/`: 85% line, 75% branch
- `infrastructure/`: 80% line, 70% branch

## Rollout Plan

### Current Status: Phase 0 Complete ✓

- [x] Boundary scanner implemented
- [x] Core contracts defined
- [x] Runtime validators created
- [x] Ports defined
- [x] Adapters implemented
- [x] Test suite established
- [x] CI workflow active

### Next: Phase 1 - I/O Extraction

**Target modules** (in order):
1. `semantic_chunking_policy.py`
2. `emebedding_policy.py`
3. `contradiction_deteccion.py`
4. `teoria_cambio.py`
5. `financiero_viabilidad_tablas.py`
6. `dereck_beach`

**Success criteria**:
- Boundary scanner reports 0 I/O operations
- All tests pass with mock adapters
- No breaking changes to outputs

## Common Patterns

### Pattern: Inject Port Dependencies

```python
# Good: Explicit dependency injection
def process_data(
    data: DataInput,
    file_port: FilePort,
    log_port: LogPort
) -> DataOutput:
    log_port.info("Processing data")
    config = file_port.read_json('config.json')
    # ... process ...
    return result

# Bad: Hidden dependencies
def process_data(data: dict) -> dict:
    # Hidden file I/O
    with open('config.json') as f:
        config = json.load(f)
    # ... process ...
```

### Pattern: Validate at Boundaries

```python
# Good: Validate at entry point
def public_api_function(input_dict: dict) -> dict:
    # Validate input
    input = InputModel(**input_dict)
    
    # Call internal functions (no validation)
    result = internal_process(input)
    
    # Validate output
    output = OutputModel(**result)
    return output.model_dump()

# Bad: Validate everywhere
def internal_function(input: dict) -> dict:
    # Unnecessary validation in internal function
    InputModel(**input)
    # ...
```

### Pattern: Factory for DI

```python
# Good: Centralized composition
# orchestrator/factory.py
def create_production_dependencies() -> dict:
    return {
        'file_port': LocalFileAdapter(base_path='/data'),
        'http_port': RequestsHttpAdapter(max_retries=3),
        'env_port': OsEnvAdapter(),
        'clock_port': SystemClockAdapter(),
    }

def create_test_dependencies() -> dict:
    return {
        'file_port': InMemoryFileAdapter(),
        'http_port': MockHttpAdapter(),
        'env_port': DictEnvAdapter({}),
        'clock_port': FixedClockAdapter(datetime.now()),
    }
```

## FAQ

### Q: Why not just use dependency injection frameworks?

**A**: We prefer explicit over implicit. Our DI is simple: pass dependencies as parameters. This makes dependencies visible in function signatures and easy to understand.

### Q: What about performance overhead from validation?

**A**: Validation only happens at boundaries (entry points), not internal functions. For hot paths, validation can be disabled with an environment flag in production.

### Q: How do I add a new port?

**A**:
1. Define protocol in `ports.py`
2. Implement adapter in `infrastructure/`
3. Add tests in `tests/test_adapters.py`
4. Update factory in `orchestrator/factory.py`

### Q: How do I add a new contract?

**A**:
1. Define TypedDict in `core_contracts.py`
2. Add validator in `contracts_runtime.py`
3. Add version to `CURRENT_VERSIONS`
4. Add tests in `tests/test_contracts.py`
5. Update SURFACE_MAP.md

---

**Document Version**: 1.0  
**Feedback**: Open an issue or PR to improve this guide
