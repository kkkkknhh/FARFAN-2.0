# Contributing to FARFAN-2.0

Thank you for contributing to FARFAN-2.0! This guide will help you understand our development process and standards.

## 🎯 Unified Orchestrator Architecture

FARFAN-2.0 uses a **single unified orchestrator** to ensure deterministic, auditable, and maintainable code. All contributions must align with this architecture.

### Core Principles

1. **Single Source of Truth** - All orchestration logic lives in `orchestration/unified_orchestrator.py`
2. **Explicit Contracts** - All component integrations use Protocol-based contracts with runtime assertions
3. **No Silent Failures** - All errors must raise explicit exceptions with structured context
4. **Deterministic Execution** - Results must be reproducible (fixed calibration, seeded RNG available)
5. **Complete Telemetry** - All decision points emit structured telemetry events
6. **Audit Trail** - All executions logged with SHA-256 provenance

## 📝 Development Workflow

### 1. Setup Development Environment

```bash
# Clone repository
git clone https://github.com/kkkkknhh/FARFAN-2.0.git
cd FARFAN-2.0

# Install dependencies
pip install -r requirements.txt

# Run tests to verify setup
pytest -v
```

### 2. Making Changes

#### For Orchestration Changes

All orchestration changes must go through the unified orchestrator:

```python
from orchestration.unified_orchestrator import UnifiedOrchestrator

# Create orchestrator with dependency injection
orchestrator = UnifiedOrchestrator(config, deterministic_mode=True)

# Inject components (all components validated against contracts)
orchestrator.inject_components(
    extraction_pipeline=your_component,
    # ... other components
)

# Execute pipeline
result = await orchestrator.execute_pipeline(pdf_path)
```

**DO NOT:**
- Create new orchestrator files
- Add orchestration logic outside `orchestration/unified_orchestrator.py`
- Use implicit wiring or magic fallbacks
- Silently catch exceptions without re-raising

**DO:**
- Add new stages to the existing unified pipeline
- Create Protocol contracts for new components
- Emit telemetry at decision points
- Add runtime contract assertions
- Write deterministic tests

#### For Component Development

All components must implement explicit Protocol contracts:

```python
from typing import Protocol

class MyComponentProtocol(Protocol):
    """Contract for MyComponent
    
    Contract:
    - MUST return structured result
    - MUST NOT return None
    - MUST raise explicit exception on failure
    """
    def my_method(self, param: str) -> Dict[str, Any]:
        ...
```

### 3. Testing

#### Contract Tests

All components must have contract validation tests:

```python
def test_component_contract_validation():
    """Test that component implements required contract"""
    
    orchestrator = UnifiedOrchestrator(config)
    
    # Should raise ContractViolationError if contract not satisfied
    with pytest.raises(ContractViolationError):
        orchestrator.inject_components(my_component=BadComponent())
```

#### Deterministic Tests

All tests must be deterministic and reproducible:

```python
@pytest.mark.asyncio
async def test_deterministic_execution():
    """Test that execution is deterministic"""
    
    # Enable deterministic mode
    orch = UnifiedOrchestrator(config, deterministic_mode=True)
    
    # Same input should produce same output
    result1 = await orch.execute_pipeline(pdf_path)
    result2 = await orch.execute_pipeline(pdf_path)
    
    assert result1.macro_score == result2.macro_score
```

#### Telemetry Tests

Verify that decision points emit telemetry:

```python
def test_telemetry_emission():
    """Test that telemetry events are emitted"""
    
    orch = UnifiedOrchestrator(config, enable_telemetry=True)
    orch._emit_telemetry('test.event', {'data': 'value'})
    
    # Verify telemetry recorded
    assert 'telemetry.test.event' in orch.metrics._metrics
```

### 4. Code Style

- Follow PEP 8
- Use type hints for all function signatures
- Document contracts in Protocol docstrings
- Add SIN_CARRETA compliance comments for critical sections

```python
def critical_method(self, param: str) -> Dict[str, Any]:
    """Process critical logic
    
    SIN_CARRETA Compliance:
    - Uses calibration constants (no hardcoded values)
    - Emits telemetry at decision point
    - Raises explicit exception on failure
    
    Args:
        param: Input parameter
        
    Returns:
        Structured result dictionary
        
    Raises:
        ValidationError: If validation fails with context
    """
    # SIN_CARRETA: Emit decision telemetry
    self._emit_telemetry('method.decision', {'param': param})
    
    # SIN_CARRETA: Use calibration constants
    if score < self.calibration.THRESHOLD:
        raise ValidationError("Score below threshold", context={'score': score})
    
    return result
```

## 🔄 Integration of Core Modules

The unified orchestrator explicitly integrates these core modules:

### 1. TeoriaCambio (Causal Graph Validation)

Integration point: `_validate_teoria_cambio()`

```python
# Validates: INSUMOS → PROCESOS → PRODUCTOS → RESULTADOS → CAUSALIDAD
teoria_result = await self._validate_teoria_cambio(causal_graph, run_id)
```

### 2. ValidadorDNP (Municipal Competency Compliance)

Integration point: `_validate_dnp_compliance()`

```python
# Validates municipal competencies, MGA indicators, PDET guidelines
dnp_result = await self._validate_dnp_compliance(chunks, run_id)
```

### 3. SMARTRecommendation (AHP Prioritization)

Integration point: `_generate_smart_recommendations()`

```python
# Generates SMART recommendations with AHP prioritization
recommendations = await self._generate_smart_recommendations(result, run_id)
```

### 4. PolicyContradictionDetectorV2 (Multi-modal Detection)

Integration point: `_stage_4_contradiction()`

```python
# Detects contradictions using semantic, NLI, and statistical methods
contradictions = await self._stage_4_contradiction(graph, chunks, run_id)
```

### 5. BayesianEngine (3-AGUJAS Inference)

Integration point: `_stage_3_bayesian()`

```python
# Uses immutable prior snapshots (breaks circular dependency)
bayesian_result = await self._stage_3_bayesian(graph, chunks, prior_snapshot, run_id)
```

## 🚫 Anti-Patterns (Will Be Rejected)

### ❌ DO NOT DO THIS:

```python
# Silent failure
try:
    result = component.process()
except Exception:
    pass  # ❌ NEVER silently catch

# Magic fallbacks
result = component.process() or {}  # ❌ No implicit fallbacks

# Hardcoded values
if score > 0.7:  # ❌ Use CALIBRATION.THRESHOLD

# Implicit wiring
self.component = SomeComponent()  # ❌ Must use dependency injection
```

### ✅ DO THIS INSTEAD:

```python
# Explicit exception handling
try:
    result = component.process()
except Exception as e:
    raise StageExecutionError(
        f"Processing failed: {str(e)}",
        context={'component': type(component).__name__}
    ) from e

# Explicit fallback with telemetry
if not result:
    self._emit_telemetry('component.fallback', {'reason': 'empty_result'})
    raise ComponentNotInjectedError("Component returned empty result")

# Use calibration constants
if score > self.calibration.COHERENCE_THRESHOLD:
    ...

# Explicit dependency injection
orchestrator.inject_components(component=my_component)
```

## 📊 CI/CD Requirements

All PRs must pass:

1. **Contract Enforcement Tests** - `pytest test_unified_orchestrator_contracts.py`
2. **Deterministic Tests** - Same input produces same output
3. **Telemetry Coverage** - All decision points emit telemetry
4. **Audit Log Validation** - All executions create audit records
5. **No Legacy Orchestrator Imports** - Only `orchestration/unified_orchestrator.py` allowed

## 🔍 Code Review Checklist

Before submitting PR, verify:

- [ ] No new orchestrator files created
- [ ] All components use Protocol contracts
- [ ] All errors raise explicit exceptions (no silent failures)
- [ ] All decision points emit telemetry
- [ ] Tests are deterministic (pass consistently)
- [ ] Uses calibration constants (no hardcoded values)
- [ ] Audit logs generated for all executions
- [ ] Documentation updated (if public API changed)

## 📚 Additional Resources

- [Unified Orchestrator Implementation](UNIFIED_ORCHESTRATOR_IMPLEMENTATION.md)
- [SIN_CARRETA Doctrine](.github/copilot-instructions.md)
- [Orchestrator README](ORCHESTRATOR_README.md)
- [Calibration Constants](infrastructure/calibration_constants.py)

## 💬 Questions?

Open an issue with the `question` label or reach out to the maintainers.

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.
