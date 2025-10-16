# Contributing to FARFAN 2.0

Thank you for your interest in contributing to FARFAN 2.0! This document provides guidelines and requirements for contributing to this project.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [SIN_CARRETA Doctrine](#sin_carreta-doctrine)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Requirements](#documentation-requirements)
- [Pull Request Process](#pull-request-process)
- [Determinism and Contracts](#determinism-and-contracts)
- [CI/CD Requirements](#cicd-requirements)

---

## Code of Conduct

This project adheres to professional standards of collaboration and respect. All contributors are expected to:

- Be respectful and constructive in all interactions
- Focus on technical merit and evidence-based discussions
- Prioritize code quality, determinism, and auditability
- Follow the SIN_CARRETA doctrine (see below)

---

## SIN_CARRETA Doctrine

**⚠️ MANDATORY COMPLIANCE ⚠️**

All contributions to FARFAN 2.0 **MUST** comply with the [SIN_CARRETA Doctrine](SIN_CARRETA_RULES.md).

### Core Principles

1. **Determinism**: Same input → same output, always
2. **Contracts**: Explicit preconditions, postconditions, and invariants
3. **Auditability**: Every decision point emits structured telemetry
4. **No Silent Failures**: All errors are explicit and logged

### Quick Checklist

Before submitting any code, verify:

- [ ] No use of `time.time()`, `datetime.now()`, or `random.*` without deterministic injection
- [ ] All public functions have type hints and docstrings
- [ ] All contract checks are preserved or enhanced
- [ ] All decision points emit telemetry
- [ ] Tests prove determinism (same input → same output)
- [ ] No silent failures or best-effort error handling
- [ ] CODE_FIX_REPORT.md updated with SIN_CARRETA compliance

**Read the full doctrine**: [SIN_CARRETA_RULES.md](SIN_CARRETA_RULES.md)

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Familiarity with the FARFAN 2.0 architecture
- Understanding of the SIN_CARRETA doctrine

### Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/FARFAN-2.0.git
   cd FARFAN-2.0
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv farfan_env
   source farfan_env/bin/activate  # On Linux/macOS
   # farfan_env\Scripts\activate   # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

5. **Download spaCy model**:
   ```bash
   python -m spacy download es_core_news_lg
   ```

---

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/*` - New features
- `fix/*` - Bug fixes
- `refactor/*` - Code refactoring (must include SIN_CARRETA rationale)
- `docs/*` - Documentation updates
- `test/*` - Test additions or improvements

### 2. Make Your Changes

- Follow the [Coding Standards](#coding-standards)
- Write tests for all new functionality
- Ensure determinism in all code
- Add telemetry for decision points
- Update documentation

### 3. Test Your Changes

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest test_your_module.py -v

# Run with coverage
python -m pytest --cov=. --cov-report=term-missing
```

### 4. Validate Determinism

```bash
# Run determinism validation
python ci_contract_enforcement.py

# Run telemetry validation
python ci_telemetry_validation.py
```

### 5. Update Documentation

- Update module README if applicable
- Update CONTRIBUTING.md if adding new patterns
- Update CODE_FIX_REPORT.md with changes

### 6. Commit Your Changes

Follow conventional commit format:

```bash
git add .
git commit -m "feat: add deterministic timestamp injection

SIN_CARRETA-RATIONALE:
- Added injectable clock for deterministic testing
- Increases complexity by 15 lines for contract enforcement
- Satisfies Primary Rule: Determinism
- Tests prove reproducibility: test_deterministic_timestamps.py

Refs: #123"
```

Commit message requirements:
- **Type**: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
- **Subject**: Clear, concise description (imperative mood)
- **Body** (if increasing complexity): Include SIN_CARRETA-RATIONALE section
- **Footer**: Reference issue numbers

---

## Coding Standards

### Python Style

- Follow **PEP 8** with the following specifics:
  - Maximum line length: 100 characters
  - Use 4 spaces for indentation (no tabs)
  - Use snake_case for functions and variables
  - Use PascalCase for classes
  - Use UPPER_CASE for constants

### Type Hints

**REQUIRED** for all public functions:

```python
def process_data(input_data: Dict[str, Any], threshold: float = 0.7) -> PhaseResult:
    """
    Process input data and return phase result.
    
    Args:
        input_data: Dictionary containing input data
        threshold: Coherence threshold (default: 0.7)
        
    Returns:
        PhaseResult with processing outcomes
        
    Raises:
        ValueError: If input_data is empty or invalid
        
    Preconditions:
        - input_data must be non-empty dict
        - threshold must be in range [0.0, 1.0]
        
    Postconditions:
        - Returns PhaseResult with status "success" or "error"
        - All outputs are deterministic for same inputs
    """
    assert input_data, "Contract violation: input_data cannot be empty"
    assert 0.0 <= threshold <= 1.0, f"Contract violation: threshold {threshold} out of range"
    
    # ... implementation ...
```

### Docstrings

**REQUIRED** for all public functions and classes:

- Use Google-style docstrings
- Include Args, Returns, Raises sections
- Include Preconditions and Postconditions
- Provide usage examples for complex functions

### Contract Enforcement

Use assertions for contract validation:

```python
def calculate_score(values: List[float]) -> float:
    """Calculate average score from values."""
    # Precondition check
    assert values, "Contract violation: values list cannot be empty"
    assert all(0 <= v <= 1 for v in values), "Contract violation: all values must be in [0, 1]"
    
    result = sum(values) / len(values)
    
    # Postcondition check
    assert 0 <= result <= 1, f"Contract violation: result {result} out of range"
    return result
```

### Determinism Requirements

#### ❌ FORBIDDEN:

```python
import time
import random
from datetime import datetime

# VIOLATION: Non-deterministic time
timestamp = datetime.now()

# VIOLATION: Non-deterministic random
value = random.random()

# VIOLATION: Non-deterministic I/O
files = os.listdir(".")  # Order not guaranteed
```

#### ✅ REQUIRED:

```python
from datetime import datetime
from typing import Callable, Optional

# Deterministic time via injection
def process_with_timestamp(
    data: Dict[str, Any],
    clock: Optional[Callable[[], datetime]] = None
) -> Dict[str, Any]:
    """Process data with injectable timestamp source."""
    if clock is None:
        clock = datetime.now  # Default for production
    
    timestamp = clock()  # Allows deterministic injection in tests
    return {"timestamp": timestamp.isoformat(), "data": data}

# Deterministic random via seeding
import random

def generate_samples(seed: int = 42) -> List[float]:
    """Generate deterministic random samples."""
    rng = random.Random(seed)  # Explicit seed
    return [rng.random() for _ in range(10)]

# Deterministic I/O via sorting
files = sorted(os.listdir("."))  # Guaranteed order
```

### Telemetry Emission

All decision points must emit telemetry:

```python
from infrastructure.telemetry import TelemetryCollector, TelemetryEventType

def analyze_data(data: Dict[str, Any], telemetry: TelemetryCollector) -> Dict[str, Any]:
    """Analyze data with telemetry emission."""
    
    # Emit start event
    telemetry.emit_event(
        event_type=TelemetryEventType.PHASE_START,
        phase_name="analyze_data",
        inputs=data
    )
    
    try:
        # Analysis logic
        result = perform_analysis(data)
        
        # Emit decision event
        telemetry.emit_event(
            event_type=TelemetryEventType.PHASE_DECISION,
            phase_name="analyze_data",
            decision={"analysis_method": "statistical", "confidence": 0.95}
        )
        
        # Emit completion event
        telemetry.emit_event(
            event_type=TelemetryEventType.PHASE_COMPLETION,
            phase_name="analyze_data",
            outputs=result
        )
        
        return result
        
    except Exception as e:
        # Emit error event
        telemetry.emit_event(
            event_type=TelemetryEventType.ERROR_OCCURRED,
            phase_name="analyze_data",
            error=str(e)
        )
        raise
```

---

## Testing Requirements

### Test Coverage

- **Minimum coverage**: 80% for new code
- **Required coverage**: 100% for contract enforcement and determinism

### Test Types

1. **Unit Tests**: Test individual functions in isolation
2. **Integration Tests**: Test module interactions
3. **Determinism Tests**: Prove same input → same output
4. **Contract Tests**: Prove assertions fire on violations

### Example Determinism Test

```python
def test_process_data_determinism():
    """Test that process_data is deterministic."""
    input_data = {"value": 42}
    
    # Fixed clock for deterministic timestamps
    fixed_clock = lambda: datetime(2024, 1, 1, 12, 0, 0)
    
    # Run twice with same inputs
    result1 = process_data(input_data, clock=fixed_clock)
    result2 = process_data(input_data, clock=fixed_clock)
    
    # Results must be identical
    assert result1 == result2, "Function is not deterministic!"
```

### Example Contract Test

```python
def test_calculate_score_contract_violation():
    """Test that calculate_score enforces contract."""
    
    # Should raise on empty list
    with pytest.raises(AssertionError, match="values list cannot be empty"):
        calculate_score([])
    
    # Should raise on out-of-range values
    with pytest.raises(AssertionError, match="all values must be in"):
        calculate_score([0.5, 1.5])  # 1.5 is out of range
```

---

## Documentation Requirements

### Module README

Update if your change affects:
- Determinism requirements
- Contract enforcement
- Auditability mechanisms
- Public API

### CODE_FIX_REPORT.md

**REQUIRED** for all code changes:

```markdown
### File: `module_name.py`

**Date**: 2024-01-15
**Changes**:
- Added deterministic timestamp injection to `process_data()`
- Enhanced contract validation in `calculate_score()`

**SIN_CARRETA Clauses**:
- Primary Rule: Determinism (injectable clock)
- Primary Rule: Contract Enforcement (runtime assertions)
- Auditability: Structured telemetry emission

**Tests**:
- `test_process_data_determinism()` - [link](test_module.py#L42)
- `test_calculate_score_contract_violation()` - [link](test_module.py#L67)

**Rationale**:
Complexity increased by 18 lines to ensure deterministic behavior.
Required for reproducible audit logs per SIN_CARRETA doctrine.
Trade-off: Slightly more verbose code for guaranteed reproducibility.
```

---

## Pull Request Process

### Before Submitting

1. ✅ All tests pass
2. ✅ CI validation passes (determinism, contracts, telemetry)
3. ✅ Documentation updated
4. ✅ CODE_FIX_REPORT.md updated
5. ✅ Commit messages follow conventional format
6. ✅ SIN_CARRETA compliance verified

### PR Template

Your PR description should include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Refactoring (includes SIN_CARRETA rationale)
- [ ] Documentation update

## SIN_CARRETA Compliance
- [ ] No new time/random operations without deterministic injection
- [ ] All public functions have type hints and docstrings
- [ ] Contract checks preserved or enhanced
- [ ] Decision points emit telemetry
- [ ] Tests prove determinism
- [ ] CODE_FIX_REPORT.md updated

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Determinism tests included
- [ ] Contract tests included

## Complexity Changes
- [ ] N/A - No complexity increase
- [ ] Includes SIN_CARRETA-RATIONALE in commit message
- [ ] Tests validate rationale claims

## Related Issues
Closes #123
```

### Review Process

1. **Automated Checks**: CI must pass (determinism, contracts, coverage)
2. **Human Review**: At least one approval required
3. **Special Approvals**:
   - Complexity increases: Require `sin-carreta/approver` label
   - Determinism changes: Require `determinism-verified` label
   - API changes: Require `contract-enforced` label

### After Approval

- PR will be merged by maintainers
- Squash merge preferred for clean history
- Commit message will include SIN_CARRETA compliance summary

---

## Determinism and Contracts

### Injectable Dependencies

All external dependencies must be injectable:

```python
# Good: Injectable dependencies
def process_file(
    file_path: str,
    file_reader: Callable[[str], str] = None,
    clock: Callable[[], datetime] = None
) -> Dict[str, Any]:
    """Process file with injectable dependencies."""
    if file_reader is None:
        file_reader = lambda p: open(p).read()
    if clock is None:
        clock = datetime.now
    
    content = file_reader(file_path)
    timestamp = clock()
    return {"content": content, "timestamp": timestamp.isoformat()}

# Test with deterministic mocks
def test_process_file_deterministic():
    mock_reader = lambda p: "test content"
    mock_clock = lambda: datetime(2024, 1, 1, 12, 0, 0)
    
    result = process_file("test.txt", file_reader=mock_reader, clock=mock_clock)
    assert result == {"content": "test content", "timestamp": "2024-01-01T12:00:00"}
```

### Immutable Data Structures

Prefer immutable data structures:

```python
from typing import NamedTuple

# Good: Immutable dataclass
class PhaseResult(NamedTuple):
    phase_name: str
    inputs: dict
    outputs: dict
    status: str
    
    def validate_contract(self) -> None:
        """Validate contract compliance."""
        assert self.phase_name, "Contract violation: phase_name cannot be empty"
        assert self.status in ["success", "error"], f"Invalid status: {self.status}"
```

---

## CI/CD Requirements

### Automated Checks

All PRs must pass:

1. **Linting**: `flake8`, `mypy` for type checking
2. **Tests**: All pytest tests must pass
3. **Coverage**: Minimum 80% coverage for new code
4. **Determinism Validation**: `ci_contract_enforcement.py`
5. **Telemetry Validation**: `ci_telemetry_validation.py`
6. **Contract Enforcement**: Assertion checks present

### CI Configuration

See `.github/workflows/sin-carreta-enforcement.yml` for full CI configuration.

---

## Questions?

- **Technical Questions**: Open an issue with `question` label
- **SIN_CARRETA Clarifications**: Reference [SIN_CARRETA_RULES.md](SIN_CARRETA_RULES.md)
- **Architecture Discussions**: Open an issue with `architecture` label

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

**Thank you for contributing to FARFAN 2.0 with determinism, contracts, and auditability!** 🎯
