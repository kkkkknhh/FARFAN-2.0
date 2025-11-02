# Contributing to FARFAN 2.0

## Welcome

Thank you for considering contributing to FARFAN 2.0! This document provides guidelines for contributing to the project, with special emphasis on our **SIN_CARRETA** (without shortcuts) doctrine.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [SIN_CARRETA Doctrine](#sin_carreta-doctrine)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Contract Enforcement](#contract-enforcement)
- [Review Process](#review-process)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)

## Code of Conduct

This project follows a strict **anti-mediocrity** principle. We value:

- **Precision over Speed**: Correct, deterministic code over quick fixes
- **Explicitness over Magic**: Clear, verbose code over clever shortcuts
- **Safety over Convenience**: Contract checks and assertions over silent failures
- **Evidence over Assumptions**: Tests and metrics over gut feelings

## SIN_CARRETA Doctrine

### Core Principles

The SIN_CARRETA (without shortcuts) doctrine is our foundational philosophy:

1. **NO Silent Best-Effort Code**
   - All errors must be explicit and logged
   - No try-except blocks without proper error handling
   - No fallback to default values without rationale

2. **NO Magic or Implicit Fallbacks**
   - All behavior must be deterministic and documented
   - No undocumented side effects
   - No "magic numbers" or unexplained constants

3. **NO Removal of Contract Checks**
   - Assertions, validations, and contract checks cannot be removed
   - Must be replaced with stronger alternatives if changed
   - Requires SIN_CARRETA-RATIONALE documentation

4. **NO Ambiguity**
   - All code must have clear, verifiable behavior
   - Function signatures must be explicit
   - Return types must be documented

### What This Means in Practice

**✅ Good (SIN_CARRETA Compliant)**
```python
def _extract_statements(self, text: str) -> PhaseResult:
    """Extract policy statements with explicit contract validation."""
    # Explicit input validation
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")
    
    if not text.strip():
        # Explicit handling of edge case with audit trail
        self.audit_logger.append("extract_statements", {
            "status": "empty_input",
            "timestamp": datetime.now().isoformat()
        })
        return PhaseResult(
            phase_name="extract_statements",
            inputs={"text_length": 0},
            outputs={"statements": []},
            metrics={"count": 0},
            timestamp=datetime.now().isoformat(),
            status="success",
            error="Empty input text"
        )
    
    # Process statements
    statements = extract_policy_statements(text)
    
    # Explicit contract validation
    assert isinstance(statements, list), "statements must be list"
    assert all(isinstance(s, dict) for s in statements), "All statements must be dict"
    
    # Record metrics
    self.metrics.record("statements.extracted", len(statements))
    
    # Return structured contract
    return PhaseResult(
        phase_name="extract_statements",
        inputs={"text_length": len(text)},
        outputs={"statements": statements},
        metrics={"count": len(statements)},
        timestamp=datetime.now().isoformat(),
        status="success"
    )
```

**❌ Bad (Violates SIN_CARRETA)**
```python
def _extract_statements(self, text):
    """Extract statements."""
    try:
        # Silent best-effort processing
        statements = extract_policy_statements(text or "")
    except:
        # Silent failure with magic fallback
        statements = []
    
    # No contract validation
    # No metrics
    # No audit trail
    return statements  # Not PhaseResult
```

## Getting Started

### Prerequisites

- Python 3.11+
- Git
- Understanding of industrial software practices

### Installation

1. Clone the repository
```bash
git clone https://github.com/kkkkknhh/FARFAN-2.0.git
cd FARFAN-2.0
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run tests to verify setup
```bash
python -m unittest discover -s . -p "test_*.py"
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Minimal Changes

- Change as few lines as possible
- Never delete/modify working code unless absolutely necessary
- Focus on surgical, precise modifications

### 3. Add Tests

All code changes MUST have corresponding tests:

```python
# test_your_feature.py
import unittest

class TestYourFeature(unittest.TestCase):
    def test_contract_validation(self):
        """Test that contract validation works"""
        result = your_function(valid_input)
        
        # Explicit assertions
        self.assertIsInstance(result, PhaseResult)
        self.assertEqual(result.status, "success")
        self.assertIn("outputs", result.__dict__)
```

### 4. Run CI Checks Locally

Before pushing, run all CI checks:

```bash
# Contract enforcement
python ci_orchestrator_contract_checker.py

# Governance standards
python ci_contract_enforcement.py

# Git diff analysis (if modifying existing code)
python ci_git_diff_contract_analyzer.py

# Run tests
python -m unittest discover
```

### 5. Document Changes

If your change:
- Removes assertions or contract checks
- Modifies audit/telemetry code
- Changes contract boundaries
- Alters async/sync behavior

Then you MUST document it in `CODE_FIX_REPORT.md` with SIN_CARRETA-RATIONALE.

### 6. Create Pull Request

```bash
git add .
git commit -m "Brief description of change

SIN_CARRETA-RATIONALE (if applicable):
Detailed explanation of why contract was changed.
What stronger alternative was implemented.
Reference to tests demonstrating correctness.
"
git push origin feature/your-feature-name
```

## Contract Enforcement

### What CI Checks

Our CI pipeline automatically checks for:

1. **Missing Assertions**: All orchestrator phases must have explicit contract checks
2. **Forbidden Removals**: No removal of assertions, audit logging, or telemetry without rationale
3. **Async/Sync Consistency**: Function changes must have matching test updates
4. **Governance Standards**: All methodological gates must pass

### Automatic Merge Blocking

CI will automatically block merge if:

- ❌ Orchestrator phase omits explicit assertions
- ❌ Code removes contract checks without SIN_CARRETA-RATIONALE
- ❌ Async function converted to sync without test updates
- ❌ Missing sin-carreta/approver label for critical changes

### How to Fix Violations

1. **Review CI logs** for detailed violation reports
2. **Add missing assertions** to phase methods
3. **Document removals** with SIN_CARRETA-RATIONALE in commit message or `CODE_FIX_REPORT.md`
4. **Update tests** for any function signature changes
5. **Request review** from sin-carreta/approver

## Review Process

### Required Reviewers

Changes to the following require sin-carreta/approver review:

- `orchestrator.py` and `orchestrator_*.py`
- `infrastructure/` directory
- `governance_standards.py`
- Calibration constants
- Audit/observability infrastructure
- GitHub Actions workflows

### Approver Checklist

sin-carreta/approvers must verify:

- [ ] Rationale is clear and justified (if applicable)
- [ ] No silent failures introduced
- [ ] Contract validation preserved or enhanced
- [ ] Determinism preserved
- [ ] Audit trail complete
- [ ] Tests demonstrate correctness
- [ ] Documentation updated
- [ ] All CI checks pass

### Getting sin-carreta/approver Label

To request sin-carreta/approver review:

1. Ensure all CI checks pass
2. Document changes in `CODE_FIX_REPORT.md` if required
3. Add comment to PR: `@sin-carreta-approvers please review`
4. Approver will add `sin-carreta/approver` label if approved

## Testing Requirements

### Minimum Test Coverage

- All new functions must have tests
- All contract changes must have validation tests
- All bug fixes must have regression tests

### Test Structure

```python
class TestFeatureName(unittest.TestCase):
    """Test suite for FeatureName"""
    
    def test_happy_path(self):
        """Test normal operation"""
        pass
    
    def test_edge_cases(self):
        """Test edge cases (empty input, null, etc)"""
        pass
    
    def test_error_handling(self):
        """Test error cases raise proper exceptions"""
        pass
    
    def test_contract_validation(self):
        """Test return types and contract structure"""
        pass
```

## Documentation Standards

### Code Comments

- Comment WHY, not WHAT
- Document complex algorithms
- Explain non-obvious edge cases
- Include SIN_CARRETA-RATIONALE for contract changes

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> PhaseResult:
    """
    Brief description of function.
    
    Longer description explaining behavior, edge cases, and contracts.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        PhaseResult with structured outputs
        
    Raises:
        ValueError: If param1 is empty
        TypeError: If param2 is not int
        
    Examples:
        >>> result = function_name("text", 42)
        >>> assert result.status == "success"
    """
```

### Commit Messages

Format:
```
Brief description (50 chars max)

Detailed explanation of WHAT changed and WHY.

SIN_CARRETA-RATIONALE (if applicable):
Explanation of contract change/removal.
Reference to replacement code.
Reference to tests.

Fixes #123
```

## Questions?

- Review `.github/copilot-instructions.md` for orchestration principles
- Review `CODE_FIX_REPORT.md` for rationale documentation format
- Open an issue with question label
- Contact sin-carreta/approver team

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
