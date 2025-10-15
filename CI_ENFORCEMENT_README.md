# CI Contract Enforcement System

## Overview

This directory contains the CI contract enforcement system for FARFAN 2.0. It implements automated gates that **BLOCK MERGE** on violations of our SIN_CARRETA (without shortcuts) doctrine.

## What Gets Enforced

### 1. Orchestrator Contract Validation
**Script**: `ci_orchestrator_contract_checker.py`

Ensures all orchestrator phases:
- ✅ Have explicit assertions or contract checks
- ✅ Return structured PhaseResult objects
- ✅ Include audit logging calls
- ✅ Have corresponding test files for async functions

**Blocks merge if**:
- Phase method lacks assertions/validations
- Phase returns non-PhaseResult data
- Orchestrator file has no audit logging
- Async function has no test file

### 2. Git Diff Contract Analysis
**Script**: `ci_git_diff_contract_analyzer.py`

Detects removal of critical code patterns:
- ❌ Assertions (`assert`, `raise`)
- ❌ Contract validation calls
- ❌ Audit logging
- ❌ Telemetry/metrics

**Blocks merge if**:
- Critical code removed WITHOUT `SIN_CARRETA-RATIONALE`

**Allows merge if**:
- Rationale present in commit message or code comments
- Stronger alternative documented
- Tests demonstrate correctness

### 3. Governance Standards Tests
**Script**: `ci_contract_enforcement.py`

Runs methodological gates:
- Hoop test failure detection
- Posterior distribution bounds
- Mechanism prior decay
- Execution isolation validation
- Immutable audit log verification
- Human-in-the-loop gate triggers

### 4. Cognitive Complexity Check
**Script**: `ci_cognitive_complexity_checker.py`

Flags high-complexity functions:
- ⚠️ Functions exceeding complexity threshold
- 📊 Requires sin-carreta/approver review if intentional

**Does NOT block merge**, but requires review.

## GitHub Actions Workflow

**File**: `.github/workflows/main.yml`

Runs on every PR and push to main/develop:

```yaml
1. Checkout code (with full history)
2. Set up Python 3.11
3. Install dependencies
4. Run Orchestrator Contract Checker → BLOCKS on violation
5. Run Git Diff Contract Analyzer → BLOCKS on removal without rationale
6. Run Governance Standards Tests → BLOCKS on test failure
7. Run Cognitive Complexity Check → WARNING only
8. Check sin-carreta/approver label → BLOCKS if required and missing
9. Comment PR with violations → Automatic feedback
```

## Review Requirements

**File**: `.github/CODEOWNERS`

Certain files require sin-carreta/approver review:

- `orchestrator.py`, `orchestrator_*.py`
- `infrastructure/` directory
- `ci_*.py` enforcement scripts
- `governance_standards.py`
- Calibration constants
- Audit/observability infrastructure
- GitHub Actions workflows
- Documentation files

## Documentation

### CODE_FIX_REPORT.md
Tracks all contract, determinism, and audit changes with:
- Rationale for changes
- Replacement code
- Test references
- Impact analysis

### CONTRIBUTING.md
Developer guide with:
- SIN_CARRETA doctrine principles
- Development workflow
- Contract enforcement rules
- Testing requirements
- Review process

## How to Use Locally

### Run All Checks Before Pushing

```bash
# Contract validation
python ci_orchestrator_contract_checker.py

# Git diff analysis (if you have changes)
python ci_git_diff_contract_analyzer.py

# Governance tests
python ci_contract_enforcement.py

# Complexity check
python ci_cognitive_complexity_checker.py

# Run all tests
python -m unittest discover -p "test_*.py"
```

### Adding SIN_CARRETA-RATIONALE

If you need to remove contract code, document it:

**In commit message**:
```
Refactor validation logic

SIN_CARRETA-RATIONALE: Moved assertion from _extract_statements()
to base Validator class. Provides stronger contract validation
with explicit error codes and structured exceptions.

Replaced: assert len(statements) > 0
With: BaseValidator.validate_non_empty(statements, error_code="EMPTY_STATEMENTS")

Tests: test_base_validator.py::test_validate_non_empty
```

**In code comment**:
```python
# SIN_CARRETA-RATIONALE: Contract moved to PhaseResult validation
# Old assertion was too fragile for edge cases
# See PhaseResult.__post_init__() for new validation
# Tests: test_orchestrator.py::test_phase_result_validation
```

### Getting sin-carreta/approver Label

1. Ensure all CI checks pass
2. Document changes in `CODE_FIX_REPORT.md` if required
3. Add comment: `@sin-carreta-approvers please review`
4. Approver will add label if approved

## Testing the Enforcement System

```bash
# Run enforcement system tests
python -m unittest test_ci_contract_enforcement -v

# Expected: 11 tests, all passing
```

## Exit Codes

All enforcement scripts use standard exit codes:

- **0**: All checks passed, merge allowed
- **1**: Violations detected, BLOCK MERGE

## Hard Refusal Clause

The enforcement system implements a **Hard Refusal Clause**:

> Any PR introducing ambiguity, removing contract checks, or following mediocre patterns will be automatically blocked. No exceptions, no shortcuts.

Violations result in:
1. ❌ CI check failure
2. 🚫 Merge blocked
3. 💬 Automatic PR comment with violation details
4. 📋 Required actions documented

## Maintenance

### Adding New Checks

1. Create new checker script: `ci_your_check.py`
2. Add to workflow: `.github/workflows/main.yml`
3. Add tests: `test_ci_your_check.py`
4. Update this README

### Modifying Thresholds

Threshold changes require sin-carreta/approver review and documentation in `CODE_FIX_REPORT.md`.

### Disabling Checks

**DO NOT disable enforcement checks**. If a check is broken:
1. Fix the check
2. If truly necessary, mark as `continue-on-error: true` temporarily
3. Create issue to fix
4. Document in `CODE_FIX_REPORT.md` with deadline

## Philosophy

This enforcement system embodies the SIN_CARRETA doctrine:

- **NO silent best-effort code** → Explicit assertions required
- **NO magic or fallbacks** → All behavior documented
- **NO contract removal** → Rationale required
- **NO ambiguity** → Clear, verifiable code only

## Support

Questions about enforcement:
1. Review `.github/copilot-instructions.md`
2. Review `CONTRIBUTING.md`
3. Review `CODE_FIX_REPORT.md`
4. Open issue with `question` label
5. Contact sin-carreta/approver team

## License

Part of FARFAN 2.0, MIT License.
