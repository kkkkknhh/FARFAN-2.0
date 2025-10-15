# CODE FIX REPORT

## Purpose

This document tracks all contract, determinism, and audit changes in the FARFAN 2.0 codebase. Any change that modifies contract boundaries, removes assertions, alters telemetry/audit logging, or affects deterministic behavior MUST be documented here with full rationale.

## SIN_CARRETA Doctrine

The "SIN_CARRETA" (without shortcuts) doctrine enforces:

1. **NO silent best-effort code** - All failures must be explicit
2. **NO magic or implicit fallbacks** - All behavior must be deterministic and documented
3. **NO removal of contract checks** - Unless replaced with stronger alternatives
4. **NO ambiguity** - All code must have clear, verifiable behavior

## Rationale Documentation Format

When documenting a change that requires SIN_CARRETA-RATIONALE, use this format:

```markdown
### [DATE] - [COMPONENT] - [CHANGE TYPE]

**Author**: [Name]
**PR**: #[number]
**Reviewers**: [sin-carreta/approver names]

**Change Description**:
[Brief description of what was changed]

**Rationale**:
[Detailed explanation of WHY the change was necessary]

**Replaced With**:
[What stronger alternative was implemented, if applicable]

**Tests Added**:
[List of test files and test cases added to validate the change]

**Impact Analysis**:
- Determinism: [Preserved/Enhanced/N/A]
- Audit Trail: [Preserved/Enhanced/N/A]
- Contract Validation: [Preserved/Enhanced/Replaced]
- Cognitive Complexity: [Increased/Decreased/Neutral]

**Verification**:
- [ ] All CI checks pass
- [ ] Contract checker validates change
- [ ] Tests demonstrate correctness
- [ ] Documentation updated
```

## Change Log

### 2025-10-15 - CI Enforcement - Initial Implementation

**Author**: GitHub Copilot
**PR**: TBD
**Reviewers**: Pending

**Change Description**:
Implemented comprehensive CI and review enforcement gates to block merges on:
- Ambiguity in orchestrator phases
- Contract removal without rationale
- Mediocre/non-SOTA design patterns

**Rationale**:
Enforce maximum non-ambiguity and anti-mediocrity principles per issue requirements. Prevent silent error handling, simplification, magic, or fallback logic from entering the codebase.

**Replaced With**:
N/A (new functionality)

**Tests Added**:
- `test_ci_contract_enforcement.py` - Tests for contract checker
- `test_git_diff_analyzer.py` - Tests for diff analyzer

**Impact Analysis**:
- Determinism: Enhanced (enforces explicit contracts)
- Audit Trail: Enhanced (blocks removal of audit code)
- Contract Validation: Enhanced (automatic validation)
- Cognitive Complexity: Increased (intentional - complexity required for safety)

**Verification**:
- [ ] All CI checks pass
- [ ] Contract checker validates orchestrator
- [ ] Tests demonstrate enforcement
- [ ] Documentation complete

---

## Guidelines for Future Changes

### When to Document Here

Document changes that:

1. **Modify Contract Boundaries**
   - Change PhaseResult structure
   - Alter function signatures in orchestrator
   - Change calibration constant values
   - Modify audit log format

2. **Remove or Replace Assertions**
   - Remove `assert` statements
   - Remove `raise` statements
   - Change validation logic
   - Remove contract checks

3. **Alter Audit/Telemetry**
   - Remove audit logging calls
   - Remove metrics collection
   - Change audit log structure
   - Modify telemetry granularity

4. **Change Async/Sync Behavior**
   - Convert async to sync functions
   - Convert sync to async functions
   - Change concurrency patterns
   - Modify worker pool behavior

5. **Increase Cognitive Complexity**
   - Add new orchestrator phases
   - Introduce new dependencies
   - Change phase ordering
   - Add new contract requirements

### Approval Process

1. **Developer**: Document change with SIN_CARRETA-RATIONALE
2. **CI**: Validate change meets contract requirements
3. **Reviewer**: sin-carreta/approver must review and approve
4. **Merge**: Only after CI passes and approver confirms

### Red Flags (Automatic Rejection)

Changes will be automatically rejected if they:

- ❌ Remove assertions without replacement
- ❌ Add silent try-except blocks without logging
- ❌ Use "magic numbers" or undocumented constants
- ❌ Simplify error handling to "best effort"
- ❌ Remove telemetry without explanation
- ❌ Skip tests for critical functionality
- ❌ Introduce non-deterministic behavior

### Examples of Good Rationale

**Example 1: Replacing Assertion**
```markdown
SIN_CARRETA-RATIONALE: Removed direct assertion in _extract_statements()

Old: assert len(statements) > 0, "No statements found"
New: Validation moved to PhaseResult contract check with explicit error code

Rationale: Assertion was too fragile (empty documents are valid).
Replaced with contract-based validation that returns structured error
in PhaseResult.outputs["validation_status"] with error code "EMPTY_INPUT".

Tests: test_orchestrator.py::test_empty_document_handling
```

**Example 2: Removing Telemetry**
```markdown
SIN_CARRETA-RATIONALE: Removed redundant metrics call in loop

Old: for statement in statements: metrics.record("statement.processed")
New: Single metrics.record("statements.batch_count", len(statements))

Rationale: Per-statement metrics caused performance degradation in
large documents (10K+ statements). Replaced with batch-level metrics
that preserve observability while improving performance by 40%.

Tests: test_metrics_performance.py::test_batch_metrics_performance
Benchmark: Shows no loss of observability, 40% faster processing
```

## Review Checklist for sin-carreta/approvers

When reviewing changes documented here:

- [ ] Rationale is clear and justified
- [ ] Replacement code is stronger or equivalent
- [ ] Tests validate new behavior
- [ ] No silent failures introduced
- [ ] Determinism preserved
- [ ] Audit trail preserved
- [ ] Documentation updated
- [ ] CI checks all pass

## Contact

Questions about SIN_CARRETA-RATIONALE requirements:
- Review `.github/copilot-instructions.md`
- Review `CONTRIBUTING.md`
- Consult with sin-carreta/approver team
