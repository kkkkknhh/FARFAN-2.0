# SIN_CARRETA Doctrine - Normative Rules for FARFAN 2.0

**Version**: 1.0  
**Status**: NORMATIVE - MANDATORY ENFORCEMENT  
**Last Updated**: 2025-10-16

---

## 📜 Primary Rule — Determinism & Contracts

**All code edits must preserve or improve determinism, contract clarity, and auditability.**

### Hard Requirements

1. **Determinism is Non-Negotiable**
   - Any edit that introduces or preserves non-determinism (time, randomness, uncontrolled I/O) is **FORBIDDEN**
   - Exception: When accompanied by deterministic stubs/mocks and corresponding tests
   - All time-dependent code must use injectable clock/timestamp sources
   - All random operations must use seeded RNGs with explicit seed management

2. **Contract Enforcement**
   - Every public function must have explicit preconditions and postconditions
   - Use assertions, decorators, or type hints to enforce contracts
   - Runtime validation is preferred over documentation-only contracts
   - Contract violations must be detectable by automated tools

3. **Auditability**
   - Every decision point must emit structured telemetry
   - Audit logs must be immutable and versioned
   - All state transitions must be traceable
   - No silent failures - every error must be logged with context

---

## 🧠 On Cognitive Complexity Refactors

**Cognitive Complexity is a tool, not an objective.**

### Refactoring Rules

1. **Preserve Explicit Logic**
   - Do NOT reduce complexity by collapsing explicit, contract-rich logic into opaque "simpler" constructs
   - Explicit guards, traces, and failure semantics must be preserved
   - Clarity and correctness trump brevity

2. **Forbidden Refactoring Patterns**
   - ❌ Removing assertions or contract checks without replacement
   - ❌ Converting sync to async (or vice versa) without test updates
   - ❌ Collapsing error handling into generic catch-all blocks
   - ❌ Hiding control flow in decorators without explicit documentation
   - ❌ Removing telemetry or observability hooks

3. **Acceptable Complexity Increases**
   - ✅ Adding assertions and runtime contract checks
   - ✅ Adding structured telemetry at decision points
   - ✅ Adding explicit error handling with recovery semantics
   - ✅ Adding deterministic stubs for external dependencies
   - ✅ Adding validation layers

### When Increasing Complexity is Required

If improving determinism, observability, or contractual integrity requires increased complexity:

1. **Add a rationale comment** (1-3 lines) in the code
2. **Reference SIN_CARRETA clause** in commit message
3. **Include contract checks** (assertion or decorator) for each new element
4. **Emit structured telemetry** for every decision point
5. **Add tests** proving determinism and contract enforcement

**Example Rationale Comment:**
```python
# SIN_CARRETA: Explicit contract enforcement added to prevent silent failures
# Increases complexity but ensures auditability and deterministic error handling
assert input_data is not None, "Contract violation: input_data cannot be None"
```

---

## 🤖 Automated Enforcement (CI Rules)

### Build Must Fail If:

1. **Contract Violations**
   - Any code change removes assertions or contract checks without replacement
   - Functions lack type hints for parameters and return values
   - Public APIs lack docstrings with preconditions/postconditions

2. **Determinism Violations**
   - `time.time()`, `datetime.now()`, or `random.*` used without deterministic injection
   - Async functions converted to sync (or vice versa) without corresponding test updates
   - Missing determinism tests for modules touching time or randomness

3. **Auditability Violations**
   - State transitions without telemetry emission
   - Error handling without structured logging
   - Missing audit trail for critical operations

4. **Test Coverage**
   - New code lacks corresponding tests
   - Determinism tests are missing for time/random operations
   - Contract enforcement tests are missing

### Build May Pass If:

Cognitive Complexity increases but the commit contains:

1. **SIN_CARRETA-RATIONALE section** in commit body explaining:
   - Why complexity increased
   - Which SIN_CARRETA clauses it satisfies
   - Trade-offs considered

2. **Tests validating**:
   - Determinism (fixed clocks, seeded RNGs)
   - Contract enforcement (assertions, type checks)
   - Audit traces (telemetry events)

---

## 👥 Reviewer Gate

### Approval Requirements

1. **Complexity Increases**
   - Any change that intentionally increases cognitive complexity must be explicitly approved
   - Reviewer must verify SIN_CARRETA-RATIONALE is present and valid
   - Reviewer must confirm tests prove determinism and contracts

2. **Required Reviewer Labels**
   - `sin-carreta/approver` - For complexity-increasing changes
   - `determinism-verified` - For changes affecting time/randomness
   - `contract-enforced` - For changes to public APIs

3. **CI Gate**
   - Merges require `sin-carreta/approver` label for complexity increases
   - Automated checks must pass before human review
   - No overrides permitted for determinism violations

---

## 📝 Documentation & Traceability

### Module Documentation

1. **Every code change must update**:
   - Module README (if it affects determinism, contracts, or auditability)
   - CONTRIBUTING.md (if it adds new patterns or requirements)
   - Inline comments for complex contract logic

2. **CODE_FIX_REPORT.md requirements**:
   - Per-file entries for all changes
   - What changed (specific functions/classes)
   - SIN_CARRETA clauses satisfied
   - Link to tests proving determinism
   - Rationale for complexity changes

### Example CODE_FIX_REPORT.md Entry:

```markdown
### File: `orchestrator.py`

**Changes**:
- Added deterministic timestamp injection to `_calculate_coherence_metrics()`
- Added contract validation in `PhaseResult.validate_contract()`

**SIN_CARRETA Clauses**:
- Primary Rule: Determinism (injectable clock)
- Primary Rule: Contract Enforcement (runtime validation)
- Auditability: Structured telemetry emission

**Tests**:
- `test_orchestrator_deterministic_timestamps()` - [line 142](test_orchestrator_auditability.py#L142)
- `test_phase_result_contract_validation()` - [line 89](test_orchestrator_auditability.py#L89)

**Rationale**:
Complexity increased by 12 lines to ensure deterministic behavior across test runs.
Required for reproducible audit logs per SIN_CARRETA doctrine.
```

---

## 🚫 Hard Refusal Clause

### Automatic Rejection

**If an automated fix would convert explicit failure semantics into warnings or silent "best-effort" behavior:**

1. **The fix must be ABORTED immediately**
2. **A blocking comment must be created** on the PR explaining:
   - Which SIN_CARRETA clause was violated
   - Why the conversion is forbidden
   - What the correct approach should be

3. **CI must block the merge** until violation is resolved

### Examples of Forbidden Conversions:

❌ **FORBIDDEN**:
```python
# Before: Explicit failure
if not validate_input(data):
    raise ValueError("Invalid input data")

# After: Silent best-effort (VIOLATION!)
if not validate_input(data):
    logger.warning("Invalid input data, using defaults")
    data = get_defaults()
```

✅ **ALLOWED**:
```python
# Before: Explicit failure
if not validate_input(data):
    raise ValueError("Invalid input data")

# After: Explicit failure with enhanced contract
assert validate_input(data), f"Contract violation: {data} failed validation"
# ... telemetry emission ...
raise ValueError(f"Invalid input data: {data}")
```

---

## 🔍 Enforcement Checklist

### For Every Code Change

- [ ] No new time/random operations without deterministic injection
- [ ] All new public functions have type hints and docstrings
- [ ] All contract checks are preserved or enhanced
- [ ] All decision points emit telemetry
- [ ] Tests prove determinism (same input → same output)
- [ ] Tests prove contract enforcement (assertions fire on violations)
- [ ] CODE_FIX_REPORT.md updated with SIN_CARRETA compliance
- [ ] Commit message references SIN_CARRETA clause if complexity increased
- [ ] No silent failures introduced
- [ ] Audit trails are immutable and traceable

### For Reviewers

- [ ] SIN_CARRETA-RATIONALE present if complexity increased
- [ ] Tests validate all claims in rationale
- [ ] No forbidden conversion patterns detected
- [ ] Documentation updated appropriately
- [ ] Appropriate labels applied (`sin-carreta/approver`, etc.)

---

## 📚 References

- **Copilot Organization Instructions**: [GitHub Docs](https://docs.github.com/en/copilot/customizing-copilot/adding-organization-custom-instructions-for-github-copilot)
- **FARFAN 2.0 Orchestrator Principles**: [copilot-instructions.md](.github/copilot-instructions.md)
- **Auditability Implementation**: [CODE_FIX_REPORT.md](CODE_FIX_REPORT.md)

---

## ⚖️ Governance

This document is **normative** and takes precedence over any conflicting guidance.

- **Effective Date**: 2025-10-16
- **Scope**: All code in FARFAN-2.0 repository
- **Enforcement**: CI/CD pipeline + human reviewers
- **Updates**: Require approval from repository maintainers

**Violations of SIN_CARRETA doctrine will result in PR rejection and required rework.**

---

**SIN_CARRETA**: **S**tructured **I**ntegrity, **N**o **C**areless **A**lterations, **R**igorous **R**ules for **E**xplicit **T**racing, **T**esting, and **A**uditability

© 2024 FARFAN-2.0 - Determinism, Contracts, and Auditability as First-Class Citizens
