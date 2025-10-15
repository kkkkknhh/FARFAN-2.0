# GitHub Copilot Instructions for FARFAN 2.0 Orchestration

## Core Orchestration Principles

### 1. Sequential Phase Execution

All analytical modules MUST execute in this strict order:

1. **extract_statements** → Extract policy statements from text
2. **detect_contradictions** → Detect contradictions across statements
3. **analyze_regulatory_constraints** → Analyze regulatory compliance
4. **calculate_coherence_metrics** → Calculate coherence scores
5. **generate_audit_summary** → Generate audit summary
6. **compile_final_report** → Compile unified report

**Never** reorder these phases. Each phase depends on outputs from previous phases.

### 2. Calibration Constants (Mathematical Invariants)

These constants MUST remain stable across all runs unless explicitly overridden:

```python
COHERENCE_THRESHOLD = 0.7
CAUSAL_INCOHERENCE_LIMIT = 5
REGULATORY_DEPTH_FACTOR = 1.3
CRITICAL_SEVERITY_THRESHOLD = 0.85
HIGH_SEVERITY_THRESHOLD = 0.70
MEDIUM_SEVERITY_THRESHOLD = 0.50
EXCELLENT_CONTRADICTION_LIMIT = 5
GOOD_CONTRADICTION_LIMIT = 10
```

**Rules:**
- Never hardcode different values in individual modules
- Always reference these from the orchestrator's calibration dictionary
- Only override through the orchestrator's initialization parameters
- Document any override with explicit justification

### 3. Data Flow Contracts

Every analytical phase MUST return a `PhaseResult` with this signature:

```python
{
  "phase_name": str,           # e.g., "detect_contradictions"
  "inputs": {...},             # All inputs to this phase
  "outputs": {...},            # All outputs from this phase
  "metrics": {...},            # Quantitative metrics
  "timestamp": str,            # ISO format timestamp
  "status": str,               # "success" or "error"
  "error": Optional[str]       # Error message if status is "error"
}
```

**Rules:**
- Never omit any of these fields
- Always use ISO 8601 format for timestamps
- Store all phase outputs under explicit keys (no generic names)
- Never merge results in a way that overwrites another phase's data

### 4. Output Key Naming Convention

Phase outputs MUST be stored under explicit, non-conflicting keys:

- **Regulatory Analysis**: `d1_q5_regulatory_analysis`
- **Audit Summary**: `harmonic_front_4_audit`
- **Coherence Metrics**: `coherence_metrics`
- **Contradictions**: `contradictions`
- **Statements**: `statements`

**Rules:**
- Never use generic keys like `results`, `data`, or `output`
- Each phase gets its own namespace in the global report
- Use descriptive prefixes (e.g., `d1_q5_`, `harmonic_front_4_`)

### 5. Error Handling and Fallbacks

When a phase fails:

1. **Log the error** with full context
2. **Generate fallback values** (empty lists, zero metrics, etc.)
3. **Add cause flag** to indicate fallback was used
4. **Continue pipeline** unless critical dependency is missing

Example fallback:

```python
if phase_result.status == "error":
    logger.warning(f"Phase {phase_name} failed: {phase_result.error}")
    # Use fallback values
    outputs = {
        "contradictions": [],
        "fallback_used": True,
        "fallback_reason": phase_result.error
    }
```

**Never:**
- Silently fail and return None
- Stop the entire pipeline for non-critical errors
- Modify or rename output keys implicitly

### 6. Audit Logging

Every phase MUST append its results to the immutable audit log:

```python
self._append_audit_log(phase_result)
```

Audit logs MUST persist to `/logs/orchestrator/` with format:

```
audit_log_{plan_name}_{timestamp}.json
```

**Rules:**
- Log immediately after phase completes
- Never modify logs after creation (immutable)
- Include full calibration constants in log
- Timestamp in ISO 8601 format

### 7. Refactoring Guidelines

When modifying analytical modules:

1. **Preserve function signatures** - existing hooks must remain stable
2. **Extract pure functions** - separate I/O from computation
3. **Maintain backward compatibility** - don't break existing callers
4. **Document dependencies** - explicit input requirements

**Protected hooks:**
- `_analyze_regulatory_constraints()`
- `_calculate_advanced_coherence_metrics()`
- `_detect_contradictions()`
- `_extract_policy_statements()`
- `_generate_audit_summary()`

**Never:**
- Collapse multiple phases into one monolithic function
- Remove or rename these methods without updating orchestrator
- Add new required dependencies without fallbacks

### 8. Deterministic Behavior Requirements

All analytical phases MUST be deterministic:

1. **No random seeds** without explicit setting
2. **Stable sorting** for any ordering operations
3. **Fixed iteration order** over dictionaries (use OrderedDict or sorted())
4. **No timestamp-based logic** except for logging

**Validation:**
- Running the same input twice MUST produce identical outputs
- Metrics MUST be reproducible to 6 decimal places
- Phase ordering MUST never change

### 9. Pre-Commit Verification Checklist

Before committing any orchestrator changes, verify:

- [ ] No phase dependency cycles exist
- [ ] All phases use calibration constants from orchestrator
- [ ] Each function produces deterministic outputs
- [ ] Phase results follow the PhaseResult contract
- [ ] Audit log is generated for test run
- [ ] No hardcoded thresholds in individual modules
- [ ] orchestrator.py compiles without errors
- [ ] Dependency validation passes (`verify_phase_dependencies()`)

Run verification:

```bash
python orchestrator.py
```

Expected output:
```
✓ Orchestrator validation PASSED - no dependency cycles detected
```

### 10. Integration with Existing Modules

When integrating the orchestrator with existing modules (e.g., `contradiction_deteccion.py`):

1. **Import the module** in orchestrator
2. **Wrap calls** in PhaseResult structure
3. **Pass calibration constants** as parameters
4. **Handle errors** gracefully with fallbacks
5. **Log everything** to audit trail

Example integration:

```python
from contradiction_deteccion import ContradictionDetector

def _detect_contradictions(self, statements, text, plan_name, dimension):
    try:
        detector = ContradictionDetector(
            coherence_threshold=self.calibration["coherence_threshold"]
        )
        results = detector.detect(text, plan_name, dimension)
        
        return PhaseResult(
            phase_name="detect_contradictions",
            inputs={"statements_count": len(statements)},
            outputs={"contradictions": results["contradictions"]},
            metrics={"total": len(results["contradictions"])},
            timestamp=datetime.now().isoformat(),
            status="success"
        )
    except Exception as e:
        return PhaseResult(
            phase_name="detect_contradictions",
            inputs={"statements_count": len(statements)},
            outputs={"contradictions": []},
            metrics={"total": 0},
            timestamp=datetime.now().isoformat(),
            status="error",
            error=str(e)
        )
```

### 11. Metrics and Quality Grading

Quality grades MUST be determined using calibration constants:

```python
# For contradictions
if total_contradictions < EXCELLENT_CONTRADICTION_LIMIT:
    quality_grade = "Excelente"
elif total_contradictions < GOOD_CONTRADICTION_LIMIT:
    quality_grade = "Bueno"
else:
    quality_grade = "Regular"

# For coherence
if coherence_score >= COHERENCE_THRESHOLD:
    quality_grade = "Cumple"
else:
    quality_grade = "No cumple"

# For causal incoherence
if causal_incoherence_count < CAUSAL_INCOHERENCE_LIMIT:
    status = "Aceptable"
else:
    status = "Requiere revisión"
```

### 12. Documentation Requirements

All orchestrator changes MUST include:

1. **Docstrings** for new functions (Google style)
2. **Type hints** for all parameters and returns
3. **Comments** for complex logic (not obvious code)
4. **Changelog entry** in docstring header if significant

Example:

```python
def _calculate_coherence_metrics(
    self,
    contradictions: List[Any],
    statements: List[Any],
    text: str
) -> PhaseResult:
    """
    Calculate advanced coherence metrics for policy document.
    
    Applies COHERENCE_THRESHOLD calibration constant to determine
    if document meets quality standards.
    
    Args:
        contradictions: List of detected contradictions
        statements: List of extracted policy statements
        text: Full document text
        
    Returns:
        PhaseResult with coherence metrics and quality grade
        
    Raises:
        ValueError: If statements list is empty
    """
```

### 13. Testing Orchestration

When testing the orchestrator:

1. **Test with minimal input** - verify phases execute in order
2. **Test with errors** - verify fallback mechanisms work
3. **Test determinism** - run twice, compare outputs
4. **Test calibration** - verify constants are used
5. **Test audit logs** - verify logs are created and immutable

Example test structure:

```python
def test_orchestrator_determinism():
    orchestrator = create_orchestrator()
    text = "Sample policy text"
    
    result1 = orchestrator.orchestrate_analysis(text, "PDM_Test", "estratégico")
    result2 = orchestrator.orchestrate_analysis(text, "PDM_Test", "estratégico")
    
    assert result1 == result2, "Orchestrator must be deterministic"
```

### 14. Common Pitfalls to Avoid

**Don't:**
- ❌ Skip phase dependency validation
- ❌ Hardcode thresholds in module functions
- ❌ Use mutable default arguments
- ❌ Modify audit logs after creation
- ❌ Return None on errors (use fallback values)
- ❌ Use global state for phase results
- ❌ Assume phase order is flexible
- ❌ Mix different timestamp formats

**Do:**
- ✅ Always validate inputs before processing
- ✅ Use calibration constants from orchestrator
- ✅ Return structured PhaseResult objects
- ✅ Log all phase transitions
- ✅ Handle errors gracefully with fallbacks
- ✅ Maintain immutable audit trail
- ✅ Follow strict phase ordering
- ✅ Use ISO 8601 timestamps consistently

### 15. Orchestrator Workflow Summary

```
Input: text, plan_name, dimension
  ↓
Phase 1: Extract Statements
  ↓ (outputs: statements)
Phase 2: Detect Contradictions
  ↓ (outputs: contradictions, temporal_conflicts)
Phase 3: Analyze Regulatory Constraints
  ↓ (outputs: d1_q5_regulatory_analysis)
Phase 4: Calculate Coherence Metrics
  ↓ (outputs: coherence_metrics)
Phase 5: Generate Audit Summary
  ↓ (outputs: harmonic_front_4_audit)
Phase 6: Compile Final Report
  ↓
Output: Unified structured report + Audit log
```

Each arrow represents a data dependency. Each phase appends to the audit log before proceeding.

---

## Quick Reference Card

| Aspect | Requirement |
|--------|-------------|
| **Phase Order** | Fixed: extract → detect → analyze → calculate → audit → compile |
| **Calibration** | Use constants from orchestrator.calibration dict |
| **Return Type** | PhaseResult dataclass with 6 required fields |
| **Error Handling** | Log + fallback + continue (never silent fail) |
| **Audit Logs** | Persist to logs/orchestrator/ after completion |
| **Output Keys** | Explicit namespaced keys (e.g., d1_q5_*, harmonic_front_4_*) |
| **Determinism** | Same input → same output (no random/timestamp logic) |
| **Timestamps** | ISO 8601 format only |
| **Quality Grades** | Based on calibration constants, not hardcoded |
| **Dependencies** | Validated with verify_phase_dependencies() |

---

## Support and Questions

For questions about orchestration:
1. Review this document first
2. Check orchestrator.py implementation
3. Validate with verify_phase_dependencies()
4. Review audit logs for execution traces

For bugs or improvements:
1. Verify against checklist in section 9
2. Ensure deterministic behavior
3. Add tests for new functionality
4. Update this document if rules change
