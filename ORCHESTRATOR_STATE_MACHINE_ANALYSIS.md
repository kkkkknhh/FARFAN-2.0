# Orchestrator State Machine Analysis and Remediation Plan

## Executive Summary

Analysis of three orchestrators reveals **significant architectural fragmentation** with inconsistent calibration constants, incomplete state machine implementations, missing backpressure mechanisms, and audit trail gaps.

**Critical Findings:**
- ❌ **Calibration Drift**: Constants exist only in `AnalyticalOrchestrator`, missing from PDM/CDAF
- ❌ **Async Inconsistency**: Only PDMOrchestrator uses async/await; others are synchronous
- ⚠️ **Queue Mechanisms**: Only PDM has asyncio.Queue; no standardized backpressure
- ⚠️ **Audit Logging**: Three different implementations with varying SHA-256 provenance levels
- ❌ **Phase Overlaps**: Redundant contradiction detection and regulatory analysis across orchestrators

---

## 1. Orchestrator Comparison Matrix

### 1.1 Core Responsibilities

| Orchestrator | Primary Responsibility | State Machine | Async | Audit Logging |
|--------------|------------------------|---------------|-------|---------------|
| **AnalyticalOrchestrator** (`orchestrator.py`) | Sequential analytical phases (contradiction, regulatory, coherence) | ✓ Enum-based (`AnalyticalPhase`) | ❌ Synchronous | ⚠️ JSON file logs (no SHA-256) |
| **PDMOrchestrator** (`orchestration/pdm_orchestrator.py`) | Phase 0-IV execution with observability | ✓ Enum-based (`PDMAnalysisState`) | ✓ Full async/await | ✓ **ImmutableAuditLogger** with SHA-256 |
| **CDAFFramework** (`dereck_beach`) | Standalone CDAF processing with Bayesian inference | ❌ Implicit (no explicit enum) | ❌ Synchronous | ⚠️ Pydantic validation logs (no SHA-256) |

### 1.2 Calibration Constants Verification

| Constant | AnalyticalOrchestrator | PDMOrchestrator | CDAFFramework |
|----------|------------------------|-----------------|---------------|
| `COHERENCE_THRESHOLD = 0.7` | ✓ Hardcoded | ❌ **MISSING** | ❌ **MISSING** |
| `CAUSAL_INCOHERENCE_LIMIT = 5` | ✓ Hardcoded | ❌ **MISSING** | ❌ **MISSING** |
| `REGULATORY_DEPTH_FACTOR = 1.3` | ✓ Hardcoded | ❌ **MISSING** | ❌ **MISSING** |
| `kl_divergence threshold` | ❌ N/A | ❌ N/A | ✓ Config-based (0.01) |
| `prior_alpha/beta` | ❌ N/A | ❌ N/A | ✓ Config-based (2.0/2.0) |

**VIOLATION**: SIN_CARRETA doctrine requires **mathematical invariants** across all orchestrators. Current state introduces non-determinism through drift.

---

## 2. Detailed State Machine Documentation

### 2.1 AnalyticalOrchestrator State Machine

```
┌──────────────────────────────────────────────────────────────────┐
│  AnalyticalOrchestrator (orchestrator.py)                         │
│  Pattern: Sequential Pipeline with Explicit Phase Enumeration     │
└──────────────────────────────────────────────────────────────────┘

States (AnalyticalPhase Enum):
  ① EXTRACT_STATEMENTS
  ② DETECT_CONTRADICTIONS
  ③ ANALYZE_REGULATORY_CONSTRAINTS
  ④ CALCULATE_COHERENCE_METRICS
  ⑤ GENERATE_AUDIT_SUMMARY
  ⑥ COMPILE_FINAL_REPORT

Phase Transitions:
  INIT → ① → ② → ③ → ④ → ⑤ → ⑥ → DONE
          ↓   ↓   ↓   ↓   ↓   ↓
       [Error: _generate_error_report() → partial_results]

Quality Gates:
  - None explicit (all phases execute regardless of prior failures)
  - ❌ MISSING: Early termination on critical failures

Timeout Configuration:
  - ❌ MISSING: No timeout enforcement

Backpressure Mechanisms:
  - ❌ MISSING: Synchronous execution (no concurrency control)

Metrics Collection:
  - ⚠️ Phase-level: statements_count, contradictions_count, coherence scores
  - ❌ MISSING: MetricsCollector integration

Audit Logging:
  - Append-only _audit_log: List[PhaseResult]
  - Persist to JSON: logs/orchestrator/audit_log_{plan_name}_{timestamp}.json
  - ❌ MISSING: SHA-256 provenance hashing
  - ❌ MISSING: Immutability guarantees (list can be mutated)
```

### 2.2 PDMOrchestrator State Machine

```
┌──────────────────────────────────────────────────────────────────┐
│  PDMOrchestrator (orchestration/pdm_orchestrator.py)              │
│  Pattern: Async Phases with Backpressure and Governance          │
└──────────────────────────────────────────────────────────────────┘

States (PDMAnalysisState Enum):
  INITIALIZED → EXTRACTING → BUILDING_DAG → INFERRING_MECHANISMS
        ↓           ↓             ↓                  ↓
     VALIDATING → FINALIZING → COMPLETED / FAILED

Phase Transitions:
  analyze_plan() entry
    ↓
  [Semaphore acquire + Timeout context]
    ↓
  ① _extract_complete(pdf_path)        # Phase I: Tide Gate
    ↓
  Quality Gate: extraction_quality.score >= min_quality_threshold
    ↓ (pass)                           ↓ (fail)
  ② _build_graph()                   raise DataQualityError
    ↓
  ③ asyncio.gather(                   # Phase III: Parallel Audits
       _infer_all_mechanisms(),
       _validate_complete()
     )
    ↓
  Human Gate: requires_manual_review?
    ↓ (yes: _trigger_manual_review_hold())
  ④ _calculate_quality_score()        # Phase IV: Final Verdict
    ↓
  D6 Alert Gate: d6_score < 0.55?
    ↓ (yes: metrics.alert("CRITICAL"))
  [State → COMPLETED]
    ↓
  ImmutableAuditLogger.append_record(sha256_source, duration, result)

Quality Gates:
  ✓ Extraction quality threshold (min_quality_threshold=0.5)
  ✓ Manual review hold (requires_manual_review flag)
  ✓ D6 dimension alert (d6_score < 0.55)

Timeout Configuration:
  ✓ worker_timeout_secs (default: 300s)
  ✓ asyncio.timeout() context manager
  ✓ Fallback: _handle_timeout() → FAILED state

Backpressure Mechanisms:
  ✓ asyncio.Queue(maxsize=queue_size) [default: 10]
  ✓ asyncio.Semaphore(max_inflight_jobs) [default: 3]
  ✓ Concurrency control: async with self.semaphore

Metrics Collection:
  ✓ MetricsCollector with:
      - record(metric_name, value)
      - increment(counter_name)
      - alert(level, message)
  ✓ Tracked metrics:
      - pipeline.duration_seconds
      - pipeline.timeout_count
      - pipeline.error_count
      - extraction.chunk_count, extraction.table_count
      - graph.node_count, graph.edge_count
      - mechanism.prior_decay_rate
      - evidence.hoop_test_fail_count
      - dimension.avg_score_D6

Audit Logging:
  ✓ ImmutableAuditLogger:
      - SHA-256 file hash (_hash_file)
      - JSONL append-only store
      - Timestamp + duration + final_state
      - ✓ Immutability: records.append() + disk persist
```

### 2.3 CDAFFramework State Machine

```
┌──────────────────────────────────────────────────────────────────┐
│  CDAFFramework (dereck_beach)                                     │
│  Pattern: Implicit State with Functional Pipeline                 │
└──────────────────────────────────────────────────────────────────┘

States (Implicit - no explicit enum):
  [INIT: ConfigLoader validation]
    ↓
  [PDF Extraction: PDFExtractor.extract_complete()]
    ↓
  [Semantic Chunking: SemanticChunker.chunk_document()]
    ↓
  [Causal Graph: CausalGraphBuilder.build_from_chunks()]
    ↓
  [Bayesian Inference: BayesianInferenceEngine.infer_mechanisms()]
    ↓
  [Audit Execution: Execute audit framework]
    ↓
  [Report Generation: Compile final report]

Phase Transitions:
  ❌ No explicit state tracking enum
  ⚠️ Implicitly managed through function call stack
  ❌ No _transition_state() logging

Quality Gates:
  ✓ Pydantic schema validation (CDAFConfigSchema)
  ✓ Mechanism necessity/sufficiency tests
  ⚠️ Implicit gates via exception handling

Timeout Configuration:
  ❌ MISSING: No timeout enforcement

Backpressure Mechanisms:
  ❌ MISSING: Synchronous execution (no async)

Metrics Collection:
  ⚠️ Self-reflection metrics:
      - update_priors_from_feedback()
      - _uncertainty_history tracking
  ❌ MISSING: Standardized MetricsCollector integration

Audit Logging:
  ⚠️ ConfigLoader with Pydantic validation errors
  ⚠️ CDAFException.to_dict() structured errors
  ❌ MISSING: ImmutableAuditLogger implementation
  ❌ MISSING: SHA-256 provenance tracking
```

---

## 3. Phase Handoff Matrix

### 3.1 Overlapping Responsibilities

| Functionality | AnalyticalOrchestrator | PDMOrchestrator | CDAFFramework |
|---------------|------------------------|-----------------|---------------|
| **PDF Extraction** | ❌ (placeholder) | ✓ `_extract_complete()` | ✓ `PDFExtractor.extract_complete()` |
| **Statement Extraction** | ✓ `_extract_statements()` | ⚠️ (delegated to pipeline) | ✓ Semantic chunking |
| **Contradiction Detection** | ✓ `_detect_contradictions()` | ❌ | ⚠️ (indirect via coherence) |
| **Regulatory Analysis** | ✓ `_analyze_regulatory_constraints()` | ❌ | ⚠️ DNP validation |
| **Causal Graph Building** | ❌ | ✓ `_build_graph()` | ✓ `CausalGraphBuilder` |
| **Bayesian Inference** | ❌ | ✓ `_infer_all_mechanisms()` | ✓ `BayesianInferenceEngine` |
| **Coherence Metrics** | ✓ `_calculate_coherence_metrics()` | ❌ | ⚠️ (KL divergence) |
| **Validation** | ❌ | ✓ `_validate_complete()` | ✓ DNP + audit framework |
| **Quality Scoring** | ⚠️ (audit summary) | ✓ `_calculate_quality_score()` | ❌ |
| **Report Generation** | ✓ `_compile_final_report()` | ⚠️ (AnalysisResult) | ✓ Output generation |

### 3.2 Identified Redundancies

#### 🔴 **REDUNDANCY 1: Contradiction Detection**
- **Location**: `AnalyticalOrchestrator._detect_contradictions()`
- **Issue**: Placeholder implementation; actual logic likely in `contradiction_deteccion.py`
- **Action**: **DELETE** placeholder; centralize in shared module

#### 🔴 **REDUNDANCY 2: Regulatory Analysis**
- **Location**: `AnalyticalOrchestrator._analyze_regulatory_constraints()`
- **Issue**: Duplicates DNP validation in CDAFFramework
- **Action**: **DELETE** placeholder; delegate to `dnp_integration.py`

#### 🔴 **REDUNDANCY 3: Quality Score Calculation**
- **Locations**: 
  - `AnalyticalOrchestrator._generate_audit_summary()` (quality_grade)
  - `PDMOrchestrator._calculate_quality_score()` (QualityScore)
- **Issue**: Two different implementations of quality assessment
- **Action**: **UNIFY** into single scoring module

### 3.3 Phase Handoff Gaps

#### 🟡 **GAP 1: Calibration Constant Propagation**
- **Problem**: AnalyticalOrchestrator defines constants, but they don't propagate to PDM/CDAF
- **Impact**: Non-deterministic behavior across orchestrators
- **Fix**: Create `calibration_constants.py` shared module

#### 🟡 **GAP 2: Metrics Aggregation**
- **Problem**: PDMOrchestrator has MetricsCollector; AnalyticalOrchestrator has ad-hoc metrics
- **Impact**: Inconsistent observability
- **Fix**: Standardize on MetricsCollector interface

#### 🟡 **GAP 3: Audit Trail Consistency**
- **Problem**: Three different audit logging approaches (JSON, JSONL, Pydantic)
- **Impact**: Cannot aggregate audit trails across orchestrators
- **Fix**: Standardize on ImmutableAuditLogger with SHA-256 provenance

---

## 4. Remediation Plan

### 4.1 DELETE Operations

#### DELETE-1: Remove Analytical Orchestrator Placeholders
**Rationale**: Placeholder methods in `AnalyticalOrchestrator` create false interfaces without implementation.

**Files to Modify**:
- `orchestrator.py`: Remove `_extract_statements()`, `_detect_contradictions()`, `_analyze_regulatory_constraints()`, `_calculate_coherence_metrics()` placeholders
- Replace with delegation to specialized modules

**Contract Preservation**:
- Maintain `PhaseResult` dataclass interface
- Keep `orchestrate_analysis()` entry point signature

#### DELETE-2: Consolidate Quality Scoring
**Rationale**: Two separate quality scoring implementations violate DRY principle.

**Files to Modify**:
- `orchestrator.py`: Remove `_generate_audit_summary()` scoring logic
- `orchestration/pdm_orchestrator.py`: Extract `_calculate_quality_score()` to shared module

**New Module**: `quality_scoring.py`

### 4.2 FILL Operations

#### FILL-1: Centralize Calibration Constants
**Rationale**: SIN_CARRETA requires mathematical invariants across all executions.

**New Module**: `infrastructure/calibration_constants.py`
```python
# infrastructure/calibration_constants.py
from dataclasses import dataclass
from typing import Final

@dataclass(frozen=True)
class CalibrationConstants:
    """Immutable calibration constants per SIN_CARRETA doctrine"""
    
    # Coherence thresholds
    COHERENCE_THRESHOLD: Final[float] = 0.7
    CAUSAL_INCOHERENCE_LIMIT: Final[int] = 5
    REGULATORY_DEPTH_FACTOR: Final[float] = 1.3
    
    # Severity thresholds
    CRITICAL_SEVERITY_THRESHOLD: Final[float] = 0.85
    HIGH_SEVERITY_THRESHOLD: Final[float] = 0.70
    MEDIUM_SEVERITY_THRESHOLD: Final[float] = 0.50
    
    # Audit quality grades
    EXCELLENT_CONTRADICTION_LIMIT: Final[int] = 5
    GOOD_CONTRADICTION_LIMIT: Final[int] = 10
    
    # Bayesian thresholds
    KL_DIVERGENCE_THRESHOLD: Final[float] = 0.01
    CONVERGENCE_MIN_EVIDENCE: Final[int] = 2
    PRIOR_ALPHA: Final[float] = 2.0
    PRIOR_BETA: Final[float] = 2.0
    LAPLACE_SMOOTHING: Final[float] = 1.0
    
    def __setattr__(self, *args, **kwargs):
        raise FrozenInstanceError("Calibration constants are immutable")

# Singleton instance
CALIBRATION: Final[CalibrationConstants] = CalibrationConstants()
```

**Files to Update**:
- `orchestrator.py`: Import CALIBRATION singleton
- `orchestration/pdm_orchestrator.py`: Import CALIBRATION singleton
- `dereck_beach`: Import CALIBRATION singleton (replace config-based)

#### FILL-2: Standardize Metrics Collection
**Rationale**: Observability must be consistent across all orchestrators.

**Files to Modify**:
- `orchestrator.py`: Integrate `MetricsCollector` from PDMOrchestrator
- `dereck_beach`: Add `MetricsCollector` instrumentation

**Pattern**:
```python
# Add to all orchestrators
self.metrics = MetricsCollector()

# Instrument all phases
self.metrics.record("phase.{phase_name}.duration", duration)
self.metrics.increment("phase.{phase_name}.count")
```

#### FILL-3: Unify Audit Logging
**Rationale**: Immutable audit trails with SHA-256 provenance are non-negotiable for governance.

**Files to Modify**:
- `orchestrator.py`: Replace `_persist_audit_log()` with `ImmutableAuditLogger`
- `dereck_beach`: Add `ImmutableAuditLogger` integration

**Shared Interface**:
```python
# All orchestrators must use
self.audit_logger = ImmutableAuditLogger(audit_store_path)
self.audit_logger.append_record(
    run_id=run_id,
    orchestrator=self.__class__.__name__,
    sha256_source=self._hash_file(source),
    calibration_snapshot=CALIBRATION.__dict__,
    ...
)
```

### 4.3 EXECUTE Operations

#### EXECUTE-1: Add Async/Await to AnalyticalOrchestrator
**Rationale**: Sync execution limits throughput and prevents parallel phase execution.

**Pattern**:
```python
# Convert to async
async def orchestrate_analysis(...) -> Dict[str, Any]:
    async with self.semaphore:
        async with self._timeout_context(timeout_secs):
            # Parallel phases where dependencies allow
            contradictions_task = asyncio.create_task(
                self._detect_contradictions(...)
            )
            regulatory_task = asyncio.create_task(
                self._analyze_regulatory_constraints(...)
            )
            results = await asyncio.gather(...)
```

#### EXECUTE-2: Add Explicit State Machine to CDAFFramework
**Rationale**: Implicit state tracking prevents audit trail completeness.

**Pattern**:
```python
class CDAFState(str, Enum):
    INITIALIZED = "initialized"
    EXTRACTING_PDF = "extracting_pdf"
    CHUNKING_SEMANTICS = "chunking_semantics"
    BUILDING_CAUSAL_GRAPH = "building_causal_graph"
    BAYESIAN_INFERENCE = "bayesian_inference"
    EXECUTING_AUDIT = "executing_audit"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"

class CDAFFramework:
    def __init__(self, ...):
        self.state = CDAFState.INITIALIZED
        self.metrics = MetricsCollector()
        self.audit_logger = ImmutableAuditLogger(...)
    
    def _transition_state(self, new_state: CDAFState) -> None:
        old_state = self.state
        self.state = new_state
        self.logger.info(f"State transition: {old_state} -> {new_state}")
        self.metrics.record("state_transitions", 1.0)
        self.audit_logger.append_record(
            event="state_transition",
            from_state=old_state.value,
            to_state=new_state.value
        )
```

#### EXECUTE-3: Add Timeout and Backpressure to AnalyticalOrchestrator
**Rationale**: Production systems require timeout enforcement and concurrency control.

**Pattern**:
```python
class AnalyticalOrchestrator:
    def __init__(self, ...):
        self.semaphore = asyncio.Semaphore(max_inflight_jobs)
        self.worker_timeout_secs = 300
    
    @asynccontextmanager
    async def _timeout_context(self, timeout_secs: float):
        try:
            async with asyncio.timeout(timeout_secs):
                yield
        except asyncio.TimeoutError:
            self.metrics.increment("pipeline.timeout_count")
            raise
```

---

## 5. Implementation Checklist

### Phase 1: Foundation (Shared Infrastructure)
- [ ] Create `infrastructure/calibration_constants.py` with frozen dataclass
- [ ] Extract `MetricsCollector` to `infrastructure/metrics_collector.py`
- [ ] Extract `ImmutableAuditLogger` to `infrastructure/audit_logger.py`
- [ ] Add SHA-256 hashing utility to audit_logger module
- [ ] Write tests for calibration constant immutability
- [ ] Write tests for audit logger append-only semantics

### Phase 2: DELETE Operations
- [ ] Remove placeholder methods from `AnalyticalOrchestrator`
- [ ] Delete duplicate quality scoring logic
- [ ] Remove hardcoded constants from all three orchestrators
- [ ] Update all imports to use shared modules

### Phase 3: FILL Operations
- [ ] Integrate `CalibrationConstants` into all orchestrators
- [ ] Add `MetricsCollector` to `AnalyticalOrchestrator` and `CDAFFramework`
- [ ] Replace audit logging in all orchestrators with `ImmutableAuditLogger`
- [ ] Add SHA-256 provenance tracking to all orchestrators
- [ ] Update configuration schemas to load from CALIBRATION singleton

### Phase 4: EXECUTE Operations
- [ ] Convert `AnalyticalOrchestrator.orchestrate_analysis()` to async
- [ ] Add asyncio.Queue and Semaphore to `AnalyticalOrchestrator`
- [ ] Add timeout context managers to all orchestrators
- [ ] Add explicit `CDAFState` enum to `CDAFFramework`
- [ ] Add `_transition_state()` method to `CDAFFramework`
- [ ] Add quality gates to `AnalyticalOrchestrator` (like PDM)

### Phase 5: Validation
- [ ] Run `python3 pretest_compilation.py` after each change
- [ ] Verify all tests pass: `python -m unittest discover -s . -p "test_*.py"`
- [ ] Validate calibration constant consistency across all modules
- [ ] Validate audit trail SHA-256 hashes match source files
- [ ] Validate metrics are collected at all decision points
- [ ] Validate state transitions are logged for all orchestrators
- [ ] Run integration test: `python demo_orchestration_complete.py`

### Phase 6: Documentation
- [ ] Update `ORCHESTRATOR_README.md` with unified state machine diagrams
- [ ] Document calibration constant usage in each orchestrator
- [ ] Document audit logger interface and SHA-256 provenance
- [ ] Add sequence diagrams for cross-orchestrator handoffs
- [ ] Update `AGENTS.md` with new shared module paths

---

## 6. Risk Assessment

### High-Risk Changes
1. **Async conversion of AnalyticalOrchestrator**: May break synchronous callers
   - **Mitigation**: Add synchronous wrapper `orchestrate_analysis_sync()`
   - **Test**: Validate all existing tests pass with wrapper

2. **Removal of placeholder methods**: May break imports
   - **Mitigation**: Add deprecation warnings before removal
   - **Test**: Search codebase for all import statements

3. **Calibration constant centralization**: May break config-based overrides
   - **Mitigation**: Add `override_calibration()` method for testing
   - **Test**: Validate override mechanism with unit tests

### Medium-Risk Changes
1. **Metrics collector standardization**: May change metric names
   - **Mitigation**: Maintain backward-compatible metric naming
   - **Test**: Compare metric outputs before/after

2. **Audit logger unification**: May change audit log format
   - **Mitigation**: Support both formats during transition
   - **Test**: Validate JSONL parsing with existing tools

### Low-Risk Changes
1. **State enum additions**: Additive only
2. **Documentation updates**: No code impact

---

## 7. Success Criteria

### Functional Criteria
✅ All calibration constants resolve to same values across orchestrators  
✅ All orchestrators use `ImmutableAuditLogger` with SHA-256 provenance  
✅ All orchestrators use `MetricsCollector` with standardized metric names  
✅ All orchestrators have explicit state machine enums with transition logging  
✅ All orchestrators enforce timeouts and backpressure limits  
✅ No placeholder methods remain in production code  

### Non-Functional Criteria
✅ All tests pass: `python -m unittest discover`  
✅ Pretest compilation succeeds: `python3 pretest_compilation.py`  
✅ No regression in processing time (±10% tolerance)  
✅ Audit logs maintain immutability (append-only verification)  
✅ SHA-256 hashes match source files (integrity check)  

### Determinism Criteria (SIN_CARRETA Compliance)
✅ Calibration constants are frozen dataclasses (immutable)  
✅ State transitions are deterministic (no time-based conditions)  
✅ Audit logs include provenance hashes for all inputs  
✅ Metrics collection does not alter execution flow  
✅ Quality gates have explicit, documented thresholds  

---

## 8. Appendices

### Appendix A: Calibration Constant Audit
```bash
# Verify constants are consistent
grep -r "COHERENCE_THRESHOLD" --include="*.py" .
grep -r "CAUSAL_INCOHERENCE_LIMIT" --include="*.py" .
grep -r "REGULATORY_DEPTH_FACTOR" --include="*.py" .

# Expected: All resolve to CalibrationConstants singleton
```

### Appendix B: Audit Logger Verification
```bash
# Verify SHA-256 provenance
python3 << EOF
import json
from pathlib import Path

audit_logs = Path("audit_logs.jsonl")
if audit_logs.exists():
    with open(audit_logs) as f:
        for line in f:
            record = json.loads(line)
            assert "sha256_source" in record, "Missing SHA-256 provenance"
            print(f"✓ {record['run_id']}: {record['sha256_source'][:16]}...")
EOF
```

### Appendix C: State Machine Coverage
```bash
# Verify all state transitions are logged
python3 << EOF
import re
from pathlib import Path

for orchestrator_file in ["orchestrator.py", "orchestration/pdm_orchestrator.py", "dereck_beach"]:
    content = Path(orchestrator_file).read_text()
    transitions = re.findall(r'_transition_state\([^)]+\)', content)
    print(f"{orchestrator_file}: {len(transitions)} transitions")
EOF
```

---

## Conclusion

This analysis reveals **critical architectural debt** that violates determinism and governance requirements. The remediation plan provides a **phased approach** to unify orchestrator implementations while preserving existing functionality. All changes must pass SIN_CARRETA compliance checks before deployment.

**Next Steps**:
1. Review and approve remediation plan
2. Execute Phase 1 (Foundation) infrastructure changes
3. Validate with `pretest_compilation.py` after each phase
4. Deploy unified orchestrators with full observability

**SIN_CARRETA Compliance Status**: ❌ **NON-COMPLIANT** (requires remediation)

