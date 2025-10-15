# Governance Standards Quick Reference

## 🚀 Quick Start

### Run All Tests
```bash
python -m unittest test_governance_standards -v
```

### Run CI Contract Enforcement
```bash
python ci_contract_enforcement.py
# Exit code 0 = PASS (merge allowed)
# Exit code 1 = FAIL (BLOCK MERGE)
```

### Run Governance Demo
```bash
python governance_standards.py
python orchestrator_governance_integration.py
```

## 📊 Audit Point Checklist

| Audit Point | Status | Test Command |
|-------------|--------|--------------|
| 5.1 Execution Isolation | ✅ | `python -m unittest test_governance_standards.TestExecutionIsolation` |
| 5.2 Immutable Audit Log | ✅ | `python -m unittest test_governance_standards.TestImmutableAuditLog` |
| 5.3 Explainability Payload | ✅ | `python -m unittest test_governance_standards.TestExplainabilityPayload` |
| 5.4 Human-in-the-Loop Gate | ✅ | `python -m unittest test_governance_standards.TestHumanInTheLoopGate` |
| 5.5 CI Contract Enforcement | ✅ | `python -m unittest test_governance_standards.TestCIContractEnforcement` |

## 🔧 Usage Examples

### 1. Basic Governance Analysis
```python
from orchestrator_governance_integration import create_governance_orchestrator

orchestrator = create_governance_orchestrator(enable_governance=True)
result = orchestrator.orchestrate_analysis_with_governance(
    text=policy_text,
    plan_name="PDM_2024",
    dimension="estratégico"
)

# Check if manual review needed
if result["governance"]["human_in_the_loop_gate"]["hold_for_manual_review"]:
    print("⚠️ Manual review required")
```

### 2. Query Audit Log
```python
from governance_standards import ImmutableAuditLog
from pathlib import Path

audit_log = ImmutableAuditLog(log_dir=Path("logs/governance_audit"))
entries = audit_log.query_by_run_id("PDM_2024_...")
is_valid, errors = audit_log.verify_chain()
```

### 3. Create Explainability Payload
```python
from governance_standards import ExplainabilityPayload

payload = ExplainabilityPayload(
    link_id="MP-001→MR-001",
    posterior_mean=0.75,
    posterior_std=0.12,
    confidence_interval=(0.55, 0.90),
    necessity_test_passed=True,
    necessity_test_missing=[],
    evidence_snippets=["Evidence 1", "Evidence 2"],
    sha256_evidence=ExplainabilityPayload.compute_evidence_hash(["Evidence 1", "Evidence 2"])
)
```

### 4. Check Isolation Metrics
```python
from governance_standards import IsolationMetrics

metrics = IsolationMetrics(
    total_executions=1000,
    failure_count=1
)
metrics.update_uptime()

if metrics.uptime_percentage >= 99.9:
    print("✅ Meets SOTA standard")
```

## 📁 File Reference

| File | Purpose |
|------|---------|
| `governance_standards.py` | Core implementation (588 lines) |
| `test_governance_standards.py` | Test suite (601 lines, 19 tests) |
| `orchestrator_governance_integration.py` | Orchestrator integration (423 lines) |
| `ci_contract_enforcement.py` | CI/CD runner (94 lines) |
| `GOVERNANCE_IMPLEMENTATION.md` | Full documentation (360 lines) |

## 🎯 Key Metrics

- **Test Coverage**: 19/19 tests passing
- **Uptime Target**: ≥99.9%
- **Audit Retention**: 5 years (1825 days)
- **Hash Algorithm**: SHA256
- **CI Exit Codes**: 0 (pass), 1 (fail/block)

## 🔒 Security Features

1. **Immutable Logs**: Read-only files (chmod 0o444)
2. **Hash Chains**: SHA256 cryptographic linking
3. **Timeout Enforcement**: Worker timeout with fail-open
4. **Manual Review Gates**: Automatic quality-based triggers
5. **Source Traceability**: SHA256 of all source documents

## ⚙️ Configuration

```python
# Execution Isolation
ExecutionIsolationConfig(
    mode=IsolationMode.DOCKER,
    worker_timeout_secs=300,      # 5 minutes
    fail_open_on_timeout=True,
    container_memory_limit="2g",
    container_cpu_limit=1.0
)

# Quality Grades (Human-in-the-Loop triggers)
# - Excelente: < 5 contradictions
# - Bueno: < 10 contradictions  
# - Regular: ≥ 10 contradictions
# - Insuficiente: Low coherence
```

## 🐛 Troubleshooting

### NumPy/SciPy Not Available
The Bayesian CI contract tests (5.5) will be skipped if NumPy/SciPy are not installed. All other tests will run normally.

### Audit Chain Verification Failed
```python
audit_log = ImmutableAuditLog()
is_valid, errors = audit_log.verify_chain()
for error in errors:
    print(f"Chain error: {error}")
```

### Uptime Below 99.9%
```python
metrics = orchestrator.isolation_metrics
print(f"Failures: {metrics.failure_count}/{metrics.total_executions}")
print(f"Timeouts: {metrics.timeout_count}")
```

## 📞 Support

1. Run tests: `python -m unittest test_governance_standards -v`
2. Check logs: `logs/orchestrator/governance/audit_log_*.json`
3. Verify chain: `ImmutableAuditLog.verify_chain()`
4. Review CI output: `python ci_contract_enforcement.py`

## 📚 References

- Full docs: `GOVERNANCE_IMPLEMENTATION.md`
- Code: `governance_standards.py`
- Tests: `test_governance_standards.py`
- Integration: `orchestrator_governance_integration.py`
