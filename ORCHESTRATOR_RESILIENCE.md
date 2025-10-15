# Orchestrator Resilience & Monitoring System

## Overview

The FARFAN orchestrator has been enhanced with comprehensive resilience and monitoring capabilities to handle failures gracefully, track execution metrics, and enable post-mortem analysis of pipeline runs.

## Architecture

### Components

1. **RiskRegistry** (`risk_registry.py`)
   - Catalogs known risks by stage
   - Pre-stage risk assessment
   - Exception-to-risk mapping
   - Mitigation strategy execution

2. **CircuitBreaker** (`circuit_breaker.py`)
   - Protects against cascading failures
   - Three states: CLOSED, OPEN, HALF_OPEN
   - Automatic recovery attempts
   - Per-stage circuit breakers

3. **PipelineCheckpoint** (`pipeline_checkpoint.py`)
   - Immutable checkpoints after each stage
   - Recovery from interruptions
   - Hash-based integrity verification
   - Checkpoint chains for traceability

4. **PipelineMetrics** (`pipeline_metrics.py`)
   - Per-stage success rates
   - Execution time tracking
   - Mitigation invocation counting
   - Alert generation
   - Complete execution traces

## Execution Flow

For each pipeline stage:

```
1. PRE-STAGE RISK ASSESSMENT
   ├─ Query RiskRegistry for applicable risks
   ├─ Evaluate risk conditions
   └─ Emit alerts for CRITICAL/HIGH risks

2. CIRCUIT BREAKER PROTECTED EXECUTION
   ├─ Check circuit breaker state
   ├─ Execute stage function
   └─ Update circuit breaker on success/failure

3. POST-STAGE CHECKPOINT
   ├─ Serialize stage context
   ├─ Compute integrity hash
   ├─ Save checkpoint to disk
   └─ Link to previous checkpoint

4. FAILURE HANDLING (if exception occurs)
   ├─ Map exception to risk definition
   ├─ Record mitigation attempt
   ├─ Emit alert based on severity
   ├─ CRITICAL: Abort immediately
   └─ Other severities: Execute mitigation strategy

5. METRICS COLLECTION
   ├─ Record stage success/failure
   ├─ Track execution time
   ├─ Count mitigation invocations
   └─ Update aggregated statistics
```

## Risk Management

### Risk Severity Levels

- **CRITICAL**: Abort execution immediately, no mitigation attempted
- **HIGH**: Execute mitigation, emit ERROR alert
- **MEDIUM**: Execute mitigation, emit WARNING alert
- **LOW**: Execute mitigation, emit INFO alert

### Risk Categories

- `DATA_QUALITY`: Input data issues (empty extraction, corrupted PDF)
- `RESOURCE_EXHAUSTION`: Memory/CPU exhaustion
- `EXTERNAL_DEPENDENCY`: Missing models, unavailable services
- `COMPUTATION_ERROR`: Algorithm failures, empty results
- `VALIDATION_FAILURE`: DNP compliance issues
- `CONFIGURATION`: Setup/config problems

### Default Risks

| Risk ID | Category | Severity | Applicable Stages |
|---------|----------|----------|-------------------|
| `PDF_CORRUPT` | DATA_QUALITY | CRITICAL | LOAD_DOCUMENT, EXTRACT_TEXT_TABLES |
| `EMPTY_EXTRACTION` | DATA_QUALITY | HIGH | EXTRACT_TEXT_TABLES |
| `NLP_MODEL_MISSING` | EXTERNAL_DEPENDENCY | CRITICAL | SEMANTIC_ANALYSIS, CAUSAL_EXTRACTION |
| `INSUFFICIENT_TEXT` | DATA_QUALITY | MEDIUM | SEMANTIC_ANALYSIS, CAUSAL_EXTRACTION |
| `MEMORY_EXHAUSTION` | RESOURCE_EXHAUSTION | HIGH | Multiple stages |
| `TIMEOUT` | COMPUTATION_ERROR | MEDIUM | Multiple stages |
| `DNP_VALIDATION_FAIL` | VALIDATION_FAILURE | LOW | DNP_VALIDATION |
| `MISSING_FINANCIAL_DATA` | DATA_QUALITY | MEDIUM | FINANCIAL_AUDIT |
| `EMPTY_CAUSAL_GRAPH` | COMPUTATION_ERROR | HIGH | CAUSAL_EXTRACTION |

## Circuit Breaker

### Configuration

```python
CircuitBreakerConfig(
    failure_threshold=2,    # Open after N failures
    success_threshold=2,    # Close after N successes in HALF_OPEN
    timeout=30.0           # Seconds before retry (OPEN -> HALF_OPEN)
)
```

### State Transitions

```
CLOSED ─(failures >= threshold)─> OPEN
   ↑                                 │
   │                                 │
   └───── HALF_OPEN ←─(timeout)─────┘
           │      ↑
           │      └─(failure)
           └─(success >= threshold)
```

## Checkpoints

### Structure

Each checkpoint contains:
- **Metadata**: Stage name, timestamp, execution time, success flag
- **Context**: Complete pipeline context serialized
- **Hash**: SHA-256 hash for integrity verification
- **Chain**: Reference to previous checkpoint

### Storage

```
checkpoints/
├── PDM2024-ANT-MED_LOAD_DOCUMENT_20241014_120000_123456.pkl
├── PDM2024-ANT-MED_LOAD_DOCUMENT_20241014_120000_123456.meta.json
├── PDM2024-ANT-MED_SEMANTIC_ANALYSIS_20241014_120005_234567.pkl
└── PDM2024-ANT-MED_SEMANTIC_ANALYSIS_20241014_120005_234567.meta.json
```

## Metrics & Traces

### Collected Metrics

**Per-Stage:**
- Success count / failure count
- Total execution time / average time
- Risk assessments performed
- Mitigation attempts
- Circuit breaker state

**Aggregated:**
- Success rates by stage
- Mitigation counts by category/severity
- Alert counts by level
- Circuit breaker transitions

### Execution Trace

Complete JSON export containing:

```json
{
  "policy_code": "PDM2024-ANT-MED",
  "start_time": "2024-10-14T12:00:00",
  "end_time": "2024-10-14T12:15:30",
  "success": true,
  "total_execution_time_ms": 930000,
  "stages": [
    {
      "stage_name": "LOAD_DOCUMENT",
      "success": true,
      "execution_time_ms": 1500,
      "risk_assessments": ["PDF_CORRUPT"],
      "mitigation_attempts": [],
      "circuit_breaker_state": "CLOSED"
    }
  ],
  "alerts": [
    {
      "level": "WARNING",
      "message": "Riesgo HIGH detectado: EMPTY_EXTRACTION",
      "context": {"stage": "EXTRACT_TEXT_TABLES"},
      "timestamp": "2024-10-14T12:00:02"
    }
  ],
  "mitigation_stats": {
    "by_severity": {
      "MEDIUM": {"total": 2, "success": 2}
    },
    "by_category": {
      "DATA_QUALITY": {"total": 2, "success": 2}
    }
  },
  "circuit_breaker_stats": {
    "LOAD_DOCUMENT": {
      "state": "CLOSED",
      "total_calls": 1,
      "total_successes": 1,
      "success_rate": 1.0
    }
  }
}
```

## Usage

### Basic Usage

```python
from pathlib import Path
from orchestrator import FARFANOrchestrator

orchestrator = FARFANOrchestrator(
    output_dir=Path("./resultados"),
    log_level="INFO"
)

context = orchestrator.process_plan(
    pdf_path=Path("plan.pdf"),
    policy_code="PDM2024-001",
    es_municipio_pdet=False
)
```

### Accessing Results

```python
# Metrics are automatically exported
# Check: resultados/metrics/execution_trace_PDM2024-001_*.json

# Checkpoints are saved incrementally
# Check: resultados/checkpoints/PDM2024-001_*.pkl

# Console output shows summary
orchestrator.metrics.print_summary()
```

### Custom Risk Handling

```python
from risk_registry import RiskDefinition, RiskSeverity, RiskCategory

# Register custom risk
custom_risk = RiskDefinition(
    risk_id="CUSTOM_RISK",
    name="Custom risk description",
    category=RiskCategory.COMPUTATION_ERROR,
    severity=RiskSeverity.MEDIUM,
    description="Detailed description",
    applicable_stages=["CUSTOM_STAGE"],
    mitigation_strategy="custom_mitigation"
)

orchestrator.risk_registry.register_risk(custom_risk)

# Register mitigation strategy
def custom_mitigation(context):
    # Implement mitigation logic
    pass

orchestrator.risk_registry.mitigation_strategies["custom_mitigation"] = custom_mitigation
```

## Testing

Run the resilience test suite:

```bash
python3 test_orchestrator_resilience.py
```

This validates:
- RiskRegistry assessment and mitigation
- CircuitBreaker state transitions
- PipelineCheckpoint save/load
- PipelineMetrics tracking and export

## Monitoring & Alerts

### Alert Levels

- **INFO**: Successful mitigations, normal events
- **WARNING**: Medium severity risks detected
- **ERROR**: High severity risks, mitigation failures
- **CRITICAL**: Critical risks, execution aborted

### Alert Triggers

1. **CRITICAL/HIGH risks detected** in pre-stage assessment
2. **Circuit breaker opens** (too many failures)
3. **Mitigation fails** for HIGH/CRITICAL risks
4. **Uncataloged exceptions** occur

## Post-Mortem Analysis

Execution traces enable:

1. **Failure Analysis**: Identify which stage failed and why
2. **Performance Profiling**: Find bottleneck stages
3. **Risk Patterns**: Detect recurring risks
4. **Mitigation Effectiveness**: Evaluate strategy success rates
5. **Circuit Breaker Behavior**: Understand failure cascades

### Example Queries

```python
import json
from pathlib import Path

# Load trace
trace_file = Path("resultados/metrics/execution_trace_PDM2024-001_20241014.json")
with open(trace_file) as f:
    trace = json.load(f)

# Find slowest stage
slowest = max(trace["stages"], key=lambda s: s["execution_time_ms"])
print(f"Slowest stage: {slowest['stage_name']} ({slowest['execution_time_ms']}ms)")

# Count mitigations by category
for category, stats in trace["mitigation_stats"]["by_category"].items():
    print(f"{category}: {stats['total']} attempts, {stats['success']} successes")

# Find CRITICAL alerts
critical_alerts = [a for a in trace["alerts"] if a["level"] == "CRITICAL"]
for alert in critical_alerts:
    print(f"CRITICAL: {alert['message']} at {alert['timestamp']}")
```

## Performance Impact

The resilience system adds minimal overhead:

- **Risk Assessment**: ~1-5ms per stage (pre-assessment)
- **Circuit Breaker**: ~0.1ms per call (state check)
- **Checkpoint**: ~50-200ms per stage (serialization + I/O)
- **Metrics**: ~0.5ms per event (in-memory tracking)

Total overhead: **~50-250ms per stage** (mostly from checkpointing)

## Best Practices

1. **Always check traces** after failed runs to understand what happened
2. **Monitor circuit breaker states** - frequent OPEN states indicate systemic issues
3. **Review mitigation success rates** - low rates suggest strategy improvements needed
4. **Clean old checkpoints** periodically to manage disk space
5. **Adjust circuit breaker thresholds** based on stage reliability
6. **Add custom risks** for domain-specific failure modes
7. **Export metrics** to external monitoring systems for production deployments

## Future Enhancements

- [ ] Integration with external monitoring (Prometheus, Datadog)
- [ ] Automatic checkpoint cleanup policies
- [ ] Circuit breaker backpressure to upstream stages
- [ ] Risk prediction using historical traces
- [ ] Distributed tracing for multi-node deployments
- [ ] Real-time dashboard for pipeline monitoring
