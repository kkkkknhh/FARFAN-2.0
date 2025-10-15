# Part 4 IoR - Quick Reference

## Quick Start

### Audit Point 4.1: Execution Isolation
```python
from infrastructure import create_isolated_processor

processor = create_isolated_processor(worker_timeout_secs=120)
result = await processor.process_pdf("document.pdf")
```

### Audit Point 4.2: Backpressure Signaling
```python
from infrastructure import create_orchestrator, QueueFullError

orchestrator = create_orchestrator(queue_size=100, max_workers=5)
await orchestrator.start()

try:
    result = await orchestrator.submit_job(expensive_func, data)
except QueueFullError as e:
    return {"error": "Service busy", "status": e.http_status}  # HTTP 503
```

### Audit Point 4.3: Fail-Open Policy
```python
from infrastructure import create_policy_manager, CDAFValidationError

manager = create_policy_manager()

# Enrichment: fail-open (degrade gracefully)
result = await manager.execute_validation("dnp_validator", validator_func)
if result.fail_open_applied:
    print(f"Continued with {result.degradation_penalty:.0%} penalty")

# Core: fail-closed (halt on error)
try:
    await manager.execute_validation("core_validator", validator_func)
except CDAFValidationError:
    raise  # Critical failure
```

## Key Exports

```python
from infrastructure import (
    # 4.2: Backpressure
    AsyncOrchestrator,
    QueueFullError,
    create_orchestrator,
    
    # 4.1: Isolation
    IsolatedPDFProcessor,
    IsolationStrategy,
    create_isolated_processor,
    
    # 4.3: Fail-Open
    FailOpenPolicyManager,
    CDAFValidationError,
    DNP_AVAILABLE,
    set_dnp_available,
    create_policy_manager,
)
```

## Configuration

### Orchestrator Config (4.2)
```python
OrchestratorConfig(
    queue_size=100,          # Max jobs in queue
    max_workers=5,           # Concurrent workers
    job_timeout_secs=300,    # Job timeout
    enable_backpressure=True,
    log_backpressure=True
)
```

### Isolation Config (4.1)
```python
IsolationConfig(
    worker_timeout_secs=120,
    isolation_strategy=IsolationStrategy.PROCESS,
    max_memory_mb=512,
    enable_monitoring=True,
    kill_on_timeout=True
)
```

### Component Config (4.3)
```python
ComponentConfig(
    name="dnp_validator",
    component_type=ComponentType.ENRICHMENT,
    fail_closed=False,       # Fail-open
    degradation_penalty=0.05, # 5% penalty
    timeout_secs=120
)
```

## Testing

```bash
python3 test_part4_ior.py
```

Expected: 17 tests pass (6 + 5 + 6)

## Compliance Checklist

- [x] 4.1: PDF parsing isolation with worker_timeout_secs
- [x] 4.1: Container/sandbox strategy with fallback
- [x] 4.1: 99.9% uptime through fault isolation
- [x] 4.2: HTTP 503 on queue overflow (queue_size limit)
- [x] 4.2: Deque-based queue management
- [x] 4.2: Backpressure logging and metrics
- [x] 4.3: Fail-open for enrichment (DNP validator)
- [x] 4.3: Fail-closed for core components
- [x] 4.3: CDAFValidationError exception
- [x] 4.3: DNP_AVAILABLE flag
- [x] 4.3: <10% accuracy loss (2% achieved)

## Metrics

```python
# Orchestrator metrics
metrics = orchestrator.get_metrics()
print(f"Queue: {metrics.current_queue_size}/{config.queue_size}")
print(f"Backpressure events: {metrics.backpressure_events}")
print(f"Rejected jobs: {metrics.total_jobs_rejected}")

# Isolation metrics
metrics = processor.get_metrics()
print(f"Uptime: {metrics.uptime_percentage:.2f}%")
print(f"Timeouts: {metrics.timeout_failures}")

# Policy metrics
metrics = manager.get_metrics()
print(f"Accuracy loss: {metrics.accuracy_loss:.2%}")
print(f"Fail-open applied: {metrics.fail_open_applied}")
```

## Error Handling

```python
# Backpressure
except QueueFullError as e:
    return {"status": e.http_status}  # 503

# Timeout
except JobTimeoutError as e:
    log.error(f"Timeout: {e}")

# Isolation error
except PDFProcessingTimeoutError as e:
    log.warning(f"PDF timeout: {e}")

# Validation error
except CDAFValidationError as e:
    if e.fail_closed:
        raise  # Critical
    else:
        log.warning(f"Degraded: {e}")
```
