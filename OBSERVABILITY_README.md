# F4.4: Comprehensive Observability Stack

## Overview

This implementation provides a complete observability stack for FARFAN 2.0, including metrics collection, structured logging, and distributed tracing. It implements all Observability Metrics specified in the Standard.

## Architecture

```
infrastructure/
├── __init__.py           # Module exports
└── observability.py      # Complete implementation
    ├── ObservabilityConfig    # Configuration dataclass
    ├── MetricsCollector       # Metrics backend (histograms, gauges, counters)
    ├── StructuredLogger       # Structured logging with context
    ├── DistributedTracer      # Distributed tracing with spans
    └── ObservabilityStack     # Main orchestration class
```

## Components

### 1. ObservabilityConfig

Configuration dataclass for the observability stack:

```python
@dataclass
class ObservabilityConfig:
    metrics_backend: str = "in_memory"
    log_level: str = "INFO"
    trace_backend: str = "in_memory"
    enable_distributed_tracing: bool = False
```

### 2. MetricsCollector

Collects three types of metrics:

- **Histograms**: Distribution of values over time (e.g., pipeline durations)
- **Gauges**: Point-in-time values (e.g., memory usage, scores)
- **Counters**: Incrementing counts (e.g., failures, events)

Methods:
- `histogram(metric_name, value, tags=None)`: Record histogram value
- `gauge(metric_name, value)`: Set gauge value
- `increment(metric_name, tags=None)`: Increment counter
- `get_count(metric_name, tags=None)`: Get counter value
- `get_summary()`: Get complete metrics summary

### 3. StructuredLogger

Provides structured logging with configurable log levels:

- `debug(message, **context)`: Debug-level logging
- `info(message, **context)`: Info-level logging
- `warning(message, **context)`: Warning-level logging
- `error(message, **context)`: Error-level logging
- `critical(message, **context)`: Critical-level logging

All methods accept arbitrary keyword arguments for structured context.

### 4. DistributedTracer

Distributed tracing with span-based operation tracking:

- `start_span(operation_name, attributes=None)`: Start new span
- `finish_span(span)`: Finish and record span
- `get_traces()`: Get all completed traces

Spans automatically track:
- Operation name
- Start/end timestamps
- Duration
- Custom attributes

### 5. ObservabilityStack

Main orchestration class implementing all Standard metrics:

#### Standard Metrics Implemented

##### pdm.pipeline.duration_seconds
```python
stack.record_pipeline_duration(duration_secs)
# Alert: HIGH if duration > 1800s (30 minutes)
```

##### pdm.posterior.nonconvergent_count (CRITICAL)
```python
stack.record_nonconvergent_chain(chain_id, reason)
# Alert: CRITICAL (every occurrence)
```

##### pdm.memory.peak_mb
```python
stack.record_memory_peak(memory_mb)
# Alert: WARNING if memory > 16000 MB (16GB)
```

##### pdm.evidence.hoop_test_fail_count
```python
stack.record_hoop_test_failure(question, missing)
# Alert: HIGH if total failures > 5
```

##### pdm.dimension.avg_score_D6
```python
stack.record_dimension_score(dimension, score)
# Alert: CRITICAL if D6 score < 0.55
```

#### Distributed Tracing

```python
with stack.trace_operation('operation_name', **attributes) as span:
    # Perform operation
    pass
# Span automatically tracked with duration
```

## Usage Examples

### Basic Usage

```python
from infrastructure.observability import ObservabilityConfig, ObservabilityStack

# Configure
config = ObservabilityConfig(
    metrics_backend='in_memory',
    log_level='INFO',
    trace_backend='in_memory'
)

# Create stack
observability = ObservabilityStack(config)

# Record metrics
observability.record_pipeline_duration(1500.0)
observability.record_dimension_score('D6', 0.65)
observability.record_memory_peak(12000.0)

# Get summary
metrics = observability.get_metrics_summary()
```

### Pipeline Integration

```python
import time

# Track overall pipeline
pipeline_start = time.time()

# Trace each phase
with observability.trace_operation('extract_document', plan='PDM_001') as span:
    # Extraction logic
    observability.record_memory_peak(8500.0)

with observability.trace_operation('bayesian_inference', chains=4) as span:
    # Inference logic
    observability.record_nonconvergent_chain('chain_3', 'R_hat=1.15')

# Record total duration
observability.record_pipeline_duration(time.time() - pipeline_start)
```

### Alert Monitoring

All alerts are automatically logged at appropriate levels:

- **CRITICAL**: System cannot proceed, requires immediate attention
- **HIGH**: Significant issue, may impact results
- **WARNING**: Issue detected, monitoring recommended
- **INFO**: Informational message

## Alert Thresholds

| Metric | Threshold | Alert Level | Rationale |
|--------|-----------|-------------|-----------|
| Pipeline Duration | > 1800s (30min) | HIGH | Performance degradation |
| Memory Peak | > 16000 MB (16GB) | WARNING | Resource exhaustion risk |
| Hoop Test Failures | > 5 failures | HIGH | Systematic evidence issues |
| D6 Score | < 0.55 | CRITICAL | Theory structure failure |
| Non-convergent Chain | Any occurrence | CRITICAL | Inference quality failure |

## Testing

### Run Tests

```bash
python3 test_observability.py
```

### Run Examples

```bash
python3 example_observability.py
```

## Integration with Existing Components

### PDM Orchestrator Integration

The ObservabilityStack can replace or extend the existing `MetricsCollector` in `orchestration/pdm_orchestrator.py`:

```python
from infrastructure.observability import ObservabilityConfig, ObservabilityStack

class PDMOrchestrator:
    def __init__(self, config):
        # Replace simple MetricsCollector with ObservabilityStack
        obs_config = ObservabilityConfig(log_level='INFO')
        self.observability = ObservabilityStack(obs_config)
        
    async def analyze_plan(self, pdf_path: str):
        start_time = time.time()
        
        with self.observability.trace_operation('analyze_plan', pdf=pdf_path):
            # Execute phases
            result = await self._execute_phases(pdf_path, run_id)
            
        # Record metrics
        duration = time.time() - start_time
        self.observability.record_pipeline_duration(duration)
        
        return result
```

### Harmonic Front Integration

Track D6 metrics from Harmonic Front 4:

```python
# In contradiction detection
observability.record_dimension_score('D6', d6_score)

# In hoop test validation
if not test_passed:
    observability.record_hoop_test_failure(question_id, missing_items)

# In Bayesian inference
if not chain_converged:
    observability.record_nonconvergent_chain(chain_id, convergence_reason)
```

## Design Principles

### 1. Backend-Agnostic
All components accept a `backend` parameter, allowing future integration with:
- Prometheus
- StatsD
- DataDog
- Custom backends

### 2. Minimal Dependencies
Implementation uses only Python standard library (no pandas, numpy, etc. required).

### 3. Type Safety
Full type hints for all public APIs using `typing` module.

### 4. Immutable Metrics
Metrics are append-only; past values are never modified.

### 5. Structured Context
All logging includes structured context for easy parsing and analysis.

### 6. Zero Configuration Default
Works out-of-box with sensible defaults; configuration is optional.

## Performance Characteristics

- **Memory**: O(n) where n = number of metric recordings
- **CPU**: O(1) for all metric recording operations
- **Storage**: In-memory by default; persistent storage requires backend integration

## Future Enhancements

### Planned Features
1. Persistent storage backend integration
2. Real-time alerting webhooks
3. Metrics aggregation and rollup
4. Trace sampling for high-volume scenarios
5. Integration with external monitoring systems (Prometheus, Grafana)

### Migration Path
The current implementation provides a migration path:

1. **Phase 1** (Current): In-memory implementation
2. **Phase 2**: Add Prometheus exporter backend
3. **Phase 3**: Add distributed tracing backend (Jaeger/Zipkin)
4. **Phase 4**: Add alerting integration (PagerDuty/Slack)

## Compliance

This implementation satisfies all requirements from F4.4:

- ✅ ObservabilityStack class with all required methods
- ✅ MetricsCollector with histogram, gauge, increment support
- ✅ StructuredLogger with configurable log levels
- ✅ DistributedTracer with span context manager
- ✅ All Standard metrics implemented:
  - ✅ pdm.pipeline.duration_seconds
  - ✅ pdm.posterior.nonconvergent_count
  - ✅ pdm.memory.peak_mb
  - ✅ pdm.evidence.hoop_test_fail_count
  - ✅ pdm.dimension.avg_score_D{N}
- ✅ All alert thresholds implemented
- ✅ trace_operation context manager
- ✅ Type hints and validation
- ✅ Comprehensive test coverage

## Files Created

1. `infrastructure/__init__.py` - Module exports
2. `infrastructure/observability.py` - Complete implementation (440 lines)
3. `test_observability.py` - Comprehensive test suite
4. `example_observability.py` - Usage examples
5. `OBSERVABILITY_README.md` - This documentation

## References

- Orchestration Module: `orchestration/pdm_orchestrator.py`
- Harmonic Front 4: `HARMONIC_FRONT_4_IMPLEMENTATION.md`
- Integration Guide: `INTEGRATION_GUIDE.md`
