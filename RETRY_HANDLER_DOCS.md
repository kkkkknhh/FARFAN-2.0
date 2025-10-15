# RetryHandler with Circuit Breaker Integration

## Overview

The `RetryHandler` class provides robust retry logic with exponential backoff and circuit breaker integration for external dependency calls in the FARFAN 2.0 pipeline. It wraps calls to:

- **PDF Parsing** (PyMuPDF operations)
- **spaCy Model Loading**
- **DNP API Calls** (when external APIs are used)
- **Embedding Service Operations** (SentenceTransformer, CrossEncoder)

## Features

### 1. Exponential Backoff with Jitter
- Configurable base delay, max retries, and exponential base
- Random jitter prevents thundering herd problem
- Max delay cap prevents excessive wait times

### 2. Circuit Breaker Pattern
- Three states: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)
- Automatic state transitions based on failure/success thresholds
- Fast-fail when circuit is OPEN to prevent cascading failures

### 3. Per-Dependency Configuration
- Individual retry configs for each dependency type
- Separate circuit breakers track failures independently
- Configurable failure thresholds and recovery timeouts

### 4. Comprehensive Tracking
- Retry attempt history with timestamps
- Circuit breaker statistics (success rate, failure counts)
- Per-operation monitoring

## Usage

### Basic Decorator Pattern

```python
from retry_handler import RetryHandler, DependencyType

handler = RetryHandler()

@handler.with_retry(DependencyType.PDF_PARSER)
def parse_pdf(path):
    import fitz
    return fitz.open(path)

# Automatically retries on failure with exponential backoff
doc = parse_pdf("plan.pdf")
```

### Context Manager Pattern

```python
with handler.retry_context(DependencyType.SPACY_MODEL, "load_model"):
    nlp = spacy.load("es_core_news_lg")
```

### Custom Configuration

```python
from retry_handler import RetryConfig

handler.configure(DependencyType.PDF_PARSER, RetryConfig(
    base_delay=1.0,          # Start with 1s delay
    max_retries=4,           # Try up to 5 times total
    exponential_base=2.0,    # Double delay each retry
    jitter_factor=0.2,       # ±20% random jitter
    max_delay=30.0,          # Cap at 30s
    failure_threshold=5,     # Open circuit after 5 failures
    recovery_timeout=60.0,   # Wait 60s before testing recovery
    success_threshold=2      # Need 2 successes to close circuit
))
```

## Integration Points

### 1. Orchestrator (`orchestrator.py`)

The `FARFANOrchestrator` is initialized with a `RetryHandler` and configures all dependencies:

```python
orchestrator = FARFANOrchestrator(
    output_dir="./results",
    retry_handler=RetryHandler()
)
```

PDF operations in `_stage_extract_document()` are wrapped:

```python
@self.retry_handler.with_retry(
    DependencyType.PDF_PARSER,
    operation_name="extract_text",
    exceptions=(IOError, OSError, RuntimeError)
)
def extract_text():
    return self.cdaf.pdf_processor.extract_text()
```

DNP validation calls are also protected:

```python
@self.retry_handler.with_retry(
    DependencyType.DNP_API,
    operation_name="validar_proyecto",
    exceptions=(ConnectionError, TimeoutError, IOError)
)
def validate_project(node_id, node):
    return self.dnp_validator.validar_proyecto_integral(...)
```

### 2. CDAF Framework (`dereck_beach`)

The `CDAFFramework` initializes with retry handler support:

```python
# In __init__:
from retry_handler import get_retry_handler, DependencyType
self.retry_handler = get_retry_handler()

# spaCy model loading with retry:
@self.retry_handler.with_retry(
    DependencyType.SPACY_MODEL,
    operation_name="load_spacy_model",
    exceptions=(OSError, IOError, ImportError)
)
def load_spacy_with_retry():
    try:
        return spacy.load("es_core_news_lg")
    except OSError:
        return spacy.load("es_core_news_sm")
```

The `PDFProcessor` accepts a retry handler and wraps PDF operations:

```python
self.pdf_processor = PDFProcessor(self.config, retry_handler=self.retry_handler)
```

### 3. Embedding System (`embeddings_policy`)

The `PolicyAnalysisEmbedder` loads models with retry protection:

```python
embedder = PolicyAnalysisEmbedder(config, retry_handler=handler)

# SentenceTransformer loading:
@retry_handler.with_retry(
    DependencyType.EMBEDDING_SERVICE,
    operation_name="load_sentence_transformer",
    exceptions=(OSError, IOError, ConnectionError, RuntimeError)
)
def load_embedding_model():
    return SentenceTransformer(config.embedding_model)
```

Text encoding operations are also wrapped:

```python
@self.retry_handler.with_retry(
    DependencyType.EMBEDDING_SERVICE,
    operation_name="encode_texts",
    exceptions=(ConnectionError, TimeoutError, RuntimeError, OSError)
)
def encode_with_retry():
    return self.embedding_model.encode(texts, ...)
```

## Default Configurations

### PDF Parser
- Base delay: 1.0s
- Max retries: 4
- Failure threshold: 5
- Recovery timeout: 60s

### spaCy Model
- Base delay: 2.0s
- Max retries: 3
- Failure threshold: 3
- Recovery timeout: 120s

### DNP API
- Base delay: 1.5s
- Max retries: 5
- Failure threshold: 7
- Recovery timeout: 90s

### Embedding Service
- Base delay: 1.0s
- Max retries: 4
- Failure threshold: 6
- Recovery timeout: 75s

## Circuit Breaker State Transitions

```
CLOSED (normal operation)
   │
   ├─> failure_count >= failure_threshold
   │
   ▼
OPEN (reject requests, fast-fail)
   │
   ├─> time >= recovery_timeout
   │
   ▼
HALF_OPEN (test recovery)
   │
   ├─> success_count >= success_threshold ──> CLOSED
   │
   └─> any failure ──> OPEN
```

## Monitoring

### Get Circuit Breaker Stats

```python
# All dependencies
all_stats = handler.get_stats()

# Specific dependency
pdf_stats = handler.get_stats(DependencyType.PDF_PARSER)
print(f"State: {pdf_stats['state']}")
print(f"Success rate: {pdf_stats['success_rate']:.1%}")
print(f"Total failures: {pdf_stats['total_failures']}")
```

### Get Retry History

```python
# Last 100 attempts for all dependencies
history = handler.get_retry_history(limit=100)

# PDF parser attempts only
pdf_history = handler.get_retry_history(DependencyType.PDF_PARSER, limit=50)

for record in pdf_history:
    print(f"{record['operation']}: attempt {record['attempt']}, "
          f"success={record['success']}, delay={record['delay']:.2f}s")
```

### Reset Circuit Breaker

```python
# Reset specific dependency
handler.reset(DependencyType.PDF_PARSER)

# Reset all
handler.reset()
```

## Exception Handling

### CircuitBreakerOpenError

Raised when attempting to call a service with an open circuit breaker:

```python
from retry_handler import CircuitBreakerOpenError

try:
    result = my_retryable_function()
except CircuitBreakerOpenError as e:
    logger.error(f"Service unavailable: {e}")
    # Handle gracefully or notify admin
```

### Wrapped Exceptions

The retry handler catches and retries on specified exceptions:

```python
@handler.with_retry(
    DependencyType.PDF_PARSER,
    exceptions=(IOError, OSError, RuntimeError)  # Only retry these
)
def my_function():
    ...
```

Other exceptions will propagate immediately without retry.

## Best Practices

1. **Configure appropriately**: Adjust retry settings based on operation characteristics
   - Fast operations (API calls): shorter delays, more retries
   - Slow operations (model loading): longer delays, fewer retries

2. **Monitor circuit breakers**: Track circuit breaker states in production
   - Alert when circuits open
   - Log recovery attempts

3. **Use specific exceptions**: Only retry on transient failures
   - Network errors: ConnectionError, TimeoutError
   - File I/O: IOError, OSError
   - Don't retry on logic errors (ValueError, TypeError)

4. **Test failure scenarios**: Ensure circuit breakers open/close correctly
   - Simulate service outages
   - Verify recovery behavior

5. **Tune failure thresholds**: Balance between resilience and fast-fail
   - Lower threshold: Fail fast, less system load
   - Higher threshold: More resilient to transient issues

## Testing

Run the test suite:

```bash
python test_retry_handler.py
```

Tests cover:
- Basic retry with success
- Retry exhaustion
- Circuit breaker opening
- Circuit breaker recovery
- Exponential backoff timing
- Context manager usage
- Retry history tracking

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RetryHandler                           │
├─────────────────────────────────────────────────────────────┤
│  Configurations:                                            │
│  - DependencyType → RetryConfig mapping                    │
│                                                              │
│  Circuit Breakers:                                          │
│  - DependencyType → CircuitBreakerStats mapping            │
│  - State: CLOSED/OPEN/HALF_OPEN                            │
│  - Failure/Success counters                                 │
│                                                              │
│  Retry Logic:                                               │
│  - Exponential backoff: delay = base * (exp_base ^ attempt)│
│  - Jitter: delay ± (delay * jitter_factor)                 │
│  - Max delay cap                                            │
│                                                              │
│  History:                                                   │
│  - Retry attempts with timestamps                          │
│  - Success/failure tracking per operation                  │
└─────────────────────────────────────────────────────────────┘
              │
              ├──> PDF Parser (fitz operations)
              ├──> spaCy Model (load operations)
              ├──> DNP API (validation calls)
              └──> Embedding Service (encode operations)
```

## Future Enhancements

1. **Adaptive retry timing**: Learn optimal delays from historical data
2. **Bulkhead pattern**: Limit concurrent operations per dependency
3. **Metrics export**: Prometheus/StatsD integration
4. **Distributed circuit breaker**: Share state across instances (Redis)
5. **Rate limiting**: Prevent overwhelming recovering services
