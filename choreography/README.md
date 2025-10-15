# FRENTE 3: COREOGRAFÍA (Comunicación Descentralizada)

Event-driven architecture for decoupled component communication in the FARFAN-2.0 PDM analysis system.

## Overview

This module implements two key patterns for event-driven communication:

1. **Event Bus for Phase Transitions (F3.1)**: Pub/sub event bus for real-time communication between extractors, validators, and auditors
2. **Streaming Evidence Pipeline (F3.2)**: Asynchronous evidence processing with incremental Bayesian updates

## Architecture

```
choreography/
├── __init__.py           # Module exports
├── event_bus.py          # F3.1: Event bus implementation
└── evidence_stream.py    # F3.2: Streaming pipeline
```

## Key Features

### Event Bus (F3.1)

- **Decoupled Communication**: Components communicate via events without direct dependencies
- **Asynchronous Execution**: Handlers execute concurrently using `asyncio.gather`
- **Event Logging**: Complete audit trail of all events
- **Type Safety**: Pydantic models for event structure validation
- **Real-time Validation**: Validators react immediately to graph construction events

### Streaming Pipeline (F3.2)

- **Memory Efficient**: Process massive PDM documents without loading entire content
- **Incremental Updates**: Bayesian posteriors updated chunk-by-chunk
- **Real-time Feedback**: Intermediate results published via event bus
- **Early Termination**: Can stop processing when sufficient evidence found
- **Async Iterator**: Native Python async/await support

## Usage Examples

### Basic Event Bus

```python
import asyncio
from choreography.event_bus import EventBus, PDMEvent

async def main():
    # Create event bus
    bus = EventBus()
    
    # Subscribe to events
    async def on_graph_update(event: PDMEvent):
        print(f"Graph updated: {event.payload}")
    
    bus.subscribe('graph.edge_added', on_graph_update)
    
    # Publish event
    await bus.publish(PDMEvent(
        event_type='graph.edge_added',
        run_id='run_123',
        payload={'source': 'A', 'target': 'B'}
    ))

asyncio.run(main())
```

### Contradiction Detection with Events

```python
from choreography.event_bus import EventBus, ContradictionDetectorV2

bus = EventBus()

# Detector subscribes to graph events automatically
detector = ContradictionDetectorV2(bus)

# When edges are added, detector checks for contradictions
# and publishes 'contradiction.detected' events if found
```

### Streaming Evidence Analysis

```python
import asyncio
from choreography.evidence_stream import (
    EvidenceStream,
    StreamingBayesianUpdater,
    MechanismPrior
)

async def main():
    # Create evidence stream from semantic chunks
    chunks = [...]  # List of SemanticChunk dictionaries
    stream = EvidenceStream(chunks)
    
    # Define prior belief
    prior = MechanismPrior(
        mechanism_name='education_quality',
        prior_mean=0.5,
        prior_std=0.2
    )
    
    # Stream and update
    updater = StreamingBayesianUpdater()
    posterior = await updater.update_from_stream(stream, prior)
    
    print(f"Posterior mean: {posterior.posterior_mean:.3f}")
    print(f"Evidence count: {posterior.evidence_count}")

asyncio.run(main())
```

### Full Integration

```python
import asyncio
from choreography.event_bus import EventBus, ContradictionDetectorV2
from choreography.evidence_stream import (
    EvidenceStream,
    StreamingBayesianUpdater,
    MechanismPrior
)

async def main():
    # Create shared event bus
    bus = EventBus()
    
    # Setup detector (subscribes to graph events)
    detector = ContradictionDetectorV2(bus)
    
    # Setup streaming updater with event publishing
    updater = StreamingBayesianUpdater(event_bus=bus)
    
    # Monitor all events
    async def log_event(event):
        print(f"Event: {event.event_type}")
    
    bus.subscribe('posterior.updated', log_event)
    bus.subscribe('contradiction.detected', log_event)
    
    # Process evidence stream
    stream = EvidenceStream(chunks)
    prior = MechanismPrior('mechanism_name', 0.5, 0.2)
    
    posterior = await updater.update_from_stream(
        stream, prior, run_id='analysis_001'
    )
    
    # Check event log
    events = bus.get_event_log(run_id='analysis_001')
    print(f"Total events: {len(events)}")

asyncio.run(main())
```

## Event Types

### Standard Events

- `graph.edge_added`: New causal edge added to graph
- `graph.node_added`: New node added to graph
- `contradiction.detected`: Contradiction found during validation
- `posterior.updated`: Bayesian posterior updated with new evidence
- `validation.completed`: Validation check completed
- `extraction.completed`: Evidence extraction completed

### Event Payload Structure

Events carry structured payloads via Pydantic models:

```python
PDMEvent(
    event_id='auto-generated-uuid',
    event_type='graph.edge_added',
    timestamp=datetime.utcnow(),
    run_id='analysis_run_123',
    payload={
        'source': 'objetivo_A',
        'target': 'resultado_B',
        'relation': 'contributes_to',
        'confidence': 0.85
    }
)
```

## Benefits

### Reduced Coupling

Components communicate via events instead of direct method calls:

- **Before**: `orchestrator.add_edge()` → `validator.check_contradiction()`
- **After**: `orchestrator.publish('graph.edge_added')` → validator auto-reacts

### Real-time Validation

Validators react immediately to graph construction:

```python
# Validator subscribes once during initialization
detector = ContradictionDetectorV2(bus)

# Then automatically validates every new edge
await bus.publish(PDMEvent(event_type='graph.edge_added', ...))
# → detector.on_edge_added() called automatically
```

### Incremental Processing

Stream large documents without memory exhaustion:

```python
# Traditional: Load entire document
all_chunks = load_document()  # May exhaust memory
posterior = analyzer.analyze(all_chunks)

# Streaming: Process incrementally
stream = EvidenceStream(chunks)
async for chunk in stream:
    # Process one chunk at a time
    posterior = update_posterior(chunk)
```

### Flexible Auditing

Add new auditors without modifying existing code:

```python
# New auditor just subscribes to relevant events
class FinancialAuditor:
    def __init__(self, bus):
        bus.subscribe('posterior.updated', self.audit_costs)
    
    async def audit_costs(self, event):
        # Verify financial claims
        pass
```

## Testing

Run the test suite:

```bash
# Standalone tests (no pytest needed)
python test_choreography_standalone.py

# With pytest (if available)
pytest test_choreography.py -v
```

Run the demonstration:

```bash
python demo_choreography.py
```

## Integration with Existing Modules

### With `dereck_beach` (CDAF Framework)

```python
from choreography.event_bus import EventBus
from dereck_beach import CDAFFramework

# Enhance CDAF with event-driven communication
bus = EventBus()
framework = CDAFFramework(config_path, output_dir)

# Subscribe to causal extraction events
bus.subscribe('mechanism.extracted', framework.validate_mechanism)
```

### With `contradiction_deteccion.py`

```python
from choreography.event_bus import EventBus
from contradiction_deteccion import PolicyContradictionDetectorV2

bus = EventBus()
detector = PolicyContradictionDetectorV2()

# Real-time contradiction checking
async def check_contradiction(event):
    result = detector.detect(event.payload['text'])
    if result['contradictions']:
        await bus.publish(PDMEvent(
            event_type='contradiction.detected',
            payload=result
        ))

bus.subscribe('text.analyzed', check_contradiction)
```

### With `emebedding_policy.py`

```python
from choreography.evidence_stream import EvidenceStream
from emebedding_policy import AdvancedSemanticChunker

# Create chunks
chunker = AdvancedSemanticChunker(config)
chunks = chunker.chunk_document(text, metadata)

# Stream for analysis
stream = EvidenceStream(chunks)
async for chunk in stream:
    # Process chunk embeddings
    pass
```

## Performance Considerations

- **Async Execution**: All event handlers run concurrently
- **Memory Efficient**: Streaming processes one chunk at a time
- **Event Log**: Grows with events; use `bus.clear_log()` periodically
- **Handler Exceptions**: Caught and logged, don't crash the bus

## Future Enhancements

- [ ] Event persistence (save to database)
- [ ] Event replay for debugging
- [ ] Event filtering and routing
- [ ] Distributed event bus (Redis/RabbitMQ)
- [ ] Event schema validation
- [ ] Metrics and monitoring

## References

- Problem Statement: FRENTE 3 in `HARMONIC_FRONT_4_IMPLEMENTATION.md`
- Related Modules: `dereck_beach`, `contradiction_deteccion.py`, `emebedding_policy.py`
- Design Pattern: Observer/Pub-Sub pattern with async/await
- Mathematical Foundation: Sequential Bayesian updating

## License

Part of FARFAN-2.0 PDM Analysis Framework
