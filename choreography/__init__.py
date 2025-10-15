"""
FRENTE 3: COREOGRAFÍA (Comunicación Descentralizada)
======================================================
Event-driven patterns for decoupled component communication.

Modules:
- event_bus: Event Bus for Phase Transitions
- evidence_stream: Streaming Evidence Pipeline
"""

from choreography.event_bus import EventBus, PDMEvent
from choreography.evidence_stream import EvidenceStream, StreamingBayesianUpdater

__all__ = [
    "EventBus",
    "PDMEvent",
    "EvidenceStream",
    "StreamingBayesianUpdater",
]
