"""
Event Bus for Phase Transitions (F3.1)
=======================================
Decoupled event bus for communication between components.
Allows extractors, validators, and auditors to subscribe to events
without direct coupling.

Architecture:
- PDMEvent: Base event model with type safety via Pydantic
- EventBus: Pub/Sub pattern with async handler execution
- Type-safe, production-ready, extensible
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ============================================================================
# EVENT MODELS
# ============================================================================


class PDMEvent(BaseModel):
    """
    Base event class for the PDM system.

    All events in the system inherit from this base class, providing
    structured event data with traceability and metadata.

    Attributes:
        event_id: Unique identifier for this event instance
        event_type: Type/category of the event (e.g., 'graph.edge_added')
        timestamp: When the event was created
        run_id: Identifier for the analysis run generating this event
        payload: Event-specific data as a dictionary
    """

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    run_id: str
    payload: Dict[str, Any]

    class Config:
        """Pydantic configuration"""

        frozen = False  # Allow modification if needed
        arbitrary_types_allowed = True


# ============================================================================
# EVENT BUS
# ============================================================================


class EventBus:
    """
    Decoupled event bus for inter-component communication.

    Implements pub/sub pattern allowing extractors, validators, and auditors
    to subscribe to events and react without direct coupling. This enables:
    - Real-time validation as data flows through the pipeline
    - Incremental contradiction detection during graph construction
    - Flexible addition of new auditors without modifying orchestrator

    Key Features:
    - Asynchronous handler execution with asyncio.gather
    - Event logging for audit trail
    - Type-safe event payloads via Pydantic
    - Support for multiple subscribers per event type

    Example:
        ```python
        bus = EventBus()

        # Subscribe to events
        async def on_edge_added(event: PDMEvent):
            print(f"New edge: {event.payload}")

        bus.subscribe('graph.edge_added', on_edge_added)

        # Publish events
        await bus.publish(PDMEvent(
            event_type='graph.edge_added',
            run_id='run_123',
            payload={'source': 'A', 'target': 'B'}
        ))
        ```
    """

    def __init__(self):
        """Initialize the event bus with empty subscribers and event log."""
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_log: List[PDMEvent] = []
        self._lock = asyncio.Lock()
        logger.info("EventBus initialized")

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """
        Register a handler for a specific event type.

        Handlers are called asynchronously when matching events are published.
        Multiple handlers can be registered for the same event type.

        Args:
            event_type: The type of event to subscribe to (e.g., 'graph.edge_added')
            handler: Async callable that accepts PDMEvent as parameter

        Example:
            ```python
            async def handle_contradiction(event: PDMEvent):
                severity = event.payload.get('severity')
                logger.warning(f"Contradiction detected: {severity}")

            bus.subscribe('contradiction.detected', handle_contradiction)
            ```
        """
        self.subscribers[event_type].append(handler)
        logger.debug(
            f"Subscribed handler {handler.__name__} to event type '{event_type}'"
        )

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """
        Unregister a handler from a specific event type.

        Args:
            event_type: The type of event to unsubscribe from
            handler: The handler to remove
        """
        if event_type in self.subscribers and handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
            logger.debug(
                f"Unsubscribed handler {handler.__name__} from event type '{event_type}'"
            )

    def _prepare_handler_task(self, handler: Callable, event: PDMEvent):
        """
        Prepare a task for handler execution.

        Args:
            handler: The handler to execute
            event: The event to pass to the handler

        Returns:
            Coroutine or task for execution, or None on error
        """
        if asyncio.iscoroutinefunction(handler):
            return handler(event)
        else:
            return asyncio.get_event_loop().run_in_executor(None, handler, event)

    def _collect_handler_tasks(self, handlers: List[Callable], event: PDMEvent) -> List:
        """
        Collect tasks from all handlers, handling preparation errors.

        Args:
            handlers: List of handlers to prepare
            event: Event to pass to handlers

        Returns:
            List of tasks ready for execution
        """
        tasks = []
        for handler in handlers:
            try:
                task = self._prepare_handler_task(handler, event)
                tasks.append(task)
            except Exception as e:
                logger.error(
                    f"Error preparing handler {handler.__name__}: {e}", exc_info=True
                )
        return tasks

    def _log_handler_exceptions(self, results: List, handlers: List[Callable]) -> None:
        """
        Log exceptions from handler execution results.

        Args:
            results: Results from asyncio.gather with return_exceptions=True
            handlers: List of handlers that were executed
        """
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                handler_name = (
                    handlers[i].__name__ if i < len(handlers) else "unknown"
                )
                logger.error(
                    f"Handler {handler_name} raised exception: {result}",
                    exc_info=result,
                )

    async def publish(self, event: PDMEvent) -> None:
        """
        Publish an event and execute all registered handlers asynchronously.

        Events are added to the event log for audit trail. All registered
        handlers for the event type are executed concurrently using
        asyncio.gather.

        Args:
            event: The event to publish

        Raises:
            Exception: If any handler raises an exception (logged but not propagated)

        Example:
            ```python
            await bus.publish(PDMEvent(
                event_type='posterior.updated',
                run_id='analysis_456',
                payload={'posterior': {'mean': 0.75, 'std': 0.1}}
            ))
            ```
        """
        async with self._lock:
            self.event_log.append(event)

        handlers = self.subscribers.get(event.event_type, [])

        if not handlers:
            logger.debug(f"No subscribers for event type '{event.event_type}'")
            return

        logger.debug(
            f"Publishing event '{event.event_type}' to {len(handlers)} handler(s)"
        )

        tasks = self._collect_handler_tasks(handlers, event)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            self._log_handler_exceptions(results, handlers)

    def get_event_log(
        self, event_type: str = None, run_id: str = None, limit: int = None
    ) -> List[PDMEvent]:
        """
        Retrieve events from the event log.

        Args:
            event_type: Filter by event type (optional)
            run_id: Filter by run ID (optional)
            limit: Maximum number of events to return (optional)

        Returns:
            List of events matching the filters
        """
        events = self.event_log

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if run_id:
            events = [e for e in events if e.run_id == run_id]

        if limit:
            events = events[-limit:]

        return events

    def clear_log(self) -> None:
        """Clear the event log. Useful for testing or memory management."""
        self.event_log.clear()
        logger.debug("Event log cleared")


# ============================================================================
# EXAMPLE USAGE: Contradiction Detector with Event Subscription
# ============================================================================


class ContradictionDetectorV2:
    """
    Example validator that subscribes to graph construction events.

    Demonstrates real-time contradiction detection by reacting to
    edge addition events during graph construction. This allows
    incremental validation without blocking the main pipeline.

    Example:
        ```python
        bus = EventBus()
        detector = ContradictionDetectorV2(event_bus=bus)

        # Detector automatically subscribes to 'graph.edge_added' events
        # When edges are added, detector checks for contradictions
        await bus.publish(PDMEvent(
            event_type='graph.edge_added',
            run_id='run_123',
            payload={
                'source': 'objetivo_A',
                'target': 'resultado_B',
                'relation': 'contributes_to'
            }
        ))
        # Detector will check this edge and publish 'contradiction.detected'
        # if issues are found
        ```
    """

    def __init__(self, event_bus: EventBus):
        """
        Initialize detector and subscribe to graph events.

        Args:
            event_bus: The event bus to subscribe to
        """
        self.event_bus = event_bus
        self.detected_contradictions: List[Dict[str, Any]] = []

        # Subscribe to graph construction events
        event_bus.subscribe("graph.edge_added", self.on_edge_added)
        logger.info("ContradictionDetectorV2 initialized and subscribed to events")

    async def on_edge_added(self, event: PDMEvent) -> None:
        """
        React to new causal link in the graph.

        Performs incremental contradiction checking when a new edge
        is added to the causal graph. If contradictions are found,
        publishes a 'contradiction.detected' event.

        Args:
            event: Event containing edge data in payload
        """
        edge_data = event.payload
        logger.debug(
            f"Checking edge: {edge_data.get('source')} -> {edge_data.get('target')}"
        )

        # Incremental contradiction check
        if self._contradicts_existing(edge_data):
            contradiction = {
                "edge": edge_data,
                "severity": "high",
                "type": "causal_inconsistency",
                "timestamp": datetime.utcnow().isoformat(),
            }

            self.detected_contradictions.append(contradiction)

            # Publish contradiction event for other components
            await self.event_bus.publish(
                PDMEvent(
                    event_type="contradiction.detected",
                    run_id=event.run_id,
                    payload=contradiction,
                )
            )

            logger.warning(
                f"Contradiction detected: {edge_data.get('source')} -> "
                f"{edge_data.get('target')}"
            )

    def _contradicts_existing(self, edge_data: Dict[str, Any]) -> bool:
        """
        Check if new edge contradicts existing graph structure.

        Placeholder for actual contradiction detection logic.
        In production, this would check for:
        - Temporal conflicts
        - Resource allocation mismatches
        - Logical incompatibilities
        - Causal cycles

        Args:
            edge_data: Data about the new edge

        Returns:
            True if contradiction detected, False otherwise
        """
        # Placeholder logic - replace with actual contradiction detection
        # For demonstration, we'll just check if source equals target
        source = edge_data.get("source", "")
        target = edge_data.get("target", "")

        # Self-loops are contradictions
        if source and target and source == target:
            return True

        # Check against known contradictions (placeholder)
        for contradiction in self.detected_contradictions:
            existing_edge = contradiction.get("edge", {})
            # If reverse edge exists, might be a contradiction
            if (
                existing_edge.get("source") == target
                and existing_edge.get("target") == source
            ):
                return True

        return False
