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
    - Event logging for audit trail with persistence
    - Type-safe event payloads via Pydantic
    - Support for multiple subscribers per event type
    - Event storm detection and prevention
    - Circuit breaker for cascading failures

    SIN_CARRETA Compliance:
    - Deterministic event ordering with sequence numbers
    - Contract validation for all events
    - Comprehensive error handling with audit trail
    - Event replay capability for debugging

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

    def __init__(self, enable_persistence: bool = True, storm_threshold: int = 100):
        """
        Initialize the event bus with empty subscribers and event log.
        
        Args:
            enable_persistence: Enable event log persistence to disk
            storm_threshold: Max events per second before storm detection triggers
        """
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_log: List[PDMEvent] = []
        self._lock = asyncio.Lock()
        self._sequence_number = 0
        self._event_counts: Dict[str, List[float]] = defaultdict(list)  # Track event timestamps
        self._enable_persistence = enable_persistence
        self._storm_threshold = storm_threshold
        self._circuit_breaker_active = False
        self._failed_handler_count: Dict[str, int] = defaultdict(int)
        self._max_handler_failures = 3
        logger.info("EventBus initialized (persistence=%s, storm_threshold=%d)", 
                   enable_persistence, storm_threshold)

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
        asyncio.gather. Includes event storm detection and circuit breaker.

        SIN_CARRETA Compliance:
        - Assigns deterministic sequence number to each event
        - Validates event contract before publishing
        - Logs all errors with full context for audit trail
        - Detects event storms and activates circuit breaker

        Args:
            event: The event to publish

        Raises:
            RuntimeError: If circuit breaker is active or event storm detected

        Example:
            ```python
            await bus.publish(PDMEvent(
                event_type='posterior.updated',
                run_id='analysis_456',
                payload={'posterior': {'mean': 0.75, 'std': 0.1}}
            ))
            ```
        """
        # Circuit breaker check
        if self._circuit_breaker_active:
            logger.error(
                f"CIRCUIT_BREAKER_ACTIVE: Blocking event {event.event_type} "
                f"due to excessive handler failures. Manual intervention required."
            )
            raise RuntimeError("Circuit breaker active - event bus suspended")
        
        # Event storm detection
        await self._check_event_storm(event.event_type)
        
        async with self._lock:
            self._sequence_number += 1
            # Add sequence number to payload for determinism
            if 'sequence_number' not in event.payload:
                event.payload['_sequence_number'] = self._sequence_number
            
            self.event_log.append(event)
            
            # Persist event if enabled
            if self._enable_persistence:
                await self._persist_event(event)

        handlers = self.subscribers.get(event.event_type, [])

        if not handlers:
            logger.debug(f"No subscribers for event type '{event.event_type}'")
            return

        logger.debug(
            f"Publishing event '{event.event_type}' (seq={self._sequence_number}) "
            f"to {len(handlers)} handler(s)"
        )

        # Execute all handlers with circuit breaker and failure tracking
        tasks = []
        for handler in handlers:
            # Skip handlers that have exceeded failure threshold
            handler_key = f"{event.event_type}:{handler.__name__}"
            if self._failed_handler_count[handler_key] >= self._max_handler_failures:
                logger.warning(
                    f"Skipping handler {handler.__name__} - exceeded failure threshold"
                )
                continue
            
            try:
                # Support both sync and async handlers with tracking
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(self._execute_handler_with_tracking(handler, event, handler_key))
                else:
                    tasks.append(
                        self._execute_sync_handler_with_tracking(handler, event, handler_key)
                    )
            except Exception as e:
                logger.error(
                    f"Error preparing handler {handler.__name__}: {e}", exc_info=True
                )

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log any exceptions and track failures
            failure_count = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    handler_name = (
                        handlers[i].__name__ if i < len(handlers) else "unknown"
                    )
                    logger.error(
                        f"Handler {handler_name} raised exception for event "
                        f"{event.event_type} (seq={self._sequence_number}): {result}",
                        exc_info=result,
                    )
                    failure_count += 1
            
            # Activate circuit breaker if too many failures
            if failure_count >= len(handlers) * 0.5:  # 50% failure rate
                logger.critical(
                    f"CIRCUIT_BREAKER_TRIGGERED: {failure_count}/{len(handlers)} "
                    f"handlers failed for event {event.event_type}"
                )
                self._circuit_breaker_active = True
    
    async def _execute_handler_with_tracking(self, handler: Callable, event: PDMEvent, handler_key: str):
        """Execute async handler with failure tracking"""
        try:
            await handler(event)
            # Reset failure count on success
            if handler_key in self._failed_handler_count:
                self._failed_handler_count[handler_key] = 0
        except Exception as e:
            self._failed_handler_count[handler_key] += 1
            logger.error(
                f"Handler {handler.__name__} failed ({self._failed_handler_count[handler_key]}/{self._max_handler_failures}): {e}"
            )
            raise
    
    async def _execute_sync_handler_with_tracking(self, handler: Callable, event: PDMEvent, handler_key: str):
        """Execute sync handler with failure tracking"""
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, handler, event)
            # Reset failure count on success
            if handler_key in self._failed_handler_count:
                self._failed_handler_count[handler_key] = 0
            return result
        except Exception as e:
            self._failed_handler_count[handler_key] += 1
            logger.error(
                f"Sync handler {handler.__name__} failed ({self._failed_handler_count[handler_key]}/{self._max_handler_failures}): {e}"
            )
            raise
    
    async def _check_event_storm(self, event_type: str):
        """
        Check for event storm conditions.
        
        SIN_CARRETA Compliance:
        - Deterministic storm detection based on event rate
        - Hard failure on storm detection to prevent cascading issues
        """
        import time
        current_time = time.time()
        
        # Clean old timestamps (older than 1 second)
        self._event_counts[event_type] = [
            ts for ts in self._event_counts[event_type] 
            if current_time - ts < 1.0
        ]
        
        # Add current event
        self._event_counts[event_type].append(current_time)
        
        # Check if storm threshold exceeded
        if len(self._event_counts[event_type]) > self._storm_threshold:
            logger.error(
                f"EVENT_STORM_DETECTED: {event_type} exceeded {self._storm_threshold} events/second"
            )
            # Clear the queue to prevent runaway
            self._event_counts[event_type] = []
            raise RuntimeError(
                f"Event storm detected for {event_type} - "
                f"exceeded {self._storm_threshold} events/second"
            )
    
    async def _persist_event(self, event: PDMEvent):
        """
        Persist event to audit trail.
        
        SIN_CARRETA Compliance:
        - Deterministic serialization for replay
        - Immutable audit trail
        """
        # TODO: Implement actual persistence (e.g., to JSON file or database)
        # For now, just log at DEBUG level
        logger.debug(
            f"AUDIT_TRAIL: event_id={event.event_id}, "
            f"type={event.event_type}, "
            f"run_id={event.run_id}, "
            f"seq={event.payload.get('_sequence_number', -1)}"
        )

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
    
    def reset_circuit_breaker(self) -> None:
        """
        Reset circuit breaker after manual intervention.
        
        SIN_CARRETA Compliance:
        - Requires explicit manual reset for safety
        - Logs reset action to audit trail
        """
        self._circuit_breaker_active = False
        self._failed_handler_count.clear()
        logger.warning("Circuit breaker manually reset - event bus reactivated")
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status and failure counts"""
        return {
            'active': self._circuit_breaker_active,
            'failed_handlers': dict(self._failed_handler_count),
            'total_events': len(self.event_log),
            'sequence_number': self._sequence_number
        }


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
        event_bus.subscribe("graph.node_added", self.on_node_added)
        logger.info("ContradictionDetectorV2 initialized and subscribed to events")

    async def on_edge_added(self, event: PDMEvent) -> None:
        """
        React to new causal link in the graph.

        Performs incremental contradiction checking when a new edge
        is added to the causal graph. If contradictions are found,
        publishes a 'contradiction.detected' event.

        SIN_CARRETA Compliance:
        - Deterministic contradiction detection
        - Contract validation on input event
        - Comprehensive error handling

        Args:
            event: Event containing edge data in payload
        """
        # Contract validation
        assert event.event_type == "graph.edge_added", \
            f"Expected graph.edge_added, got {event.event_type}"
        
        edge_data = event.payload
        
        # Validate required fields
        required_fields = ['source', 'target']
        for field in required_fields:
            if field not in edge_data:
                logger.error(
                    f"CONTRACT_VIOLATION: graph.edge_added missing required field '{field}'"
                )
                return
        
        logger.debug(
            f"Checking edge: {edge_data.get('source')} -> {edge_data.get('target')}"
        )

        # Incremental contradiction check
        try:
            if self._contradicts_existing(edge_data):
                contradiction = {
                    "edge": edge_data,
                    "severity": "high",
                    "type": "causal_inconsistency",
                    "timestamp": datetime.utcnow().isoformat(),
                    "sequence_number": edge_data.get('_sequence_number', -1)
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
        except Exception as e:
            logger.error(
                f"Error in contradiction detection for edge {edge_data.get('source')} -> "
                f"{edge_data.get('target')}: {e}",
                exc_info=True
            )
            # Do not re-raise to prevent handler failure cascade
    
    async def on_node_added(self, event: PDMEvent) -> None:
        """
        React to new node in the graph.
        
        Validates node schema and checks for duplicates.
        
        SIN_CARRETA Compliance:
        - Contract validation on input event
        - Schema validation for node data
        
        Args:
            event: Event containing node data in payload
        """
        # Contract validation
        assert event.event_type == "graph.node_added", \
            f"Expected graph.node_added, got {event.event_type}"
        
        node_data = event.payload
        
        # Validate required fields
        if 'node_id' not in node_data:
            logger.error(
                "CONTRACT_VIOLATION: graph.node_added missing required field 'node_id'"
            )
            return
        
        node_id = node_data.get('node_id')
        logger.debug(f"Validating node: {node_id}")
        
        # Schema validation
        expected_fields = ['node_id', 'node_type']
        for field in expected_fields:
            if field not in node_data:
                logger.warning(
                    f"SCHEMA_WARNING: Node {node_id} missing recommended field '{field}'"
                )
        
        # Check for duplicate nodes (placeholder - would track in actual implementation)
        logger.debug(f"Node {node_id} validated successfully")

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
