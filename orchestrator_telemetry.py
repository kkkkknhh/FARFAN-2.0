#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator Telemetry and Tracing
===================================

Intent: Emit telemetry events at all decision, start, and completion boundaries
Mechanism: Structured event emission with trace context and audit IDs
Constraint: No silent failures, all events are logged and traceable

SIN_CARRETA compliance: Explicit tracing, deterministic event ordering
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================================
# ENUMERATIONS
# ============================================================================


class EventType(Enum):
    """Telemetry event types"""

    PHASE_START = auto()
    PHASE_COMPLETE = auto()
    PHASE_ERROR = auto()
    DECISION_POINT = auto()
    CONTRACT_VALIDATION = auto()
    MODULE_INVOCATION = auto()
    AUDIT_CHECKPOINT = auto()


class EventSeverity(Enum):
    """Event severity levels"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass(frozen=True)
class TraceContext:
    """
    Intent: Immutable trace context for event correlation
    Mechanism: UUID-based trace and span IDs with hierarchical structure
    Constraint: All IDs must be valid UUIDs
    """

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        """Validate UUID format"""
        for field_name in ["trace_id", "span_id", "run_id"]:
            value = getattr(self, field_name)
            try:
                uuid.UUID(value)
            except ValueError:
                raise ValueError(f"{field_name} must be a valid UUID, got: {value}")

        if self.parent_span_id is not None:
            try:
                uuid.UUID(self.parent_span_id)
            except ValueError:
                raise ValueError(
                    f"parent_span_id must be a valid UUID, got: {self.parent_span_id}"
                )


@dataclass(frozen=True)
class TelemetryEvent:
    """
    Intent: Immutable telemetry event with full context
    Mechanism: Structured event with trace context and metadata
    Constraint: All required fields must be present
    """

    event_type: EventType
    trace_context: TraceContext
    phase_name: str
    timestamp: str
    severity: EventSeverity = EventSeverity.INFO
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_type": self.event_type.name,
            "trace_id": self.trace_context.trace_id,
            "span_id": self.trace_context.span_id,
            "parent_span_id": self.trace_context.parent_span_id,
            "run_id": self.trace_context.run_id,
            "phase_name": self.phase_name,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "message": self.message,
            "metadata": self.metadata,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


# ============================================================================
# TELEMETRY COLLECTOR
# ============================================================================


class TelemetryCollector:
    """
    Intent: Collect and persist telemetry events with trace context
    Mechanism: In-memory buffer with periodic flush to disk
    Constraint: Events are never lost, always persisted

    SIN_CARRETA compliance: Deterministic event ordering, immutable events
    """

    def __init__(
        self, log_dir: Path, buffer_size: int = 100, enable_console: bool = True
    ):
        """
        Initialize telemetry collector

        Args:
            log_dir: Directory for telemetry logs
            buffer_size: Number of events before auto-flush
            enable_console: Whether to log events to console
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.buffer_size = buffer_size
        self.enable_console = enable_console

        # Event buffer (not frozen, internal state)
        self._events: List[TelemetryEvent] = []

        # Logger
        self.logger = logging.getLogger(f"{__name__}.TelemetryCollector")

        # Trace context stack (for nested spans)
        self._trace_stack: List[TraceContext] = []

    def start_trace(self, trace_id: Optional[str] = None) -> TraceContext:
        """
        Start a new trace context

        Args:
            trace_id: Optional trace ID (generates UUID if None)

        Returns:
            New trace context
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4())

        span_id = str(uuid.uuid4())
        parent_span_id = self._trace_stack[-1].span_id if self._trace_stack else None

        context = TraceContext(
            trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id
        )

        self._trace_stack.append(context)
        return context

    def end_trace(self) -> None:
        """End current trace context"""
        if self._trace_stack:
            self._trace_stack.pop()

    def get_current_trace(self) -> Optional[TraceContext]:
        """Get current trace context"""
        return self._trace_stack[-1] if self._trace_stack else None

    def emit_event(
        self,
        event_type: EventType,
        phase_name: str,
        trace_context: Optional[TraceContext] = None,
        severity: EventSeverity = EventSeverity.INFO,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> TelemetryEvent:
        """
        Emit a telemetry event

        Args:
            event_type: Type of event
            phase_name: Phase name
            trace_context: Trace context (uses current if None)
            severity: Event severity
            message: Event message
            metadata: Additional metadata
            duration_ms: Duration in milliseconds
            error: Error message if applicable

        Returns:
            The emitted event
        """
        # Use current trace context if not provided
        if trace_context is None:
            trace_context = self.get_current_trace()
            if trace_context is None:
                # Create ephemeral trace context
                trace_context = TraceContext(
                    trace_id=str(uuid.uuid4()), span_id=str(uuid.uuid4())
                )

        # Create event
        event = TelemetryEvent(
            event_type=event_type,
            trace_context=trace_context,
            phase_name=phase_name,
            timestamp=datetime.now().isoformat(),
            severity=severity,
            message=message,
            metadata=metadata or {},
            duration_ms=duration_ms,
            error=error,
        )

        # Add to buffer
        self._events.append(event)

        # Console logging
        if self.enable_console:
            log_level = {
                EventSeverity.DEBUG: logging.DEBUG,
                EventSeverity.INFO: logging.INFO,
                EventSeverity.WARNING: logging.WARNING,
                EventSeverity.ERROR: logging.ERROR,
                EventSeverity.CRITICAL: logging.CRITICAL,
            }[severity]

            self.logger.log(
                log_level,
                f"[{event_type.name}] {phase_name}: {message} "
                f"(trace={trace_context.trace_id[:8]}, span={trace_context.span_id[:8]})",
            )

        # Auto-flush if buffer full
        if len(self._events) >= self.buffer_size:
            self.flush()

        return event

    def flush(self) -> None:
        """Flush events to disk"""
        if not self._events:
            return

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = self.log_dir / f"telemetry_{timestamp}.jsonl"

        # Write events as JSON Lines
        with open(filename, "w", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(event.to_dict()) + "\n")

        self.logger.debug(f"Flushed {len(self._events)} events to {filename}")

        # Clear buffer
        self._events = []

    def get_events(self) -> List[TelemetryEvent]:
        """Get all buffered events (for testing)"""
        return list(self._events)

    def clear(self) -> None:
        """Clear all buffered events (for testing)"""
        self._events = []

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - flush events"""
        self.flush()
        return False


# ============================================================================
# TELEMETRY DECORATORS
# ============================================================================


def traced_phase(collector: TelemetryCollector):
    """
    Decorator to automatically trace phase execution

    Usage:
        @traced_phase(collector)
        def my_phase(self, ...):
            ...
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract self and phase name
            if args and hasattr(args[0], "__class__"):
                phase_name = func.__name__
            else:
                phase_name = func.__name__

            # Start trace
            trace_context = collector.start_trace()

            # Emit start event
            collector.emit_event(
                EventType.PHASE_START,
                phase_name,
                trace_context=trace_context,
                message=f"Starting phase: {phase_name}",
            )

            start_time = time.time()

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000

                # Emit completion event
                collector.emit_event(
                    EventType.PHASE_COMPLETE,
                    phase_name,
                    trace_context=trace_context,
                    message=f"Completed phase: {phase_name}",
                    duration_ms=duration_ms,
                )

                return result

            except Exception as e:
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000

                # Emit error event
                collector.emit_event(
                    EventType.PHASE_ERROR,
                    phase_name,
                    trace_context=trace_context,
                    severity=EventSeverity.ERROR,
                    message=f"Phase failed: {phase_name}",
                    error=str(e),
                    duration_ms=duration_ms,
                )

                raise

            finally:
                # End trace
                collector.end_trace()

        return wrapper

    return decorator


# ============================================================================
# AUDIT ID GENERATION
# ============================================================================


def generate_audit_id(run_id: str, phase_name: str, timestamp: str) -> str:
    """
    Generate deterministic audit ID

    Intent: Create unique, reproducible audit identifier
    Mechanism: SHA256 hash of run_id + phase_name + timestamp
    Constraint: Same inputs always produce same ID

    Args:
        run_id: Orchestration run ID
        phase_name: Phase name
        timestamp: ISO format timestamp

    Returns:
        32-character hex audit ID
    """
    content = f"{run_id}:{phase_name}:{timestamp}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]


# ============================================================================
# DECISION POINT TRACKING
# ============================================================================


class DecisionTracker:
    """
    Intent: Track all decision points in orchestration
    Mechanism: Record decision criteria, outcomes, and rationale
    Constraint: All decisions must be explicit and traceable
    """

    def __init__(self, collector: TelemetryCollector):
        self.collector = collector
        self._decisions: List[Dict[str, Any]] = []

    def record_decision(
        self,
        phase_name: str,
        decision_point: str,
        criteria: Dict[str, Any],
        outcome: str,
        rationale: str,
        trace_context: Optional[TraceContext] = None,
    ) -> None:
        """
        Record a decision point

        Args:
            phase_name: Phase where decision occurs
            decision_point: Name of decision point
            criteria: Decision criteria
            outcome: Decision outcome
            rationale: Explanation of decision
            trace_context: Trace context
        """
        decision = {
            "phase_name": phase_name,
            "decision_point": decision_point,
            "criteria": criteria,
            "outcome": outcome,
            "rationale": rationale,
            "timestamp": datetime.now().isoformat(),
        }

        self._decisions.append(decision)

        # Emit telemetry event
        self.collector.emit_event(
            EventType.DECISION_POINT,
            phase_name,
            trace_context=trace_context,
            message=f"Decision: {decision_point} -> {outcome}",
            metadata=decision,
        )

    def get_decisions(self) -> List[Dict[str, Any]]:
        """Get all recorded decisions"""
        return list(self._decisions)
