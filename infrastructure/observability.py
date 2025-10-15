#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F4.4: Comprehensive Observability Stack
Stack completo de métricas, logging, y tracing.
Implementa todos los Observability Metrics del Standard.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional


@dataclass
class ObservabilityConfig:
    """Configuration for observability stack"""

    metrics_backend: str = "in_memory"
    log_level: str = "INFO"
    trace_backend: str = "in_memory"
    enable_distributed_tracing: bool = False


class MetricsCollector:
    """
    Collects metrics with support for histograms, gauges, and counters.
    Backend-agnostic implementation with in-memory storage.
    """

    def __init__(self, backend: str = "in_memory"):
        self.backend = backend
        self.logger = logging.getLogger(self.__class__.__name__)
        self.histograms: Dict[str, List[float]] = {}
        self.gauges: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        self.tags_store: Dict[str, List[Dict[str, Any]]] = {}

    def histogram(
        self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a histogram metric (distribution of values over time).

        Args:
            metric_name: Name of the metric (e.g., 'pdm.pipeline.duration_seconds')
            value: Value to record
            tags: Optional tags for metric categorization
        """
        if metric_name not in self.histograms:
            self.histograms[metric_name] = []
            self.tags_store[metric_name] = []

        self.histograms[metric_name].append(value)
        if tags:
            self.tags_store[metric_name].append({"value": value, "tags": tags})

        self.logger.debug(f"Histogram recorded: {metric_name}={value} {tags or {}}")

    def gauge(self, metric_name: str, value: float) -> None:
        """
        Record a gauge metric (point-in-time value).

        Args:
            metric_name: Name of the metric (e.g., 'pdm.memory.peak_mb')
            value: Current value
        """
        self.gauges[metric_name] = value
        self.logger.debug(f"Gauge set: {metric_name}={value}")

    def increment(
        self, metric_name: str, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increment a counter metric.

        Args:
            metric_name: Name of the counter (e.g., 'pdm.posterior.nonconvergent_count')
            tags: Optional tags for categorization
        """
        key = self._make_counter_key(metric_name, tags)
        self.counters[key] = self.counters.get(key, 0) + 1
        self.logger.debug(
            f"Counter incremented: {metric_name} {tags or {}} -> {self.counters[key]}"
        )

    def get_count(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> int:
        """
        Get current count for a counter.

        Args:
            metric_name: Name of the counter
            tags: Optional tags to filter by

        Returns:
            Current counter value
        """
        key = self._make_counter_key(metric_name, tags)
        return self.counters.get(key, 0)

    def _make_counter_key(
        self, metric_name: str, tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Create a unique key for counters with tags"""
        if not tags:
            return metric_name
        tag_str = ",".join(f"{k}:{v}" for k, v in sorted(tags.items()))
        return f"{metric_name}[{tag_str}]"

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        return {
            "histograms": {
                name: {
                    "count": len(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "mean": sum(values) / len(values) if values else 0,
                }
                for name, values in self.histograms.items()
            },
            "gauges": self.gauges,
            "counters": self.counters,
        }


class StructuredLogger:
    """
    Structured logging with configurable log levels.
    Provides context-aware logging for observability.
    """

    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level
        self.logger = logging.getLogger("StructuredLogger")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Ensure handler exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def debug(self, message: str, **context) -> None:
        """Log debug message with context"""
        self.logger.debug(self._format_message(message, context))

    def info(self, message: str, **context) -> None:
        """Log info message with context"""
        self.logger.info(self._format_message(message, context))

    def warning(self, message: str, **context) -> None:
        """Log warning message with context"""
        self.logger.warning(self._format_message(message, context))

    def error(self, message: str, **context) -> None:
        """Log error message with context"""
        self.logger.error(self._format_message(message, context))

    def critical(self, message: str, **context) -> None:
        """Log critical message with context"""
        self.logger.critical(self._format_message(message, context))

    def _format_message(self, message: str, context: Dict[str, Any]) -> str:
        """Format message with structured context"""
        if not context:
            return message
        context_str = " | ".join(f"{k}={v}" for k, v in context.items())
        return f"{message} | {context_str}"


class Span:
    """Represents a single tracing span"""

    def __init__(self, operation_name: str, attributes: Dict[str, Any]):
        self.operation_name = operation_name
        self.attributes = attributes
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def finish(self) -> None:
        """Finish the span and calculate duration"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.logger.debug(
            f"Span finished: {self.operation_name} duration={self.duration:.3f}s {self.attributes}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary"""
        return {
            "operation": self.operation_name,
            "attributes": self.attributes,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
        }


class DistributedTracer:
    """
    Distributed tracing for operation tracking.
    Provides span-based tracing for pipeline observability.
    """

    def __init__(self, backend: str = "in_memory"):
        self.backend = backend
        self.logger = logging.getLogger(self.__class__.__name__)
        self.spans: List[Span] = []
        self.active_spans: List[Span] = []

    def start_span(
        self, operation_name: str, attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """
        Start a new tracing span.

        Args:
            operation_name: Name of the operation being traced
            attributes: Additional attributes for context

        Returns:
            Span object
        """
        span = Span(operation_name, attributes or {})
        self.active_spans.append(span)
        self.logger.debug(f"Span started: {operation_name} {attributes or {}}")
        return span

    def finish_span(self, span: Span) -> None:
        """Finish a span and store it"""
        span.finish()
        if span in self.active_spans:
            self.active_spans.remove(span)
        self.spans.append(span)

    def get_traces(self) -> List[Dict[str, Any]]:
        """Get all completed traces"""
        return [span.to_dict() for span in self.spans]


class ObservabilityStack:
    """
    Stack completo de métricas, logging, y tracing.
    Implementa todos los Observability Metrics del Standard.
    """

    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.metrics_collector = MetricsCollector(config.metrics_backend)
        self.logger = StructuredLogger(config.log_level)
        self.tracer = DistributedTracer(config.trace_backend)

    # Métricas críticas del Standard

    def record_pipeline_duration(self, duration_secs: float) -> None:
        """
        Record pipeline duration metric: pdm.pipeline.duration_seconds

        Triggers HIGH alert if duration exceeds 30 minutes (1800 seconds).

        Args:
            duration_secs: Pipeline duration in seconds
        """
        self.metrics_collector.histogram(
            "pdm.pipeline.duration_seconds", duration_secs, tags={"phase": "complete"}
        )
        if duration_secs > 1800:
            self.alert(
                "HIGH", f"Pipeline duration exceeded 30min: {duration_secs:.1f}s"
            )

    def record_nonconvergent_chain(self, chain_id: str, reason: str) -> None:
        """
        Record non-convergent MCMC chain: pdm.posterior.nonconvergent_count (CRITICAL)

        This is a CRITICAL metric as non-convergent chains indicate fundamental
        issues with Bayesian inference quality.

        Args:
            chain_id: Identifier for the MCMC chain
            reason: Reason for convergence failure
        """
        self.metrics_collector.increment(
            "pdm.posterior.nonconvergent_count", tags={"chain_id": chain_id}
        )
        self.alert("CRITICAL", f"MCMC chain {chain_id} failed to converge: {reason}")

    def record_memory_peak(self, memory_mb: float) -> None:
        """
        Record peak memory usage: pdm.memory.peak_mb

        Triggers WARNING alert if peak memory exceeds 16GB (16000 MB).

        Args:
            memory_mb: Peak memory usage in megabytes
        """
        self.metrics_collector.gauge("pdm.memory.peak_mb", memory_mb)
        if memory_mb > 16000:
            self.alert("WARNING", f"Peak memory usage: {memory_mb:.1f}MB")

    def record_hoop_test_failure(self, question: str, missing: List[str]) -> None:
        """
        Record hoop test failure: pdm.evidence.hoop_test_fail_count

        Hoop tests are necessary conditions. Multiple failures (>5) trigger HIGH alert
        as they indicate systematic evidence quality issues.

        Args:
            question: Question identifier that failed
            missing: List of missing evidence items
        """
        self.metrics_collector.increment(
            "pdm.evidence.hoop_test_fail_count", tags={"question": question}
        )
        current_count = self.metrics_collector.get_count(
            "pdm.evidence.hoop_test_fail_count"
        )
        if current_count > 5:
            self.alert(
                "HIGH",
                f"Multiple hoop test failures detected: {current_count} failures",
            )

    def record_dimension_score(self, dimension: str, score: float) -> None:
        """
        Record dimension quality score: pdm.dimension.avg_score_D{N}

        Special handling for D6 (Theory of Change) - scores below 0.55 trigger
        CRITICAL alert as they indicate fundamental theory structure issues.

        Args:
            dimension: Dimension identifier (e.g., 'D6')
            score: Quality score (0.0 to 1.0)
        """
        self.metrics_collector.gauge(f"pdm.dimension.avg_score_{dimension}", score)
        if dimension == "D6" and score < 0.55:
            self.alert("CRITICAL", f"D6 score below threshold: {score:.2f}")

    @contextmanager
    def trace_operation(
        self, operation_name: str, **attributes
    ) -> Generator[Span, None, None]:
        """
        Distributed tracing context manager.

        Usage:
            with observability.trace_operation('extract_chunks', plan='PDM_001') as span:
                # perform operation
                pass

        Args:
            operation_name: Name of the operation to trace
            **attributes: Additional attributes for context

        Yields:
            Span object for the operation
        """
        span = self.tracer.start_span(operation_name, attributes=attributes)
        try:
            yield span
        finally:
            self.tracer.finish_span(span)

    def alert(self, level: str, message: str) -> None:
        """
        Generate an alert with specified level.

        Alert levels:
        - CRITICAL: System cannot proceed, requires immediate attention
        - HIGH: Significant issue, may impact results
        - WARNING: Issue detected, monitoring recommended
        - INFO: Informational message

        Args:
            level: Alert level (CRITICAL, HIGH, WARNING, INFO)
            message: Alert message
        """
        log_method = {
            "CRITICAL": self.logger.critical,
            "HIGH": self.logger.error,
            "WARNING": self.logger.warning,
            "INFO": self.logger.info,
        }.get(level, self.logger.warning)

        log_method(f"ALERT [{level}]: {message}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.

        Returns:
            Dictionary containing all metrics data
        """
        return self.metrics_collector.get_summary()

    def get_traces_summary(self) -> List[Dict[str, Any]]:
        """
        Get all distributed traces.

        Returns:
            List of trace dictionaries
        """
        return self.tracer.get_traces()
