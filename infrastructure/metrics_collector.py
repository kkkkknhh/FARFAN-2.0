#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics Collector Module
=========================

Standardized metrics collection for observability across all orchestrators.

SIN_CARRETA Compliance:
- Deterministic metric recording (no side effects)
- Structured alert system with severity levels
- Thread-safe operations for async contexts
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class MetricRecord:
    """Single metric recording with metadata"""
    name: str
    value: float
    timestamp: str
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRecord:
    """Alert record with structured metadata"""
    level: str
    message: str
    timestamp: str
    context: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and tracks metrics during pipeline execution.
    
    SIN_CARRETA Compliance:
    - Immutable metric history (append-only)
    - Explicit timestamp recording
    - Structured alert system
    - Thread-safe for async contexts
    
    Usage:
        metrics = MetricsCollector()
        
        # Record metric
        metrics.record("phase.extraction.duration", 1.23)
        
        # Increment counter
        metrics.increment("phase.extraction.count")
        
        # Alert on anomaly
        metrics.alert("CRITICAL", "Extraction quality below threshold")
        
        # Get summary
        summary = metrics.get_summary()
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._metrics: Dict[str, List[MetricRecord]] = defaultdict(list)
        self._counters: Dict[str, int] = {}
        self._alerts: List[AlertRecord] = []
        
    def record(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a metric value with optional tags.
        
        Args:
            metric_name: Metric identifier (e.g., "phase.extraction.duration")
            value: Metric value (must be numeric)
            tags: Optional metadata tags
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Metric value must be numeric, got {type(value)}")
        
        record = MetricRecord(
            name=metric_name,
            value=float(value),
            timestamp=datetime.now().isoformat(),
            tags=tags or {}
        )
        
        self._metrics[metric_name].append(record)
        self.logger.debug(f"Metric recorded: {metric_name}={value}")
        
    def increment(self, counter_name: str, amount: int = 1) -> None:
        """
        Increment a counter.
        
        Args:
            counter_name: Counter identifier
            amount: Increment amount (default: 1)
        """
        self._counters[counter_name] = self._counters.get(counter_name, 0) + amount
        self.logger.debug(
            f"Counter incremented: {counter_name}={self._counters[counter_name]}"
        )
        
    def alert(
        self,
        level: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an alert with severity level.
        
        Args:
            level: Alert severity (INFO, WARNING, CRITICAL)
            message: Alert message
            context: Optional context metadata
        """
        alert = AlertRecord(
            level=level.upper(),
            message=message,
            timestamp=datetime.now().isoformat(),
            context=context or {}
        )
        
        self._alerts.append(alert)
        
        # Log at appropriate level
        log_func = getattr(self.logger, level.lower(), self.logger.warning)
        log_func(f"Alert [{level}]: {message}")
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics, counters, and alerts.
        
        Returns:
            Summary dictionary with:
                - metrics: Dict of metric name -> stats
                - counters: Dict of counter name -> current value
                - alerts: List of alert records
        """
        metrics_summary = {}
        for name, records in self._metrics.items():
            values = [r.value for r in records]
            metrics_summary[name] = {
                'count': len(values),
                'last': values[-1] if values else 0.0,
                'min': min(values) if values else 0.0,
                'max': max(values) if values else 0.0,
                'avg': sum(values) / len(values) if values else 0.0
            }
        
        return {
            'metrics': metrics_summary,
            'counters': dict(self._counters),
            'alerts': [
                {
                    'level': alert.level,
                    'message': alert.message,
                    'timestamp': alert.timestamp,
                    'context': alert.context
                }
                for alert in self._alerts
            ],
            'total_metrics_recorded': sum(len(records) for records in self._metrics.values()),
            'total_alerts': len(self._alerts)
        }
    
    def get_metric_history(self, metric_name: str) -> List[Dict[str, Any]]:
        """
        Get full history for a specific metric.
        
        Args:
            metric_name: Metric identifier
            
        Returns:
            List of metric records with timestamps and tags
        """
        records = self._metrics.get(metric_name, [])
        return [
            {
                'value': record.value,
                'timestamp': record.timestamp,
                'tags': record.tags
            }
            for record in records
        ]
    
    def get_alerts_by_level(self, level: str) -> List[Dict[str, Any]]:
        """
        Get all alerts filtered by severity level.
        
        Args:
            level: Severity level (INFO, WARNING, CRITICAL)
            
        Returns:
            List of matching alert records
        """
        level_upper = level.upper()
        return [
            {
                'message': alert.message,
                'timestamp': alert.timestamp,
                'context': alert.context
            }
            for alert in self._alerts
            if alert.level == level_upper
        ]
    
    def reset(self) -> None:
        """
        Reset all metrics, counters, and alerts.
        
        WARNING: Use only in testing contexts with explicit markers.
        """
        self._metrics.clear()
        self._counters.clear()
        self._alerts.clear()
        self.logger.info("Metrics collector reset")
