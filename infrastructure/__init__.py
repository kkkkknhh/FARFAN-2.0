#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infrastructure Module
Provides infrastructure components for resilient external service integration:
- Circuit Breaker pattern for cascading failure prevention
- Resilient DNP Validator with fail-open policy
- Service health monitoring and metrics
- Resource pool management and computational infrastructure

Author: AI Systems Architect
Version: 1.0.0
"""

from infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerMetrics,
    CircuitOpenError,
    CircuitState,
    SyncCircuitBreaker,
)
from infrastructure.resilient_dnp_validator import (
    DNPAPIClient,
    PDMData,
    ResilientDNPValidator,
    ValidationResult,
    create_resilient_validator,
)
from infrastructure.resource_pool import (
    BayesianInferenceEngine,
    ResourceConfig,
    ResourcePool,
    Worker,
    WorkerMemoryError,
    WorkerTimeoutError,
)

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "CircuitBreakerMetrics",
    "SyncCircuitBreaker",
    # Resilient DNP Validator
    "ResilientDNPValidator",
    "ValidationResult",
    "PDMData",
    "DNPAPIClient",
    "create_resilient_validator",
    # Resource Pool
    "ResourceConfig",
    "Worker",
    "ResourcePool",
    "WorkerTimeoutError",
    "WorkerMemoryError",
    "BayesianInferenceEngine",
]

__version__ = "1.0.0"
