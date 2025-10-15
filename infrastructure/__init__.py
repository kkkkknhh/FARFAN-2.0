#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infrastructure Module
Provides infrastructure components for resilient external service integration:
- Circuit Breaker pattern for cascading failure prevention
- Resilient DNP Validator with fail-open policy
- Service health monitoring and metrics
- Resource pool management and computational infrastructure
- Async orchestrator with backpressure signaling (Audit Point 4.2)
- PDF processing isolation (Audit Point 4.1)
- Fail-open policy framework (Audit Point 4.3)

Author: AI Systems Architect
Version: 1.0.0
"""

from infrastructure.async_orchestrator import (
    AsyncOrchestrator,
    JobStatus,
    JobTimeoutError,
    OrchestratorConfig,
    OrchestratorMetrics,
    QueueFullError,
    create_orchestrator,
)
from infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerMetrics,
    CircuitOpenError,
    CircuitState,
    SyncCircuitBreaker,
)
from infrastructure.fail_open_policy import (
    DNP_AVAILABLE,
    CDAFValidationError,
    ComponentConfig,
    ComponentType,
    CoreValidationError,
    EnrichmentValidationWarning,
    FailOpenMetrics,
    FailOpenPolicyManager,
    FailureMode,
    create_default_components,
    create_policy_manager,
    is_dnp_available,
    set_dnp_available,
)
from infrastructure.pdf_isolation import (
    IsolatedPDFProcessor,
    IsolationConfig,
    IsolationMetrics,
    IsolationStrategy,
    PDFProcessingIsolationError,
    PDFProcessingTimeoutError,
    ProcessingResult,
    create_isolated_processor,
)
from infrastructure.resilient_dnp_validator import (
    DNPAPIClient,
    PDMData,
    ResilientDNPValidator,
    ValidationResult,
    create_resilient_validator,
)

# Resource pool imports are optional (requires psutil)
try:
    from infrastructure.resource_pool import (
        BayesianInferenceEngine,
        ResourceConfig,
        ResourcePool,
        Worker,
        WorkerMemoryError,
        WorkerTimeoutError,
    )
    RESOURCE_POOL_AVAILABLE = True
except ImportError:
    # Graceful degradation if psutil is not available
    RESOURCE_POOL_AVAILABLE = False
    BayesianInferenceEngine = None
    ResourceConfig = None
    ResourcePool = None
    Worker = None
    WorkerMemoryError = None
    WorkerTimeoutError = None

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
    # Async Orchestrator (Audit Point 4.2)
    "AsyncOrchestrator",
    "OrchestratorConfig",
    "OrchestratorMetrics",
    "QueueFullError",
    "JobTimeoutError",
    "JobStatus",
    "create_orchestrator",
    # PDF Isolation (Audit Point 4.1)
    "IsolatedPDFProcessor",
    "IsolationConfig",
    "IsolationStrategy",
    "PDFProcessingTimeoutError",
    "PDFProcessingIsolationError",
    "ProcessingResult",
    "IsolationMetrics",
    "create_isolated_processor",
    # Fail-Open Policy (Audit Point 4.3)
    "FailOpenPolicyManager",
    "ComponentConfig",
    "ComponentType",
    "FailureMode",
    "CDAFValidationError",
    "CoreValidationError",
    "EnrichmentValidationWarning",
    "FailOpenMetrics",
    "DNP_AVAILABLE",
    "set_dnp_available",
    "is_dnp_available",
    "create_policy_manager",
    "create_default_components",
]

__version__ = "1.0.0"
