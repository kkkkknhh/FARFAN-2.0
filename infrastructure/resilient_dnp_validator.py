#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resilient DNP Validator with Circuit Breaker Integration
========================================================

Wraps the DNP validator with circuit breaker protection to implement
fail-open policy for external service failures. This ensures the pipeline
continues even when DNP validation services are unavailable.

Design Principles:
- Fail-open policy: Continue with penalty score if service unavailable
- Circuit breaker integration for cascading failure prevention
- Graceful degradation with observability
- Backwards compatible with existing ValidadorDNP interface

Author: AI Systems Architect
Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)

# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class ValidationResult:
    """
    Result of DNP validation with fail-open policy support.

    Attributes:
        status: 'passed', 'failed', 'skipped' (when circuit is open)
        score: Compliance score (0.0-1.0)
        score_penalty: Penalty applied for skipped validation (0.0-0.1)
        reason: Human-readable explanation
        details: Additional validation details
        circuit_state: Current circuit breaker state
    """

    status: str  # 'passed', 'failed', 'skipped'
    score: float = 0.0
    score_penalty: float = 0.0
    reason: str = ""
    details: Dict[str, Any] = None
    circuit_state: Optional[str] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "status": self.status,
            "score": self.score,
            "score_penalty": self.score_penalty,
            "reason": self.reason,
            "details": self.details,
            "circuit_state": self.circuit_state,
        }


@dataclass
class PDMData:
    """
    Plan de Desarrollo Municipal data structure for validation.

    This is a simplified interface - in production, this would match
    the actual PDM data structure from the CDAF framework.
    """

    sector: str
    descripcion: str
    indicadores_propuestos: List[str]
    presupuesto: float = 0.0
    es_rural: bool = False
    poblacion_victimas: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============================================================================
# DNP API Client Interface
# ============================================================================


class DNPAPIClient:
    """
    Abstract interface for DNP API client.

    In production, this would be implemented with actual HTTP calls
    to the DNP validation service.
    """

    async def validate_compliance(self, data: PDMData) -> Dict[str, Any]:
        """
        Validate PDM data against DNP standards.

        Args:
            data: PDM data to validate

        Returns:
            Validation results dictionary

        Raises:
            ConnectionError: If service is unavailable
            TimeoutError: If request times out
            Exception: For other service errors
        """
        raise NotImplementedError("DNPAPIClient must be implemented with actual API")


# ============================================================================
# Resilient DNP Validator
# ============================================================================


class ResilientDNPValidator:
    """
    DNP Validator with circuit breaker protection and fail-open policy.

    This class wraps DNP validation calls with a circuit breaker to prevent
    cascading failures. When the circuit is OPEN (service unavailable), it
    implements a fail-open policy: validation is skipped with a minor penalty
    rather than blocking the entire pipeline.

    Fail-open Policy:
    - Service available: Full validation with complete scoring
    - Service degraded: Continue with 5% score penalty
    - Circuit OPEN: Skip validation, apply penalty, log warning

    Args:
        dnp_api_client: Client for DNP validation API
        failure_threshold: Failures before opening circuit (default: 3)
        recovery_timeout: Seconds before attempting recovery (default: 120)
        fail_open_penalty: Score penalty when skipping validation (default: 0.05)

    Example:
        >>> validator = ResilientDNPValidator(dnp_client)
        >>> result = await validator.validate(pdm_data)
        >>> if result.status == 'skipped':
        >>>     logger.warning(f"Validation skipped: {result.reason}")
    """

    def __init__(
        self,
        dnp_api_client: DNPAPIClient,
        failure_threshold: int = 3,
        recovery_timeout: int = 120,
        fail_open_penalty: float = 0.05,
    ):
        """
        Initialize resilient validator with circuit breaker.

        Args:
            dnp_api_client: DNP API client instance
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before recovery attempt
            fail_open_penalty: Penalty score for skipped validation (0.0-0.1)
        """
        if not isinstance(dnp_api_client, DNPAPIClient):
            raise TypeError("dnp_api_client must be instance of DNPAPIClient")

        if not 0.0 <= fail_open_penalty <= 0.1:
            raise ValueError("fail_open_penalty must be between 0.0 and 0.1")

        self.client = dnp_api_client
        self.fail_open_penalty = fail_open_penalty

        # Initialize circuit breaker with custom configuration
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=Exception,  # Catch all service exceptions
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"ResilientDNPValidator initialized with fail-open policy "
            f"(penalty={fail_open_penalty}, threshold={failure_threshold})"
        )

    async def validate(self, data: PDMData) -> ValidationResult:
        """
        Validate PDM data with circuit breaker protection.

        Implements fail-open policy: if validation service is unavailable,
        continue processing with a minor score penalty rather than failing.

        Args:
            data: PDM data to validate

        Returns:
            ValidationResult with status, score, and details

        Flow:
            1. Attempt validation through circuit breaker
            2. If successful: return full validation results
            3. If circuit OPEN: skip validation with penalty
            4. If service error: track failure and re-raise
        """
        circuit_state = self.circuit_breaker.get_state()

        try:
            # Attempt validation through circuit breaker
            self.logger.info(
                f"Attempting DNP validation (circuit: {circuit_state.value})"
            )

            result_dict = await self.circuit_breaker.call(
                self.client.validate_compliance, data
            )

            # Successful validation
            self.logger.info("DNP validation completed successfully")
            return ValidationResult(
                status="passed" if result_dict.get("cumple", False) else "failed",
                score=result_dict.get("score_total", 0.0) / 100.0,  # Normalize to 0-1
                score_penalty=0.0,
                reason=result_dict.get("nivel_cumplimiento", "Validation completed"),
                details=result_dict,
                circuit_state=self.circuit_breaker.get_state().value,
            )

        except CircuitOpenError as e:
            # Circuit is OPEN - implement fail-open policy
            self.logger.warning(
                f"DNP validation skipped - circuit breaker is OPEN. "
                f"Failures: {e.failure_count}, applying {self.fail_open_penalty} penalty"
            )

            return ValidationResult(
                status="skipped",
                score=1.0 - self.fail_open_penalty,  # Apply penalty to perfect score
                score_penalty=self.fail_open_penalty,
                reason="External service unavailable - circuit breaker OPEN",
                details={
                    "failure_count": e.failure_count,
                    "last_failure_time": e.last_failure_time,
                    "fail_open_policy": "enabled",
                },
                circuit_state=CircuitState.OPEN.value,
            )

        except Exception as e:
            # Unexpected error during validation - log and re-raise
            # Circuit breaker will track this failure
            self.logger.error(
                f"DNP validation failed with {type(e).__name__}: {str(e)}"
            )

            # Return failed result with details
            return ValidationResult(
                status="failed",
                score=0.0,
                score_penalty=0.0,
                reason=f"Validation error: {type(e).__name__}",
                details={"error": str(e), "error_type": type(e).__name__},
                circuit_state=self.circuit_breaker.get_state().value,
            )

    def get_circuit_metrics(self) -> Dict[str, Any]:
        """
        Get circuit breaker metrics for monitoring and observability.

        Returns:
            Dictionary with circuit state and call metrics
        """
        metrics = self.circuit_breaker.get_metrics()
        return {
            "state": self.circuit_breaker.get_state().value,
            "total_calls": metrics.total_calls,
            "successful_calls": metrics.successful_calls,
            "failed_calls": metrics.failed_calls,
            "rejected_calls": metrics.rejected_calls,
            "state_transitions": metrics.state_transitions,
            "last_state_change": metrics.last_state_change,
            "failure_count": self.circuit_breaker.failure_count,
            "last_failure_time": self.circuit_breaker.last_failure_time,
        }

    def reset_circuit(self):
        """
        Manually reset circuit breaker to CLOSED state.

        Use this for administrative override or testing purposes.
        """
        self.logger.warning("Manual circuit breaker reset requested")
        self.circuit_breaker.reset()


# ============================================================================
# Factory Functions
# ============================================================================


def create_resilient_validator(
    dnp_api_client: DNPAPIClient, **kwargs
) -> ResilientDNPValidator:
    """
    Factory function to create ResilientDNPValidator with sensible defaults.

    Args:
        dnp_api_client: DNP API client instance
        **kwargs: Additional configuration (failure_threshold, recovery_timeout, etc.)

    Returns:
        Configured ResilientDNPValidator instance
    """
    return ResilientDNPValidator(dnp_api_client, **kwargs)
