#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fail-Open Policy Framework - Audit Point 4.3
============================================

Implements fail-open vs fail-closed policy configuration for graceful
degradation in the CDAF validation framework.

Design Principles:
- Fail-open for enrichment components (DNP validator, external APIs)
- Fail-closed for core validation components
- Configurable degradation strategies
- Penalty-based scoring for skipped validations
- Graceful degradation <10% accuracy loss

Audit Point 4.3 Compliance:
- Fail-open policy for non-core components (e.g., ValidadorDNP)
- fail_closed=True for core components, False for enrichment
- Simulation of DNP failure with continuation vs halt
- Graceful degradation with minimal accuracy loss
- CDAFValidationError exception handling

Author: AI Systems Architect
Version: 1.0.0
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

# ============================================================================
# Exceptions
# ============================================================================


class CDAFValidationError(Exception):
    """
    Base exception for CDAF validation errors.

    This exception is raised when core validation components fail
    and fail_closed=True is configured.
    """

    def __init__(
        self,
        message: str,
        component: str,
        fail_closed: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.component = component
        self.fail_closed = fail_closed
        self.details = details or {}
        super().__init__(f"[{component}] {message}")


class CoreValidationError(CDAFValidationError):
    """Raised when core validation fails (fail_closed=True)"""

    pass


class EnrichmentValidationWarning(CDAFValidationError):
    """Raised when enrichment validation fails (fail_closed=False)"""

    pass


# ============================================================================
# Data Structures
class ComponentType(str, Enum):
    """Type of validation component"""

    CORE = "core"  # Core validation - fail_closed=True
    ENRICHMENT = "enrichment"  # Enrichment - fail_closed=False


class FailureMode(str, Enum):
    """Failure handling mode"""

    FAIL_CLOSED = "fail_closed"  # Halt on error
    FAIL_OPEN = "fail_open"  # Continue with degradation


@dataclass
class ComponentConfig:
    """Configuration for a validation component"""

    name: str
    component_type: ComponentType
    fail_closed: bool
    degradation_penalty: float = 0.05  # 5% penalty for skipped validation
    max_retries: int = 0
    timeout_secs: int = 30

    def __post_init__(self):
        """Validate configuration"""
        if not 0.0 <= self.degradation_penalty <= 0.1:
            raise ValueError(
                f"degradation_penalty must be between 0.0 and 0.1, "
                f"got {self.degradation_penalty}"
            )
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}"
            )
        if self.timeout_secs <= 0:
            raise ValueError(f"timeout_secs must be positive, got {self.timeout_secs}")


@dataclass
class ValidationResult:
    """Result from validation component"""

    component: str
    success: bool
    score: float = 0.0
    degradation_penalty: float = 0.0
    status: str = "unknown"
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    fail_open_applied: bool = False


@dataclass
class FailOpenMetrics:
    """Metrics for fail-open policy"""

    total_validations: int = 0
    core_failures: int = 0
    enrichment_failures: int = 0
    fail_open_applied: int = 0
    avg_degradation: float = 0.0
    accuracy_loss: float = 0.0


# ============================================================================
# DNP Availability Flag
# ============================================================================

# Global flag to indicate DNP service availability
# In production, this would be set based on service health checks
DNP_AVAILABLE = True


def set_dnp_available(available: bool):
    """
    Set DNP service availability flag.

    Args:
        available: True if DNP service is available, False otherwise
    """
    global DNP_AVAILABLE
    DNP_AVAILABLE = available


def is_dnp_available() -> bool:
    """
    Check if DNP service is available.

    Returns:
        True if DNP service is available, False otherwise
    """
    return DNP_AVAILABLE


# ============================================================================
# Fail-Open Policy Manager
# ============================================================================


class FailOpenPolicyManager:
    """
    Manages fail-open vs fail-closed policies for validation components.

    This class coordinates validation components with different failure
    policies. Core components use fail-closed (halt on error), while
    enrichment components use fail-open (degrade gracefully).

    Features:
    - Configurable fail-open/fail-closed policies
    - Penalty-based scoring for degraded validation
    - Graceful degradation with <10% accuracy loss
    - Metrics and observability

    Args:
        components: Dictionary of component configurations

    Example:
        >>> components = {
        >>>     "core_validator": ComponentConfig(
        >>>         name="core_validator",
        >>>         component_type=ComponentType.CORE,
        >>>         fail_closed=True
        >>>     ),
        >>>     "dnp_validator": ComponentConfig(
        >>>         name="dnp_validator",
        >>>         component_type=ComponentType.ENRICHMENT,
        >>>         fail_closed=False,
        >>>         degradation_penalty=0.05
        >>>     )
        >>> }
        >>> manager = FailOpenPolicyManager(components)
    """

    def __init__(self, components: Dict[str, ComponentConfig]):
        """
        Initialize fail-open policy manager.

        Args:
            components: Dictionary mapping component names to configurations
        """
        self.components = components
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics = FailOpenMetrics()

        # Log configuration
        core_components = [
            name
            for name, cfg in components.items()
            if cfg.component_type == ComponentType.CORE
        ]
        enrichment_components = [
            name
            for name, cfg in components.items()
            if cfg.component_type == ComponentType.ENRICHMENT
        ]

        self.logger.info(
            f"FailOpenPolicyManager initialized: "
            f"core={core_components}, enrichment={enrichment_components}"
        )

    async def execute_validation(
        self, component_name: str, validator_func: Callable, *args, **kwargs
    ) -> ValidationResult:
        """
        Execute validation with fail-open/fail-closed policy.

        This method wraps validator execution with policy enforcement.
        If the validator fails and fail_closed=False, it returns a degraded
        result instead of raising an exception.

        Args:
            component_name: Name of the validation component
            validator_func: Async validator function
            *args: Positional arguments for validator
            **kwargs: Keyword arguments for validator

        Returns:
            ValidationResult with success/failure status

        Raises:
            CoreValidationError: If core component fails (fail_closed=True)
        """
        if component_name not in self.components:
            raise ValueError(f"Unknown component: {component_name}")

        config = self.components[component_name]
        self.metrics.total_validations += 1

        self.logger.info(
            f"Executing validation: {component_name} (fail_closed={config.fail_closed})"
        )

        try:
            # Execute validator with timeout
            result = await asyncio.wait_for(
                validator_func(*args, **kwargs), timeout=config.timeout_secs
            )

            # Successful validation
            if isinstance(result, ValidationResult):
                return result
            else:
                # Convert dict result to ValidationResult
                return ValidationResult(
                    component=component_name,
                    success=True,
                    score=result.get("score", 1.0),
                    status="passed",
                    reason=result.get("reason", "Validation passed"),
                    details=result,
                )

        except asyncio.TimeoutError:
            error_msg = f"Validation timeout after {config.timeout_secs}s"
            self.logger.warning(f"{component_name}: {error_msg}")

            # Apply fail-open or fail-closed policy
            if config.fail_closed:
                # Core component - halt with error
                self.metrics.core_failures += 1
                raise CoreValidationError(
                    error_msg, component=component_name, fail_closed=True
                )
            else:
                # Enrichment component - degrade gracefully
                self.metrics.enrichment_failures += 1
                self.metrics.fail_open_applied += 1
                return self._create_degraded_result(config, error_msg)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.logger.error(f"{component_name} failed: {error_msg}")

            # Apply fail-open or fail-closed policy
            if config.fail_closed:
                # Core component - halt with error
                self.metrics.core_failures += 1
                raise CoreValidationError(
                    error_msg,
                    component=component_name,
                    fail_closed=True,
                    details={"error_type": type(e).__name__, "error": str(e)},
                )
            else:
                # Enrichment component - degrade gracefully
                self.metrics.enrichment_failures += 1
                self.metrics.fail_open_applied += 1
                return self._create_degraded_result(config, error_msg)

    def _create_degraded_result(
        self, config: ComponentConfig, error_msg: str
    ) -> ValidationResult:
        """
        Create degraded validation result with penalty.

        Args:
            config: Component configuration
            error_msg: Error message

        Returns:
            ValidationResult with degradation penalty applied
        """
        self.logger.warning(
            f"Applying fail-open policy for {config.name}: "
            f"penalty={config.degradation_penalty}"
        )

        # Update degradation metrics
        self._update_degradation_metrics(config.degradation_penalty)

        return ValidationResult(
            component=config.name,
            success=False,
            score=1.0 - config.degradation_penalty,  # Apply penalty to perfect score
            degradation_penalty=config.degradation_penalty,
            status="skipped",
            reason=f"Validation failed - fail-open policy applied: {error_msg}",
            details={"error": error_msg, "fail_open_policy": "enabled"},
            fail_open_applied=True,
        )

    def _update_degradation_metrics(self, penalty: float):
        """Update average degradation metrics"""
        if self.metrics.fail_open_applied > 0:
            # Calculate cumulative average degradation
            total_degradation = (
                self.metrics.avg_degradation * (self.metrics.fail_open_applied - 1)
                + penalty
            )
            self.metrics.avg_degradation = (
                total_degradation / self.metrics.fail_open_applied
            )

            # Calculate overall accuracy loss
            self.metrics.accuracy_loss = (
                self.metrics.avg_degradation
                * self.metrics.fail_open_applied
                / max(self.metrics.total_validations, 1)
            )

    def get_metrics(self) -> FailOpenMetrics:
        """
        Get fail-open policy metrics.

        Returns:
            FailOpenMetrics with current statistics
        """
        return self.metrics

    def verify_graceful_degradation(self) -> Dict[str, Any]:
        """
        Verify that graceful degradation is working correctly.

        Returns:
            Dictionary with degradation verification results
        """
        return {
            "total_validations": self.metrics.total_validations,
            "fail_open_applied": self.metrics.fail_open_applied,
            "avg_degradation": self.metrics.avg_degradation,
            "accuracy_loss": self.metrics.accuracy_loss,
            "accuracy_loss_target": 0.10,  # <10% target
            "meets_target": self.metrics.accuracy_loss < 0.10,
            "core_failures": self.metrics.core_failures,
            "enrichment_failures": self.metrics.enrichment_failures,
        }


# ============================================================================
# Default Component Configurations
# ============================================================================


def create_default_components() -> Dict[str, ComponentConfig]:
    """
    Create default component configurations for CDAF framework.

    Returns:
        Dictionary of default component configurations
    """
    return {
        "core_validator": ComponentConfig(
            name="core_validator",
            component_type=ComponentType.CORE,
            fail_closed=True,
            timeout_secs=60,
        ),
        "dnp_validator": ComponentConfig(
            name="dnp_validator",
            component_type=ComponentType.ENRICHMENT,
            fail_closed=False,
            degradation_penalty=0.05,  # 5% penalty
            timeout_secs=120,
        ),
        "external_api": ComponentConfig(
            name="external_api",
            component_type=ComponentType.ENRICHMENT,
            fail_closed=False,
            degradation_penalty=0.03,  # 3% penalty
            timeout_secs=30,
        ),
    }


# ============================================================================
# Factory Functions
# ============================================================================


def create_policy_manager(
    components: Optional[Dict[str, ComponentConfig]] = None,
) -> FailOpenPolicyManager:
    """
    Factory function to create FailOpenPolicyManager.

    Args:
        components: Optional component configurations (uses defaults if not provided)

    Returns:
        Configured FailOpenPolicyManager instance
    """
    if components is None:
        components = create_default_components()

    return FailOpenPolicyManager(components)
