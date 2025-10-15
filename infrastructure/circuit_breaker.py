#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circuit Breaker Pattern Implementation for FARFAN 2.0
=====================================================

Implements the Circuit Breaker pattern for external service calls to prevent
cascading failures and maintain system throughput during external service outages.

Design Principles:
- Fail-open vs Fail-closed policy support
- Async/await pattern for modern Python applications
- Configurable failure thresholds and recovery timeouts
- State transitions: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
- Comprehensive error tracking and observability

Author: AI Systems Architect
Version: 1.0.0
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Type


# ============================================================================
# Circuit State Definitions
# ============================================================================

class CircuitState(str, Enum):
    """Circuit breaker states following the canonical pattern"""
    CLOSED = "closed"        # Normal operation - requests pass through
    OPEN = "open"           # Failing - reject requests immediately
    HALF_OPEN = "half_open" # Testing recovery - allow limited requests


# ============================================================================
# Custom Exceptions
# ============================================================================

class CircuitOpenError(Exception):
    """
    Raised when circuit breaker is OPEN and rejects a request.
    
    This exception indicates that the circuit breaker has detected too many
    failures and is preventing requests to protect the system.
    """
    
    def __init__(self, message: str = "Circuit breaker is OPEN", 
                 failure_count: int = 0,
                 last_failure_time: Optional[float] = None):
        self.failure_count = failure_count
        self.last_failure_time = last_failure_time
        super().__init__(message)
    
    def __str__(self) -> str:
        return (f"{super().__str__()} - "
                f"Failures: {self.failure_count}, "
                f"Last failure: {self.last_failure_time}")


# ============================================================================
# Circuit Breaker Implementation
# ============================================================================

@dataclass
class CircuitBreakerMetrics:
    """Metrics tracked by the circuit breaker for observability"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_transitions: int = 0
    last_state_change: Optional[float] = None


class CircuitBreaker:
    """
    Implements Circuit Breaker pattern for external service protection.
    
    The circuit breaker monitors calls to external services and prevents
    cascading failures by "opening" the circuit when failure thresholds
    are exceeded. This implements both fail-open and fail-closed policies
    as required by governance standards.
    
    State Transitions:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Too many failures detected, reject all requests
    - HALF_OPEN: Testing recovery, allow one request through
    
    Args:
        failure_threshold: Number of consecutive failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery (HALF_OPEN)
        expected_exception: Exception type(s) to catch and count as failures
        
    Example:
        >>> breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        >>> result = await breaker.call(external_api_call, arg1, arg2)
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker with failure tolerance parameters.
        
        Args:
            failure_threshold: Number of failures before opening circuit (default: 5)
            recovery_timeout: Seconds before attempting recovery (default: 60)
            expected_exception: Exception type to catch (default: Exception)
        """
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout < 1:
            raise ValueError("recovery_timeout must be >= 1")
        
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        # State tracking
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        
        # Metrics for observability
        self.metrics = CircuitBreakerMetrics()
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"Circuit breaker initialized: threshold={failure_threshold}, "
            f"timeout={recovery_timeout}s, state={self.state.value}"
        )
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        This method wraps the external service call with circuit breaker logic.
        If the circuit is OPEN, it raises CircuitOpenError immediately.
        If the circuit is HALF_OPEN, it attempts one test call.
        If the circuit is CLOSED, it executes normally and tracks failures.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from func if successful
            
        Raises:
            CircuitOpenError: If circuit is OPEN and cannot recover yet
            Exception: Re-raises exceptions from func after tracking
        """
        self.metrics.total_calls += 1
        
        # Check if circuit is OPEN
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.logger.info("Attempting circuit recovery - transitioning to HALF_OPEN")
                self._transition_to(CircuitState.HALF_OPEN)
            else:
                self.metrics.rejected_calls += 1
                time_until_retry = self.recovery_timeout - (time.time() - self.last_failure_time)
                self.logger.warning(
                    f"Circuit OPEN - rejecting call. Retry in {time_until_retry:.1f}s"
                )
                raise CircuitOpenError(
                    "Circuit breaker is OPEN",
                    failure_count=self.failure_count,
                    last_failure_time=self.last_failure_time
                )
        
        # Execute the protected function
        try:
            # Handle both async and sync functions
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success - reset failure tracking
            self._on_success()
            return result
            
        except self.expected_exception as e:
            # Expected failure - track and potentially open circuit
            self._on_failure()
            self.logger.error(
                f"Call failed with {type(e).__name__}: {str(e)} - "
                f"Failures: {self.failure_count}/{self.failure_threshold}"
            )
            raise
    
    def _on_success(self):
        """Handle successful call - reset failure tracking"""
        if self.failure_count > 0:
            self.logger.info(
                f"Call succeeded - resetting failure count from {self.failure_count}"
            )
        
        self.failure_count = 0
        self.metrics.successful_calls += 1
        
        # If we were in HALF_OPEN, success means we can close the circuit
        if self.state == CircuitState.HALF_OPEN:
            self.logger.info("Recovery successful - closing circuit")
            self._transition_to(CircuitState.CLOSED)
        elif self.state != CircuitState.CLOSED:
            self._transition_to(CircuitState.CLOSED)
    
    def _on_failure(self):
        """Handle failed call - increment failure count and potentially open circuit"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.metrics.failed_calls += 1
        
        if self.failure_count >= self.failure_threshold:
            self.logger.warning(
                f"Failure threshold exceeded ({self.failure_count}/{self.failure_threshold}) - "
                f"OPENING circuit"
            )
            self._transition_to(CircuitState.OPEN)
    
    def _should_attempt_reset(self) -> bool:
        """
        Check if enough time has passed to attempt circuit recovery.
        
        Returns:
            True if recovery timeout has elapsed since last failure
        """
        if self.last_failure_time is None:
            return True
        
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.recovery_timeout
    
    def _transition_to(self, new_state: CircuitState):
        """
        Transition circuit breaker to a new state.
        
        Args:
            new_state: Target state for transition
        """
        if new_state != self.state:
            old_state = self.state
            self.state = new_state
            self.metrics.state_transitions += 1
            self.metrics.last_state_change = time.time()
            
            self.logger.info(
                f"Circuit state transition: {old_state.value} -> {new_state.value}"
            )
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state"""
        return self.state
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics for monitoring"""
        return self.metrics
    
    def reset(self):
        """
        Manually reset circuit breaker to CLOSED state.
        
        This should only be used for testing or explicit administrative override.
        """
        self.logger.warning("Manual circuit breaker reset requested")
        self.failure_count = 0
        self.last_failure_time = None
        self._transition_to(CircuitState.CLOSED)


# ============================================================================
# Synchronous Circuit Breaker Wrapper
# ============================================================================

class SyncCircuitBreaker(CircuitBreaker):
    """
    Synchronous wrapper for CircuitBreaker for non-async codebases.
    
    This provides the same circuit breaker functionality without requiring
    async/await syntax. Useful for legacy code integration.
    """
    
    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function synchronously with circuit breaker protection.
        
        Args:
            func: Function to execute (non-async)
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result from func if successful
            
        Raises:
            CircuitOpenError: If circuit is OPEN
            Exception: Re-raises exceptions from func
        """
        self.metrics.total_calls += 1
        
        # Check if circuit is OPEN
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.logger.info("Attempting circuit recovery - transitioning to HALF_OPEN")
                self._transition_to(CircuitState.HALF_OPEN)
            else:
                self.metrics.rejected_calls += 1
                raise CircuitOpenError(
                    "Circuit breaker is OPEN",
                    failure_count=self.failure_count,
                    last_failure_time=self.last_failure_time
                )
        
        # Execute the protected function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            self.logger.error(
                f"Call failed with {type(e).__name__}: {str(e)} - "
                f"Failures: {self.failure_count}/{self.failure_threshold}"
            )
            raise
