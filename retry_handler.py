#!/usr/bin/env python3
"""
Retry Handler with Exponential Backoff and Circuit Breaker Integration
Wraps external dependency calls with configurable retry logic for:
- PDF parsing (PyMuPDF/pdfplumber operations)
- spaCy model loading
- DNP API calls
- Embedding service operations
"""

import logging
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Types of external dependencies tracked by the retry handler"""

    PDF_PARSER = "pdf_parser"
    SPACY_MODEL = "spacy_model"
    DNP_API = "dnp_api"
    EMBEDDING_SERVICE = "embedding_service"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""

    base_delay: float = 1.0  # Base delay in seconds
    max_retries: int = 3  # Maximum retry attempts
    exponential_base: float = 2.0  # Exponential backoff base
    jitter_factor: float = 0.1  # Random jitter (0.0-1.0)
    max_delay: float = 60.0  # Maximum delay cap

    # Circuit breaker settings
    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 2  # Successes needed to close circuit


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""

    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_state_change: float = field(default_factory=time.time)
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0


@dataclass
class RetryAttempt:
    """Record of a retry attempt"""

    dependency: DependencyType
    operation: str
    attempt_number: int
    delay: float
    timestamp: float = field(default_factory=time.time)
    success: bool = False
    error: Optional[str] = None


T = TypeVar("T")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""

    pass


class RetryHandler:
    """
    Centralized retry handler with exponential backoff and circuit breaker integration.

    Features:
    - Configurable base delay, max retries, and jitter
    - Exponential backoff with jitter to prevent thundering herd
    - Per-dependency circuit breaker tracking
    - Automatic circuit breaker state transitions
    - Retry attempt logging and statistics

    Usage:
        handler = RetryHandler()

        # As decorator
        @handler.with_retry(DependencyType.PDF_PARSER)
        def parse_pdf(path):
            ...

        # As context manager
        with handler.retry_context(DependencyType.SPACY_MODEL):
            nlp = spacy.load("es_core_news_lg")
    """

    def __init__(self, default_config: Optional[RetryConfig] = None):
        """
        Initialize retry handler.

        Args:
            default_config: Default retry configuration (uses defaults if None)
        """
        self.default_config = default_config or RetryConfig()
        self.configs: Dict[DependencyType, RetryConfig] = {}
        self.circuit_breakers: Dict[DependencyType, CircuitBreakerStats] = {}
        self.retry_history: List[RetryAttempt] = []

        # Initialize circuit breakers for all dependency types
        for dep_type in DependencyType:
            self.circuit_breakers[dep_type] = CircuitBreakerStats()

        logger.info(
            "RetryHandler initialized with default config: "
            f"base_delay={self.default_config.base_delay}s, "
            f"max_retries={self.default_config.max_retries}, "
            f"failure_threshold={self.default_config.failure_threshold}"
        )

    def configure(self, dependency: DependencyType, config: RetryConfig):
        """
        Configure retry behavior for a specific dependency.

        Args:
            dependency: The dependency type to configure
            config: The retry configuration
        """
        self.configs[dependency] = config
        logger.info(
            f"Configured {dependency.value}: max_retries={config.max_retries}, "
            f"base_delay={config.base_delay}s"
        )

    def get_config(self, dependency: DependencyType) -> RetryConfig:
        """Get configuration for a dependency (or default)"""
        return self.configs.get(dependency, self.default_config)

    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """
        Calculate delay with exponential backoff and jitter.

        Args:
            attempt: Current attempt number (0-indexed)
            config: Retry configuration

        Returns:
            Delay in seconds
        """
        # Exponential backoff: base_delay * (exponential_base ^ attempt)
        exponential_delay = config.base_delay * (config.exponential_base**attempt)

        # Apply max delay cap
        exponential_delay = min(exponential_delay, config.max_delay)

        # Add random jitter: delay * (1 ± jitter_factor)
        jitter_range = exponential_delay * config.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)

        final_delay = max(0, exponential_delay + jitter)

        return final_delay

    def _check_circuit_breaker(self, dependency: DependencyType, config: RetryConfig):
        """
        Check circuit breaker state and handle transitions.

        Args:
            dependency: The dependency to check
            config: Retry configuration

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        stats = self.circuit_breakers[dependency]
        stats.total_requests += 1

        current_time = time.time()

        # Check if we should transition from OPEN to HALF_OPEN
        if stats.state == CircuitBreakerState.OPEN:
            if stats.last_failure_time is not None:
                time_since_failure = current_time - stats.last_failure_time
                if time_since_failure >= config.recovery_timeout:
                    self._transition_state(dependency, CircuitBreakerState.HALF_OPEN)
                    logger.info(
                        f"{dependency.value}: Circuit breaker transitioned to HALF_OPEN"
                    )
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker for {dependency.value} is OPEN. "
                        f"Retry in {config.recovery_timeout - time_since_failure:.1f}s"
                    )
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker for {dependency.value} is OPEN"
                )

    def _record_success(self, dependency: DependencyType):
        """
        Record a successful operation.

        Args:
            dependency: The dependency that succeeded
        """
        stats = self.circuit_breakers[dependency]
        stats.success_count += 1
        stats.total_successes += 1
        stats.failure_count = 0  # Reset consecutive failures

        config = self.get_config(dependency)

        # Handle state transitions based on success
        if stats.state == CircuitBreakerState.HALF_OPEN:
            if stats.success_count >= config.success_threshold:
                self._transition_state(dependency, CircuitBreakerState.CLOSED)
                logger.info(
                    f"{dependency.value}: Circuit breaker CLOSED after recovery"
                )
        elif stats.state == CircuitBreakerState.OPEN:
            # Should not happen, but handle gracefully
            self._transition_state(dependency, CircuitBreakerState.HALF_OPEN)

    def _record_failure(self, dependency: DependencyType, error: Exception):
        """
        Record a failed operation and handle circuit breaker transitions.

        Args:
            dependency: The dependency that failed
            error: The exception that occurred
        """
        stats = self.circuit_breakers[dependency]
        stats.failure_count += 1
        stats.total_failures += 1
        stats.last_failure_time = time.time()
        stats.success_count = 0  # Reset success count

        config = self.get_config(dependency)

        # Transition to OPEN if failure threshold exceeded
        if stats.state == CircuitBreakerState.CLOSED:
            if stats.failure_count >= config.failure_threshold:
                self._transition_state(dependency, CircuitBreakerState.OPEN)
                logger.error(
                    f"{dependency.value}: Circuit breaker OPEN after "
                    f"{stats.failure_count} consecutive failures"
                )
        elif stats.state == CircuitBreakerState.HALF_OPEN:
            # Any failure in HALF_OPEN goes back to OPEN
            self._transition_state(dependency, CircuitBreakerState.OPEN)
            logger.warning(
                f"{dependency.value}: Circuit breaker back to OPEN "
                f"after failure during recovery"
            )

    def _transition_state(
        self, dependency: DependencyType, new_state: CircuitBreakerState
    ):
        """
        Transition circuit breaker to a new state.

        Args:
            dependency: The dependency to transition
            new_state: The new state
        """
        stats = self.circuit_breakers[dependency]
        old_state = stats.state
        stats.state = new_state
        stats.last_state_change = time.time()

        if new_state == CircuitBreakerState.CLOSED:
            stats.failure_count = 0
            stats.success_count = 0

        logger.info(
            f"{dependency.value}: Circuit breaker {old_state.value} → {new_state.value}"
        )

    def with_retry(
        self,
        dependency: DependencyType,
        operation_name: Optional[str] = None,
        exceptions: tuple = (Exception,),
    ) -> Callable:
        """
        Decorator to wrap a function with retry logic.

        Args:
            dependency: Type of dependency being accessed
            operation_name: Name of the operation (defaults to function name)
            exceptions: Tuple of exceptions to catch and retry

        Returns:
            Decorated function

        Example:
            @handler.with_retry(DependencyType.PDF_PARSER)
            def parse_pdf(path):
                return fitz.open(path)
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                op_name = operation_name or func.__name__
                config = self.get_config(dependency)

                # Check circuit breaker before attempting
                self._check_circuit_breaker(dependency, config)

                last_exception = None

                for attempt in range(config.max_retries + 1):
                    try:
                        # Attempt the operation
                        result = func(*args, **kwargs)

                        # Record success
                        self._record_success(dependency)
                        self.retry_history.append(
                            RetryAttempt(
                                dependency=dependency,
                                operation=op_name,
                                attempt_number=attempt,
                                delay=0.0,
                                success=True,
                            )
                        )

                        if attempt > 0:
                            logger.info(
                                f"{dependency.value}.{op_name}: Success on attempt {attempt + 1}"
                            )

                        return result

                    except exceptions as e:
                        last_exception = e

                        # Record failure
                        self._record_failure(dependency, e)

                        # Check if we should retry
                        if attempt < config.max_retries:
                            delay = self._calculate_delay(attempt, config)

                            self.retry_history.append(
                                RetryAttempt(
                                    dependency=dependency,
                                    operation=op_name,
                                    attempt_number=attempt,
                                    delay=delay,
                                    success=False,
                                    error=str(e),
                                )
                            )

                            logger.warning(
                                f"{dependency.value}.{op_name}: Attempt {attempt + 1} failed: {e}. "
                                f"Retrying in {delay:.2f}s..."
                            )

                            time.sleep(delay)
                        else:
                            # Max retries exhausted
                            self.retry_history.append(
                                RetryAttempt(
                                    dependency=dependency,
                                    operation=op_name,
                                    attempt_number=attempt,
                                    delay=0.0,
                                    success=False,
                                    error=str(e),
                                )
                            )

                            logger.error(
                                f"{dependency.value}.{op_name}: Failed after {config.max_retries + 1} attempts"
                            )

                # Re-raise the last exception after exhausting retries
                raise last_exception

            return wrapper

        return decorator

    @contextmanager
    def retry_context(
        self,
        dependency: DependencyType,
        operation_name: str = "operation",
        exceptions: tuple = (Exception,),
    ):
        """
        Context manager for retry logic.

        Args:
            dependency: Type of dependency being accessed
            operation_name: Name of the operation
            exceptions: Tuple of exceptions to catch and retry

        Yields:
            None

        Example:
            with handler.retry_context(DependencyType.SPACY_MODEL):
                nlp = spacy.load("es_core_news_lg")
        """
        config = self.get_config(dependency)

        # Check circuit breaker
        self._check_circuit_breaker(dependency, config)

        last_exception = None

        for attempt in range(config.max_retries + 1):
            try:
                yield

                # If we get here, operation succeeded
                self._record_success(dependency)
                self.retry_history.append(
                    RetryAttempt(
                        dependency=dependency,
                        operation=operation_name,
                        attempt_number=attempt,
                        delay=0.0,
                        success=True,
                    )
                )

                if attempt > 0:
                    logger.info(
                        f"{dependency.value}.{operation_name}: Success on attempt {attempt + 1}"
                    )

                return

            except exceptions as e:
                last_exception = e
                self._record_failure(dependency, e)

                if attempt < config.max_retries:
                    delay = self._calculate_delay(attempt, config)

                    self.retry_history.append(
                        RetryAttempt(
                            dependency=dependency,
                            operation=operation_name,
                            attempt_number=attempt,
                            delay=delay,
                            success=False,
                            error=str(e),
                        )
                    )

                    logger.warning(
                        f"{dependency.value}.{operation_name}: Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    time.sleep(delay)
                else:
                    self.retry_history.append(
                        RetryAttempt(
                            dependency=dependency,
                            operation=operation_name,
                            attempt_number=attempt,
                            delay=0.0,
                            success=False,
                            error=str(e),
                        )
                    )

                    logger.error(
                        f"{dependency.value}.{operation_name}: Failed after {config.max_retries + 1} attempts"
                    )

        if last_exception:
            raise last_exception

    def get_stats(self, dependency: Optional[DependencyType] = None) -> Dict[str, Any]:
        """
        Get statistics for circuit breakers.

        Args:
            dependency: Specific dependency to get stats for (or all if None)

        Returns:
            Dictionary of statistics
        """
        if dependency:
            stats = self.circuit_breakers[dependency]
            return {
                "dependency": dependency.value,
                "state": stats.state.value,
                "failure_count": stats.failure_count,
                "success_count": stats.success_count,
                "total_requests": stats.total_requests,
                "total_failures": stats.total_failures,
                "total_successes": stats.total_successes,
                "success_rate": (
                    stats.total_successes / stats.total_requests
                    if stats.total_requests > 0
                    else 0.0
                ),
                "last_failure_time": stats.last_failure_time,
                "last_state_change": stats.last_state_change,
            }
        else:
            return {dep.value: self.get_stats(dep) for dep in DependencyType}

    def get_retry_history(
        self, dependency: Optional[DependencyType] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get retry attempt history.

        Args:
            dependency: Filter by dependency (or all if None)
            limit: Maximum number of records to return

        Returns:
            List of retry attempt records
        """
        history = self.retry_history

        if dependency:
            history = [h for h in history if h.dependency == dependency]

        history = history[-limit:]

        return [
            {
                "dependency": h.dependency.value,
                "operation": h.operation,
                "attempt": h.attempt_number,
                "delay": h.delay,
                "timestamp": h.timestamp,
                "success": h.success,
                "error": h.error,
            }
            for h in history
        ]

    def reset(self, dependency: Optional[DependencyType] = None):
        """
        Reset circuit breaker state.

        Args:
            dependency: Specific dependency to reset (or all if None)
        """
        if dependency:
            self.circuit_breakers[dependency] = CircuitBreakerStats()
            logger.info(f"Reset circuit breaker for {dependency.value}")
        else:
            for dep in DependencyType:
                self.circuit_breakers[dep] = CircuitBreakerStats()
            logger.info("Reset all circuit breakers")


# Global singleton instance
_global_handler: Optional[RetryHandler] = None


def get_retry_handler() -> RetryHandler:
    """
    Get the global retry handler singleton.

    Returns:
        Global RetryHandler instance
    """
    global _global_handler
    if _global_handler is None:
        _global_handler = RetryHandler()
    return _global_handler


def configure_global_handler(config: RetryConfig):
    """
    Configure the global retry handler.

    Args:
        config: Default retry configuration
    """
    global _global_handler
    _global_handler = RetryHandler(default_config=config)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    print("=== RetryHandler Demo ===\n")

    handler = RetryHandler()

    # Configure specific dependencies
    handler.configure(
        DependencyType.PDF_PARSER,
        RetryConfig(base_delay=0.5, max_retries=3, failure_threshold=3),
    )

    # Example 1: Decorator usage
    @handler.with_retry(DependencyType.PDF_PARSER)
    def flaky_pdf_parse(fail_count: int = 0):
        """Simulates a flaky PDF parsing operation"""
        if fail_count > 0:
            fail_count -= 1
            raise IOError("PDF file temporarily locked")
        return "PDF parsed successfully"

    # Example 2: Context manager usage
    def test_context_manager():
        try:
            with handler.retry_context(DependencyType.SPACY_MODEL, "load_model"):
                raise RuntimeError("Model download failed")
        except RuntimeError:
            print("Context manager caught exception after retries\n")

    # Run examples
    print("1. Testing successful operation:")
    result = flaky_pdf_parse()
    print(f"Result: {result}\n")

    print("2. Testing retries:")
    try:
        flaky_pdf_parse(fail_count=2)
    except:
        pass

    print("\n3. Testing context manager:")
    test_context_manager()

    print("\n4. Circuit Breaker Stats:")
    stats = handler.get_stats()
    for dep, dep_stats in stats.items():
        print(f"{dep}:")
        print(f"  State: {dep_stats['state']}")
        print(f"  Total Requests: {dep_stats['total_requests']}")
        print(f"  Successes: {dep_stats['total_successes']}")
        print(f"  Failures: {dep_stats['total_failures']}")
        print(f"  Success Rate: {dep_stats['success_rate']:.1%}")
