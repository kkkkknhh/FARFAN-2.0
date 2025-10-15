#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Circuit Breaker Implementation
=============================================

Comprehensive tests for circuit breaker pattern and resilient DNP validator.

Author: AI Systems Architect
Version: 1.0.0
"""

import asyncio
import time
from typing import Any

from infrastructure import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    DNPAPIClient,
    PDMData,
    ResilientDNPValidator,
    ValidationResult,
)

# ============================================================================
# Mock External Services
# ============================================================================


class MockDNPAPIClient(DNPAPIClient):
    """Mock DNP API client for testing"""

    def __init__(self, should_fail: bool = False, failure_count: int = 0):
        self.should_fail = should_fail
        self.failure_count = failure_count
        self.call_count = 0

    async def validate_compliance(self, data: PDMData) -> dict:
        """Mock validation that can be configured to fail"""
        self.call_count += 1

        # Simulate failures for first N calls
        if self.call_count <= self.failure_count or self.should_fail:
            raise ConnectionError("DNP service unavailable")

        # Return success response
        return {
            "cumple": True,
            "score_total": 85.0,
            "nivel_cumplimiento": "BUENO",
            "detalles": "Mock validation passed",
        }


async def failing_function(*args, **kwargs):
    """Mock function that always fails"""
    raise ValueError("Simulated failure")


async def succeeding_function(*args, **kwargs):
    """Mock function that always succeeds"""
    return "success"


# ============================================================================
# Circuit Breaker Tests
# ============================================================================


async def test_circuit_breaker_initialization():
    """Test circuit breaker initialization and validation"""
    print("\n" + "=" * 70)
    print("TEST: Circuit Breaker Initialization")
    print("=" * 70)

    # Test default initialization
    breaker = CircuitBreaker()
    assert breaker.failure_threshold == 5
    assert breaker.recovery_timeout == 60
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0
    print("✓ Default initialization successful")

    # Test custom initialization
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
    assert breaker.failure_threshold == 3
    assert breaker.recovery_timeout == 30
    print("✓ Custom initialization successful")

    # Test validation
    try:
        CircuitBreaker(failure_threshold=0)
        assert False, "Should raise ValueError"
    except ValueError:
        print("✓ Input validation working")

    print("✅ Circuit Breaker Initialization: PASSED\n")


async def test_circuit_breaker_closed_state():
    """Test circuit breaker in CLOSED state (normal operation)"""
    print("=" * 70)
    print("TEST: Circuit Breaker CLOSED State")
    print("=" * 70)

    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)

    # Test successful calls
    result = await breaker.call(succeeding_function)
    assert result == "success"
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0
    print("✓ Successful call in CLOSED state")

    # Test single failure doesn't open circuit
    try:
        await breaker.call(failing_function)
    except ValueError:
        pass

    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 1
    print("✓ Single failure tracked, circuit remains CLOSED")

    # Successful call resets failure count
    await breaker.call(succeeding_function)
    assert breaker.failure_count == 0
    print("✓ Success resets failure count")

    print("✅ Circuit Breaker CLOSED State: PASSED\n")


async def test_circuit_breaker_opens_on_threshold():
    """Test circuit breaker opens after failure threshold"""
    print("=" * 70)
    print("TEST: Circuit Breaker OPEN on Threshold")
    print("=" * 70)

    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)

    # Trigger failures to reach threshold
    for i in range(3):
        try:
            await breaker.call(failing_function)
        except ValueError:
            print(f"✓ Failure {i + 1}/3 tracked")

    assert breaker.state == CircuitState.OPEN
    assert breaker.failure_count == 3
    print("✓ Circuit OPENED after 3 failures")

    # Verify circuit rejects calls
    try:
        await breaker.call(succeeding_function)
        assert False, "Should raise CircuitOpenError"
    except CircuitOpenError as e:
        print(f"✓ Circuit rejects calls: {e.failure_count} failures")

    metrics = breaker.get_metrics()
    assert metrics.rejected_calls > 0
    print("✓ Rejected calls tracked in metrics")

    print("✅ Circuit Breaker OPEN on Threshold: PASSED\n")


async def test_circuit_breaker_half_open_recovery():
    """Test circuit breaker HALF_OPEN state and recovery"""
    print("=" * 70)
    print("TEST: Circuit Breaker HALF_OPEN Recovery")
    print("=" * 70)

    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

    # Open the circuit
    for _ in range(2):
        try:
            await breaker.call(failing_function)
        except ValueError:
            pass

    assert breaker.state == CircuitState.OPEN
    print("✓ Circuit OPENED")

    # Wait for recovery timeout
    await asyncio.sleep(1.1)
    print("✓ Recovery timeout elapsed")

    # Next call should transition to HALF_OPEN
    result = await breaker.call(succeeding_function)
    assert result == "success"
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0
    print("✓ Circuit recovered: HALF_OPEN -> CLOSED")

    print("✅ Circuit Breaker HALF_OPEN Recovery: PASSED\n")


async def test_circuit_breaker_metrics():
    """Test circuit breaker metrics collection"""
    print("=" * 70)
    print("TEST: Circuit Breaker Metrics")
    print("=" * 70)

    breaker = CircuitBreaker(failure_threshold=2)

    # Execute various calls
    await breaker.call(succeeding_function)
    try:
        await breaker.call(failing_function)
    except ValueError:
        pass

    metrics = breaker.get_metrics()
    assert metrics.total_calls == 2
    assert metrics.successful_calls == 1
    assert metrics.failed_calls == 1
    print(
        f"✓ Metrics tracked: {metrics.total_calls} total, "
        f"{metrics.successful_calls} success, {metrics.failed_calls} failed"
    )

    print("✅ Circuit Breaker Metrics: PASSED\n")


# ============================================================================
# Resilient DNP Validator Tests
# ============================================================================


async def test_resilient_validator_success():
    """Test resilient validator with successful validation"""
    print("=" * 70)
    print("TEST: Resilient Validator Success")
    print("=" * 70)

    mock_client = MockDNPAPIClient(should_fail=False)
    validator = ResilientDNPValidator(mock_client, failure_threshold=3)

    data = PDMData(
        sector="salud",
        descripcion="Programa de atención primaria",
        indicadores_propuestos=["IND001", "IND002"],
    )

    result = await validator.validate(data)

    assert result.status == "passed"
    assert result.score > 0.0
    assert result.score_penalty == 0.0
    assert result.circuit_state == CircuitState.CLOSED.value
    print(f"✓ Validation passed: score={result.score}, status={result.status}")

    print("✅ Resilient Validator Success: PASSED\n")


async def test_resilient_validator_fail_open_policy():
    """Test resilient validator fail-open policy when circuit opens"""
    print("=" * 70)
    print("TEST: Resilient Validator Fail-Open Policy")
    print("=" * 70)

    # Create client that always fails
    mock_client = MockDNPAPIClient(should_fail=True)
    validator = ResilientDNPValidator(
        mock_client, failure_threshold=2, recovery_timeout=1, fail_open_penalty=0.05
    )

    data = PDMData(
        sector="educacion",
        descripcion="Programa educativo",
        indicadores_propuestos=["IND003"],
    )

    # First two calls will fail and open circuit
    for i in range(2):
        result = await validator.validate(data)
        print(f"✓ Call {i + 1}: status={result.status}, score={result.score}")

    # Third call should get skipped (fail-open)
    result = await validator.validate(data)
    assert result.status == "skipped"
    assert result.score == 1.0 - 0.05  # Perfect score minus penalty
    assert result.score_penalty == 0.05
    assert result.circuit_state == CircuitState.OPEN.value
    assert "circuit breaker OPEN" in result.reason
    print(
        f"✓ Fail-open policy applied: score={result.score}, penalty={result.score_penalty}"
    )

    print("✅ Resilient Validator Fail-Open Policy: PASSED\n")


async def test_resilient_validator_recovery():
    """Test resilient validator recovery after failures"""
    print("=" * 70)
    print("TEST: Resilient Validator Recovery")
    print("=" * 70)

    # Create client that fails first 2 times, then succeeds
    mock_client = MockDNPAPIClient(should_fail=False, failure_count=2)
    validator = ResilientDNPValidator(
        mock_client, failure_threshold=3, recovery_timeout=1
    )

    data = PDMData(
        sector="infraestructura",
        descripcion="Vías terciarias",
        indicadores_propuestos=["IND004"],
    )

    # First 2 calls fail
    for i in range(2):
        result = await validator.validate(data)
        print(f"✓ Call {i + 1}: status={result.status}")

    # Third call succeeds
    result = await validator.validate(data)
    assert result.status == "passed"
    assert result.score > 0.0
    assert result.score_penalty == 0.0
    print(f"✓ Recovery successful: status={result.status}, score={result.score}")

    print("✅ Resilient Validator Recovery: PASSED\n")


async def test_resilient_validator_metrics():
    """Test resilient validator metrics collection"""
    print("=" * 70)
    print("TEST: Resilient Validator Metrics")
    print("=" * 70)

    mock_client = MockDNPAPIClient(should_fail=False)
    validator = ResilientDNPValidator(mock_client)

    data = PDMData(
        sector="agua", descripcion="Acueducto rural", indicadores_propuestos=["IND005"]
    )

    # Execute some validations
    await validator.validate(data)
    await validator.validate(data)

    metrics = validator.get_circuit_metrics()
    assert metrics["state"] == CircuitState.CLOSED.value
    assert metrics["total_calls"] == 2
    assert metrics["successful_calls"] == 2
    assert metrics["failed_calls"] == 0
    print(
        f"✓ Metrics: {metrics['total_calls']} calls, "
        f"{metrics['successful_calls']} successful"
    )

    print("✅ Resilient Validator Metrics: PASSED\n")


# ============================================================================
# Test Runner
# ============================================================================


async def run_all_tests():
    """Run all tests sequentially"""
    print("\n" + "=" * 70)
    print("CIRCUIT BREAKER TEST SUITE")
    print("=" * 70)

    # Circuit Breaker tests
    await test_circuit_breaker_initialization()
    await test_circuit_breaker_closed_state()
    await test_circuit_breaker_opens_on_threshold()
    await test_circuit_breaker_half_open_recovery()
    await test_circuit_breaker_metrics()

    # Resilient DNP Validator tests
    await test_resilient_validator_success()
    await test_resilient_validator_fail_open_policy()
    await test_resilient_validator_recovery()
    await test_resilient_validator_metrics()

    print("=" * 70)
    print("ALL TESTS PASSED ✅")
    print("=" * 70)


if __name__ == "__main__":
    # Configure logging
    import logging

    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during tests
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run tests
    asyncio.run(run_all_tests())
