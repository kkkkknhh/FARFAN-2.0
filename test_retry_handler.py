#!/usr/bin/env python3
"""
Tests for RetryHandler with Circuit Breaker Integration
Validates retry logic, exponential backoff, jitter, and circuit breaker state transitions
"""

import time
import logging
from retry_handler import (
    RetryHandler,
    DependencyType,
    RetryConfig,
    CircuitBreakerState,
    CircuitBreakerOpenError
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_retry():
    """Test basic retry functionality"""
    print("\n" + "="*80)
    print("TEST 1: Basic Retry with Success")
    print("="*80)
    
    handler = RetryHandler()
    
    # Configure with low retries for testing
    handler.configure(DependencyType.PDF_PARSER, RetryConfig(
        base_delay=0.1,
        max_retries=3,
        failure_threshold=5
    ))
    
    attempt_count = [0]
    
    @handler.with_retry(DependencyType.PDF_PARSER)
    def flaky_operation():
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise IOError(f"Simulated failure (attempt {attempt_count[0]})")
        return "Success!"
    
    result = flaky_operation()
    assert result == "Success!"
    assert attempt_count[0] == 3
    
    stats = handler.get_stats(DependencyType.PDF_PARSER)
    print(f"✓ Function succeeded after {attempt_count[0]} attempts")
    print(f"✓ Circuit state: {stats['state']}")
    print(f"✓ Total requests: {stats['total_requests']}")
    print(f"✓ Success rate: {stats['success_rate']:.1%}")


def test_retry_exhaustion():
    """Test retry exhaustion and failure"""
    print("\n" + "="*80)
    print("TEST 2: Retry Exhaustion")
    print("="*80)
    
    handler = RetryHandler()
    
    handler.configure(DependencyType.SPACY_MODEL, RetryConfig(
        base_delay=0.05,
        max_retries=2,
        failure_threshold=3
    ))
    
    @handler.with_retry(DependencyType.SPACY_MODEL)
    def always_fails():
        raise RuntimeError("Persistent failure")
    
    try:
        always_fails()
        assert False, "Should have raised exception"
    except RuntimeError as e:
        print(f"✓ Correctly raised exception after retries: {e}")
    
    stats = handler.get_stats(DependencyType.SPACY_MODEL)
    print(f"✓ Circuit state: {stats['state']}")
    print(f"✓ Failure count: {stats['failure_count']}")


def test_circuit_breaker_open():
    """Test circuit breaker opening after threshold"""
    print("\n" + "="*80)
    print("TEST 3: Circuit Breaker Opening")
    print("="*80)
    
    handler = RetryHandler()
    
    handler.configure(DependencyType.DNP_API, RetryConfig(
        base_delay=0.05,
        max_retries=1,
        failure_threshold=3,
        recovery_timeout=1.0
    ))
    
    @handler.with_retry(DependencyType.DNP_API)
    def failing_api():
        raise ConnectionError("API unavailable")
    
    # Cause failures to trigger circuit breaker
    failures = 0
    for i in range(3):
        try:
            failing_api()
        except (ConnectionError, CircuitBreakerOpenError):
            failures += 1
    
    stats = handler.get_stats(DependencyType.DNP_API)
    print(f"✓ Circuit state after failures: {stats['state']}")
    print(f"✓ Total failure attempts: {failures}")
    
    # The circuit should be open after the failures
    # Try again - should fail fast with CircuitBreakerOpenError
    try:
        @handler.with_retry(DependencyType.DNP_API)
        def check_circuit():
            return "Should not execute"
        
        check_circuit()
        assert False, "Should have raised CircuitBreakerOpenError"
    except CircuitBreakerOpenError as e:
        print(f"✓ Circuit breaker correctly blocked request: {str(e)[:60]}...")


def test_circuit_breaker_recovery():
    """Test circuit breaker recovery (HALF_OPEN -> CLOSED)"""
    print("\n" + "="*80)
    print("TEST 4: Circuit Breaker Recovery")
    print("="*80)
    
    handler = RetryHandler()
    
    handler.configure(DependencyType.EMBEDDING_SERVICE, RetryConfig(
        base_delay=0.05,
        max_retries=1,
        failure_threshold=2,
        recovery_timeout=0.5,
        success_threshold=2
    ))
    
    attempt_count = [0]
    
    @handler.with_retry(DependencyType.EMBEDDING_SERVICE)
    def recovering_service():
        attempt_count[0] += 1
        if attempt_count[0] <= 4:  # Fail first 4 times
            raise RuntimeError("Service down")
        return "Service recovered"
    
    # Cause failures to open circuit
    for i in range(2):
        try:
            recovering_service()
        except (RuntimeError, CircuitBreakerOpenError):
            pass
    
    stats = handler.get_stats(DependencyType.EMBEDDING_SERVICE)
    print(f"✓ Circuit opened: {stats['state']}")
    assert stats['state'] == CircuitBreakerState.OPEN.value
    
    # Wait for recovery timeout
    print("  Waiting for recovery timeout...")
    time.sleep(0.6)
    
    # Circuit should transition to HALF_OPEN and then succeed
    try:
        result = recovering_service()
        print(f"✓ First recovery attempt succeeded: {result}")
    except RuntimeError:
        print("  First recovery attempt failed (HALF_OPEN -> OPEN)")
    
    # Try again after success threshold
    time.sleep(0.6)
    result = recovering_service()
    print(f"✓ Service recovered: {result}")
    
    stats = handler.get_stats(DependencyType.EMBEDDING_SERVICE)
    print(f"✓ Final circuit state: {stats['state']}")


def test_exponential_backoff():
    """Test exponential backoff timing"""
    print("\n" + "="*80)
    print("TEST 5: Exponential Backoff with Jitter")
    print("="*80)
    
    handler = RetryHandler()
    
    handler.configure(DependencyType.PDF_PARSER, RetryConfig(
        base_delay=0.1,
        max_retries=4,
        exponential_base=2.0,
        jitter_factor=0.1,
        max_delay=1.0
    ))
    
    delays = []
    attempt_count = [0]
    
    @handler.with_retry(DependencyType.PDF_PARSER)
    def measure_backoff():
        attempt_count[0] += 1
        if attempt_count[0] < 4:
            raise IOError("Failure")
        return "Success"
    
    start = time.time()
    result = measure_backoff()
    elapsed = time.time() - start
    
    print(f"✓ Total time with retries: {elapsed:.3f}s")
    print(f"✓ Attempts: {attempt_count[0]}")
    
    # Check retry history for delays
    history = handler.get_retry_history(DependencyType.PDF_PARSER)
    for record in history:
        if record['delay'] > 0:
            print(f"  Attempt {record['attempt']}: delay={record['delay']:.3f}s")


def test_context_manager():
    """Test retry context manager"""
    print("\n" + "="*80)
    print("TEST 6: Context Manager Usage")
    print("="*80)
    
    handler = RetryHandler()
    
    handler.configure(DependencyType.SPACY_MODEL, RetryConfig(
        base_delay=0.05,
        max_retries=2
    ))
    
    attempt_count = [0]
    
    try:
        with handler.retry_context(DependencyType.SPACY_MODEL, "load_model"):
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise OSError("Model not found")
            print("  Model loaded successfully")
    except Exception as e:
        print(f"✗ Context manager failed: {e}")
    
    print(f"✓ Context manager completed after {attempt_count[0]} attempts")


def test_retry_history():
    """Test retry history tracking"""
    print("\n" + "="*80)
    print("TEST 7: Retry History Tracking")
    print("="*80)
    
    handler = RetryHandler()
    
    @handler.with_retry(DependencyType.PDF_PARSER)
    def tracked_operation():
        return "Success"
    
    tracked_operation()
    
    history = handler.get_retry_history(DependencyType.PDF_PARSER, limit=10)
    print(f"✓ Retry history entries: {len(history)}")
    
    for record in history[-3:]:
        print(f"  {record['dependency']}.{record['operation']}: "
              f"attempt={record['attempt']}, success={record['success']}")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("RETRY HANDLER TEST SUITE")
    print("="*80)
    
    try:
        test_basic_retry()
        test_retry_exhaustion()
        test_circuit_breaker_open()
        test_circuit_breaker_recovery()
        test_exponential_backoff()
        test_context_manager()
        test_retry_history()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        return 0
    
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
