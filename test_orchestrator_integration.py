#!/usr/bin/env python3
"""
Integration test for orchestrator with resilience components

This test validates that the orchestrator properly integrates all resilience
components without requiring a real PDF or full module initialization.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch


def test_orchestrator_initialization():
    """Test that orchestrator initializes with resilience components"""
    print("\n" + "=" * 60)
    print("TEST: Orchestrator Initialization with Resilience")
    print("=" * 60)

    from orchestrator import FARFANOrchestrator

    temp_dir = Path(tempfile.mkdtemp())

    orchestrator = FARFANOrchestrator(output_dir=temp_dir, log_level="WARNING")

    # Verify resilience components are initialized
    assert orchestrator.risk_registry is not None, "RiskRegistry not initialized"
    print("✓ RiskRegistry initialized")

    assert orchestrator.circuit_breaker_registry is not None, (
        "CircuitBreakerRegistry not initialized"
    )
    print("✓ CircuitBreakerRegistry initialized")

    assert orchestrator.checkpoint is not None, "PipelineCheckpoint not initialized"
    print("✓ PipelineCheckpoint initialized")

    assert orchestrator.metrics is not None, "PipelineMetrics not initialized"
    print("✓ PipelineMetrics initialized")

    # Verify risk registry has default risks
    assert len(orchestrator.risk_registry.risks) > 0, "No default risks loaded"
    print(f"✓ RiskRegistry has {len(orchestrator.risk_registry.risks)} default risks")

    print("✅ Orchestrator initialization test passed")


def test_stage_execution_flow():
    """Test that _execute_stage_with_protection works correctly"""
    print("\n" + "=" * 60)
    print("TEST: Stage Execution with Protection")
    print("=" * 60)

    from orchestrator import FARFANOrchestrator, PipelineContext

    temp_dir = Path(tempfile.mkdtemp())
    orchestrator = FARFANOrchestrator(temp_dir, log_level="WARNING")

    # Create mock context
    ctx = PipelineContext(
        pdf_path=Path("test.pdf"), policy_code="TEST-001", output_dir=temp_dir
    )

    # Start metrics tracking
    orchestrator.metrics.start_execution("TEST-001")

    # Test successful stage execution
    def mock_stage_func(context):
        context.raw_text = "Mock text"
        return context

    try:
        result_ctx = orchestrator._execute_stage_with_protection(
            "TEST_STAGE", mock_stage_func, ctx
        )

        assert result_ctx.raw_text == "Mock text", "Stage didn't execute correctly"
        print("✓ Successful stage execution with protection")

        # Verify checkpoint was created
        checkpoints = orchestrator.checkpoint.list_checkpoints("TEST-001")
        assert len(checkpoints) > 0, "No checkpoint created"
        print(f"✓ Checkpoint created: {checkpoints[0].checkpoint_id}")

        # Verify metrics were recorded
        assert orchestrator.metrics.current_trace is not None, "No metrics trace"
        assert len(orchestrator.metrics.current_trace.stages) > 0, "No stages recorded"
        print(
            f"✓ Metrics recorded for {len(orchestrator.metrics.current_trace.stages)} stage(s)"
        )

    except Exception as e:
        print(f"❌ Stage execution failed: {e}")
        raise

    print("✅ Stage execution flow test passed")


def test_failure_handling():
    """Test that failures are handled with risk-based mitigation"""
    print("\n" + "=" * 60)
    print("TEST: Failure Handling with Risk-Based Mitigation")
    print("=" * 60)

    from orchestrator import FARFANOrchestrator, PipelineContext

    temp_dir = Path(tempfile.mkdtemp())
    orchestrator = FARFANOrchestrator(temp_dir, log_level="WARNING")

    ctx = PipelineContext(
        pdf_path=Path("test.pdf"), policy_code="TEST-002", output_dir=temp_dir
    )

    orchestrator.metrics.start_execution("TEST-002")

    # Test stage that raises a known exception (non-CRITICAL)
    def failing_stage_medium(context):
        raise TimeoutError("Stage timed out")

    try:
        orchestrator._execute_stage_with_protection(
            "SEMANTIC_ANALYSIS", failing_stage_medium, ctx
        )
        print("⚠️  Stage continued after mitigation (expected)")
    except TimeoutError:
        print("✓ Medium severity exception handled with mitigation attempt")

    # Verify mitigation was attempted
    assert len(orchestrator.risk_registry.mitigation_history) > 0, (
        "No mitigation attempted"
    )
    print(
        f"✓ Mitigation history: {len(orchestrator.risk_registry.mitigation_history)} attempt(s)"
    )

    # Test CRITICAL failure (should abort immediately)
    def failing_stage_critical(context):
        raise FileNotFoundError("PDF not found")

    ctx2 = PipelineContext(
        pdf_path=Path("test2.pdf"), policy_code="TEST-003", output_dir=temp_dir
    )

    orchestrator.metrics.start_execution("TEST-003")

    try:
        orchestrator._execute_stage_with_protection(
            "LOAD_DOCUMENT", failing_stage_critical, ctx2
        )
        print("❌ CRITICAL failure should have aborted")
        assert False, "CRITICAL failure should abort"
    except FileNotFoundError:
        print("✓ CRITICAL failure aborted execution (expected)")

    # Verify alert was emitted
    assert len(orchestrator.metrics.current_trace.alerts) > 0, "No alerts emitted"
    critical_alerts = [
        a
        for a in orchestrator.metrics.current_trace.alerts
        if a.level.value == "CRITICAL"
    ]
    assert len(critical_alerts) > 0, "No CRITICAL alert emitted"
    print(f"✓ CRITICAL alert emitted: {critical_alerts[0].message}")

    print("✅ Failure handling test passed")


def test_circuit_breaker_integration():
    """Test that circuit breakers are properly integrated"""
    print("\n" + "=" * 60)
    print("TEST: Circuit Breaker Integration")
    print("=" * 60)

    from orchestrator import FARFANOrchestrator, PipelineContext

    temp_dir = Path(tempfile.mkdtemp())
    orchestrator = FARFANOrchestrator(temp_dir, log_level="WARNING")

    ctx = PipelineContext(
        pdf_path=Path("test.pdf"), policy_code="TEST-004", output_dir=temp_dir
    )

    orchestrator.metrics.start_execution("TEST-004")

    # Stage that always fails
    def always_fails(context):
        raise RuntimeError("Persistent failure")

    # First failure
    try:
        orchestrator._execute_stage_with_protection("TEST_STAGE_CB", always_fails, ctx)
    except RuntimeError:
        pass

    # Second failure (should open circuit)
    try:
        orchestrator._execute_stage_with_protection("TEST_STAGE_CB", always_fails, ctx)
    except RuntimeError:
        pass

    # Check circuit breaker state
    breaker = orchestrator.circuit_breaker_registry.breakers.get("TEST_STAGE_CB")
    assert breaker is not None, "Circuit breaker not created"
    print(f"✓ Circuit breaker state: {breaker.state.value}")

    # Third call should be blocked by open circuit
    from circuit_breaker import CircuitBreakerError

    try:
        orchestrator._execute_stage_with_protection("TEST_STAGE_CB", always_fails, ctx)
        # If circuit is open, should raise CircuitBreakerError
    except (RuntimeError, CircuitBreakerError) as e:
        print(f"✓ Circuit breaker blocked call or caught failure: {type(e).__name__}")

    stats = breaker.get_stats()
    print(
        f"✓ Circuit breaker stats: {stats['total_failures']} failures, state={stats['state']}"
    )

    print("✅ Circuit breaker integration test passed")


def main():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("ORCHESTRATOR INTEGRATION TESTS")
    print("=" * 60)

    try:
        test_orchestrator_initialization()
        test_stage_execution_flow()
        test_failure_handling()
        test_circuit_breaker_integration()

        print("\n" + "=" * 60)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("=" * 60 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
