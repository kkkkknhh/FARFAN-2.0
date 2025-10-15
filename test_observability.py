#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for F4.4: Comprehensive Observability Stack
Validates all observability metrics and functionality.
"""

import sys

import pytest

from infrastructure.observability import (
    DistributedTracer,
    MetricsCollector,
    ObservabilityConfig,
    ObservabilityStack,
    StructuredLogger,
)


def test_metrics_collector():
    """Test MetricsCollector functionality"""
    print("Testing MetricsCollector...")
    collector = MetricsCollector()

    # Test histogram
    collector.histogram("test.metric", 1.5, tags={"phase": "test"})
    collector.histogram("test.metric", 2.5, tags={"phase": "test"})

    # Test gauge
    collector.gauge("test.gauge", 100.0)

    # Test counter
    collector.increment("test.counter", tags={"type": "a"})
    collector.increment("test.counter", tags={"type": "a"})
    assert collector.get_count("test.counter", tags={"type": "a"}) == 2

    # Test summary
    summary = collector.get_summary()
    assert "histograms" in summary
    assert "gauges" in summary
    assert "counters" in summary

    print("✓ MetricsCollector tests passed")


def test_structured_logger():
    """Test StructuredLogger functionality"""
    print("Testing StructuredLogger...")
    logger = StructuredLogger("DEBUG")

    logger.debug("Debug message", context="test")
    logger.info("Info message", key="value")
    logger.warning("Warning message")
    logger.error("Error message", error_code=500)
    logger.critical("Critical message")

    print("✓ StructuredLogger tests passed")


def test_distributed_tracer():
    """Test DistributedTracer functionality"""
    print("Testing DistributedTracer...")
    tracer = DistributedTracer()

    # Start and finish a span
    span = tracer.start_span("test_operation", attributes={"test": "value"})
    assert span.operation_name == "test_operation"
    assert span.attributes["test"] == "value"

    tracer.finish_span(span)
    assert span.duration is not None

    # Get traces
    traces = tracer.get_traces()
    assert len(traces) == 1
    assert traces[0]["operation"] == "test_operation"

    print("✓ DistributedTracer tests passed")


def test_observability_stack():
    """Test complete ObservabilityStack"""
    print("Testing ObservabilityStack...")

    config = ObservabilityConfig(
        metrics_backend="in_memory", log_level="INFO", trace_backend="in_memory"
    )
    stack = ObservabilityStack(config)

    # Test pipeline duration recording
    stack.record_pipeline_duration(500.0)  # 500 seconds - no alert
    stack.record_pipeline_duration(2000.0)  # 2000 seconds - should alert

    # Test nonconvergent chain
    stack.record_nonconvergent_chain("chain_001", "R_hat > 1.1")

    # Test memory peak
    stack.record_memory_peak(8000.0)  # 8GB - no alert
    stack.record_memory_peak(17000.0)  # 17GB - should alert

    # Test hoop test failures
    for i in range(7):
        stack.record_hoop_test_failure(f"Q{i}", ["evidence_missing"])

    # Test dimension scores
    stack.record_dimension_score("D1", 0.75)
    stack.record_dimension_score("D6", 0.50)  # Should alert

    # Test trace operation context manager
    with stack.trace_operation("test_extraction", plan="PDM_001") as span:
        # Simulate some work
        import time

        time.sleep(0.1)

    # Get summaries
    metrics_summary = stack.get_metrics_summary()
    traces_summary = stack.get_traces_summary()

    assert "histograms" in metrics_summary
    assert "gauges" in metrics_summary
    assert "counters" in metrics_summary
    assert len(traces_summary) > 0

    # Verify specific metrics
    assert "pdm.pipeline.duration_seconds" in metrics_summary["histograms"]
    assert "pdm.memory.peak_mb" in metrics_summary["gauges"]
    assert metrics_summary["gauges"]["pdm.dimension.avg_score_D6"] == pytest.approx(0.50, rel=1e-9, abs=1e-12)  # replaced float equality with pytest.approx

    print("✓ ObservabilityStack tests passed")


def test_standard_metrics():
    """Test all Standard metrics mentioned in problem statement"""
    print("Testing Standard Metrics compliance...")

    config = ObservabilityConfig()
    stack = ObservabilityStack(config)

    # pdm.pipeline.duration_seconds
    stack.record_pipeline_duration(1500.0)

    # pdm.posterior.nonconvergent_count (CRITICAL)
    stack.record_nonconvergent_chain("chain_1", "Convergence failure")

    # pdm.memory.peak_mb
    stack.record_memory_peak(12000.0)

    # pdm.evidence.hoop_test_fail_count
    stack.record_hoop_test_failure("D1-Q5", ["regulatory_analysis"])

    # pdm.dimension.avg_score_D6
    stack.record_dimension_score("D6", 0.65)

    # Verify all metrics recorded
    summary = stack.get_metrics_summary()

    assert "pdm.pipeline.duration_seconds" in summary["histograms"]
    assert any("pdm.posterior.nonconvergent_count" in k for k in summary["counters"])
    assert "pdm.memory.peak_mb" in summary["gauges"]
    assert any("pdm.evidence.hoop_test_fail_count" in k for k in summary["counters"])
    assert "pdm.dimension.avg_score_D6" in summary["gauges"]

    print("✓ Standard Metrics compliance verified")


def main():
    """Run all tests"""
    print("=" * 70)
    print("F4.4: Comprehensive Observability Stack - Test Suite")
    print("=" * 70)
    print()

    try:
        test_metrics_collector()
        print()

        test_structured_logger()
        print()

        test_distributed_tracer()
        print()

        test_observability_stack()
        print()

        test_standard_metrics()
        print()

        print("=" * 70)
        print("ALL TESTS PASSED ✅")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
