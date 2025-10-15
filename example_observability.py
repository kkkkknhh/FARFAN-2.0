#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of F4.4: Comprehensive Observability Stack
Demonstrates integration with pipeline analysis.
"""

from infrastructure.observability import ObservabilityConfig, ObservabilityStack


def example_pipeline_with_observability():
    """Example of using ObservabilityStack in a pipeline"""

    # Configure observability
    config = ObservabilityConfig(
        metrics_backend="in_memory", log_level="INFO", trace_backend="in_memory"
    )

    observability = ObservabilityStack(config)

    print("=" * 70)
    print("Example: PDM Pipeline with Comprehensive Observability")
    print("=" * 70)
    print()

    # Simulate pipeline execution
    import time

    # Track pipeline duration
    pipeline_start = time.time()

    # Phase 1: Extraction
    with observability.trace_operation(
        "extract_document", plan="PDM_EXAMPLE_001"
    ) as span:
        print("Phase 1: Extracting document...")
        time.sleep(0.1)
        # Record memory peak during extraction
        observability.record_memory_peak(8500.0)

    # Phase 2: Graph construction
    with observability.trace_operation("build_causal_graph", nodes=45) as span:
        print("Phase 2: Building causal graph...")
        time.sleep(0.1)

    # Phase 3: Bayesian inference
    with observability.trace_operation("bayesian_inference", chains=4) as span:
        print("Phase 3: Running Bayesian inference...")
        time.sleep(0.1)

        # Simulate convergence check - one chain fails
        observability.record_nonconvergent_chain(
            "chain_3", "R_hat=1.15 exceeds threshold of 1.1"
        )

    # Phase 4: Evidence validation
    with observability.trace_operation("validate_evidence") as span:
        print("Phase 4: Validating evidence...")
        time.sleep(0.1)

        # Record hoop test failures
        observability.record_hoop_test_failure(
            "D1-Q5", ["regulatory_constraint_missing"]
        )
        observability.record_hoop_test_failure("D3-Q3", ["budget_traceability_weak"])

    # Phase 5: Quality scoring
    with observability.trace_operation("calculate_quality_scores") as span:
        print("Phase 5: Calculating quality scores...")
        time.sleep(0.1)

        # Record dimension scores
        observability.record_dimension_score("D1", 0.72)
        observability.record_dimension_score("D2", 0.68)
        observability.record_dimension_score("D3", 0.75)
        observability.record_dimension_score("D4", 0.70)
        observability.record_dimension_score("D5", 0.65)
        observability.record_dimension_score("D6", 0.52)  # Below threshold - alerts

    # Record total pipeline duration
    pipeline_duration = time.time() - pipeline_start
    observability.record_pipeline_duration(pipeline_duration)

    print()
    print("=" * 70)
    print("Observability Summary")
    print("=" * 70)

    # Get metrics summary
    metrics = observability.get_metrics_summary()

    print("\n📊 Metrics Summary:")
    print(f"  Histograms: {list(metrics['histograms'].keys())}")
    print(f"  Gauges: {list(metrics['gauges'].keys())}")
    print(f"  Counters: {list(metrics['counters'].keys())}")

    # Get traces
    traces = observability.get_traces_summary()
    print(f"\n🔍 Traces Recorded: {len(traces)} operations")
    for trace in traces:
        print(f"  - {trace['operation']}: {trace['duration']:.3f}s")

    # Show specific critical metrics
    print("\n⚠️  Critical Metrics:")
    if "pdm.memory.peak_mb" in metrics["gauges"]:
        print(f"  Peak Memory: {metrics['gauges']['pdm.memory.peak_mb']:.1f} MB")

    if "pdm.dimension.avg_score_D6" in metrics["gauges"]:
        d6_score = metrics["gauges"]["pdm.dimension.avg_score_D6"]
        status = "❌ BELOW THRESHOLD" if d6_score < 0.55 else "✓ OK"
        print(f"  D6 Score: {d6_score:.2f} {status}")

    nonconvergent = sum(1 for k in metrics["counters"] if "nonconvergent" in k)
    if nonconvergent > 0:
        print(f"  Non-convergent Chains: {nonconvergent} ⚠️")

    print()
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


def example_alert_thresholds():
    """Demonstrate alert threshold behavior"""

    config = ObservabilityConfig(log_level="WARNING")
    obs = ObservabilityStack(config)

    print("\n" + "=" * 70)
    print("Alert Threshold Examples")
    print("=" * 70)
    print()

    print("1. Pipeline Duration Alerts:")
    print("   - 1500s (25 min): No alert")
    obs.record_pipeline_duration(1500.0)
    print("   - 2000s (33 min): HIGH alert triggered ⚠️")
    obs.record_pipeline_duration(2000.0)

    print("\n2. Memory Peak Alerts:")
    print("   - 12GB: No alert")
    obs.record_memory_peak(12000.0)
    print("   - 18GB: WARNING alert triggered ⚠️")
    obs.record_memory_peak(18000.0)

    print("\n3. Hoop Test Failure Alerts:")
    print("   - Failures 1-5: No alert")
    for i in range(5):
        obs.record_hoop_test_failure(f"Q{i}", ["missing"])
    print("   - Failure 6+: HIGH alert triggered ⚠️")
    obs.record_hoop_test_failure("Q6", ["missing"])

    print("\n4. D6 Score Alerts:")
    print("   - D6=0.60: No alert")
    obs.record_dimension_score("D6", 0.60)
    print("   - D6=0.50: CRITICAL alert triggered ⚠️")
    obs.record_dimension_score("D6", 0.50)

    print("\n5. Non-convergent Chain Alerts:")
    print("   - Every non-convergent chain: CRITICAL alert ⚠️")
    obs.record_nonconvergent_chain("chain_1", "R_hat > 1.1")

    print()


if __name__ == "__main__":
    example_pipeline_with_observability()
    example_alert_thresholds()
