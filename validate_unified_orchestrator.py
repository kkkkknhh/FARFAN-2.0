#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation Script for Unified Orchestrator
==========================================
Validates the unified orchestrator implementation and integration.
"""

import asyncio
import sys
from pathlib import Path


def test_imports():
    """Test all imports are available"""
    print("Testing imports...")
    try:
        from orchestration.unified_orchestrator import (
            MetricsCollector,
            PipelineStage,
            PriorSnapshot,
            UnifiedOrchestrator,
            UnifiedResult,
        )

        print("✓ UnifiedOrchestrator imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_orchestrator_creation():
    """Test orchestrator can be instantiated"""
    print("\nTesting orchestrator creation...")
    try:
        from dataclasses import dataclass

        @dataclass
        class MockSelfReflection:
            enable_prior_learning: bool = True
            prior_history_path: str = "/tmp/test_priors.json"
            feedback_weight: float = 0.1
            min_documents_for_learning: int = 1

        @dataclass
        class MockConfig:
            self_reflection: MockSelfReflection = MockSelfReflection()
            prior_decay_factor: float = 0.9

        from orchestration.unified_orchestrator import UnifiedOrchestrator

        config = MockConfig()
        orchestrator = UnifiedOrchestrator(config)

        print(f"✓ Orchestrator created: {orchestrator}")
        return True
    except Exception as e:
        print(f"✗ Creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_prior_snapshot():
    """Test prior snapshot creation"""
    print("\nTesting prior snapshot...")
    try:
        from dataclasses import dataclass

        @dataclass
        class MockSelfReflection:
            enable_prior_learning: bool = True
            prior_history_path: str = "/tmp/test_priors.json"
            feedback_weight: float = 0.1
            min_documents_for_learning: int = 1

        @dataclass
        class MockConfig:
            self_reflection: MockSelfReflection = MockSelfReflection()
            prior_decay_factor: float = 0.9

        from orchestration.unified_orchestrator import UnifiedOrchestrator

        config = MockConfig()
        orchestrator = UnifiedOrchestrator(config)

        snapshot = orchestrator._create_prior_snapshot("test_run")

        print(f"✓ Snapshot created: {snapshot.run_id}")
        print(f"  Priors: {list(snapshot.priors.keys())}")
        return True
    except Exception as e:
        print(f"✗ Snapshot failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_metrics_collector():
    """Test metrics collector"""
    print("\nTesting metrics collector...")
    try:
        import time

        from orchestration.unified_orchestrator import (
            MetricsCollector,
            PipelineStage,
            StageMetrics,
        )

        collector = MetricsCollector()

        # Record some metrics
        collector.record("test_metric", 1.5)
        collector.increment("test_counter")

        # Add stage metric
        stage_metric = StageMetrics(
            stage=PipelineStage.STAGE_1_EXTRACTION,
            start_time=time.time(),
            end_time=time.time() + 1.0,
            duration_seconds=1.0,
            items_processed=10,
        )
        collector.add_stage_metric(stage_metric)

        summary = collector.get_summary()

        print(f"✓ Metrics collector working")
        print(f"  Stages tracked: {len(summary['stage_metrics'])}")
        return True
    except Exception as e:
        print(f"✗ Metrics collector failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_pipeline_stages():
    """Test individual pipeline stages"""
    print("\nTesting pipeline stages...")
    try:
        from dataclasses import dataclass

        @dataclass
        class MockSelfReflection:
            enable_prior_learning: bool = True
            prior_history_path: str = "/tmp/test_priors.json"
            feedback_weight: float = 0.1
            min_documents_for_learning: int = 1

        @dataclass
        class MockConfig:
            self_reflection: MockSelfReflection = MockSelfReflection()
            prior_decay_factor: float = 0.9

        from orchestration.unified_orchestrator import UnifiedOrchestrator

        config = MockConfig()
        orchestrator = UnifiedOrchestrator(config)

        # Test stage 0
        pdf_data = await orchestrator._stage_0_ingestion("/tmp/test.pdf", "test_run")
        print(f"✓ Stage 0 (Ingestion): {pdf_data['loaded']}")

        # Test stage 1 (with fallback)
        extraction = await orchestrator._stage_1_extraction(pdf_data, "test_run")
        print(f"✓ Stage 1 (Extraction): {len(extraction['chunks'])} chunks")

        return True
    except Exception as e:
        print(f"✗ Pipeline stages failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_circular_dependency_resolution():
    """Test circular dependency is resolved"""
    print("\nTesting circular dependency resolution...")
    try:
        from dataclasses import dataclass

        @dataclass
        class MockSelfReflection:
            enable_prior_learning: bool = True
            prior_history_path: str = "/tmp/test_priors_circ.json"
            feedback_weight: float = 0.1
            min_documents_for_learning: int = 1

        @dataclass
        class MockConfig:
            self_reflection: MockSelfReflection = MockSelfReflection()
            prior_decay_factor: float = 0.9

        from orchestration.unified_orchestrator import UnifiedOrchestrator

        config = MockConfig()
        orchestrator = UnifiedOrchestrator(config)

        # Create snapshot 1
        snapshot1 = orchestrator._create_prior_snapshot("run_1")
        initial_prior = snapshot1.priors["tecnico"]

        # Simulate update to prior store
        orchestrator.prior_store.update_mechanism_prior(
            "tecnico", initial_prior * 0.8, "penalty"
        )

        # Create snapshot 2
        snapshot2 = orchestrator._create_prior_snapshot("run_2")

        # Verify snapshots are different
        assert snapshot1.priors["tecnico"] != snapshot2.priors["tecnico"]
        print(f"✓ Circular dependency resolved")
        print(f"  Snapshot 1 prior: {snapshot1.priors['tecnico']:.3f}")
        print(f"  Snapshot 2 prior: {snapshot2.priors['tecnico']:.3f}")

        return True
    except Exception as e:
        print(f"✗ Circular dependency test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("=" * 60)
    print("UNIFIED ORCHESTRATOR VALIDATION")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Orchestrator Creation", test_orchestrator_creation),
        ("Prior Snapshot", test_prior_snapshot),
        ("Metrics Collector", test_metrics_collector),
        ("Circular Dependency", test_circular_dependency_resolution),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} raised exception: {e}")
            results.append((name, False))

    # Async tests
    print("\nRunning async tests...")
    try:
        result = asyncio.run(test_pipeline_stages())
        results.append(("Pipeline Stages", result))
    except Exception as e:
        print(f"✗ Pipeline stages raised exception: {e}")
        results.append(("Pipeline Stages", False))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All validation tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
