#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for orchestration module
Tests PDMOrchestrator and AdaptiveLearningLoop
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from orchestration.learning_loop import (
    AdaptiveLearningLoop,
    Feedback,
    FeedbackExtractor,
    PriorHistoryStore,
)
from orchestration.pdm_orchestrator import (
    AnalysisResult,
    ExtractionResult,
    MechanismResult,
    PDMAnalysisState,
    PDMOrchestrator,
    QualityScore,
    ValidationResult,
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class MockSelfReflectionConfig:
    """Mock self-reflection configuration"""

    enable_prior_learning: bool = True
    prior_history_path: str = "/tmp/test_prior_history.json"


@dataclass
class MockConfig:
    """Mock configuration for testing"""

    queue_size: int = 10
    max_inflight_jobs: int = 3
    worker_timeout_secs: int = 60
    min_quality_threshold: float = 0.5
    prior_decay_factor: float = 0.9

    def __post_init__(self):
        self.self_reflection = MockSelfReflectionConfig()


def test_orchestrator_initialization():
    """Test PDMOrchestrator initialization"""
    print("Testing PDMOrchestrator initialization...")

    config = MockConfig()
    orchestrator = PDMOrchestrator(config)

    assert orchestrator.state == PDMAnalysisState.INITIALIZED
    assert orchestrator.metrics is not None
    assert orchestrator.audit_logger is not None

    print("✓ PDMOrchestrator initialized successfully")


def test_state_transitions():
    """Test state machine transitions"""
    print("Testing state transitions...")

    config = MockConfig()
    orchestrator = PDMOrchestrator(config)

    initial_state = orchestrator.state
    assert initial_state == PDMAnalysisState.INITIALIZED

    orchestrator._transition_state(PDMAnalysisState.EXTRACTING)
    assert orchestrator.state == PDMAnalysisState.EXTRACTING

    orchestrator._transition_state(PDMAnalysisState.BUILDING_DAG)
    assert orchestrator.state == PDMAnalysisState.BUILDING_DAG

    orchestrator._transition_state(PDMAnalysisState.COMPLETED)
    assert orchestrator.state == PDMAnalysisState.COMPLETED

    print("✓ State transitions working correctly")


def test_learning_loop_initialization():
    """Test AdaptiveLearningLoop initialization"""
    print("Testing AdaptiveLearningLoop initialization...")

    config = MockConfig()
    learning_loop = AdaptiveLearningLoop(config)

    assert learning_loop.enabled == True
    assert learning_loop.prior_store is not None
    assert learning_loop.feedback_extractor is not None

    print("✓ AdaptiveLearningLoop initialized successfully")


def test_prior_store_operations():
    """Test PriorHistoryStore operations"""
    print("Testing PriorHistoryStore operations...")

    import os
    import time

    # Use unique filename to avoid collision with previous test runs
    test_file = Path(f"/tmp/test_priors_{int(time.time() * 1000)}.json")

    # Ensure file doesn't exist
    if test_file.exists():
        test_file.unlink()

    store = PriorHistoryStore(test_file)

    # Get default prior (should be fresh)
    prior = store.get_mechanism_prior("test_mechanism")
    assert prior.alpha == pytest.approx(2.0, rel=1e-6, abs=1e-9)
    assert prior.beta == pytest.approx(2.0, rel=1e-6, abs=1e-9)

    # Update prior
    store.update_mechanism_prior(
        mechanism_type="test_mechanism", new_alpha=1.8, reason="Test decay"
    )

    updated_prior = store.get_mechanism_prior("test_mechanism")
    assert updated_prior.alpha == pytest.approx(1.8, rel=1e-6, abs=1e-9)
    assert updated_prior.update_count == 1

    # Save snapshot
    store.save_snapshot()
    assert len(store.snapshots) == 1

    # Clean up
    if test_file.exists():
        test_file.unlink()

    print("✓ PriorHistoryStore operations working correctly")


def test_feedback_extraction():
    """Test feedback extraction from analysis results"""
    print("Testing feedback extraction...")

    # Create mock analysis result
    @dataclass
    class MockAnalysisResult:
        mechanism_results: list
        quality_score: QualityScore

    mechanism_results = [
        MechanismResult(
            type="causal_link",
            necessity_test={"passed": False, "missing": ["evidence1"]},
            posterior_mean=0.5,
        ),
        MechanismResult(
            type="inference",
            necessity_test={"passed": True, "missing": []},
            posterior_mean=0.8,
        ),
    ]

    quality_score = QualityScore(
        overall_score=0.7,
        dimension_scores={
            "D1": 0.7,
            "D2": 0.6,
            "D3": 0.8,
            "D4": 0.7,
            "D5": 0.65,
            "D6": 0.75,
        },
    )

    result = MockAnalysisResult(
        mechanism_results=mechanism_results, quality_score=quality_score
    )

    extractor = FeedbackExtractor()
    feedback = extractor.extract_from_result(result)

    assert len(feedback.failed_mechanism_types) == 1
    assert len(feedback.passed_mechanism_types) == 1
    assert feedback.overall_quality == pytest.approx(0.7, rel=1e-6, abs=1e-9)
    assert "causal_link" in feedback.failed_mechanism_types

    print("✓ Feedback extraction working correctly")


def test_prior_updates():
    """Test prior updates based on feedback"""
    print("Testing prior updates...")

    config = MockConfig()
    learning_loop = AdaptiveLearningLoop(config)

    # Create mock analysis result with failures
    @dataclass
    class MockAnalysisResult:
        mechanism_results: list
        quality_score: QualityScore

    mechanism_results = [
        MechanismResult(
            type="failed_mech",
            necessity_test={"passed": False, "missing": ["data"]},
            posterior_mean=0.3,
        )
    ]

    result = MockAnalysisResult(
        mechanism_results=mechanism_results,
        quality_score=QualityScore(overall_score=0.5),
    )

    # Get initial prior
    initial_alpha = learning_loop.get_current_prior("failed_mech")

    # Update priors
    learning_loop.extract_and_update_priors(result)

    # Get updated prior
    updated_alpha = learning_loop.get_current_prior("failed_mech")

    # Should be decayed
    assert updated_alpha < initial_alpha
    assert (
        abs(updated_alpha - initial_alpha * 0.9) < 0.01
    )  # Allow small floating point error

    print("✓ Prior updates working correctly")


async def test_async_analyze_plan():
    """Test async analyze_plan method"""
    print("Testing async analyze_plan...")

    config = MockConfig()
    orchestrator = PDMOrchestrator(config)

    # Create a dummy PDF file for testing
    test_pdf = Path("/tmp/test_plan.pdf")
    test_pdf.write_text("dummy content")

    try:
        # This will use fallback implementations
        result = await orchestrator.analyze_plan(str(test_pdf))

        assert result is not None
        assert result.run_id is not None
        assert result.quality_score is not None
        assert orchestrator.state in [
            PDMAnalysisState.COMPLETED,
            PDMAnalysisState.FAILED,
        ]

        print("✓ Async analyze_plan working correctly")
    finally:
        # Clean up
        if test_pdf.exists():
            test_pdf.unlink()


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("Running Orchestration Module Tests")
    print("=" * 70)
    print()

    try:
        # Synchronous tests
        test_orchestrator_initialization()
        test_state_transitions()
        test_learning_loop_initialization()
        test_prior_store_operations()
        test_feedback_extraction()
        test_prior_updates()

        # Async tests
        asyncio.run(test_async_analyze_plan())

        print()
        print("=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)

        return True

    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"Test failed: {e}")
        print("=" * 70)
        return False
    except Exception as e:
        print()
        print("=" * 70)
        print(f"Test error: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
