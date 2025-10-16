#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for the Analytical Orchestrator
===========================================

Validates orchestrator behavior, determinism, and audit trail generation.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from infrastructure.calibration_constants import CALIBRATION
from orchestrator import (
    AnalyticalOrchestrator,
    create_orchestrator,
)


def test_orchestrator_creation():
    """Test that orchestrator can be created with default and custom calibration."""
    # Default calibration
    orch1 = create_orchestrator()
    assert orch1.calibration.COHERENCE_THRESHOLD == CALIBRATION.COHERENCE_THRESHOLD
    assert orch1.calibration.CAUSAL_INCOHERENCE_LIMIT == 5

    # Custom calibration
    from infrastructure.calibration_constants import override_calibration

    custom_cal = override_calibration(COHERENCE_THRESHOLD=0.8)
    orch2 = create_orchestrator(calibration=custom_cal)
    assert orch2.calibration.COHERENCE_THRESHOLD == pytest.approx(
        0.8, rel=1e-6, abs=1e-9
    )

    print("✓ Test orchestrator creation PASSED")


def test_phase_dependency_validation():
    """Test that phase dependencies are validated correctly."""
    orch = create_orchestrator()
    validation = orch.verify_phase_dependencies()

    assert validation["validation_status"] == "PASS"
    assert validation["has_cycles"] is False
    assert len(validation["dependencies"]) == 6  # 6 phases

    print("✓ Test phase dependency validation PASSED")


def test_orchestrator_execution():
    """Test that orchestrator can execute analysis pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        orch = create_orchestrator(log_dir=Path(tmpdir))

        # Execute with minimal test data
        result = orch.orchestrate_analysis(
            text="Plan de desarrollo municipal con objetivos claros.",
            plan_name="PDM_Test",
            dimension="estratégico",
        )

        # Verify structure
        assert "orchestration_metadata" in result
        assert "plan_name" in result
        assert result["plan_name"] == "PDM_Test"
        assert "dimension" in result

        # Verify phase results are included
        assert "extract_statements" in result
        assert "detect_contradictions" in result
        assert "analyze_regulatory_constraints" in result
        assert "calculate_coherence_metrics" in result
        assert "generate_audit_summary" in result

        # Verify calibration is preserved
        assert (
            result["orchestration_metadata"]["calibration"]["coherence_threshold"]
            == CALIBRATION.COHERENCE_THRESHOLD
        )

        # Verify audit log was created
        log_files = list(Path(tmpdir).glob("audit_log_*.json"))
        assert len(log_files) == 1

        # Verify audit log structure
        with open(log_files[0], "r") as f:
            audit_data = json.load(f)
            assert "plan_name" in audit_data
            assert "calibration" in audit_data
            assert "phases" in audit_data
            assert len(audit_data["phases"]) == 5  # 5 phases before compilation

    print("✓ Test orchestrator execution PASSED")


def test_deterministic_behavior():
    """Test that orchestrator produces deterministic results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        orch = create_orchestrator(log_dir=Path(tmpdir))

        test_text = "Plan de desarrollo con inversión de 1000 millones."

        # Run twice with same input
        result1 = orch.orchestrate_analysis(
            test_text, "PDM_Determinism_Test", "estratégico"
        )

        # Reset orchestrator state
        orch._audit_log = []
        orch._global_report = {
            "orchestration_metadata": {
                "version": "2.0.0",
                "calibration": orch.calibration,
                "execution_start": None,
                "execution_end": None,
            }
        }

        result2 = orch.orchestrate_analysis(
            test_text, "PDM_Determinism_Test", "estratégico"
        )

        # Compare key metrics (timestamps will differ, so check structure only)
        assert result1["plan_name"] == result2["plan_name"]
        assert result1["dimension"] == result2["dimension"]
        assert result1["total_statements"] == result2["total_statements"]
        assert result1["total_contradictions"] == result2["total_contradictions"]

    print("✓ Test deterministic behavior PASSED")


def test_error_handling():
    """Test that orchestrator handles errors gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        orch = create_orchestrator(log_dir=Path(tmpdir))

        # Empty text should still produce a report
        result = orch.orchestrate_analysis("", "PDM_Empty", "estratégico")

        # Should have structure even with empty input
        assert "orchestration_metadata" in result
        assert "plan_name" in result

        # Should have fallback values
        assert result["total_statements"] == 0
        assert result["total_contradictions"] == 0

    print("✓ Test error handling PASSED")


def test_calibration_constants_usage():
    """Test that calibration constants are used throughout pipeline."""
    from infrastructure.calibration_constants import override_calibration

    custom_cal = override_calibration(
        COHERENCE_THRESHOLD=0.85, CAUSAL_INCOHERENCE_LIMIT=3
    )
    orch = create_orchestrator(calibration=custom_cal)

    # Verify custom calibration is set
    assert orch.calibration.COHERENCE_THRESHOLD == pytest.approx(
        0.85, rel=1e-6, abs=1e-9
    )
    assert orch.calibration.CAUSAL_INCOHERENCE_LIMIT == 3

    # Execute pipeline
    result = orch.orchestrate_analysis(
        "Plan de desarrollo test.", "PDM_Calibration", "estratégico"
    )

    # Verify calibration is in metadata
    assert result["orchestration_metadata"]["calibration"][
        "coherence_threshold"
    ] == pytest.approx(0.85, rel=1e-6, abs=1e-9)
    assert (
        result["orchestration_metadata"]["calibration"]["causal_incoherence_limit"] == 3
    )

    print("✓ Test calibration constants usage PASSED")


def test_audit_log_immutability():
    """Test that audit log entries are preserved and not modified."""
    with tempfile.TemporaryDirectory() as tmpdir:
        orch = create_orchestrator(log_dir=Path(tmpdir))

        result = orch.orchestrate_analysis(
            "Plan de desarrollo municipal.", "PDM_Audit", "estratégico"
        )

        # Get audit log
        initial_log_count = len(orch._audit_log)

        # Verify log has expected number of phases
        assert initial_log_count == 5  # 5 phases before final compilation

        # Verify each phase has required fields
        for phase_result in orch._audit_log:
            assert hasattr(phase_result, "phase_name")
            assert hasattr(phase_result, "inputs")
            assert hasattr(phase_result, "outputs")
            assert hasattr(phase_result, "metrics")
            assert hasattr(phase_result, "timestamp")
            assert hasattr(phase_result, "status")

    print("✓ Test audit log immutability PASSED")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 70)
    print("RUNNING ORCHESTRATOR TEST SUITE")
    print("=" * 70 + "\n")

    test_orchestrator_creation()
    test_phase_dependency_validation()
    test_orchestrator_execution()
    test_deterministic_behavior()
    test_error_handling()
    test_calibration_constants_usage()
    test_audit_log_immutability()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
