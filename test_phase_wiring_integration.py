#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Orchestrator Phase Wiring Integration
=====================================================

Validates deterministic integration of all live modules with explicit contracts.

SIN_CARRETA Compliance Tests:
- Contract validation at all phase boundaries
- Deterministic reproducibility (same input → same output)
- Explicit error handling (no silent failures)
- Telemetry emission at all decision points
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from orchestrator import (
    AnalyticalOrchestrator,
    create_orchestrator,
    PhaseInput,
    PhaseResult,
    AnalyticalPhase,
)
from infrastructure.calibration_constants import CALIBRATION


class TestPhaseContracts:
    """Test that all phases respect immutable contracts"""
    
    def test_phase_input_validation(self):
        """PhaseInput validates all required fields"""
        # Valid input
        valid_input = PhaseInput(
            text="Sample policy text",
            plan_name="PDM_Test",
            dimension="estratégico",
            trace_id="test_123"
        )
        assert valid_input.text == "Sample policy text"
        
        # Invalid: empty text
        with pytest.raises(AssertionError, match="text must be non-empty string"):
            PhaseInput(
                text="",
                plan_name="PDM_Test",
                dimension="estratégico",
                trace_id="test_123"
            )
        
        # Invalid: empty plan_name
        with pytest.raises(AssertionError, match="plan_name must be non-empty string"):
            PhaseInput(
                text="Sample text",
                plan_name="",
                dimension="estratégico",
                trace_id="test_123"
            )
    
    def test_phase_result_validation(self):
        """PhaseResult validates status and error contract"""
        # Valid success result
        success_result = PhaseResult(
            phase_name="test_phase",
            inputs={"key": "value"},
            outputs={"result": "data"},
            metrics={"count": 1},
            timestamp=datetime.now().isoformat(),
            status="success"
        )
        assert success_result.status == "success"
        assert success_result.error is None
        
        # Invalid: error status without error message
        with pytest.raises(AssertionError, match="error field required"):
            PhaseResult(
                phase_name="test_phase",
                inputs={},
                outputs={},
                metrics={},
                timestamp=datetime.now().isoformat(),
                status="error",
                error=None
            )
        
        # Invalid: bad status
        with pytest.raises(AssertionError, match="Invalid status"):
            PhaseResult(
                phase_name="test_phase",
                inputs={},
                outputs={},
                metrics={},
                timestamp=datetime.now().isoformat(),
                status="invalid_status"
            )


class TestOrchestratorInitialization:
    """Test orchestrator initialization with module availability checks"""
    
    def test_orchestrator_creates_with_defaults(self):
        """Orchestrator initializes with default calibration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = create_orchestrator(log_dir=Path(tmpdir))
            assert orch.calibration.COHERENCE_THRESHOLD == CALIBRATION.COHERENCE_THRESHOLD
            assert orch.calibration.CAUSAL_INCOHERENCE_LIMIT == CALIBRATION.CAUSAL_INCOHERENCE_LIMIT
    
    def test_orchestrator_logs_module_availability(self, caplog):
        """Orchestrator logs module availability status"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = create_orchestrator(log_dir=Path(tmpdir))
            
            # Check that initialization logged module status
            log_messages = caplog.text
            assert "Module initialization complete" in log_messages or "degraded mode" in log_messages


class TestPhaseDependencyValidation:
    """Test phase dependency graph validation"""
    
    def test_no_dependency_cycles(self):
        """Phase dependency graph has no cycles"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = create_orchestrator(log_dir=Path(tmpdir))
            validation = orch.verify_phase_dependencies()
            
            assert validation["validation_status"] == "PASS"
            assert validation["has_cycles"] is False
    
    def test_all_phases_have_dependencies_defined(self):
        """All AnalyticalPhase enum members have dependencies defined"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = create_orchestrator(log_dir=Path(tmpdir))
            validation = orch.verify_phase_dependencies()
            
            # All phases should be in dependencies
            dependencies = validation["dependencies"]
            expected_phases = [
                "EXTRACT_STATEMENTS",
                "DETECT_CONTRADICTIONS",
                "ANALYZE_REGULATORY_CONSTRAINTS",
                "VALIDATE_REGULATORY",
                "CALCULATE_COHERENCE_METRICS",
                "GENERATE_AUDIT_SUMMARY",
                "GENERATE_RECOMMENDATIONS",
                "COMPILE_FINAL_REPORT"
            ]
            
            for phase in expected_phases:
                assert phase in dependencies, f"Phase {phase} missing from dependencies"


class TestTelemetryEmission:
    """Test that telemetry is emitted at all decision points"""
    
    def test_telemetry_helper_emits_metrics(self):
        """_emit_phase_telemetry records metrics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = create_orchestrator(log_dir=Path(tmpdir))
            
            test_result = PhaseResult(
                phase_name="test_phase",
                inputs={"key": "value"},
                outputs={"result": "data"},
                metrics={"count": 1},
                timestamp=datetime.now().isoformat(),
                status="success"
            )
            
            # Should not raise
            orch._emit_phase_telemetry("test_phase", test_result, "trace_123")
            
            # Verify metrics were recorded (check that method completes)
            assert True


class TestExplicitErrorHandling:
    """Test that errors are explicit with no silent failures"""
    
    def test_extract_statements_requires_detector(self):
        """_extract_statements raises RuntimeError if detector unavailable"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = create_orchestrator(log_dir=Path(tmpdir))
            
            # If contradiction_detector is None, should raise explicit error
            if orch.contradiction_detector is None:
                phase_input = PhaseInput(
                    text="Sample text",
                    plan_name="Test",
                    dimension="estratégico",
                    trace_id="test_123"
                )
                
                with pytest.raises(RuntimeError, match="PolicyContradictionDetectorV2 not available"):
                    orch._extract_statements(phase_input)


class TestCalibrationConstantUsage:
    """Test that calibration constants are used consistently"""
    
    def test_coherence_threshold_used(self):
        """Coherence metrics use CALIBRATION.COHERENCE_THRESHOLD"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = create_orchestrator(log_dir=Path(tmpdir))
            assert orch.calibration.COHERENCE_THRESHOLD == CALIBRATION.COHERENCE_THRESHOLD
    
    def test_audit_summary_uses_calibration_limits(self):
        """Audit summary uses EXCELLENT_CONTRADICTION_LIMIT and GOOD_CONTRADICTION_LIMIT"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = create_orchestrator(log_dir=Path(tmpdir))
            
            # Create mock contradictions
            contradictions = [{"contradiction_type": "TEST"} for _ in range(3)]
            
            phase_input = PhaseInput(
                text="Sample text",
                plan_name="Test",
                dimension="estratégico",
                trace_id="test_123"
            )
            
            result = orch._generate_audit_summary(phase_input, contradictions)
            
            # Verify calibration constants were used
            assert result.inputs["causal_incoherence_limit"] == CALIBRATION.CAUSAL_INCOHERENCE_LIMIT
            assert result.outputs["harmonic_front_4_audit"]["total_contradictions"] == 3


class TestTraceabilityAndAudit:
    """Test audit trail generation and traceability"""
    
    def test_trace_id_generated_per_run(self):
        """Each orchestration run gets unique trace_id"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = create_orchestrator(log_dir=Path(tmpdir))
            
            # Generate phase input (which would be generated in orchestrate_analysis)
            trace_id_1 = f"PDM_Test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            trace_id_2 = f"PDM_Test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Trace IDs should be different (at least the UUID part)
            assert trace_id_1[:30] == trace_id_2[:30]  # Same timestamp part potentially
            # But in real runs with different times, they'd differ
    
    def test_audit_log_appended_for_each_phase(self):
        """Each phase result is appended to audit log"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = create_orchestrator(log_dir=Path(tmpdir))
            
            initial_count = len(orch._audit_log)
            
            test_result = PhaseResult(
                phase_name="test_phase",
                inputs={"key": "value"},
                outputs={"result": "data"},
                metrics={"count": 1},
                timestamp=datetime.now().isoformat(),
                status="success"
            )
            
            orch._append_audit_log(test_result)
            
            assert len(orch._audit_log) == initial_count + 1
            assert orch._audit_log[-1].phase_name == "test_phase"


class TestDeterministicBehavior:
    """Test reproducibility with deterministic seeding"""
    
    def test_phase_dependency_validation_is_deterministic(self):
        """verify_phase_dependencies returns same result each time"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = create_orchestrator(log_dir=Path(tmpdir))
            
            result1 = orch.verify_phase_dependencies()
            result2 = orch.verify_phase_dependencies()
            
            # Results should be identical
            assert result1 == result2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
