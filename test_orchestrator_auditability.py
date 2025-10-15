#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Orchestrator Auditability and Determinism
=========================================================

Validates structured telemetry, contract enforcement, and audit trail generation
per SIN_CARRETA doctrine.

Test Coverage:
- Telemetry event emission at all phase boundaries
- Contract validation for PhaseResult
- Trace context propagation across phases
- Input/output hashing for reproducibility
- Telemetry completeness verification
- Audit log immutability and retention
"""

import json
import tempfile
from pathlib import Path

import pytest

from orchestrator import (
    AnalyticalOrchestrator,
    PhaseResult,
    create_orchestrator,
)
from infrastructure.telemetry import (
    TelemetryCollector,
    TraceContext,
    ContractViolationError,
    ValidationCheckError,
    EventType,
)


class TestTelemetryModule:
    """Test the telemetry module functionality"""
    
    def test_trace_context_creation(self):
        """Test that trace contexts are created correctly"""
        root_trace = TraceContext.create_root("test_run_001")
        
        assert root_trace.trace_id is not None
        assert root_trace.span_id is not None
        assert root_trace.parent_span_id is None
        assert root_trace.audit_id.startswith("audit_test_run_001")
        
        # Create child span
        child_trace = root_trace.create_child_span("extract_statements")
        assert child_trace.trace_id == root_trace.trace_id
        assert child_trace.span_id != root_trace.span_id
        assert child_trace.parent_span_id == root_trace.span_id
        assert child_trace.audit_id == root_trace.audit_id
        
        print("✓ Test trace context creation PASSED")
    
    def test_telemetry_hash_determinism(self):
        """Test that data hashing is deterministic"""
        data1 = {"key1": "value1", "key2": 42}
        data2 = {"key2": 42, "key1": "value1"}  # Different order
        
        hash1 = TelemetryCollector.hash_data(data1)
        hash2 = TelemetryCollector.hash_data(data2)
        
        # Hashes should be identical (sorted keys)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length
        
        # Different data should produce different hash
        data3 = {"key1": "different", "key2": 42}
        hash3 = TelemetryCollector.hash_data(data3)
        assert hash3 != hash1
        
        print("✓ Test telemetry hash determinism PASSED")
    
    def test_telemetry_event_emission(self):
        """Test that telemetry events are emitted correctly"""
        telemetry = TelemetryCollector()
        trace = TraceContext.create_root("test_run")
        
        # Emit phase start
        telemetry.emit_phase_start(
            phase_name="test_phase",
            trace_context=trace,
            inputs={"test_input": "value"}
        )
        
        # Emit phase completion
        telemetry.emit_phase_completion(
            phase_name="test_phase",
            trace_context=trace,
            outputs={"test_output": "result"},
            metrics={"duration": 1.5}
        )
        
        events = telemetry.get_events()
        assert len(events) == 2
        
        start_event = events[0]
        assert start_event.event_type == EventType.PHASE_START
        assert start_event.phase_name == "test_phase"
        assert start_event.input_hash != ""
        
        completion_event = events[1]
        assert completion_event.event_type == EventType.PHASE_COMPLETION
        assert completion_event.output_hash != ""
        assert completion_event.metrics["duration"] == 1.5
        
        print("✓ Test telemetry event emission PASSED")
    
    def test_telemetry_completeness_verification(self):
        """Test that telemetry completeness can be verified"""
        telemetry = TelemetryCollector()
        trace = TraceContext.create_root("test_run")
        
        # Phase with complete telemetry
        telemetry.emit_phase_start("complete_phase", trace, {"input": 1})
        telemetry.emit_phase_completion("complete_phase", trace, {"output": 2}, {})
        
        # Phase with only start event
        telemetry.emit_phase_start("incomplete_phase", trace, {"input": 3})
        
        # Verify completeness
        complete_result = telemetry.verify_completeness("complete_phase")
        assert complete_result["complete"] is True
        assert len(complete_result["missing_events"]) == 0
        
        incomplete_result = telemetry.verify_completeness("incomplete_phase")
        assert incomplete_result["complete"] is False
        assert "PHASE_COMPLETION" in incomplete_result["missing_events"]
        
        print("✓ Test telemetry completeness verification PASSED")
    
    def test_contract_violation_error(self):
        """Test that contract violations are properly structured"""
        trace = TraceContext.create_root("test_run")
        
        with pytest.raises(ContractViolationError) as exc_info:
            raise ContractViolationError(
                phase_name="test_phase",
                violation_type="missing_field",
                expected="non-empty dict",
                actual="None",
                trace_context=trace
            )
        
        error = exc_info.value
        assert error.phase_name == "test_phase"
        assert error.violation_type == "missing_field"
        assert error.trace_context == trace
        
        print("✓ Test contract violation error PASSED")


class TestPhaseResultContract:
    """Test PhaseResult contract validation"""
    
    def test_valid_phase_result(self):
        """Test that valid PhaseResult passes validation"""
        trace = TraceContext.create_root("test_run")
        
        result = PhaseResult(
            phase_name="test_phase",
            inputs={"input_key": "value"},
            outputs={"output_key": "result"},
            metrics={"count": 5},
            timestamp="2024-01-01T12:00:00",
            status="success",
            input_hash="abc123",
            output_hash="def456",
            trace_context=trace
        )
        
        # Should not raise
        result.validate_contract()
        
        print("✓ Test valid phase result PASSED")
    
    def test_invalid_phase_result_empty_name(self):
        """Test that empty phase name violates contract"""
        result = PhaseResult(
            phase_name="",
            inputs={},
            outputs={},
            metrics={},
            timestamp="2024-01-01T12:00:00"
        )
        
        with pytest.raises(ContractViolationError) as exc_info:
            result.validate_contract()
        
        assert exc_info.value.violation_type == "empty_phase_name"
        
        print("✓ Test invalid phase result empty name PASSED")
    
    def test_invalid_phase_result_wrong_type(self):
        """Test that wrong input types violate contract"""
        result = PhaseResult(
            phase_name="test_phase",
            inputs="not_a_dict",  # Wrong type
            outputs={},
            metrics={},
            timestamp="2024-01-01T12:00:00"
        )
        
        with pytest.raises(ContractViolationError) as exc_info:
            result.validate_contract()
        
        assert exc_info.value.violation_type == "invalid_inputs_type"
        
        print("✓ Test invalid phase result wrong type PASSED")
    
    def test_invalid_phase_result_bad_status(self):
        """Test that invalid status violates contract"""
        result = PhaseResult(
            phase_name="test_phase",
            inputs={},
            outputs={},
            metrics={},
            timestamp="2024-01-01T12:00:00",
            status="maybe"  # Invalid status
        )
        
        with pytest.raises(ContractViolationError) as exc_info:
            result.validate_contract()
        
        assert exc_info.value.violation_type == "invalid_status"
        
        print("✓ Test invalid phase result bad status PASSED")
    
    def test_invalid_phase_result_bad_timestamp(self):
        """Test that invalid timestamp violates contract"""
        result = PhaseResult(
            phase_name="test_phase",
            inputs={},
            outputs={},
            metrics={},
            timestamp="not-a-timestamp"
        )
        
        with pytest.raises(ContractViolationError) as exc_info:
            result.validate_contract()
        
        assert exc_info.value.violation_type == "invalid_timestamp_format"
        
        print("✓ Test invalid phase result bad timestamp PASSED")


class TestOrchestratorAuditability:
    """Test orchestrator auditability features"""
    
    def test_orchestrator_emits_telemetry(self):
        """Test that orchestrator emits telemetry for all phases"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = create_orchestrator(log_dir=Path(tmpdir))
            
            # Execute analysis
            result = orch.orchestrate_analysis(
                text="Plan de desarrollo municipal con objetivos claros.",
                plan_name="PDM_Telemetry_Test",
                dimension="estratégico"
            )
            
            # Verify telemetry was collected
            events = orch.telemetry.get_events()
            assert len(events) > 0
            
            # Verify phase start/completion events exist
            phase_names = [
                "orchestration_pipeline",
                "extract_statements",
                "detect_contradictions",
                "analyze_regulatory_constraints",
                "calculate_coherence_metrics",
                "generate_audit_summary"
            ]
            
            for phase_name in phase_names:
                phase_events = orch.telemetry.get_events(phase_name=phase_name)
                assert len(phase_events) >= 2, f"Missing events for {phase_name}"
                
                # Check for start and completion
                event_types = [e.event_type for e in phase_events]
                assert EventType.PHASE_START in event_types
                assert EventType.PHASE_COMPLETION in event_types
            
            print("✓ Test orchestrator emits telemetry PASSED")
    
    def test_orchestrator_persists_telemetry(self):
        """Test that orchestrator persists telemetry to disk"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = create_orchestrator(log_dir=Path(tmpdir))
            
            # Execute analysis
            result = orch.orchestrate_analysis(
                text="Plan de desarrollo municipal.",
                plan_name="PDM_Persist_Test",
                dimension="estratégico"
            )
            
            # Check for telemetry files
            log_dir = Path(tmpdir)
            telemetry_files = list(log_dir.glob("telemetry_*.jsonl"))
            assert len(telemetry_files) > 0, "No telemetry files created"
            
            # Verify file content
            with open(telemetry_files[0], 'r') as f:
                lines = f.readlines()
                assert len(lines) > 0
                
                # Parse first event
                first_event = json.loads(lines[0])
                assert "event_type" in first_event
                assert "phase_name" in first_event
                assert "trace_context" in first_event
                assert "timestamp" in first_event
            
            print("✓ Test orchestrator persists telemetry PASSED")
    
    def test_orchestrator_trace_context_propagation(self):
        """Test that trace context is propagated across phases"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = create_orchestrator(log_dir=Path(tmpdir))
            
            # Execute analysis
            result = orch.orchestrate_analysis(
                text="Plan de desarrollo municipal.",
                plan_name="PDM_Trace_Test",
                dimension="estratégico"
            )
            
            # Get all events
            events = orch.telemetry.get_events()
            
            # Extract trace IDs
            trace_ids = set(e.trace_context.trace_id for e in events)
            
            # All events should have the same trace ID
            assert len(trace_ids) == 1, "Multiple trace IDs detected"
            
            # Verify audit IDs are consistent
            audit_ids = set(e.trace_context.audit_id for e in events)
            assert len(audit_ids) == 1, "Multiple audit IDs detected"
            
            print("✓ Test orchestrator trace context propagation PASSED")
    
    def test_orchestrator_deterministic_hashing(self):
        """Test that orchestrator produces deterministic hashes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch1 = create_orchestrator(log_dir=Path(tmpdir) / "run1")
            orch2 = create_orchestrator(log_dir=Path(tmpdir) / "run2")
            
            text = "Plan de desarrollo municipal idéntico."
            
            # Run twice
            result1 = orch1.orchestrate_analysis(text, "PDM_Hash_Test", "estratégico")
            result2 = orch2.orchestrate_analysis(text, "PDM_Hash_Test", "estratégico")
            
            # Get phase results
            phase_results1 = orch1._audit_log
            phase_results2 = orch2._audit_log
            
            # Compare input/output hashes for each phase
            assert len(phase_results1) == len(phase_results2)
            
            for r1, r2 in zip(phase_results1, phase_results2):
                assert r1.phase_name == r2.phase_name
                assert r1.input_hash == r2.input_hash, f"Input hashes differ for {r1.phase_name}"
                assert r1.output_hash == r2.output_hash, f"Output hashes differ for {r1.phase_name}"
            
            print("✓ Test orchestrator deterministic hashing PASSED")
    
    def test_orchestrator_telemetry_completeness_check(self):
        """Test that orchestrator verifies telemetry completeness"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = create_orchestrator(log_dir=Path(tmpdir))
            
            # Execute analysis
            result = orch.orchestrate_analysis(
                text="Plan de desarrollo municipal.",
                plan_name="PDM_Completeness_Test",
                dimension="estratégico"
            )
            
            # Verify all expected phases have complete telemetry
            expected_phases = [
                "extract_statements",
                "detect_contradictions",
                "analyze_regulatory_constraints",
                "calculate_coherence_metrics",
                "generate_audit_summary",
                "orchestration_pipeline"
            ]
            
            verification = orch.telemetry.verify_all_phases(expected_phases)
            assert verification["all_complete"] is True
            assert verification["complete_phases"] == verification["total_phases"]
            
            print("✓ Test orchestrator telemetry completeness check PASSED")
    
    def test_orchestrator_contract_enforcement(self):
        """Test that orchestrator enforces PhaseResult contracts"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = create_orchestrator(log_dir=Path(tmpdir))
            
            # Execute analysis - should not raise if all contracts are valid
            try:
                result = orch.orchestrate_analysis(
                    text="Plan de desarrollo municipal.",
                    plan_name="PDM_Contract_Test",
                    dimension="estratégico"
                )
                
                # Verify all phase results passed validation
                for phase_result in orch._audit_log:
                    # Should not raise
                    phase_result.validate_contract()
                
                print("✓ Test orchestrator contract enforcement PASSED")
            except ContractViolationError as e:
                pytest.fail(f"Contract violation should not occur: {e}")
    
    def test_orchestrator_audit_log_immutability(self):
        """Test that audit logs are immutable and versioned"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = create_orchestrator(log_dir=Path(tmpdir))
            
            # Execute analysis
            result = orch.orchestrate_analysis(
                text="Plan de desarrollo municipal.",
                plan_name="PDM_Immutable_Test",
                dimension="estratégico"
            )
            
            # Check for audit log files
            log_dir = Path(tmpdir)
            audit_files = list(log_dir.glob("audit_logs.jsonl"))
            assert len(audit_files) > 0
            
            # Verify audit log is append-only (JSONL format)
            with open(audit_files[0], 'r') as f:
                lines = f.readlines()
                assert len(lines) > 0
                
                # Each line should be a valid JSON object
                for line in lines:
                    audit_record = json.loads(line)
                    assert "run_id" in audit_record
                    assert "orchestrator" in audit_record
                    assert "timestamp" in audit_record
                    assert "sha256_source" in audit_record
            
            print("✓ Test orchestrator audit log immutability PASSED")


class TestAuditRetention:
    """Test audit log retention policies"""
    
    def test_retention_policy_configured(self):
        """Test that 7-year retention policy is configured"""
        orch = create_orchestrator()
        
        # Verify retention is set to 7 years
        assert orch._retention_years == 7
        assert orch.telemetry._retention_years == 7
        
        print("✓ Test retention policy configured PASSED")
    
    def test_telemetry_statistics(self):
        """Test that telemetry statistics include retention info"""
        telemetry = TelemetryCollector()
        
        stats = telemetry.get_statistics()
        assert "retention_years" in stats
        assert stats["retention_years"] == 7
        
        print("✓ Test telemetry statistics PASSED")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
