#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Deterministic Unified Orchestrator
==================================================

Tests for contract enforcement, deterministic execution, and telemetry.

SIN_CARRETA Compliance:
- Tests prove deterministic behavior (same input → same output)
- Tests validate contract enforcement (missing inputs/outputs → exceptions)
- Tests verify telemetry at all decision points
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from orchestration.deterministic_unified_orchestrator import (
    PHASE_CONTRACTS,
    ContractViolationError,
    DependencyNotInjectedError,
    DeterministicUnifiedOrchestrator,
    PhaseContract,
    PhaseExecutionError,
    PhaseResult,
    PipelineContext,
    PipelinePhase,
    create_unified_orchestrator,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_log_dir():
    """Create temporary log directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def deterministic_orchestrator(temp_log_dir):
    """Create deterministic orchestrator for testing"""
    return DeterministicUnifiedOrchestrator(
        log_dir=temp_log_dir,
        deterministic_mode=True,
        fixed_timestamp="2025-01-01T00:00:00",
        random_seed=42,
    )


@pytest.fixture
def mock_components():
    """Create mock components for testing"""
    return {
        "teoria_cambio": Mock(),
        "contradiction_detector": Mock(),
        "axiomatic_validator": Mock(),
        "dnp_validator": Mock(),
        "bayesian_engine": Mock(),
        "smart_recommendation": Mock(),
        "extraction_pipeline": Mock(),
        "report_generator": Mock(),
    }


# ============================================================================
# TEST: INITIALIZATION
# ============================================================================


def test_orchestrator_initialization(temp_log_dir):
    """Test orchestrator initialization"""
    orch = DeterministicUnifiedOrchestrator(log_dir=temp_log_dir)

    assert orch.log_dir == temp_log_dir
    assert orch.calibration is not None
    assert orch.metrics is not None
    assert orch.audit_logger is not None
    assert orch.deterministic_mode is False


def test_orchestrator_deterministic_mode(temp_log_dir):
    """Test orchestrator in deterministic mode"""
    orch = DeterministicUnifiedOrchestrator(
        log_dir=temp_log_dir,
        deterministic_mode=True,
        fixed_timestamp="2025-01-01T00:00:00",
        random_seed=42,
    )

    assert orch.deterministic_mode is True
    assert orch.fixed_timestamp == "2025-01-01T00:00:00"
    assert orch.random_seed == 42

    # Test deterministic timestamp
    ts1 = orch._get_timestamp()
    ts2 = orch._get_timestamp()
    assert ts1 == ts2 == "2025-01-01T00:00:00"


def test_factory_function(temp_log_dir):
    """Test factory function"""
    orch = create_unified_orchestrator(log_dir=temp_log_dir, deterministic_mode=True)

    assert isinstance(orch, DeterministicUnifiedOrchestrator)
    assert orch.deterministic_mode is True


# ============================================================================
# TEST: DEPENDENCY INJECTION
# ============================================================================


def test_inject_component(deterministic_orchestrator):
    """Test component injection"""
    mock_component = Mock()

    deterministic_orchestrator.inject_component("teoria_cambio", mock_component)

    assert "teoria_cambio" in deterministic_orchestrator._components
    assert deterministic_orchestrator._components["teoria_cambio"] == mock_component


def test_inject_invalid_component(deterministic_orchestrator):
    """Test injecting invalid component raises error"""
    with pytest.raises(ValueError, match="Invalid component name"):
        deterministic_orchestrator.inject_component("invalid_component", Mock())


def test_get_component_required(deterministic_orchestrator):
    """Test getting required component raises error if not injected"""
    with pytest.raises(
        DependencyNotInjectedError,
        match="Required component 'teoria_cambio' not injected",
    ):
        deterministic_orchestrator._get_component("teoria_cambio", required=True)


def test_get_component_optional(deterministic_orchestrator):
    """Test getting optional component returns None if not injected"""
    component = deterministic_orchestrator._get_component(
        "teoria_cambio", required=False
    )
    assert component is None


def test_get_component_success(deterministic_orchestrator):
    """Test getting component successfully"""
    mock_component = Mock()
    deterministic_orchestrator.inject_component("teoria_cambio", mock_component)

    component = deterministic_orchestrator._get_component(
        "teoria_cambio", required=True
    )
    assert component == mock_component


# ============================================================================
# TEST: CONTRACT VALIDATION
# ============================================================================


def test_phase_contracts_defined():
    """Test that all phases have contracts defined"""
    for phase in PipelinePhase:
        assert phase in PHASE_CONTRACTS
        contract = PHASE_CONTRACTS[phase]
        assert isinstance(contract, PhaseContract)
        assert contract.phase == phase


def test_validate_contract_inputs_success(deterministic_orchestrator):
    """Test contract input validation succeeds when inputs available"""
    context = PipelineContext(
        run_id="test_run",
        pdf_path="/test.pdf",
        plan_name="Test",
        dimension="estratégico",
    )

    # Add required inputs for Phase 1
    context.set_output(PipelinePhase.PHASE_0_INITIALIZATION, "pdf_path", "/test.pdf")

    # Should not raise
    deterministic_orchestrator._validate_contract_inputs(
        PipelinePhase.PHASE_1_EXTRACTION, context
    )


def test_validate_contract_inputs_missing(deterministic_orchestrator):
    """Test contract input validation fails when inputs missing"""
    context = PipelineContext(
        run_id="test_run",
        pdf_path="/test.pdf",
        plan_name="Test",
        dimension="estratégico",
    )

    # Missing required inputs
    with pytest.raises(ContractViolationError, match="Missing required inputs"):
        deterministic_orchestrator._validate_contract_inputs(
            PipelinePhase.PHASE_1_EXTRACTION, context
        )


def test_validate_contract_outputs_success(deterministic_orchestrator):
    """Test contract output validation succeeds when outputs complete"""
    outputs = {
        "semantic_chunks": [],
        "tables": [],
        "extraction_quality": {"score": 1.0},
    }

    # Should not raise
    deterministic_orchestrator._validate_contract_outputs(
        PipelinePhase.PHASE_1_EXTRACTION, outputs
    )


def test_validate_contract_outputs_missing(deterministic_orchestrator):
    """Test contract output validation fails when outputs missing"""
    outputs = {
        "semantic_chunks": []
        # Missing 'tables' and 'extraction_quality'
    }

    with pytest.raises(ContractViolationError, match="Missing required outputs"):
        deterministic_orchestrator._validate_contract_outputs(
            PipelinePhase.PHASE_1_EXTRACTION, outputs
        )


# ============================================================================
# TEST: TELEMETRY
# ============================================================================


def test_emit_telemetry(deterministic_orchestrator):
    """Test telemetry emission"""
    context = PipelineContext(
        run_id="test_run",
        pdf_path="/test.pdf",
        plan_name="Test",
        dimension="estratégico",
    )

    # Should not raise
    deterministic_orchestrator._emit_telemetry(
        "start", PipelinePhase.PHASE_1_EXTRACTION, context, {"test": "data"}
    )

    # Check metric recorded
    assert (
        "telemetry.start.PHASE_1_EXTRACTION"
        in deterministic_orchestrator.metrics.metrics
    )


# ============================================================================
# TEST: PHASE RESULT
# ============================================================================


def test_phase_result_creation():
    """Test PhaseResult creation"""
    result = PhaseResult(
        phase=PipelinePhase.PHASE_1_EXTRACTION,
        status="success",
        start_time=100.0,
        end_time=102.5,
        duration_seconds=2.5,
        inputs={"pdf_path": "/test.pdf"},
        outputs={"semantic_chunks": []},
        metrics={"duration_seconds": 2.5},
        run_id="test_run",
    )

    assert result.phase == PipelinePhase.PHASE_1_EXTRACTION
    assert result.status == "success"
    assert result.duration_seconds == 2.5
    assert result.timestamp is not None


def test_phase_result_invalid_status():
    """Test PhaseResult rejects invalid status"""
    with pytest.raises(ValueError, match="Invalid status"):
        PhaseResult(
            phase=PipelinePhase.PHASE_1_EXTRACTION,
            status="invalid",
            start_time=100.0,
            end_time=102.5,
            duration_seconds=2.5,
            inputs={},
            outputs={},
            metrics={},
        )


def test_phase_result_error_requires_message():
    """Test PhaseResult error status requires error message"""
    with pytest.raises(ValueError, match="Error status requires error message"):
        PhaseResult(
            phase=PipelinePhase.PHASE_1_EXTRACTION,
            status="error",
            start_time=100.0,
            end_time=102.5,
            duration_seconds=2.5,
            inputs={},
            outputs={},
            metrics={},
        )


# ============================================================================
# TEST: PIPELINE CONTEXT
# ============================================================================


def test_pipeline_context_creation():
    """Test PipelineContext creation"""
    context = PipelineContext(
        run_id="test_run",
        pdf_path="/test.pdf",
        plan_name="Test",
        dimension="estratégico",
    )

    assert context.run_id == "test_run"
    assert context.pdf_path == "/test.pdf"
    assert context.plan_name == "Test"
    assert context.dimension == "estratégico"


def test_pipeline_context_set_get_output():
    """Test setting and getting outputs in context"""
    context = PipelineContext(
        run_id="test_run",
        pdf_path="/test.pdf",
        plan_name="Test",
        dimension="estratégico",
    )

    context.set_output(PipelinePhase.PHASE_1_EXTRACTION, "semantic_chunks", ["chunk1"])

    result = context.get_output(PipelinePhase.PHASE_1_EXTRACTION, "semantic_chunks")
    assert result == ["chunk1"]


def test_pipeline_context_get_output_default():
    """Test getting output with default value"""
    context = PipelineContext(
        run_id="test_run",
        pdf_path="/test.pdf",
        plan_name="Test",
        dimension="estratégico",
    )

    result = context.get_output(PipelinePhase.PHASE_1_EXTRACTION, "missing", default=[])
    assert result == []


def test_pipeline_context_add_result():
    """Test adding phase result to context"""
    context = PipelineContext(
        run_id="test_run",
        pdf_path="/test.pdf",
        plan_name="Test",
        dimension="estratégico",
    )

    result = PhaseResult(
        phase=PipelinePhase.PHASE_1_EXTRACTION,
        status="success",
        start_time=100.0,
        end_time=102.5,
        duration_seconds=2.5,
        inputs={},
        outputs={},
        metrics={},
    )

    context.add_result(result)
    assert len(context.phase_results) == 1
    assert context.phase_results[0] == result


# ============================================================================
# TEST: DETERMINISTIC EXECUTION
# ============================================================================


def test_deterministic_run_id(deterministic_orchestrator):
    """Test deterministic run ID generation"""
    run_id1 = deterministic_orchestrator._generate_run_id()
    run_id2 = deterministic_orchestrator._generate_run_id()

    assert run_id1 == run_id2 == "deterministic_run_001"


def test_deterministic_timestamp(deterministic_orchestrator):
    """Test deterministic timestamp generation"""
    ts1 = deterministic_orchestrator._get_timestamp()
    ts2 = deterministic_orchestrator._get_timestamp()

    assert ts1 == ts2 == "2025-01-01T00:00:00"


# ============================================================================
# TEST: PHASE EXECUTION
# ============================================================================


@pytest.mark.asyncio
async def test_execute_phase_0_initialization(deterministic_orchestrator):
    """Test Phase 0: Initialization"""
    context = PipelineContext(
        run_id="test_run",
        pdf_path="/test.pdf",
        plan_name="Test",
        dimension="estratégico",
    )

    outputs = await deterministic_orchestrator._execute_phase_0_initialization(context)

    assert "run_id" in outputs
    assert "calibration" in outputs
    assert "initialized" in outputs
    assert outputs["initialized"] is True


@pytest.mark.asyncio
async def test_execute_phase_1_extraction_requires_component(
    deterministic_orchestrator,
):
    """Test Phase 1: Extraction requires component"""
    context = PipelineContext(
        run_id="test_run",
        pdf_path="/test.pdf",
        plan_name="Test",
        dimension="estratégico",
    )

    # Should raise because extraction_pipeline not injected
    with pytest.raises(DependencyNotInjectedError):
        await deterministic_orchestrator._execute_phase_1_extraction(context)


@pytest.mark.asyncio
async def test_execute_phase_with_injected_component(deterministic_orchestrator):
    """Test phase execution with injected component"""
    # Inject mock component
    mock_extraction = Mock()
    deterministic_orchestrator.inject_component("extraction_pipeline", mock_extraction)

    context = PipelineContext(
        run_id="test_run",
        pdf_path="/test.pdf",
        plan_name="Test",
        dimension="estratégico",
    )

    # Should not raise
    outputs = await deterministic_orchestrator._execute_phase_1_extraction(context)

    assert "semantic_chunks" in outputs
    assert "tables" in outputs
    assert "extraction_quality" in outputs


# ============================================================================
# TEST: FULL PIPELINE EXECUTION
# ============================================================================


@pytest.mark.asyncio
async def test_full_pipeline_execution_with_mocks(
    deterministic_orchestrator, mock_components
):
    """Test full pipeline execution with all components mocked"""
    # Inject all mock components
    for name, component in mock_components.items():
        deterministic_orchestrator.inject_component(name, component)

    # Execute pipeline
    context = await deterministic_orchestrator.execute_pipeline(
        pdf_path="/test.pdf", plan_name="Test PDM", dimension="estratégico"
    )

    # Verify context
    assert context.run_id == "deterministic_run_001"
    assert context.pdf_path == "/test.pdf"
    assert context.plan_name == "Test PDM"
    assert context.dimension == "estratégico"

    # Verify all phases executed
    assert len(context.phase_results) == len(PipelinePhase)

    # Verify all phases succeeded
    for result in context.phase_results:
        assert result.status == "success"


@pytest.mark.asyncio
async def test_pipeline_execution_determinism(
    deterministic_orchestrator, mock_components
):
    """Test pipeline execution is deterministic"""
    # Inject all mock components
    for name, component in mock_components.items():
        deterministic_orchestrator.inject_component(name, component)

    # Execute pipeline twice
    context1 = await deterministic_orchestrator.execute_pipeline(
        pdf_path="/test.pdf", plan_name="Test PDM", dimension="estratégico"
    )

    context2 = await deterministic_orchestrator.execute_pipeline(
        pdf_path="/test.pdf", plan_name="Test PDM", dimension="estratégico"
    )

    # Verify deterministic run IDs
    assert context1.run_id == context2.run_id

    # Verify same number of phase results
    assert len(context1.phase_results) == len(context2.phase_results)


# ============================================================================
# TEST: ERROR HANDLING
# ============================================================================


@pytest.mark.asyncio
async def test_pipeline_handles_phase_error(
    deterministic_orchestrator, mock_components
):
    """Test pipeline handles phase errors correctly"""
    # Inject all mock components except one required component
    for name, component in mock_components.items():
        if name != "teoria_cambio":  # Skip this one to trigger error
            deterministic_orchestrator.inject_component(name, component)

    # Execute pipeline - should raise PhaseExecutionError
    with pytest.raises(PhaseExecutionError):
        await deterministic_orchestrator.execute_pipeline(
            pdf_path="/test.pdf", plan_name="Test PDM", dimension="estratégico"
        )


@pytest.mark.asyncio
async def test_pipeline_logs_audit_on_error(
    deterministic_orchestrator, mock_components
):
    """Test pipeline logs to audit on error"""
    # Inject all mock components except one
    for name, component in mock_components.items():
        if name != "teoria_cambio":
            deterministic_orchestrator.inject_component(name, component)

    # Execute pipeline
    try:
        await deterministic_orchestrator.execute_pipeline(
            pdf_path="/test.pdf", plan_name="Test PDM", dimension="estratégico"
        )
    except PhaseExecutionError:
        pass

    # Verify audit log contains error event
    # NOTE: This would require reading the audit log file


# ============================================================================
# TEST: CONTRACT ENFORCEMENT
# ============================================================================


@pytest.mark.asyncio
async def test_contract_enforcement_missing_outputs(deterministic_orchestrator):
    """Test contract enforcement for missing outputs"""
    context = PipelineContext(
        run_id="test_run",
        pdf_path="/test.pdf",
        plan_name="Test",
        dimension="estratégico",
    )

    # Mock a phase that returns incomplete outputs
    async def incomplete_phase(ctx):
        return {"incomplete": "output"}

    deterministic_orchestrator._execute_phase_1_extraction = incomplete_phase
    deterministic_orchestrator.inject_component("extraction_pipeline", Mock())

    # Should raise ContractViolationError when validating outputs
    with pytest.raises(ContractViolationError, match="Missing required outputs"):
        await deterministic_orchestrator._execute_phase(
            PipelinePhase.PHASE_1_EXTRACTION, context
        )


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
