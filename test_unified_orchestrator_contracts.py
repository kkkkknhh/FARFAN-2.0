#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contract Enforcement Tests for Unified Orchestrator
===================================================

SIN_CARRETA Compliance Tests:
- Explicit contract validation
- Deterministic execution
- Structured exception handling
- Telemetry emission
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import networkx as nx
import pytest

from orchestration.unified_orchestrator import (
    ComponentNotInjectedError,
    ContractViolationError,
    StageExecutionError,
    UnifiedOrchestrator,
    UnifiedResult,
    ValidationError,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def config():
    """Mock configuration"""

    @dataclass
    class MockSelfReflection:
        enable_prior_learning: bool = True
        prior_history_path: str = "/tmp/test_prior_history.json"
        feedback_weight: float = 0.1
        min_documents_for_learning: int = 1

    @dataclass
    class MockConfig:
        self_reflection: MockSelfReflection = MockSelfReflection()
        prior_decay_factor: float = 0.9

    return MockConfig()


@pytest.fixture
def orchestrator(config, tmp_path):
    """Create orchestrator in deterministic mode"""
    return UnifiedOrchestrator(
        config,
        log_dir=tmp_path / "logs",
        enable_telemetry=True,
        deterministic_mode=True,
    )


# ============================================================================
# CONTRACT VALIDATION TESTS
# ============================================================================


def test_contract_validation_missing_method(orchestrator):
    """Test that component without required method raises ContractViolationError"""

    class BadComponent:
        """Component missing extract_complete method"""

        pass

    bad_component = BadComponent()

    with pytest.raises(ContractViolationError) as exc_info:
        orchestrator.inject_components(extraction_pipeline=bad_component)

    assert "ExtractionPipelineProtocol" in str(exc_info.value)
    assert "extract_complete" in str(exc_info.value)


def test_contract_validation_non_callable_method(orchestrator):
    """Test that component with non-callable method raises ContractViolationError"""

    class BadComponent:
        """Component with non-callable extract_complete"""

        extract_complete = "not_a_function"

    bad_component = BadComponent()

    with pytest.raises(ContractViolationError) as exc_info:
        orchestrator.inject_components(extraction_pipeline=bad_component)

    assert "not callable" in str(exc_info.value)


def test_contract_validation_valid_component(orchestrator):
    """Test that valid component passes contract validation"""

    class GoodComponent:
        async def extract_complete(self, pdf_path: str):
            return Mock(semantic_chunks=[], tables=[])

    good_component = GoodComponent()

    # Should not raise
    orchestrator.inject_components(extraction_pipeline=good_component)
    assert orchestrator.extraction_pipeline is not None


# ============================================================================
# DETERMINISTIC MODE TESTS
# ============================================================================


def test_deterministic_mode_enabled(config, tmp_path):
    """Test that deterministic mode sets up correctly"""
    orch = UnifiedOrchestrator(
        config, log_dir=tmp_path / "logs", deterministic_mode=True
    )

    assert orch.deterministic_mode is True

    # Run ID should be deterministic
    run_id_1 = orch._generate_run_id()
    run_id_2 = orch._generate_run_id()

    # IDs should have predictable pattern (not timestamp-based)
    assert "test" in run_id_1
    assert "test" in run_id_2


def test_deterministic_mode_disabled(config, tmp_path):
    """Test that normal mode uses timestamps"""
    orch = UnifiedOrchestrator(
        config, log_dir=tmp_path / "logs", deterministic_mode=False
    )

    assert orch.deterministic_mode is False

    run_id = orch._generate_run_id()

    # Should contain timestamp pattern
    assert "unified_" in run_id


# ============================================================================
# TELEMETRY TESTS
# ============================================================================


def test_telemetry_emission(orchestrator):
    """Test that telemetry events are emitted correctly"""

    orchestrator._emit_telemetry(
        "test.event", {"run_id": "test_123", "data": "test_value"}
    )

    # Check metrics were recorded
    assert "telemetry.test.event" in orchestrator.metrics._metrics


def test_telemetry_disabled(config, tmp_path):
    """Test that telemetry can be disabled"""
    orch = UnifiedOrchestrator(
        config, log_dir=tmp_path / "logs", enable_telemetry=False
    )

    orch._emit_telemetry("test.event", {"data": "test"})

    # Should not record when disabled
    assert len(orch.metrics._metrics) == 0


# ============================================================================
# AUDIT LOGGING TESTS
# ============================================================================


def test_audit_logger_initialized(orchestrator, tmp_path):
    """Test that audit logger is properly initialized"""

    assert orchestrator.audit_logger is not None
    assert orchestrator.log_dir == tmp_path / "logs"

    # Verify audit file can be created
    audit_file = orchestrator.log_dir / "unified_orchestrator_audit.jsonl"
    assert audit_file.parent.exists()


# ============================================================================
# EXCEPTION HANDLING TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_explicit_exception_on_stage_failure(orchestrator):
    """Test that stage failures raise explicit StageExecutionError"""

    # Mock component that raises exception
    class FailingComponent:
        async def extract_complete(self, pdf_path: str):
            raise RuntimeError("Simulated extraction failure")

    orchestrator.inject_components(extraction_pipeline=FailingComponent())

    with pytest.raises(StageExecutionError) as exc_info:
        await orchestrator.execute_pipeline("/fake/path.pdf")

    # Verify structured error context
    assert "Pipeline execution failed" in str(exc_info.value)
    assert exc_info.value.context is not None
    assert "run_id" in exc_info.value.context
    assert "pdf_path" in exc_info.value.context


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_component_injection_and_execution(orchestrator):
    """Test full component injection and pipeline execution"""

    # Create mock components that satisfy contracts
    class MockExtraction:
        async def extract_complete(self, pdf_path: str):
            @dataclass
            class ExtractionResult:
                semantic_chunks: list = None
                tables: list = None

                def __post_init__(self):
                    if self.semantic_chunks is None:
                        self.semantic_chunks = [{"text": "chunk1", "id": "1"}]
                    if self.tables is None:
                        self.tables = []

            return ExtractionResult()

    class MockCausalBuilder:
        async def build_graph(self, chunks, tables):
            graph = nx.DiGraph()
            graph.add_edge("A", "B")
            return graph

    class MockBayesian:
        async def infer_all_mechanisms(self, graph, chunks):
            @dataclass
            class MechanismResult:
                type: str = "test"
                necessity_test: dict = None
                posterior_mean: float = 0.7

                def __post_init__(self):
                    if self.necessity_test is None:
                        self.necessity_test = {"passed": True, "missing": []}

            return [MechanismResult()]

    class MockContradiction:
        def detect(self, text: str, plan_name: str, dimension: str):
            return {"contradictions": []}

    class MockValidator:
        def validate_complete(self, graph, chunks, tables):
            @dataclass
            class ValidationResult:
                passed: bool = True
                requires_manual_review: bool = False

            return ValidationResult()

    class MockScorer:
        def calculate_all_levels(
            self, graph, mechanism_results, validation_result, contradictions
        ):
            return {"micro": {"P1-D1-Q1": 0.7}, "meso": {"C1": 0.7}, "macro": 0.7}

    class MockReporter:
        async def generate(self, result, pdf_path, run_id):
            return Path("/tmp/report.json")

    # Inject all components
    orchestrator.inject_components(
        extraction_pipeline=MockExtraction(),
        causal_builder=MockCausalBuilder(),
        bayesian_engine=MockBayesian(),
        contradiction_detector=MockContradiction(),
        validator=MockValidator(),
        scorer=MockScorer(),
        report_generator=MockReporter(),
    )

    # Execute pipeline
    result = await orchestrator.execute_pipeline("/fake/path.pdf")

    # Verify result
    assert result.success is True
    assert result.macro_score == 0.7
    assert len(result.stage_metrics) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
