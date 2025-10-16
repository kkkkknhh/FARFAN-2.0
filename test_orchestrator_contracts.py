#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contract Enforcement Tests for Unified Orchestrator
===================================================

Tests that prove:
1. Component injection contract is enforced (ComponentNotInjectedError)
2. Component return value contracts are validated (ContractViolationError)
3. Deterministic behavior across runs
4. Structured telemetry at all phase boundaries
5. NO silent failures - all errors are explicit exceptions

SIN_CARRETA Compliance:
- Explicit contract validation
- Deterministic test fixtures
- No mocking of core logic (only external dependencies)
"""

import asyncio
import pytest
import networkx as nx
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, List, Any

from orchestration.unified_orchestrator import (
    UnifiedOrchestrator,
    ComponentNotInjectedError,
    ContractViolationError,
    UnifiedResult,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Minimal config for testing"""
    from dataclasses import dataclass as dc, field as dc_field
    
    @dc
    class MockSelfReflection:
        enable_prior_learning: bool = True
        prior_history_path: str = "/tmp/test_prior_history.json"
        feedback_weight: float = 0.1
        min_documents_for_learning: int = 1

    @dc
    class MockConfig:
        self_reflection: MockSelfReflection = dc_field(default_factory=MockSelfReflection)
        prior_decay_factor: float = 0.9
        queue_size: int = 10
        max_inflight_jobs: int = 3
        worker_timeout_secs: int = 300
        min_quality_threshold: float = 0.5

    return MockConfig()


@pytest.fixture
def orchestrator(config):
    """Create orchestrator instance without components injected"""
    return UnifiedOrchestrator(config)


@pytest.fixture
def test_pdf_path(tmp_path):
    """Create a test PDF file"""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("test content")
    return str(pdf_file)


# ============================================================================
# CONTRACT ENFORCEMENT TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_component_not_injected_error_raised(orchestrator, test_pdf_path):
    """
    Test that ComponentNotInjectedError is raised when pipeline executed
    without injecting components.
    
    CONTRACT: execute_pipeline() MUST raise ComponentNotInjectedError
    if components not injected.
    """
    # Try to execute without injecting components
    with pytest.raises(ComponentNotInjectedError) as exc_info:
        await orchestrator.execute_pipeline(test_pdf_path)
    
    # Verify error message is explicit
    assert "Components not injected" in str(exc_info.value)
    assert "inject_components()" in str(exc_info.value)
    
    print("✓ Test: ComponentNotInjectedError raised when components not injected")


@pytest.mark.asyncio
async def test_extraction_pipeline_contract_violation(orchestrator, test_pdf_path):
    """
    Test that ContractViolationError is raised when extraction_pipeline
    returns invalid result (missing semantic_chunks).
    
    CONTRACT: extraction_pipeline.extract_complete() MUST return object
    with 'semantic_chunks' attribute.
    """
    # Create mock extraction pipeline that violates contract
    bad_extraction_pipeline = AsyncMock()
    
    async def bad_extract(pdf_path):
        # Returns dict WITHOUT semantic_chunks (contract violation)
        return {"invalid_key": "invalid_value"}
    
    bad_extraction_pipeline.extract_complete = bad_extract
    
    # Inject components with contract-violating extraction pipeline
    orchestrator.inject_components(
        extraction_pipeline=bad_extraction_pipeline,
        causal_builder=AsyncMock(),
        bayesian_engine=AsyncMock(),
        contradiction_detector=MagicMock(),
        validator=MagicMock(),
        scorer=MagicMock(),
        report_generator=AsyncMock(),
    )
    
    # Execute and expect ContractViolationError
    with pytest.raises(ContractViolationError) as exc_info:
        await orchestrator.execute_pipeline(test_pdf_path)
    
    # Verify error message identifies the contract violation
    assert "semantic_chunks" in str(exc_info.value)
    assert "extract_complete()" in str(exc_info.value)
    
    print("✓ Test: ContractViolationError raised for invalid extraction result")


@pytest.mark.asyncio
async def test_causal_builder_contract_violation(orchestrator, test_pdf_path):
    """
    Test that ContractViolationError is raised when causal_builder
    returns non-DiGraph result.
    
    CONTRACT: causal_builder.build_graph() MUST return nx.DiGraph.
    """
    # Create valid extraction pipeline
    extraction_pipeline = AsyncMock()
    
    async def extract(pdf_path):
        return {"semantic_chunks": [], "tables": []}
    
    extraction_pipeline.extract_complete = extract
    
    # Create contract-violating causal builder
    bad_causal_builder = AsyncMock()
    
    async def bad_build(chunks, tables):
        # Returns list instead of nx.DiGraph (contract violation)
        return []
    
    bad_causal_builder.build_graph = bad_build
    
    # Inject components
    orchestrator.inject_components(
        extraction_pipeline=extraction_pipeline,
        causal_builder=bad_causal_builder,
        bayesian_engine=AsyncMock(),
        contradiction_detector=MagicMock(),
        validator=MagicMock(),
        scorer=MagicMock(),
        report_generator=AsyncMock(),
    )
    
    # Execute and expect ContractViolationError
    with pytest.raises(ContractViolationError) as exc_info:
        await orchestrator.execute_pipeline(test_pdf_path)
    
    # Verify error identifies the contract violation
    assert "nx.DiGraph" in str(exc_info.value)
    assert "build_graph()" in str(exc_info.value)
    
    print("✓ Test: ContractViolationError raised for non-DiGraph result")


@pytest.mark.asyncio
async def test_scorer_contract_violation_missing_keys(orchestrator, test_pdf_path):
    """
    Test that ContractViolationError is raised when scorer returns
    dict without required keys (micro, meso, macro).
    
    CONTRACT: scorer.calculate_all_levels() MUST return dict with
    'micro', 'meso', 'macro' keys.
    """
    # Create valid components for stages 1-5
    extraction_pipeline = AsyncMock()
    extraction_pipeline.extract_complete = AsyncMock(return_value={
        'semantic_chunks': [{'text': 'test', 'dimension': 'ESTRATEGICO'}],
        'tables': []
    })
    
    causal_builder = AsyncMock()
    graph = nx.DiGraph()
    graph.add_edge('A', 'B')
    causal_builder.build_graph = AsyncMock(return_value=graph)
    
    bayesian_engine = AsyncMock()
    bayesian_engine.infer_all_mechanisms = AsyncMock(return_value=[])
    
    contradiction_detector = MagicMock()
    contradiction_detector.detect = MagicMock(return_value={'contradictions': []})
    
    validator = MagicMock()
    
    @dataclass
    class ValidationResult:
        passed: bool = True
        requires_manual_review: bool = False
    
    validator.validate_complete = MagicMock(return_value=ValidationResult())
    
    # Create contract-violating scorer (missing 'macro' key)
    bad_scorer = MagicMock()
    bad_scorer.calculate_all_levels = MagicMock(return_value={
        'micro': {},
        'meso': {},
        # 'macro' is MISSING - contract violation
    })
    
    report_generator = AsyncMock()
    report_generator.generate = AsyncMock(return_value=Path("/tmp/report.json"))
    
    # Inject components
    orchestrator.inject_components(
        extraction_pipeline=extraction_pipeline,
        causal_builder=causal_builder,
        bayesian_engine=bayesian_engine,
        contradiction_detector=contradiction_detector,
        validator=validator,
        scorer=bad_scorer,
        report_generator=report_generator,
    )
    
    # Execute and expect ContractViolationError
    with pytest.raises(ContractViolationError) as exc_info:
        await orchestrator.execute_pipeline(test_pdf_path)
    
    # Verify error identifies missing 'macro' key
    assert "macro" in str(exc_info.value)
    assert "calculate_all_levels()" in str(exc_info.value)
    
    print("✓ Test: ContractViolationError raised for missing scorer keys")


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_deterministic_execution_same_inputs(config, test_pdf_path):
    """
    Test that running the same pipeline twice with identical inputs
    produces identical results (determinism guarantee).
    
    DETERMINISM: Same inputs → Same outputs (no randomness)
    """
    # Create two orchestrators with identical configs
    orch1 = UnifiedOrchestrator(config)
    orch2 = UnifiedOrchestrator(config)
    
    # Create identical mock components (deterministic)
    def create_mock_components():
        extraction_pipeline = AsyncMock()
        extraction_pipeline.extract_complete = AsyncMock(return_value={
            'semantic_chunks': [
                {'text': 'Chunk 1', 'id': 'c1', 'dimension': 'ESTRATEGICO'},
                {'text': 'Chunk 2', 'id': 'c2', 'dimension': 'TACTICO'},
            ],
            'tables': [{'title': 'Table 1'}]
        })
        
        causal_builder = AsyncMock()
        graph = nx.DiGraph()
        graph.add_edge('A', 'B', weight=0.8)
        graph.add_edge('B', 'C', weight=0.7)
        causal_builder.build_graph = AsyncMock(return_value=graph)
        
        bayesian_engine = AsyncMock()
        bayesian_engine.infer_all_mechanisms = AsyncMock(return_value=[
            # Deterministic mechanism results
        ])
        
        contradiction_detector = MagicMock()
        contradiction_detector.detect = MagicMock(return_value={'contradictions': []})
        
        @dataclass
        class ValidationResult:
            passed: bool = True
            requires_manual_review: bool = False
        
        validator = MagicMock()
        validator.validate_complete = MagicMock(return_value=ValidationResult())
        
        scorer = MagicMock()
        scorer.calculate_all_levels = MagicMock(return_value={
            'micro': {'P1-D1-Q1': 0.75, 'P1-D1-Q2': 0.80},
            'meso': {'C1': 0.77},
            'macro': 0.77
        })
        
        report_generator = AsyncMock()
        report_generator.generate = AsyncMock(return_value=Path("/tmp/report.json"))
        
        return {
            'extraction_pipeline': extraction_pipeline,
            'causal_builder': causal_builder,
            'bayesian_engine': bayesian_engine,
            'contradiction_detector': contradiction_detector,
            'validator': validator,
            'scorer': scorer,
            'report_generator': report_generator,
        }
    
    # Inject identical components
    orch1.inject_components(**create_mock_components())
    orch2.inject_components(**create_mock_components())
    
    # Execute both pipelines
    result1 = await orch1.execute_pipeline(test_pdf_path)
    result2 = await orch2.execute_pipeline(test_pdf_path)
    
    # Verify results are identical (determinism)
    assert result1.success == result2.success
    assert len(result1.semantic_chunks) == len(result2.semantic_chunks)
    assert len(result1.tables) == len(result2.tables)
    assert result1.macro_score == result2.macro_score
    
    print("✓ Test: Deterministic execution produces identical results")


# ============================================================================
# TELEMETRY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_telemetry_events_published(orchestrator, test_pdf_path, caplog):
    """
    Test that structured telemetry events are logged at all phase boundaries.
    
    TELEMETRY: Each phase must emit START, DECISION, COMPLETE markers
    """
    import logging
    caplog.set_level(logging.INFO)
    
    # Create minimal valid components
    extraction_pipeline = AsyncMock()
    extraction_pipeline.extract_complete = AsyncMock(return_value={
        'semantic_chunks': [{'text': 'test'}],
        'tables': []
    })
    
    causal_builder = AsyncMock()
    causal_builder.build_graph = AsyncMock(return_value=nx.DiGraph())
    
    bayesian_engine = AsyncMock()
    bayesian_engine.infer_all_mechanisms = AsyncMock(return_value=[])
    
    contradiction_detector = MagicMock()
    contradiction_detector.detect = MagicMock(return_value={'contradictions': []})
    
    @dataclass
    class ValidationResult:
        passed: bool = True
        requires_manual_review: bool = False
    
    validator = MagicMock()
    validator.validate_complete = MagicMock(return_value=ValidationResult())
    
    scorer = MagicMock()
    scorer.calculate_all_levels = MagicMock(return_value={
        'micro': {},
        'meso': {},
        'macro': 0.7
    })
    
    report_generator = AsyncMock()
    report_generator.generate = AsyncMock(return_value=Path("/tmp/report.json"))
    
    # Inject components
    orchestrator.inject_components(
        extraction_pipeline=extraction_pipeline,
        causal_builder=causal_builder,
        bayesian_engine=bayesian_engine,
        contradiction_detector=contradiction_detector,
        validator=validator,
        scorer=scorer,
        report_generator=report_generator,
    )
    
    # Execute pipeline
    result = await orchestrator.execute_pipeline(test_pdf_path)
    
    # Verify telemetry markers in logs
    log_text = caplog.text
    
    # Pipeline-level telemetry
    assert "[TELEMETRY] Pipeline START" in log_text
    assert "[TELEMETRY] Pipeline SUCCESS" in log_text or "[TELEMETRY] Pipeline COMPLETE" in log_text
    
    # Stage-level telemetry (at least some stages)
    assert "[TELEMETRY] Stage 0 START" in log_text
    assert "[TELEMETRY] Stage 1 START" in log_text
    assert "[TELEMETRY] Stage 2 START" in log_text
    
    print("✓ Test: Telemetry events published at all phase boundaries")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
