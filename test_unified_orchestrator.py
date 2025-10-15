#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Tests for Unified Orchestrator
==========================================
Tests covering PDF→extraction→graph→bayesian→validation→scoring→report
with deterministic fixture data and quantitative verification.
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock
from dataclasses import dataclass

import networkx as nx
import numpy as np

from orchestration.unified_orchestrator import (
    UnifiedOrchestrator,
    PipelineStage,
    UnifiedResult,
    PriorSnapshot
)
from orchestration.pdm_orchestrator import (
    ExtractionResult,
    MechanismResult,
    ValidationResult,
    QualityScore
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
        queue_size: int = 10
        max_inflight_jobs: int = 3
        worker_timeout_secs: int = 300
        min_quality_threshold: float = 0.5
    
    return MockConfig()


@pytest.fixture
def orchestrator(config):
    """Create orchestrator instance"""
    return UnifiedOrchestrator(config)


@pytest.fixture
def mock_extraction_pipeline():
    """Mock extraction pipeline"""
    pipeline = AsyncMock()
    
    async def extract_complete(pdf_path):
        return ExtractionResult(
            semantic_chunks=[
                {'text': 'Chunk 1', 'id': 'chunk_1', 'dimension': 'ESTRATEGICO'},
                {'text': 'Chunk 2', 'id': 'chunk_2', 'dimension': 'DIAGNOSTICO'}
            ],
            tables=[
                {'title': 'Table 1', 'headers': ['A', 'B'], 'rows': [[1, 2]]}
            ],
            extraction_quality={'score': 0.8}
        )
    
    pipeline.extract_complete = extract_complete
    return pipeline


@pytest.fixture
def mock_causal_builder():
    """Mock causal builder"""
    builder = AsyncMock()
    
    async def build_graph(chunks, tables):
        graph = nx.DiGraph()
        graph.add_edge('producto_1', 'resultado_1', weight=0.8)
        graph.add_edge('resultado_1', 'impacto_1', weight=0.7)
        return graph
    
    builder.build_graph = build_graph
    return builder


@pytest.fixture
def mock_bayesian_engine():
    """Mock Bayesian engine"""
    engine = AsyncMock()
    
    async def infer_all_mechanisms(graph, chunks):
        return [
            MechanismResult(
                type='tecnico',
                necessity_test={'passed': True, 'missing': []},
                posterior_mean=0.75
            ),
            MechanismResult(
                type='administrativo',
                necessity_test={'passed': False, 'missing': ['timeline']},
                posterior_mean=0.45
            )
        ]
    
    engine.infer_all_mechanisms = infer_all_mechanisms
    return engine


@pytest.fixture
def mock_validator():
    """Mock validator"""
    validator = Mock()
    
    def validate_complete(graph, chunks, tables):
        from validators.axiomatic_validator import AxiomaticValidationResult
        result = AxiomaticValidationResult()
        result.is_valid = True
        result.structural_valid = True
        result.contradiction_density = 0.02
        result.regulatory_score = 75.0
        return result
    
    validator.validate_complete = validate_complete
    return validator


# ============================================================================
# TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_unified_pipeline_execution(orchestrator, mock_extraction_pipeline, 
                                         mock_causal_builder, mock_bayesian_engine,
                                         mock_validator):
    """Test complete pipeline execution"""
    # Inject components
    orchestrator.inject_components(
        extraction_pipeline=mock_extraction_pipeline,
        causal_builder=mock_causal_builder,
        bayesian_engine=mock_bayesian_engine,
        validator=mock_validator
    )
    
    # Execute pipeline
    result = await orchestrator.execute_pipeline('/tmp/test.pdf')
    
    # Verify success
    assert result.success == True
    assert result.run_id.startswith('unified_')
    
    # Verify all stages completed
    stage_names = [m.stage.name for m in result.stage_metrics]
    assert 'STAGE_0_INGESTION' in stage_names
    assert 'STAGE_1_EXTRACTION' in stage_names
    assert 'STAGE_2_GRAPH_BUILD' in stage_names
    assert 'STAGE_3_BAYESIAN' in stage_names
    assert 'STAGE_5_VALIDATION' in stage_names
    assert 'STAGE_8_LEARNING' in stage_names
    
    # Verify extraction outputs
    assert len(result.semantic_chunks) == 2
    assert len(result.tables) == 1
    
    # Verify graph
    assert result.causal_graph is not None
    assert result.causal_graph.number_of_edges() > 0
    
    # Verify Bayesian outputs
    assert len(result.mechanism_results) == 2
    assert any(m.type == 'tecnico' for m in result.mechanism_results)
    
    # Verify validation
    assert result.validation_result is not None
    assert result.validation_result.is_valid == True


@pytest.mark.asyncio
async def test_prior_snapshot_immutability(orchestrator):
    """Test that prior snapshots are immutable during run"""
    # Create initial snapshot
    run_id = "test_run_1"
    snapshot1 = orchestrator._create_prior_snapshot(run_id)
    
    # Modify prior store
    orchestrator.prior_store.update_mechanism_prior(
        'tecnico', 5.0, 'test update'
    )
    
    # Create new snapshot
    run_id2 = "test_run_2"
    snapshot2 = orchestrator._create_prior_snapshot(run_id2)
    
    # Verify snapshots are different
    assert snapshot1.priors != snapshot2.priors
    assert snapshot1.run_id != snapshot2.run_id
    
    # Verify snapshot1 unchanged
    assert snapshot1.priors['tecnico'] != 5.0


@pytest.mark.asyncio
async def test_circular_dependency_resolution(orchestrator, mock_extraction_pipeline,
                                              mock_causal_builder, mock_bayesian_engine,
                                              mock_validator):
    """Test that circular validation→scoring→prior loop is broken"""
    orchestrator.inject_components(
        extraction_pipeline=mock_extraction_pipeline,
        causal_builder=mock_causal_builder,
        bayesian_engine=mock_bayesian_engine,
        validator=mock_validator
    )
    
    # Get initial priors
    initial_prior = orchestrator.prior_store.get_mechanism_prior('administrativo')
    initial_alpha = initial_prior.alpha
    
    # Execute pipeline (will detect failures and compute penalties)
    result = await orchestrator.execute_pipeline('/tmp/test.pdf')
    
    # Verify penalty factors were computed
    assert 'administrativo' in result.penalty_factors
    assert result.penalty_factors['administrativo'] < 1.0  # Penalty applied
    
    # Verify priors updated for NEXT run
    updated_prior = orchestrator.prior_store.get_mechanism_prior('administrativo')
    assert updated_prior.alpha < initial_alpha  # Decayed due to failure
    
    # Verify update count incremented
    assert updated_prior.update_count > initial_prior.update_count


@pytest.mark.asyncio
async def test_metrics_collection(orchestrator, mock_extraction_pipeline):
    """Test metrics collection at each stage"""
    orchestrator.inject_components(extraction_pipeline=mock_extraction_pipeline)
    
    result = await orchestrator.execute_pipeline('/tmp/test.pdf')
    
    # Verify stage metrics collected
    assert len(result.stage_metrics) > 0
    
    # Verify each metric has timing data
    for metric in result.stage_metrics:
        assert metric.duration_seconds > 0
        assert metric.start_time > 0
        assert metric.end_time > 0
        assert metric.end_time > metric.start_time
    
    # Verify bottleneck identification
    summary = orchestrator.get_metrics_summary()
    assert 'bottlenecks' in summary
    assert len(summary['bottlenecks']) > 0


@pytest.mark.asyncio
async def test_event_bus_integration(orchestrator, mock_extraction_pipeline):
    """Test event bus integration"""
    orchestrator.inject_components(extraction_pipeline=mock_extraction_pipeline)
    
    # Subscribe to events
    events_received = []
    
    async def event_handler(event):
        events_received.append(event.event_type)
    
    orchestrator.event_bus.subscribe('stage.extraction.complete', event_handler)
    orchestrator.event_bus.subscribe('stage.learning.complete', event_handler)
    
    # Execute pipeline
    await orchestrator.execute_pipeline('/tmp/test.pdf')
    
    # Verify events were published
    assert 'stage.extraction.complete' in events_received
    assert 'stage.learning.complete' in events_received


@pytest.mark.asyncio
async def test_bayesian_convergence_tracking(orchestrator, mock_extraction_pipeline,
                                             mock_causal_builder, mock_bayesian_engine):
    """Test Bayesian posterior convergence tracking"""
    orchestrator.inject_components(
        extraction_pipeline=mock_extraction_pipeline,
        causal_builder=mock_causal_builder,
        bayesian_engine=mock_bayesian_engine
    )
    
    result = await orchestrator.execute_pipeline('/tmp/test.pdf')
    
    # Verify mechanism results have posteriors
    for mech in result.mechanism_results:
        assert hasattr(mech, 'posterior_mean')
        assert 0.0 <= mech.posterior_mean <= 1.0


@pytest.mark.asyncio
async def test_scoring_consistency(orchestrator, mock_extraction_pipeline,
                                   mock_causal_builder, mock_bayesian_engine,
                                   mock_validator):
    """Test MICRO→MESO→MACRO scoring consistency"""
    # Create mock scorer
    mock_scorer = Mock()
    mock_scorer.calculate_all_levels = Mock(return_value={
        'micro': {f'P{i}-D{j}-Q{k}': 0.7 
                 for i in range(1, 3)  # Simplified
                 for j in range(1, 3) 
                 for k in range(1, 2)},
        'meso': {'C1': 0.7, 'C2': 0.7},
        'macro': 0.7
    })
    
    orchestrator.inject_components(
        extraction_pipeline=mock_extraction_pipeline,
        causal_builder=mock_causal_builder,
        bayesian_engine=mock_bayesian_engine,
        validator=mock_validator,
        scorer=mock_scorer
    )
    
    result = await orchestrator.execute_pipeline('/tmp/test.pdf')
    
    # Verify scoring levels present
    assert len(result.micro_scores) > 0
    assert len(result.meso_scores) > 0
    assert result.macro_score > 0.0
    
    # Verify scorer was called
    assert mock_scorer.calculate_all_levels.called


@pytest.mark.asyncio
async def test_harmonic_front_4_penalty_learning(orchestrator, mock_extraction_pipeline,
                                                 mock_causal_builder, mock_bayesian_engine):
    """Test Harmonic Front 4 penalty factor learning"""
    orchestrator.inject_components(
        extraction_pipeline=mock_extraction_pipeline,
        causal_builder=mock_causal_builder,
        bayesian_engine=mock_bayesian_engine
    )
    
    # Execute pipeline
    result = await orchestrator.execute_pipeline('/tmp/test.pdf')
    
    # Verify penalty factors computed
    assert len(result.penalty_factors) > 0
    
    # Verify failed mechanism types penalized
    assert 'administrativo' in result.penalty_factors
    penalty = result.penalty_factors['administrativo']
    assert 0.5 <= penalty < 1.0  # Penalty reduces prior


@pytest.mark.asyncio
async def test_deterministic_fixture_reproducibility(orchestrator, mock_extraction_pipeline,
                                                     mock_causal_builder, mock_bayesian_engine):
    """Test reproducibility with deterministic fixtures"""
    orchestrator.inject_components(
        extraction_pipeline=mock_extraction_pipeline,
        causal_builder=mock_causal_builder,
        bayesian_engine=mock_bayesian_engine
    )
    
    # Run 1
    result1 = await orchestrator.execute_pipeline('/tmp/test.pdf')
    
    # Run 2 (same inputs)
    result2 = await orchestrator.execute_pipeline('/tmp/test.pdf')
    
    # Verify structure is same (not exact values due to learning)
    assert len(result1.semantic_chunks) == len(result2.semantic_chunks)
    assert len(result1.mechanism_results) == len(result2.mechanism_results)


@pytest.mark.asyncio
async def test_pipeline_failure_handling(orchestrator):
    """Test pipeline handles failures gracefully"""
    # Inject failing extraction pipeline
    failing_pipeline = AsyncMock()
    failing_pipeline.extract_complete = AsyncMock(
        side_effect=Exception("Extraction failed")
    )
    
    orchestrator.inject_components(extraction_pipeline=failing_pipeline)
    
    # Execute pipeline
    result = await orchestrator.execute_pipeline('/tmp/test.pdf')
    
    # Verify failure recorded
    assert result.success == False
    assert result.total_duration > 0


# ============================================================================
# QUANTITATIVE VERIFICATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_async_profiling_bottleneck_identification(orchestrator, 
                                                         mock_extraction_pipeline):
    """Test async profiling identifies bottlenecks"""
    orchestrator.inject_components(extraction_pipeline=mock_extraction_pipeline)
    
    result = await orchestrator.execute_pipeline('/tmp/test.pdf')
    
    # Get bottlenecks
    bottlenecks = orchestrator.metrics.get_bottlenecks(top_n=3)
    
    # Verify bottlenecks identified
    assert len(bottlenecks) > 0
    
    # Verify format: (stage_name, duration)
    for stage_name, duration in bottlenecks:
        assert isinstance(stage_name, str)
        assert isinstance(duration, float)
        assert duration > 0


@pytest.mark.asyncio
async def test_metrics_collector_phase_timing(orchestrator, mock_extraction_pipeline):
    """Test MetricsCollector captures all phase transitions"""
    orchestrator.inject_components(extraction_pipeline=mock_extraction_pipeline)
    
    result = await orchestrator.execute_pipeline('/tmp/test.pdf')
    
    # Verify all expected stages have metrics
    expected_stages = [
        'STAGE_0_INGESTION',
        'STAGE_1_EXTRACTION',
        'STAGE_2_GRAPH_BUILD',
        'STAGE_3_BAYESIAN',
        'STAGE_5_VALIDATION',
        'STAGE_6_SCORING',
        'STAGE_7_REPORT',
        'STAGE_8_LEARNING'
    ]
    
    captured_stages = [m.stage.name for m in result.stage_metrics]
    
    for expected in expected_stages:
        assert expected in captured_stages, f"Missing {expected}"


def test_prior_snapshot_correctness():
    """Test prior snapshot captures correct values"""
    from orchestration.learning_loop import PriorHistoryStore
    
    store = PriorHistoryStore(Path('/tmp/test_priors.json'))
    
    # Set known priors
    store.update_mechanism_prior('tecnico', 3.5, 'test')
    store.update_mechanism_prior('administrativo', 2.5, 'test')
    
    # Save snapshot
    store.save_snapshot()
    
    # Verify snapshot captured values
    history = store.get_history()
    assert len(history) > 0
    
    latest = history[-1]
    assert 'tecnico' in latest['priors']
    assert latest['priors']['tecnico']['alpha'] == 3.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
