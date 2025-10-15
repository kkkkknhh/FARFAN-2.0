#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Pipeline Orchestrator - FARFAN 2.0
==========================================
Consolidates PDMOrchestrator, AnalyticalOrchestrator, and CDAFFramework
into single 9-stage phase-separated pipeline with circular dependency resolution.

Resolves:
- Overlapping responsibilities between 3 orchestrators
- Circular dependency in validation→scoring→prior updates
- Missing integration between Harmonic Front 4 penalty learning and AxiomaticValidator
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import pandas as pd

# Component imports
from choreography.event_bus import EventBus, PDMEvent
from choreography.evidence_stream import EvidenceStream, StreamingBayesianUpdater
from orchestration.learning_loop import AdaptiveLearningLoop, PriorHistoryStore

logger = logging.getLogger(__name__)


# ============================================================================
# PIPELINE STAGES - 9-Stage Unified Model
# ============================================================================

class PipelineStage(Enum):
    """9-stage unified pipeline covering all orchestrator functionality"""
    STAGE_0_INGESTION = auto()           # PDF loading
    STAGE_1_EXTRACTION = auto()          # SemanticChunk + tables
    STAGE_2_GRAPH_BUILD = auto()         # Causal DAG construction
    STAGE_3_BAYESIAN = auto()            # 3 AGUJAS inference
    STAGE_4_CONTRADICTION = auto()       # Contradiction detection
    STAGE_5_VALIDATION = auto()          # Axiomatic validation
    STAGE_6_SCORING = auto()             # MICRO→MESO→MACRO
    STAGE_7_REPORT = auto()              # Report generation
    STAGE_8_LEARNING = auto()            # Penalty factor learning


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class StageMetrics:
    """Metrics collected at each stage boundary"""
    stage: PipelineStage
    start_time: float
    end_time: float
    duration_seconds: float
    items_processed: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PriorSnapshot:
    """Immutable prior snapshot for circular dependency resolution"""
    timestamp: str
    run_id: str
    priors: Dict[str, float]
    source: str = "history_store"


@dataclass
class UnifiedResult:
    """Complete pipeline result"""
    run_id: str
    success: bool
    
    # Extraction outputs
    semantic_chunks: List[Any] = field(default_factory=list)
    tables: List[Any] = field(default_factory=list)
    
    # Graph outputs
    causal_graph: Optional[nx.DiGraph] = None
    
    # Bayesian outputs
    mechanism_results: List[Any] = field(default_factory=list)
    posteriors: Dict[str, Any] = field(default_factory=dict)
    
    # Validation outputs
    validation_result: Optional[Any] = None
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Scoring outputs
    micro_scores: Dict[str, float] = field(default_factory=dict)
    meso_scores: Dict[str, float] = field(default_factory=dict)
    macro_score: float = 0.0
    
    # Learning outputs
    penalty_factors: Dict[str, float] = field(default_factory=dict)
    
    # Metrics
    stage_metrics: List[StageMetrics] = field(default_factory=list)
    total_duration: float = 0.0
    
    # Report
    report_path: Optional[Path] = None


# ============================================================================
# METRICS COLLECTOR
# ============================================================================

class MetricsCollector:
    """Enhanced metrics collector with async profiling"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics: Dict[str, List[float]] = {}
        self.counters: Dict[str, int] = {}
        self.stage_metrics: List[StageMetrics] = []
        
    def record(self, metric_name: str, value: float) -> None:
        """Record a metric value"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
        
    def increment(self, counter_name: str) -> None:
        """Increment a counter"""
        self.counters[counter_name] = self.counters.get(counter_name, 0) + 1
        
    def add_stage_metric(self, stage_metric: StageMetrics) -> None:
        """Add stage-level metric"""
        self.stage_metrics.append(stage_metric)
        self.logger.info(
            f"Stage {stage_metric.stage.name} completed in "
            f"{stage_metric.duration_seconds:.2f}s "
            f"({stage_metric.items_processed} items)"
        )
    
    def get_bottlenecks(self, top_n: int = 3) -> List[tuple]:
        """Identify top N slowest stages"""
        sorted_stages = sorted(
            self.stage_metrics, 
            key=lambda x: x.duration_seconds, 
            reverse=True
        )
        return [(s.stage.name, s.duration_seconds) for s in sorted_stages[:top_n]]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete metrics summary"""
        return {
            'stage_metrics': [
                {
                    'stage': m.stage.name,
                    'duration': m.duration_seconds,
                    'items': m.items_processed,
                    'errors': len(m.errors)
                }
                for m in self.stage_metrics
            ],
            'bottlenecks': self.get_bottlenecks(),
            'counters': self.counters,
            'total_stages': len(self.stage_metrics)
        }


# ============================================================================
# UNIFIED ORCHESTRATOR
# ============================================================================

class UnifiedOrchestrator:
    """
    Unified 9-stage pipeline orchestrator.
    
    Consolidates:
    - PDMOrchestrator (Phase 0-IV)
    - AnalyticalOrchestrator (6 phases)
    - CDAFFramework (9 stages)
    
    Resolves circular dependencies via immutable prior snapshots.
    """
    
    def __init__(self, config: Any):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        
        # Event bus for inter-component communication
        self.event_bus = EventBus()
        
        # Metrics collection
        self.metrics = MetricsCollector()
        
        # Prior history with snapshot support
        self.learning_loop = AdaptiveLearningLoop(config)
        self.prior_store = self.learning_loop.prior_store
        
        # Component placeholders (dependency injection)
        self.extraction_pipeline = None
        self.causal_builder = None
        self.bayesian_engine = None
        self.contradiction_detector = None
        self.validator = None
        self.scorer = None
        self.report_generator = None
        
        self.logger.info("UnifiedOrchestrator initialized")
    
    def inject_components(
        self,
        extraction_pipeline=None,
        causal_builder=None,
        bayesian_engine=None,
        contradiction_detector=None,
        validator=None,
        scorer=None,
        report_generator=None
    ):
        """Dependency injection for all components"""
        self.extraction_pipeline = extraction_pipeline
        self.causal_builder = causal_builder
        self.bayesian_engine = bayesian_engine
        self.contradiction_detector = contradiction_detector
        self.validator = validator
        self.scorer = scorer
        self.report_generator = report_generator
        self.logger.info("Components injected")
    
    async def execute_pipeline(self, pdf_path: str) -> UnifiedResult:
        """
        Execute complete 9-stage pipeline with:
        - Immutable prior snapshots (breaks circular dependency)
        - Async profiling at each stage
        - Event bus integration
        - Comprehensive metrics
        """
        run_id = self._generate_run_id()
        start_time = time.time()
        
        result = UnifiedResult(run_id=run_id, success=False)
        
        try:
            # SNAPSHOT PRIORS (breaks circular dependency)
            prior_snapshot = self._create_prior_snapshot(run_id)
            
            # STAGE 0: PDF Ingestion
            pdf_data = await self._stage_0_ingestion(pdf_path, run_id)
            
            # STAGE 1: Extraction
            extraction_result = await self._stage_1_extraction(pdf_data, run_id)
            result.semantic_chunks = extraction_result['chunks']
            result.tables = extraction_result['tables']
            
            # STAGE 2: Graph Construction
            result.causal_graph = await self._stage_2_graph_build(
                result.semantic_chunks, result.tables, run_id
            )
            
            # STAGE 3: Bayesian Inference (uses snapshot priors)
            bayesian_result = await self._stage_3_bayesian(
                result.causal_graph, result.semantic_chunks, prior_snapshot, run_id
            )
            result.mechanism_results = bayesian_result['mechanisms']
            result.posteriors = bayesian_result['posteriors']
            
            # STAGE 4: Contradiction Detection
            result.contradictions = await self._stage_4_contradiction(
                result.causal_graph, result.semantic_chunks, run_id
            )
            
            # STAGE 5: Axiomatic Validation
            result.validation_result = await self._stage_5_validation(
                result.causal_graph, result.semantic_chunks, result.tables, run_id
            )
            
            # STAGE 6: Scoring (MICRO→MESO→MACRO)
            scoring_result = await self._stage_6_scoring(
                result.causal_graph,
                result.mechanism_results,
                result.validation_result,
                result.contradictions,
                run_id
            )
            result.micro_scores = scoring_result['micro']
            result.meso_scores = scoring_result['meso']
            result.macro_score = scoring_result['macro']
            
            # STAGE 7: Report Generation
            result.report_path = await self._stage_7_report(
                result, pdf_path, run_id
            )
            
            # STAGE 8: Learning Loop (penalty factors for NEXT run)
            result.penalty_factors = await self._stage_8_learning(
                result, run_id
            )
            
            result.success = True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            result.success = False
        
        finally:
            result.total_duration = time.time() - start_time
            result.stage_metrics = self.metrics.stage_metrics
            self.logger.info(
                f"Pipeline completed: success={result.success}, "
                f"duration={result.total_duration:.2f}s"
            )
        
        return result
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        return f"unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self) % 10000}"
    
    def _create_prior_snapshot(self, run_id: str) -> PriorSnapshot:
        """
        Create immutable prior snapshot.
        This breaks the circular dependency:
        - Current run uses THIS snapshot
        - Validation penalties update store for NEXT run
        """
        self.prior_store.save_snapshot()
        
        priors = {}
        for mech_type in ['administrativo', 'tecnico', 'financiero', 'politico', 'mixto']:
            prior = self.prior_store.get_mechanism_prior(mech_type)
            priors[mech_type] = prior.alpha
        
        snapshot = PriorSnapshot(
            timestamp=datetime.now().isoformat(),
            run_id=run_id,
            priors=priors
        )
        
        self.logger.info(f"Created prior snapshot for run {run_id}")
        return snapshot
    
    @asynccontextmanager
    async def _stage_context(self, stage: PipelineStage):
        """Context manager for stage timing"""
        start = time.time()
        stage_metric = StageMetrics(
            stage=stage,
            start_time=start,
            end_time=0,
            duration_seconds=0
        )
        
        try:
            yield stage_metric
        finally:
            stage_metric.end_time = time.time()
            stage_metric.duration_seconds = stage_metric.end_time - stage_metric.start_time
            self.metrics.add_stage_metric(stage_metric)
    
    async def _stage_0_ingestion(self, pdf_path: str, run_id: str) -> Dict[str, Any]:
        """Stage 0: PDF Ingestion"""
        async with self._stage_context(PipelineStage.STAGE_0_INGESTION) as metric:
            self.logger.info(f"Stage 0: Ingesting PDF {pdf_path}")
            
            # Publish event
            await self.event_bus.publish(PDMEvent(
                event_type='stage.ingestion.start',
                run_id=run_id,
                payload={'pdf_path': pdf_path}
            ))
            
            # Placeholder: actual PDF loading would go here
            pdf_data = {'path': pdf_path, 'loaded': True}
            metric.items_processed = 1
            
            return pdf_data
    
    async def _stage_1_extraction(
        self, pdf_data: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        """Stage 1: Semantic Extraction"""
        async with self._stage_context(PipelineStage.STAGE_1_EXTRACTION) as metric:
            self.logger.info("Stage 1: Extracting semantic chunks and tables")
            
            if self.extraction_pipeline:
                result = await self.extraction_pipeline.extract_complete(
                    pdf_data['path']
                )
                chunks = result.semantic_chunks if hasattr(result, 'semantic_chunks') else []
                tables = result.tables if hasattr(result, 'tables') else []
            else:
                # Fallback
                chunks = [{'text': 'Placeholder chunk', 'id': 'chunk_0'}]
                tables = []
            
            metric.items_processed = len(chunks) + len(tables)
            
            await self.event_bus.publish(PDMEvent(
                event_type='stage.extraction.complete',
                run_id=run_id,
                payload={'chunks': len(chunks), 'tables': len(tables)}
            ))
            
            return {'chunks': chunks, 'tables': tables}
    
    async def _stage_2_graph_build(
        self, chunks: List[Any], tables: List[Any], run_id: str
    ) -> nx.DiGraph:
        """Stage 2: Causal Graph Construction"""
        async with self._stage_context(PipelineStage.STAGE_2_GRAPH_BUILD) as metric:
            self.logger.info("Stage 2: Building causal graph")
            
            if self.causal_builder:
                graph = await self.causal_builder.build_graph(chunks, tables)
            else:
                # Fallback
                graph = nx.DiGraph()
                graph.add_edge('A', 'B', weight=1.0)
            
            metric.items_processed = graph.number_of_edges()
            
            await self.event_bus.publish(PDMEvent(
                event_type='stage.graph.complete',
                run_id=run_id,
                payload={
                    'nodes': graph.number_of_nodes(),
                    'edges': graph.number_of_edges()
                }
            ))
            
            return graph
    
    async def _stage_3_bayesian(
        self,
        graph: nx.DiGraph,
        chunks: List[Any],
        prior_snapshot: PriorSnapshot,
        run_id: str
    ) -> Dict[str, Any]:
        """Stage 3: Bayesian Inference (3 AGUJAS)"""
        async with self._stage_context(PipelineStage.STAGE_3_BAYESIAN) as metric:
            self.logger.info("Stage 3: Running Bayesian inference with snapshot priors")
            
            # Use streaming updater for incremental inference
            updater = StreamingBayesianUpdater(event_bus=self.event_bus)
            
            mechanisms = []
            posteriors = {}
            
            if self.bayesian_engine:
                # Use injected engine
                result = await self.bayesian_engine.infer_all_mechanisms(graph, chunks)
                mechanisms = result if isinstance(result, list) else []
            else:
                # Fallback: minimal mechanism result
                from orchestration.pdm_orchestrator import MechanismResult
                mechanisms = [
                    MechanismResult(
                        type='fallback',
                        necessity_test={'passed': True, 'missing': []},
                        posterior_mean=0.7
                    )
                ]
            
            metric.items_processed = len(mechanisms)
            
            await self.event_bus.publish(PDMEvent(
                event_type='stage.bayesian.complete',
                run_id=run_id,
                payload={'mechanisms': len(mechanisms)}
            ))
            
            return {'mechanisms': mechanisms, 'posteriors': posteriors}
    
    async def _stage_4_contradiction(
        self, graph: nx.DiGraph, chunks: List[Any], run_id: str
    ) -> List[Dict[str, Any]]:
        """Stage 4: Contradiction Detection"""
        async with self._stage_context(PipelineStage.STAGE_4_CONTRADICTION) as metric:
            self.logger.info("Stage 4: Detecting contradictions")
            
            contradictions = []
            
            if self.contradiction_detector:
                # Use injected detector
                for chunk in chunks:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.contradiction_detector.detect,
                        chunk.get('text', '') if isinstance(chunk, dict) else str(chunk),
                        'PDM',
                        'estratégico'
                    )
                    if result and 'contradictions' in result:
                        contradictions.extend(result['contradictions'])
            
            metric.items_processed = len(contradictions)
            
            await self.event_bus.publish(PDMEvent(
                event_type='stage.contradiction.complete',
                run_id=run_id,
                payload={'contradictions': len(contradictions)}
            ))
            
            return contradictions
    
    async def _stage_5_validation(
        self,
        graph: nx.DiGraph,
        chunks: List[Any],
        tables: List[Any],
        run_id: str
    ) -> Any:
        """Stage 5: Axiomatic Validation"""
        async with self._stage_context(PipelineStage.STAGE_5_VALIDATION) as metric:
            self.logger.info("Stage 5: Running axiomatic validation")
            
            if self.validator:
                # Convert chunks to SemanticChunk format expected by validator
                semantic_chunks = []
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        from validators.axiomatic_validator import SemanticChunk
                        semantic_chunks.append(SemanticChunk(
                            text=chunk.get('text', ''),
                            dimension=chunk.get('dimension', 'ESTRATEGICO')
                        ))
                
                validation_result = self.validator.validate_complete(
                    graph, semantic_chunks, tables
                )
            else:
                # Fallback
                from orchestration.pdm_orchestrator import ValidationResult
                validation_result = ValidationResult(
                    requires_manual_review=False,
                    passed=True
                )
            
            metric.items_processed = 1
            
            await self.event_bus.publish(PDMEvent(
                event_type='stage.validation.complete',
                run_id=run_id,
                payload={
                    'passed': getattr(validation_result, 'passed', True),
                    'requires_review': getattr(validation_result, 'requires_manual_review', False)
                }
            ))
            
            return validation_result
    
    async def _stage_6_scoring(
        self,
        graph: nx.DiGraph,
        mechanism_results: List[Any],
        validation_result: Any,
        contradictions: List[Dict[str, Any]],
        run_id: str
    ) -> Dict[str, Any]:
        """Stage 6: Scoring Aggregation (MICRO→MESO→MACRO)"""
        async with self._stage_context(PipelineStage.STAGE_6_SCORING) as metric:
            self.logger.info("Stage 6: Calculating scores (MICRO→MESO→MACRO)")
            
            if self.scorer:
                scoring_result = self.scorer.calculate_all_levels(
                    graph=graph,
                    mechanism_results=mechanism_results,
                    validation_result=validation_result,
                    contradictions=contradictions
                )
            else:
                # Fallback: simple scoring
                micro_scores = {f'P{i}-D{j}-Q{k}': 0.7 
                               for i in range(1, 11) 
                               for j in range(1, 7) 
                               for k in range(1, 6)}
                meso_scores = {f'C{i}': 0.7 for i in range(1, 5)}
                macro_score = 0.7
                
                scoring_result = {
                    'micro': micro_scores,
                    'meso': meso_scores,
                    'macro': macro_score
                }
            
            metric.items_processed = len(scoring_result.get('micro', {}))
            
            await self.event_bus.publish(PDMEvent(
                event_type='stage.scoring.complete',
                run_id=run_id,
                payload={
                    'micro_count': len(scoring_result.get('micro', {})),
                    'macro_score': scoring_result.get('macro', 0.0)
                }
            ))
            
            return scoring_result
    
    async def _stage_7_report(
        self, result: UnifiedResult, pdf_path: str, run_id: str
    ) -> Path:
        """Stage 7: Report Generation"""
        async with self._stage_context(PipelineStage.STAGE_7_REPORT) as metric:
            self.logger.info("Stage 7: Generating final report")
            
            if self.report_generator:
                report_path = await self.report_generator.generate(
                    result, pdf_path, run_id
                )
            else:
                # Fallback: save basic JSON report
                report_dir = Path("reports")
                report_dir.mkdir(exist_ok=True)
                report_path = report_dir / f"report_{run_id}.json"
                
                import json
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'run_id': run_id,
                        'success': result.success,
                        'macro_score': result.macro_score,
                        'metrics': self.metrics.get_summary()
                    }, f, indent=2)
            
            metric.items_processed = 1
            
            await self.event_bus.publish(PDMEvent(
                event_type='stage.report.complete',
                run_id=run_id,
                payload={'report_path': str(report_path)}
            ))
            
            return report_path
    
    async def _stage_8_learning(
        self, result: UnifiedResult, run_id: str
    ) -> Dict[str, float]:
        """Stage 8: Adaptive Learning Loop (penalty factors for NEXT run)"""
        async with self._stage_context(PipelineStage.STAGE_8_LEARNING) as metric:
            self.logger.info("Stage 8: Computing penalty factors from failures")
            
            penalty_factors = {}
            
            # Extract failures from mechanism results
            failed_mechanisms = {}
            for mech in result.mechanism_results:
                mech_type = getattr(mech, 'type', 'unknown')
                necessity_test = getattr(mech, 'necessity_test', {})
                
                if isinstance(necessity_test, dict):
                    passed = necessity_test.get('passed', True)
                else:
                    passed = getattr(necessity_test, 'passed', True)
                
                if not passed:
                    failed_mechanisms[mech_type] = failed_mechanisms.get(mech_type, 0) + 1
            
            # Calculate penalty factors (more failures = lower prior for next run)
            total_mechanisms = len(result.mechanism_results) or 1
            for mech_type, fail_count in failed_mechanisms.items():
                failure_rate = fail_count / total_mechanisms
                # Penalty: reduce prior proportionally to failure rate
                penalty_factors[mech_type] = max(0.5, 1.0 - failure_rate)
            
            # Apply penalties to prior store for NEXT run
            if penalty_factors:
                feedback_data = {'penalty_factors': penalty_factors}
                
                # Update priors via learning loop
                from dataclasses import dataclass as dc
                @dc
                class MockAnalysisResult:
                    mechanism_results: List[Any]
                    quality_score: Any
                
                @dc  
                class MockQualityScore:
                    overall_score: float
                
                mock_result = MockAnalysisResult(
                    mechanism_results=result.mechanism_results,
                    quality_score=MockQualityScore(overall_score=result.macro_score)
                )
                
                self.learning_loop.extract_and_update_priors(mock_result)
            
            metric.items_processed = len(penalty_factors)
            
            await self.event_bus.publish(PDMEvent(
                event_type='stage.learning.complete',
                run_id=run_id,
                payload={'penalty_factors': penalty_factors}
            ))
            
            return penalty_factors
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return self.metrics.get_summary()
