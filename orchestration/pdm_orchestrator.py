#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPRECATED: PDM Orchestrator with Explicit State Machine
=========================================================

⚠️  DEPRECATION NOTICE ⚠️
------------------------
This module is DEPRECATED and maintained only for backward compatibility.

**NEW CODE MUST USE:** orchestration.unified_orchestrator.UnifiedOrchestrator

**MIGRATION PATH:**
```python
# OLD (DEPRECATED):
from orchestration.pdm_orchestrator import PDMOrchestrator
orch = PDMOrchestrator(config)
result = await orch.analyze_pdm(pdf_path)

# NEW (REQUIRED):
from orchestration.unified_orchestrator import UnifiedOrchestrator
orch = UnifiedOrchestrator(config)
orch.inject_components(...)  # Inject all required components
result = await orch.execute_pipeline(pdf_path)
```

**RATIONALE:**
This orchestrator has been superseded by the unified 9-stage pipeline which:
- Consolidates Phase 0-IV with analytical orchestrator phases
- Eliminates overlapping responsibilities
- Provides explicit contract enforcement (ComponentNotInjectedError)
- Implements structured telemetry at all decision points
- Resolves circular dependency via immutable prior snapshots

**DEPRECATION TIMELINE:**
- Current: Maintained for backward compatibility with deprecation warnings
- Next release: Will raise DeprecationWarning on import
- Future release: Will be removed entirely

For details, see:
- UNIFIED_ORCHESTRATOR_IMPLEMENTATION.md
- orchestration/unified_orchestrator.py

---

Original Implementation (preserved for compatibility):
Implements Phase 0-IV execution with observability, backpressure, and audit logging

SIN_CARRETA Compliance:
- Uses centralized calibration constants
- Immutable audit logging with SHA-256 provenance
- Deterministic state machine transitions
"""

import asyncio
import logging
import time
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from infrastructure.audit_logger import ImmutableAuditLogger

# SIN_CARRETA Compliance: Use centralized infrastructure
from infrastructure.calibration_constants import CALIBRATION
from infrastructure.metrics_collector import MetricsCollector

# Emit deprecation warning on import
warnings.warn(
    "orchestration.pdm_orchestrator is DEPRECATED. Use orchestration.unified_orchestrator.UnifiedOrchestrator instead. "
    "This module will be removed in a future release. "
    "See UNIFIED_ORCHESTRATOR_IMPLEMENTATION.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)


class PDMAnalysisState(str, Enum):
    """Analysis state machine states"""

    INITIALIZED = "initialized"
    EXTRACTING = "extracting"
    BUILDING_DAG = "building_dag"
    INFERRING_MECHANISMS = "inferring_mechanisms"
    VALIDATING = "validating"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExtractionResult:
    """Results from extraction phase"""

    semantic_chunks: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    extraction_quality: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.extraction_quality:
            self.extraction_quality = {"score": 1.0}


@dataclass
class MechanismResult:
    """Results from mechanism inference"""

    type: str
    necessity_test: Dict[str, Any]
    posterior_mean: float = 0.0

    def __post_init__(self):
        if not self.necessity_test:
            self.necessity_test = {"passed": True, "missing": []}


@dataclass
class ValidationResult:
    """Results from validation phase"""

    requires_manual_review: bool = False
    hold_reason: Optional[str] = None
    passed: bool = True
    warnings: List[str] = field(default_factory=list)


@dataclass
class QualityScore:
    """Quality score with dimension breakdown"""

    overall_score: float
    dimension_scores: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.dimension_scores:
            self.dimension_scores = {
                "D1": 0.7,
                "D2": 0.7,
                "D3": 0.7,
                "D4": 0.7,
                "D5": 0.7,
                "D6": 0.7,
            }


@dataclass
class AnalysisResult:
    """Complete analysis result"""

    run_id: str
    quality_score: QualityScore
    causal_graph: Any  # networkx.DiGraph
    mechanism_results: List[MechanismResult]
    validation_results: ValidationResult
    recommendations: List[str] = field(default_factory=list)

    def to_audit_dict(self) -> Dict[str, Any]:
        """Convert to audit-friendly dictionary"""
        return {
            "run_id": self.run_id,
            "quality_score": self.quality_score.overall_score,
            "dimension_scores": self.quality_score.dimension_scores,
            "graph_nodes": (
                self.causal_graph.number_of_nodes() if self.causal_graph else 0
            ),
            "graph_edges": (
                self.causal_graph.number_of_edges() if self.causal_graph else 0
            ),
            "mechanism_count": len(self.mechanism_results),
            "validation_passed": self.validation_results.passed,
            "recommendation_count": len(self.recommendations),
        }


class DataQualityError(Exception):
    """Exception raised when data quality is below threshold"""

    pass


# MetricsCollector and ImmutableAuditLogger now imported from infrastructure


class PDMOrchestrator:
    """
    Orquestador maestro que ejecuta Phase 0-IV con observabilidad completa.
    Implementa backpressure, timeouts, y audit logging.
    """

    def __init__(self, config: Any):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.state = PDMAnalysisState.INITIALIZED
        self.metrics = MetricsCollector()

        # Audit logging
        audit_store_path = getattr(config, "audit_store_path", None)
        if hasattr(config, "self_reflection") and hasattr(
            config.self_reflection, "prior_history_path"
        ):
            audit_store_path = (
                Path(config.self_reflection.prior_history_path).parent
                / "audit_logs.jsonl"
            )
        self.audit_logger = ImmutableAuditLogger(audit_store_path)

        # Queue management (Backpressure Standard)
        queue_size = getattr(config, "queue_size", 10)
        max_inflight_jobs = getattr(config, "max_inflight_jobs", 3)
        self.job_queue = asyncio.Queue(maxsize=queue_size)
        self.active_jobs: Dict[str, Any] = {}
        self.semaphore = asyncio.Semaphore(max_inflight_jobs)

        # Component placeholders (will be initialized by subclass or dependency injection)
        self.extraction_pipeline = None
        self.causal_builder = None
        self.bayesian_engine = None
        self.validator = None
        self.scorer = None

        self.logger.info(f"PDMOrchestrator initialized with state: {self.state}")

    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}_{id(self) % 10000}"

    def _transition_state(self, new_state: PDMAnalysisState) -> None:
        """Transition to new state with logging"""
        old_state = self.state
        self.state = new_state
        self.logger.info(f"State transition: {old_state} -> {new_state}")
        self.metrics.record("state_transitions", 1.0)

    @asynccontextmanager
    async def _timeout_context(self, timeout_secs: float):
        """Context manager for timeout handling"""
        try:
            async with asyncio.timeout(timeout_secs):
                yield
        except asyncio.TimeoutError:
            raise

    async def analyze_plan(self, pdf_path: str) -> AnalysisResult:
        """
        Entry point principal. Ejecuta análisis completo con:
        - State tracking
        - Timeout enforcement
        - Resource management
        - Immutable audit trail
        """
        run_id = self._generate_run_id()
        start_time = time.time()

        try:
            async with self.semaphore:  # Concurrency control
                worker_timeout = getattr(self.config, "worker_timeout_secs", 300)
                async with self._timeout_context(worker_timeout):
                    result = await self._execute_phases(pdf_path, run_id)

        except asyncio.TimeoutError:
            self.metrics.increment("pipeline.timeout_count")
            result = self._handle_timeout(run_id, pdf_path)

        except Exception as e:
            self.metrics.increment("pipeline.error_count")
            result = self._handle_failure(run_id, pdf_path, e)

        finally:
            duration = time.time() - start_time
            self.metrics.record("pipeline.duration_seconds", duration)

            # Immutable Audit Log (Governance Standard)
            self.audit_logger.append_record(
                run_id=run_id,
                orchestrator="PDMOrchestrator",
                sha256_source=ImmutableAuditLogger.hash_file(pdf_path),
                event="analyze_plan_complete",
                duration_seconds=duration,
                final_state=self.state.value,
                result_summary=result.to_audit_dict(),
            )

        return result

    async def _execute_phases(self, pdf_path: str, run_id: str) -> AnalysisResult:
        """Ejecuta pipeline Phase 0-IV con state transitions."""

        # PHASE I: Extraction (Tide Gate)
        self._transition_state(PDMAnalysisState.EXTRACTING)
        extraction = await self._extract_complete(pdf_path)
        self.metrics.record("extraction.chunk_count", len(extraction.semantic_chunks))
        self.metrics.record("extraction.table_count", len(extraction.tables))

        # Quality gate check (SIN_CARRETA compliance)
        min_quality_threshold = getattr(
            self.config, "min_quality_threshold", CALIBRATION.MIN_QUALITY_THRESHOLD
        )
        if extraction.extraction_quality.get("score", 1.0) < min_quality_threshold:
            raise DataQualityError(
                f"Extraction quality too low: {extraction.extraction_quality}"
            )

        # PHASE II: DAG Construction (Core Synthesis)
        self._transition_state(PDMAnalysisState.BUILDING_DAG)
        causal_graph = await self._build_graph(
            extraction.semantic_chunks, extraction.tables
        )
        self.metrics.record("graph.node_count", causal_graph.number_of_nodes())
        self.metrics.record("graph.edge_count", causal_graph.number_of_edges())

        # PHASE III: Concurrent Audits (Async Deep Dive)
        self._transition_state(PDMAnalysisState.INFERRING_MECHANISMS)

        # Launch parallel tasks
        mechanism_task = asyncio.create_task(
            self._infer_all_mechanisms(causal_graph, extraction.semantic_chunks)
        )
        validation_task = asyncio.create_task(
            self._validate_complete(
                causal_graph, extraction.semantic_chunks, extraction.tables
            )
        )

        mechanism_results, validation_results = await asyncio.gather(
            mechanism_task, validation_task
        )

        # Observability metrics
        self.metrics.record(
            "mechanism.prior_decay_rate", self._compute_prior_decay(mechanism_results)
        )
        self.metrics.record(
            "evidence.hoop_test_fail_count",
            sum(
                1 for r in mechanism_results if not r.necessity_test.get("passed", True)
            ),
        )

        # Human gating check (Governance Standard)
        if validation_results.requires_manual_review:
            self._trigger_manual_review_hold(run_id, validation_results.hold_reason)

        # PHASE IV: Final Convergence (Verdict)
        self._transition_state(PDMAnalysisState.FINALIZING)
        final_score = self._calculate_quality_score(
            causal_graph=causal_graph,
            mechanism_results=mechanism_results,
            validation_results=validation_results,
            extraction_quality=extraction.extraction_quality,
        )

        # D6 dimension alert (Observability) - SIN_CARRETA compliance
        d6_score = final_score.dimension_scores.get("D6", 0.7)
        self.metrics.record("dimension.avg_score_D6", d6_score)
        if d6_score < CALIBRATION.D6_ALERT_THRESHOLD:
            self.metrics.alert(
                "CRITICAL",
                f"D6_SCORE_BELOW_THRESHOLD: {d6_score} < {CALIBRATION.D6_ALERT_THRESHOLD}",
            )

        self._transition_state(PDMAnalysisState.COMPLETED)

        return AnalysisResult(
            run_id=run_id,
            quality_score=final_score,
            causal_graph=causal_graph,
            mechanism_results=mechanism_results,
            validation_results=validation_results,
            recommendations=self._generate_recommendations(
                final_score, validation_results
            ),
        )

    async def _extract_complete(self, pdf_path: str) -> ExtractionResult:
        """Phase I: Extract complete data from PDF"""
        if self.extraction_pipeline:
            # Use injected extraction pipeline
            return await self.extraction_pipeline.extract_complete(pdf_path)

        # Fallback: basic extraction
        self.logger.warning("No extraction_pipeline configured, using fallback")
        return ExtractionResult(
            semantic_chunks=[{"text": "Sample chunk", "source": pdf_path}],
            tables=[],
            extraction_quality={"score": 0.8},
        )

    async def _build_graph(
        self, semantic_chunks: List[Dict], tables: List[Dict]
    ) -> Any:
        """Phase II: Build causal graph"""
        if self.causal_builder:
            return await self.causal_builder.build_graph(semantic_chunks, tables)

        # Fallback: create empty graph
        self.logger.warning("No causal_builder configured, using fallback")
        import networkx as nx

        return nx.DiGraph()

    async def _infer_all_mechanisms(
        self, causal_graph: Any, semantic_chunks: List[Dict]
    ) -> List[MechanismResult]:
        """Phase III: Infer mechanisms"""
        if self.bayesian_engine:
            return await self.bayesian_engine.infer_all_mechanisms(
                causal_graph, semantic_chunks
            )

        # Fallback
        self.logger.warning("No bayesian_engine configured, using fallback")
        return [
            MechanismResult(
                type="fallback", necessity_test={"passed": True, "missing": []}
            )
        ]

    async def _validate_complete(
        self, causal_graph: Any, semantic_chunks: List[Dict], tables: List[Dict]
    ) -> ValidationResult:
        """Phase III: Validate complete"""
        if self.validator:
            return await self.validator.validate_complete(
                causal_graph, semantic_chunks, tables
            )

        # Fallback
        self.logger.warning("No validator configured, using fallback")
        return ValidationResult(requires_manual_review=False, passed=True)

    def _calculate_quality_score(
        self,
        causal_graph: Any,
        mechanism_results: List[MechanismResult],
        validation_results: ValidationResult,
        extraction_quality: Dict[str, Any],
    ) -> QualityScore:
        """Phase IV: Calculate quality score"""
        if self.scorer:
            return self.scorer.calculate_quality_score(
                causal_graph=causal_graph,
                mechanism_results=mechanism_results,
                validation_results=validation_results,
                extraction_quality=extraction_quality,
            )

        # Fallback: simple score calculation
        self.logger.warning("No scorer configured, using fallback")
        base_score = extraction_quality.get("score", 0.7)
        validation_score = 1.0 if validation_results.passed else 0.5
        overall = (base_score + validation_score) / 2.0

        return QualityScore(
            overall_score=overall,
            dimension_scores={
                "D1": overall,
                "D2": overall,
                "D3": overall,
                "D4": overall,
                "D5": overall,
                "D6": overall,
            },
        )

    def _compute_prior_decay(self, mechanism_results: List[MechanismResult]) -> float:
        """Compute prior decay rate from mechanism results"""
        if not mechanism_results:
            return 0.0

        # Calculate average posterior mean as proxy for decay
        posteriors = [r.posterior_mean for r in mechanism_results]
        return sum(posteriors) / len(posteriors) if posteriors else 0.0

    def _trigger_manual_review_hold(self, run_id: str, reason: Optional[str]) -> None:
        """Trigger manual review hold"""
        self.logger.warning(f"Manual review required for run {run_id}: {reason}")
        self.metrics.alert("WARNING", f"MANUAL_REVIEW_HOLD: {reason}")

    def _generate_recommendations(
        self, final_score: QualityScore, validation_results: ValidationResult
    ) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []

        if final_score.overall_score < 0.6:
            recommendations.append(
                "Overall quality score is below acceptable threshold (0.6)"
            )

        for dim, score in final_score.dimension_scores.items():
            if score < 0.55:
                recommendations.append(
                    f"Dimension {dim} score ({score:.2f}) is critically low"
                )

        if not validation_results.passed:
            recommendations.append("Validation failed - review validation warnings")

        if validation_results.warnings:
            recommendations.append(
                f"Address {len(validation_results.warnings)} validation warnings"
            )

        return recommendations

    def _handle_timeout(self, run_id: str) -> AnalysisResult:
        """Handle timeout scenario"""
        self._transition_state(PDMAnalysisState.FAILED)
        self.logger.error(f"Analysis timeout for run {run_id}")

        import networkx as nx

        return AnalysisResult(
            run_id=run_id,
            quality_score=QualityScore(overall_score=0.0),
            causal_graph=nx.DiGraph(),
            mechanism_results=[],
            validation_results=ValidationResult(passed=False),
            recommendations=[
                "Analysis timed out - consider increasing worker_timeout_secs"
            ],
        )

    def _handle_failure(self, run_id: str, error: Exception) -> AnalysisResult:
        """Handle failure scenario"""
        self._transition_state(PDMAnalysisState.FAILED)
        self.logger.error(f"Analysis failed for run {run_id}: {error}", exc_info=True)

        import networkx as nx

        return AnalysisResult(
            run_id=run_id,
            quality_score=QualityScore(overall_score=0.0),
            causal_graph=nx.DiGraph(),
            mechanism_results=[],
            validation_results=ValidationResult(passed=False),
            recommendations=[f"Analysis failed with error: {str(error)}"],
        )

    def load_ontology(self) -> Dict[str, Any]:
        """Load ontology for validator"""
        # This would load from config or file
        return {}
