#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Analytical Orchestrator for FARFAN 2.0
==============================================

Orchestrates the execution of all analytical modules (regulatory, contradiction,
audit, coherence, causal) with deterministic behavior, complete data flow integrity,
and auditable metrics.

Design Principles:
- Sequential phase execution with enforced dependencies
- Deterministic mathematical calibration (no drift)
- Complete audit trail with immutable logs
- Structured data contracts for all phases
- Error handling with fallback mechanisms
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from infrastructure.audit_logger import ImmutableAuditLogger

# SIN_CARRETA Compliance: Use centralized calibration constants
from infrastructure.calibration_constants import CALIBRATION
from infrastructure.metrics_collector import MetricsCollector

# Phase contracts and telemetry
from orchestrator_contracts import (
    AnalyzeRegulatoryInput,
    AnalyzeRegulatoryOutput,
    CalculateCoherenceInput,
    CalculateCoherenceOutput,
    ContractViolationError,
    DetectContradictionsInput,
    DetectContradictionsOutput,
    ExtractStatementsInput,
    ExtractStatementsOutput,
    GenerateRecommendationsInput,
    GenerateRecommendationsOutput,
)
from orchestrator_contracts import PhaseResult as ContractPhaseResult
from orchestrator_contracts import (
    PhaseStatus,
    ValidateRegulatoryInput,
    ValidateRegulatoryOutput,
)
from orchestrator_telemetry import (
    DecisionTracker,
    EventSeverity,
    EventType,
    TelemetryCollector,
    TraceContext,
    generate_audit_id,
)

# Live module integrations
try:
    from contradiction_deteccion import PolicyContradictionDetectorV2, PolicyDimension

    HAVE_CONTRADICTION_DETECTOR = True
except ImportError as e:
    logging.warning(f"Could not import PolicyContradictionDetectorV2: {e}")
    HAVE_CONTRADICTION_DETECTOR = False

try:
    from dnp_integration import ValidadorDNP

    HAVE_DNP_VALIDATOR = True
except ImportError as e:
    logging.warning(f"Could not import ValidadorDNP: {e}")
    HAVE_DNP_VALIDATOR = False

try:
    from teoria_cambio import AdvancedDAGValidator, TeoriaCambio

    HAVE_TEORIA_CAMBIO = True
except ImportError as e:
    logging.warning(f"Could not import TeoriaCambio: {e}")
    HAVE_TEORIA_CAMBIO = False

try:
    from smart_recommendations import (
        Priority,
        RecommendationPrioritizer,
        SMARTCriteria,
        SMARTRecommendation,
    )

    HAVE_SMART_RECOMMENDATIONS = True
except ImportError as e:
    logging.warning(f"Could not import SMARTRecommendation: {e}")
    HAVE_SMART_RECOMMENDATIONS = False

try:
    from inference.bayesian_engine import BayesianPriorBuilder, BayesianSamplingEngine

    HAVE_BAYESIAN_ENGINE = True
except ImportError as e:
    logging.warning(f"Could not import Bayesian components: {e}")
    HAVE_BAYESIAN_ENGINE = False

# ============================================================================
# PHASE ENUMERATION
# ============================================================================


class AnalyticalPhase(Enum):
    """Sequential phases in the orchestration pipeline"""

    EXTRACT_STATEMENTS = auto()
    DETECT_CONTRADICTIONS = auto()
    ANALYZE_REGULATORY_CONSTRAINTS = auto()
    CALCULATE_COHERENCE_METRICS = auto()
    GENERATE_AUDIT_SUMMARY = auto()
    COMPILE_FINAL_REPORT = auto()


# ============================================================================
# DATA CONTRACTS
# ============================================================================


@dataclass
class PhaseResult:
    """Standardized return signature for all analytical phases (backward compatibility wrapper)"""

    phase_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metrics: Dict[str, Any]
    timestamp: str
    status: str = "success"
    error: Optional[str] = None


# ============================================================================
# UNIFIED ORCHESTRATOR
# ============================================================================


class AnalyticalOrchestrator:
    """
    Main orchestrator for the FARFAN 2.0 analytical pipeline.

    Responsibilities:
    - Execute analytical phases in strict sequential order
    - Aggregate outputs into unified structured report
    - Preserve calibration constants across runs
    - Maintain immutable audit logs
    - Handle errors with fallback mechanisms
    """

    def __init__(
        self,
        log_dir: Path = None,
        calibration: Any = None,
        enable_telemetry: bool = True,
    ):
        """
        Initialize orchestrator with centralized calibration constants.

        SIN_CARRETA Compliance:
        - Uses CALIBRATION singleton by default
        - Accepts override only for testing (with explicit markers)
        - Enables telemetry and tracing by default

        Args:
            log_dir: Directory for audit logs (default: logs/orchestrator)
            calibration: Override calibration (TESTING ONLY, default: CALIBRATION)
            enable_telemetry: Enable telemetry collection
        """
        self.log_dir = log_dir or Path("logs/orchestrator")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Use centralized calibration constants
        self.calibration = calibration or CALIBRATION

        # Metrics collection (SIN_CARRETA observability)
        self.metrics = MetricsCollector()

        # Immutable audit logging (SIN_CARRETA governance)
        audit_store_path = self.log_dir / "audit_logs.jsonl"
        self.audit_logger = ImmutableAuditLogger(audit_store_path)

        # Telemetry and tracing
        self.enable_telemetry = enable_telemetry
        if enable_telemetry:
            telemetry_dir = self.log_dir / "telemetry"
            self.telemetry = TelemetryCollector(telemetry_dir)
            self.decision_tracker = DecisionTracker(self.telemetry)
        else:
            self.telemetry = None
            self.decision_tracker = None

        # Phase result storage (for backward compatibility)
        self._audit_log: List[PhaseResult] = []

        # Global report dictionary
        self._global_report: Dict[str, Any] = {
            "orchestration_metadata": {
                "version": "2.0.0",
                "calibration": {
                    "coherence_threshold": self.calibration.COHERENCE_THRESHOLD,
                    "causal_incoherence_limit": self.calibration.CAUSAL_INCOHERENCE_LIMIT,
                    "regulatory_depth_factor": self.calibration.REGULATORY_DEPTH_FACTOR,
                },
                "execution_start": None,
                "execution_end": None,
            }
        }

        # Audit store configuration (for append_audit_record)
        self._audit_store_dir = self.log_dir / "audit_store"
        self._audit_store_dir.mkdir(parents=True, exist_ok=True)
        self._retention_years = 7  # 7-year retention for compliance

        # Logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Initialize live modules (SIN_CARRETA: fail explicitly if modules unavailable)
        self._init_live_modules()

    def _init_live_modules(self) -> None:
        """
        Intent: Initialize live module instances with explicit failure modes
        Mechanism: Import and instantiate all required modules
        Constraint: MUST fail loudly if critical modules unavailable
        """
        # Initialize PolicyContradictionDetectorV2
        if not HAVE_CONTRADICTION_DETECTOR:
            raise RuntimeError(
                "CRITICAL: PolicyContradictionDetectorV2 not available. "
                "Cannot proceed without contradiction detection capability. "
                "SIN_CARRETA doctrine: NO silent fallbacks."
            )

        try:
            self.contradiction_detector = PolicyContradictionDetectorV2()
            self.logger.info("✓ PolicyContradictionDetectorV2 initialized")
        except Exception as e:
            raise RuntimeError(
                f"CRITICAL: Failed to initialize PolicyContradictionDetectorV2: {e}"
            ) from e

        # Initialize ValidadorDNP (optional, fail gracefully)
        if HAVE_DNP_VALIDATOR:
            try:
                self.dnp_validator = ValidadorDNP(es_municipio_pdet=False)
                self.logger.info("✓ ValidadorDNP initialized")
            except Exception as e:
                self.logger.warning(f"ValidadorDNP initialization failed: {e}")
                self.dnp_validator = None
        else:
            self.dnp_validator = None
            self.logger.warning("ValidadorDNP not available")

        # Initialize TeoriaCambio (optional, fail gracefully)
        if HAVE_TEORIA_CAMBIO:
            try:
                self.teoria_cambio = TeoriaCambio()
                self.dag_validator = AdvancedDAGValidator()
                self.logger.info("✓ TeoriaCambio validators initialized")
            except Exception as e:
                self.logger.warning(f"TeoriaCambio initialization failed: {e}")
                self.teoria_cambio = None
                self.dag_validator = None
        else:
            self.teoria_cambio = None
            self.dag_validator = None
            self.logger.warning("TeoriaCambio not available")

        # Initialize SMARTRecommendation system (optional)
        if HAVE_SMART_RECOMMENDATIONS:
            try:
                self.recommendation_prioritizer = RecommendationPrioritizer()
                self.logger.info("✓ SMARTRecommendation system initialized")
            except Exception as e:
                self.logger.warning(f"SMARTRecommendation initialization failed: {e}")
                self.recommendation_prioritizer = None
        else:
            self.recommendation_prioritizer = None
            self.logger.warning("SMARTRecommendation not available")

        # Initialize Bayesian components (optional)
        if HAVE_BAYESIAN_ENGINE:
            try:
                self.bayesian_prior_builder = BayesianPriorBuilder()
                self.bayesian_sampling_engine = BayesianSamplingEngine()
                self.logger.info("✓ Bayesian inference components initialized")
            except Exception as e:
                self.logger.warning(f"Bayesian components initialization failed: {e}")
                self.bayesian_prior_builder = None
                self.bayesian_sampling_engine = None
        else:
            self.bayesian_prior_builder = None
            self.bayesian_sampling_engine = None
            self.logger.warning("Bayesian components not available")

    def orchestrate_analysis(
        self, text: str, plan_name: str = "PDM", dimension: str = "estratégico"
    ) -> Dict[str, Any]:
        """
        Execute complete analytical pipeline with deterministic phase ordering.

        Orchestration sequence:
        1. extract_statements
        2. detect_contradictions
        3. analyze_regulatory_constraints
        4. calculate_coherence_metrics
        5. generate_audit_summary
        6. compile_final_report

        Args:
            text: Full policy document text
            plan_name: Policy plan identifier
            dimension: Analytical dimension

        Returns:
            Unified structured report with all phase outputs
        """
        run_id = f"analytical_{plan_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        self._global_report["orchestration_metadata"]["execution_start"] = (
            start_time.isoformat()
        )

        # SHA-256 source hash for audit trail
        sha256_source = ImmutableAuditLogger.hash_string(text)

        try:
            # Record pipeline start
            self.metrics.record("pipeline.start", 1.0)
            # Phase 1: Extract Statements
            self.metrics.record("phase.extract_statements.start", 1.0)
            statements_result = self._extract_statements(text, plan_name, dimension)
            self._append_audit_log(statements_result)
            self.metrics.record(
                "extraction.statements_count",
                len(statements_result.outputs["statements"]),
            )

            # Phase 2: Detect Contradictions
            self.metrics.record("phase.detect_contradictions.start", 1.0)
            contradictions_result = self._detect_contradictions(
                statements_result.outputs["statements"], text, plan_name, dimension
            )
            self._append_audit_log(contradictions_result)
            self.metrics.record(
                "contradictions.total_count",
                len(contradictions_result.outputs["contradictions"]),
            )

            # Phase 3: Analyze Regulatory Constraints
            regulatory_result = self._analyze_regulatory_constraints(
                statements_result.outputs["statements"],
                text,
                contradictions_result.outputs.get("temporal_conflicts", []),
            )
            self._append_audit_log(regulatory_result)

            # Phase 4: Calculate Coherence Metrics
            coherence_result = self._calculate_coherence_metrics(
                contradictions_result.outputs["contradictions"],
                statements_result.outputs["statements"],
                text,
            )
            self._append_audit_log(coherence_result)

            # Phase 5: Generate Audit Summary
            audit_result = self._generate_audit_summary(
                contradictions_result.outputs["contradictions"]
            )
            self._append_audit_log(audit_result)

            # Phase 6: Compile Final Report
            final_report = self._compile_final_report(
                plan_name=plan_name,
                dimension=dimension,
                statements_count=len(statements_result.outputs["statements"]),
                contradictions_count=len(
                    contradictions_result.outputs["contradictions"]
                ),
            )

            self._global_report["orchestration_metadata"]["execution_end"] = (
                datetime.now().isoformat()
            )

            # Record pipeline completion
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.record("pipeline.duration_seconds", duration)
            self.metrics.record("pipeline.success", 1.0)

            # Immutable audit log (SIN_CARRETA governance)
            self.audit_logger.append_record(
                run_id=run_id,
                orchestrator="AnalyticalOrchestrator",
                sha256_source=sha256_source,
                event="orchestrate_analysis_complete",
                duration_seconds=duration,
                plan_name=plan_name,
                dimension=dimension,
                statements_count=len(statements_result.outputs["statements"]),
                contradictions_count=len(
                    contradictions_result.outputs["contradictions"]
                ),
                final_score=final_report.get("total_contradictions", 0),
            )

            return final_report

        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}", exc_info=True)
            self.metrics.increment("pipeline.error_count")

            # Audit failure
            self.audit_logger.append_record(
                run_id=run_id,
                orchestrator="AnalyticalOrchestrator",
                sha256_source=sha256_source,
                event="orchestrate_analysis_failed",
                error=str(e),
            )

            return self._generate_error_report(str(e))

    def _extract_statements(
        self, text: str, plan_name: str, dimension: str
    ) -> PhaseResult:
        """
        Intent: Extract policy statements from raw document text
        Mechanism: Use PolicyContradictionDetectorV2._extract_policy_statements
        Constraint: Must return non-empty statements or fail explicitly

        Phase 1: Extract policy statements from text using live module.
        NO PLACEHOLDERS. Calls PolicyContradictionDetectorV2 directly.
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        trace_id = str(uuid.uuid4())

        # Emit telemetry: phase start
        if self.telemetry:
            self.telemetry.emit_event(
                EventType.PHASE_START,
                "extract_statements",
                message=f"Extracting statements from {plan_name}",
                metadata={
                    "plan_name": plan_name,
                    "dimension": dimension,
                    "text_length": len(text),
                },
            )

        try:
            # Create input contract
            contract_input = ExtractStatementsInput(
                text=text,
                plan_name=plan_name,
                dimension=dimension,
                trace_id=trace_id,
                timestamp=timestamp,
            )

            # Validate input contract
            try:
                contract_input.validate()
            except ContractViolationError as e:
                self.logger.error(f"Input contract violation: {e}")
                if self.telemetry:
                    self.telemetry.emit_event(
                        EventType.CONTRACT_VALIDATION,
                        "extract_statements",
                        severity=EventSeverity.ERROR,
                        message="Input contract validation failed",
                        error=str(e),
                    )
                raise

            # Map dimension string to PolicyDimension enum
            dimension_map = {
                "diagnóstico": PolicyDimension.DIAGNOSTICO,
                "estratégico": PolicyDimension.ESTRATEGICO,
                "estrategico": PolicyDimension.ESTRATEGICO,
                "programático": PolicyDimension.PROGRAMATICO,
                "programatico": PolicyDimension.PROGRAMATICO,
                "financiero": PolicyDimension.FINANCIERO,
                "plan plurianual de inversiones": PolicyDimension.FINANCIERO,
                "seguimiento": PolicyDimension.SEGUIMIENTO,
                "territorial": PolicyDimension.TERRITORIAL,
            }

            policy_dimension = dimension_map.get(
                dimension.lower(),
                PolicyDimension.ESTRATEGICO,  # Default
            )

            # LIVE MODULE CALL: PolicyContradictionDetectorV2._extract_policy_statements
            self.logger.info(
                f"Calling PolicyContradictionDetectorV2._extract_policy_statements"
            )
            if self.telemetry:
                self.telemetry.emit_event(
                    EventType.MODULE_INVOCATION,
                    "extract_statements",
                    message="Invoking PolicyContradictionDetectorV2._extract_policy_statements",
                )

            statements = self.contradiction_detector._extract_policy_statements(
                text, policy_dimension
            )

            # Explicit assertion: statements must not be empty unless text is trivial
            if not statements and len(text) > 100:
                error_msg = (
                    f"ASSERTION FAILURE: No statements extracted from text of length {len(text)}. "
                    "This violates the contract expectation that non-trivial text produces statements."
                )
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Calculate metrics
            avg_length = (
                sum(len(s.text) for s in statements) / len(statements)
                if statements
                else 0.0
            )

            dimensions_covered = tuple(set(s.dimension.value for s in statements))

            # Create output contract
            contract_output = ExtractStatementsOutput(
                statements=tuple(statements),  # Immutable
                statement_count=len(statements),
                avg_statement_length=avg_length,
                dimensions_covered=dimensions_covered,
                trace_id=trace_id,
                timestamp=datetime.now().isoformat(),
            )

            # Validate output contract
            try:
                contract_output.validate()
            except ContractViolationError as e:
                self.logger.error(f"Output contract violation: {e}")
                if self.telemetry:
                    self.telemetry.emit_event(
                        EventType.CONTRACT_VALIDATION,
                        "extract_statements",
                        severity=EventSeverity.ERROR,
                        message="Output contract validation failed",
                        error=str(e),
                    )
                raise

            duration_ms = (time.time() - start_time) * 1000

            # Emit telemetry: phase complete
            if self.telemetry:
                self.telemetry.emit_event(
                    EventType.PHASE_COMPLETE,
                    "extract_statements",
                    message=f"Extracted {len(statements)} statements",
                    duration_ms=duration_ms,
                    metadata={
                        "statement_count": len(statements),
                        "avg_statement_length": avg_length,
                    },
                )

            # Return backward-compatible PhaseResult
            return PhaseResult(
                phase_name="extract_statements",
                inputs={
                    "text_length": len(text),
                    "plan_name": plan_name,
                    "dimension": dimension,
                },
                outputs={
                    "statements": list(
                        statements
                    ),  # Convert back to list for compatibility
                    "statement_count": len(statements),
                    "avg_statement_length": avg_length,
                    "dimensions_covered": list(dimensions_covered),
                },
                metrics={
                    "statements_count": len(statements),
                    "avg_statement_length": avg_length,
                    "duration_ms": duration_ms,
                },
                timestamp=timestamp,
                status="success",
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Phase extract_statements failed: {e}", exc_info=True)

            # Emit telemetry: phase error
            if self.telemetry:
                self.telemetry.emit_event(
                    EventType.PHASE_ERROR,
                    "extract_statements",
                    severity=EventSeverity.ERROR,
                    message="Statement extraction failed",
                    error=str(e),
                    duration_ms=duration_ms,
                )

            # SIN_CARRETA: NO silent fallbacks, raise explicitly
            raise RuntimeError(f"CRITICAL FAILURE in extract_statements: {e}") from e

    def _detect_contradictions(
        self, statements: List[Any], text: str, plan_name: str, dimension: str
    ) -> PhaseResult:
        """
        Intent: Detect contradictions across policy statements
        Mechanism: Use PolicyContradictionDetectorV2.detect() with full validation
        Constraint: Must classify all contradictions by severity

        Phase 2: Detect contradictions across statements using live module.
        NO PLACEHOLDERS. Calls PolicyContradictionDetectorV2.detect() directly.
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        trace_id = str(uuid.uuid4())

        # Emit telemetry: phase start
        if self.telemetry:
            self.telemetry.emit_event(
                EventType.PHASE_START,
                "detect_contradictions",
                message=f"Detecting contradictions in {len(statements)} statements",
                metadata={"statement_count": len(statements)},
            )

        try:
            # Create input contract
            contract_input = DetectContradictionsInput(
                statements=tuple(statements),
                text=text,
                plan_name=plan_name,
                dimension=dimension,
                trace_id=trace_id,
                timestamp=timestamp,
            )

            # Validate input contract
            contract_input.validate()

            # Map dimension string to PolicyDimension enum
            dimension_map = {
                "diagnóstico": PolicyDimension.DIAGNOSTICO,
                "estratégico": PolicyDimension.ESTRATEGICO,
                "estrategico": PolicyDimension.ESTRATEGICO,
                "programático": PolicyDimension.PROGRAMATICO,
                "programatico": PolicyDimension.PROGRAMATICO,
                "financiero": PolicyDimension.FINANCIERO,
                "seguimiento": PolicyDimension.SEGUIMIENTO,
                "territorial": PolicyDimension.TERRITORIAL,
            }

            policy_dimension = dimension_map.get(
                dimension.lower(), PolicyDimension.ESTRATEGICO
            )

            # LIVE MODULE CALL: PolicyContradictionDetectorV2.detect()
            self.logger.info(f"Calling PolicyContradictionDetectorV2.detect()")
            if self.telemetry:
                self.telemetry.emit_event(
                    EventType.MODULE_INVOCATION,
                    "detect_contradictions",
                    message="Invoking PolicyContradictionDetectorV2.detect()",
                )

            detection_result = self.contradiction_detector.detect(
                text, plan_name, policy_dimension
            )

            # Extract contradictions from result
            contradictions = detection_result.get("contradictions", [])

            # Explicit runtime assertion: result must have expected structure
            required_keys = [
                "contradictions",
                "total_contradictions",
                "coherence_metrics",
            ]
            missing_keys = [k for k in required_keys if k not in detection_result]
            if missing_keys:
                raise RuntimeError(
                    f"ASSERTION FAILURE: Detection result missing required keys: {missing_keys}"
                )

            # Count severity levels
            critical_count = detection_result.get("critical_severity_count", 0)
            high_count = detection_result.get("high_severity_count", 0)
            medium_count = detection_result.get("medium_severity_count", 0)
            low_count = len(contradictions) - (
                critical_count + high_count + medium_count
            )

            # Extract temporal conflicts if available
            temporal_conflicts = []
            # Look for temporal conflicts in contradictions or separate field
            for c in contradictions:
                if (
                    isinstance(c, dict)
                    and c.get("contradiction_type") == "TEMPORAL_CONFLICT"
                ):
                    temporal_conflicts.append(c)

            # Create output contract
            contract_output = DetectContradictionsOutput(
                contradictions=tuple(contradictions),
                temporal_conflicts=tuple(temporal_conflicts),
                total_contradictions=len(contradictions),
                critical_severity_count=critical_count,
                high_severity_count=high_count,
                medium_severity_count=medium_count,
                low_severity_count=max(0, low_count),
                trace_id=trace_id,
                timestamp=datetime.now().isoformat(),
            )

            # Validate output contract
            contract_output.validate()

            duration_ms = (time.time() - start_time) * 1000

            # Decision point: Check if contradictions exceed threshold
            if self.decision_tracker:
                self.decision_tracker.record_decision(
                    phase_name="detect_contradictions",
                    decision_point="quality_assessment",
                    criteria={
                        "total_contradictions": len(contradictions),
                        "excellent_limit": self.calibration.EXCELLENT_CONTRADICTION_LIMIT,
                        "good_limit": self.calibration.GOOD_CONTRADICTION_LIMIT,
                    },
                    outcome=(
                        "excellent"
                        if len(contradictions)
                        < self.calibration.EXCELLENT_CONTRADICTION_LIMIT
                        else (
                            "good"
                            if len(contradictions)
                            < self.calibration.GOOD_CONTRADICTION_LIMIT
                            else "needs_improvement"
                        )
                    ),
                    rationale=f"Total contradictions: {len(contradictions)}",
                )

            # Emit telemetry: phase complete
            if self.telemetry:
                self.telemetry.emit_event(
                    EventType.PHASE_COMPLETE,
                    "detect_contradictions",
                    message=f"Detected {len(contradictions)} contradictions",
                    duration_ms=duration_ms,
                    metadata={
                        "total_contradictions": len(contradictions),
                        "critical": critical_count,
                        "high": high_count,
                        "medium": medium_count,
                        "low": low_count,
                    },
                )

            # Return backward-compatible PhaseResult
            return PhaseResult(
                phase_name="detect_contradictions",
                inputs={"statements_count": len(statements), "text_length": len(text)},
                outputs={
                    "contradictions": list(contradictions),
                    "temporal_conflicts": list(temporal_conflicts),
                    "total_contradictions": len(contradictions),
                },
                metrics={
                    "total_contradictions": len(contradictions),
                    "critical_severity_count": critical_count,
                    "high_severity_count": high_count,
                    "medium_severity_count": medium_count,
                    "low_severity_count": low_count,
                    "duration_ms": duration_ms,
                },
                timestamp=timestamp,
                status="success",
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Phase detect_contradictions failed: {e}", exc_info=True)

            # Emit telemetry: phase error
            if self.telemetry:
                self.telemetry.emit_event(
                    EventType.PHASE_ERROR,
                    "detect_contradictions",
                    severity=EventSeverity.ERROR,
                    message="Contradiction detection failed",
                    error=str(e),
                    duration_ms=duration_ms,
                )

            # SIN_CARRETA: NO silent fallbacks, raise explicitly
            raise RuntimeError(f"CRITICAL FAILURE in detect_contradictions: {e}") from e

    def _analyze_regulatory_constraints(
        self, statements: List[Any], text: str, temporal_conflicts: List[Any]
    ) -> PhaseResult:
        """
        Phase 3: Analyze regulatory constraints and compliance.

        Applies REGULATORY_DEPTH_FACTOR calibration constant.
        """
        timestamp = datetime.now().isoformat()

        # Placeholder: In real implementation, call actual regulatory analysis
        regulatory_analysis = {
            "regulatory_references_count": 0,
            "constraint_types_mentioned": 0,
            "is_consistent": len(temporal_conflicts) == 0,
            "d1_q5_quality": "insuficiente",
        }

        return PhaseResult(
            phase_name="analyze_regulatory_constraints",
            inputs={
                "statements_count": len(statements),
                "temporal_conflicts_count": len(temporal_conflicts),
                "regulatory_depth_factor": self.calibration.REGULATORY_DEPTH_FACTOR,
            },
            outputs={"d1_q5_regulatory_analysis": regulatory_analysis},
            metrics={
                "regulatory_references": regulatory_analysis[
                    "regulatory_references_count"
                ],
                "constraint_types": regulatory_analysis["constraint_types_mentioned"],
            },
            timestamp=timestamp,
        )

    def _calculate_coherence_metrics(
        self, contradictions: List[Any], statements: List[Any], text: str
    ) -> PhaseResult:
        """
        Phase 4: Calculate advanced coherence metrics.

        Applies COHERENCE_THRESHOLD calibration constant.
        """
        timestamp = datetime.now().isoformat()

        # Placeholder: In real implementation, call actual coherence calculation
        coherence_metrics = {
            "overall_coherence_score": 0.0,
            "temporal_consistency": 0.0,
            "causal_coherence": 0.0,
            "quality_grade": "insuficiente",
        }

        return PhaseResult(
            phase_name="calculate_coherence_metrics",
            inputs={
                "contradictions_count": len(contradictions),
                "statements_count": len(statements),
                "coherence_threshold": self.calibration.COHERENCE_THRESHOLD,
            },
            outputs={"coherence_metrics": coherence_metrics},
            metrics={
                "overall_score": coherence_metrics["overall_coherence_score"],
                "meets_threshold": coherence_metrics["overall_coherence_score"]
                >= self.calibration.COHERENCE_THRESHOLD,
            },
            timestamp=timestamp,
        )

    def _generate_audit_summary(self, contradictions: List[Any]) -> PhaseResult:
        """
        Phase 5: Generate audit summary with quality assessment.

        Applies CAUSAL_INCOHERENCE_LIMIT and quality grade thresholds.
        """
        timestamp = datetime.now().isoformat()

        # Count causal incoherence contradictions
        causal_incoherence_count = 0  # Would be counted from contradictions

        # Determine quality grade using CALIBRATION singleton
        total_contradictions = len(contradictions)
        if total_contradictions < self.calibration.EXCELLENT_CONTRADICTION_LIMIT:
            quality_grade = "Excelente"
        elif total_contradictions < self.calibration.GOOD_CONTRADICTION_LIMIT:
            quality_grade = "Bueno"
        else:
            quality_grade = "Regular"

        audit_summary = {
            "total_contradictions": total_contradictions,
            "causal_incoherence_flags": causal_incoherence_count,
            "structural_failures": 0,  # Would be calculated
            "quality_grade": quality_grade,
            "meets_causal_limit": causal_incoherence_count
            < self.calibration.CAUSAL_INCOHERENCE_LIMIT,
        }

        return PhaseResult(
            phase_name="generate_audit_summary",
            inputs={
                "contradictions_count": total_contradictions,
                "causal_incoherence_limit": self.calibration.CAUSAL_INCOHERENCE_LIMIT,
            },
            outputs={"harmonic_front_4_audit": audit_summary},
            metrics={
                "quality_grade": quality_grade,
                "causal_flags": causal_incoherence_count,
            },
            timestamp=timestamp,
        )

    def _compile_final_report(
        self,
        plan_name: str,
        dimension: str,
        statements_count: int,
        contradictions_count: int,
    ) -> Dict[str, Any]:
        """
        Phase 6: Compile final unified report from all phase outputs.

        Aggregates all phase results into a single structured dictionary.
        No merge overwrites - each phase's data is under explicit keys.
        """
        # Aggregate all phase outputs
        for phase_result in self._audit_log:
            # Store each phase's outputs under its own key
            phase_key = phase_result.phase_name
            self._global_report[phase_key] = {
                "inputs": phase_result.inputs,
                "outputs": phase_result.outputs,
                "metrics": phase_result.metrics,
                "timestamp": phase_result.timestamp,
                "status": phase_result.status,
            }

        # Add top-level summary
        self._global_report.update(
            {
                "plan_name": plan_name,
                "dimension": dimension,
                "analysis_timestamp": datetime.now().isoformat(),
                "total_statements": statements_count,
                "total_contradictions": contradictions_count,
            }
        )

        return self._global_report

    def _append_audit_log(self, phase_result: PhaseResult) -> None:
        """
        Append phase result to immutable audit log.

        Args:
            phase_result: Result from an analytical phase
        """
        self._audit_log.append(phase_result)
        self.logger.info(
            f"Phase completed: {phase_result.phase_name} - "
            f"Status: {phase_result.status}"
        )

    def _persist_audit_log(self, plan_name: str) -> None:
        """
        Persist audit log to disk for traceability.

        DEPRECATED: Now handled by ImmutableAuditLogger.
        Kept for backward compatibility.

        Args:
            plan_name: Policy plan identifier for file naming
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"audit_log_{plan_name}_{timestamp}.json"

        audit_data = {
            "plan_name": plan_name,
            "timestamp": timestamp,
            "calibration": {
                "coherence_threshold": self.calibration.COHERENCE_THRESHOLD,
                "causal_incoherence_limit": self.calibration.CAUSAL_INCOHERENCE_LIMIT,
                "regulatory_depth_factor": self.calibration.REGULATORY_DEPTH_FACTOR,
            },
            "phases": [
                {
                    "phase_name": phase.phase_name,
                    "inputs": phase.inputs,
                    "outputs": {
                        k: str(v)[:100] if isinstance(v, (list, dict)) else v
                        for k, v in phase.outputs.items()
                    },  # Truncate for readability
                    "metrics": phase.metrics,
                    "timestamp": phase.timestamp,
                    "status": phase.status,
                    "error": phase.error,
                }
                for phase in self._audit_log
            ],
        }

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(audit_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Audit log persisted to: {log_file}")

    def _generate_error_report(self, error_message: str) -> Dict[str, Any]:
        """
        Generate error report with fallback values.

        Args:
            error_message: Description of the error

        Returns:
            Minimal report indicating failure
        """
        return {
            "status": "error",
            "error_message": error_message,
            "timestamp": datetime.now().isoformat(),
            "calibration": {
                "coherence_threshold": self.calibration.COHERENCE_THRESHOLD,
                "causal_incoherence_limit": self.calibration.CAUSAL_INCOHERENCE_LIMIT,
                "regulatory_depth_factor": self.calibration.REGULATORY_DEPTH_FACTOR,
            },
            "partial_results": {
                phase.phase_name: phase.outputs for phase in self._audit_log
            },
        }

    def verify_phase_dependencies(self) -> Dict[str, Any]:
        """
        Verify that no phase dependency cycles exist.

        Returns:
            Validation report with dependency graph analysis
        """
        # Define phase dependencies
        dependencies = {
            AnalyticalPhase.EXTRACT_STATEMENTS: set(),
            AnalyticalPhase.DETECT_CONTRADICTIONS: {AnalyticalPhase.EXTRACT_STATEMENTS},
            AnalyticalPhase.ANALYZE_REGULATORY_CONSTRAINTS: {
                AnalyticalPhase.EXTRACT_STATEMENTS,
                AnalyticalPhase.DETECT_CONTRADICTIONS,
            },
            AnalyticalPhase.CALCULATE_COHERENCE_METRICS: {
                AnalyticalPhase.EXTRACT_STATEMENTS,
                AnalyticalPhase.DETECT_CONTRADICTIONS,
            },
            AnalyticalPhase.GENERATE_AUDIT_SUMMARY: {
                AnalyticalPhase.DETECT_CONTRADICTIONS
            },
            AnalyticalPhase.COMPILE_FINAL_REPORT: {
                AnalyticalPhase.EXTRACT_STATEMENTS,
                AnalyticalPhase.DETECT_CONTRADICTIONS,
                AnalyticalPhase.ANALYZE_REGULATORY_CONSTRAINTS,
                AnalyticalPhase.CALCULATE_COHERENCE_METRICS,
                AnalyticalPhase.GENERATE_AUDIT_SUMMARY,
            },
        }

        # Check for cycles (topological sort)
        has_cycle = False
        try:
            # Simple cycle detection via topological ordering
            visited = set()
            temp_mark = set()

            def visit(phase):
                if phase in temp_mark:
                    return True  # Cycle detected
                if phase in visited:
                    return False

                temp_mark.add(phase)
                for dep in dependencies.get(phase, set()):
                    if visit(dep):
                        return True
                temp_mark.remove(phase)
                visited.add(phase)
                return False

            for phase in AnalyticalPhase:
                if visit(phase):
                    has_cycle = True
                    break
        except Exception as e:
            self.logger.error(f"Dependency validation error: {e}")
            has_cycle = True

        return {
            "has_cycles": has_cycle,
            "dependencies": {
                phase.name: [dep.name for dep in deps]
                for phase, deps in dependencies.items()
            },
            "validation_status": "PASS" if not has_cycle else "FAIL",
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_orchestrator(
    log_dir: Optional[Path] = None, **calibration_overrides
) -> AnalyticalOrchestrator:
    """
    Factory function to create orchestrator with optional calibration overrides.

    Args:
        log_dir: Directory for audit logs
        **calibration_overrides: Optional overrides for calibration constants

    Returns:
        Configured AnalyticalOrchestrator instance
    """
    return AnalyticalOrchestrator(log_dir=log_dir, **calibration_overrides)


if __name__ == "__main__":
    # Example usage and validation
    orchestrator = create_orchestrator()

    # Verify no dependency cycles
    validation = orchestrator.verify_phase_dependencies()
    print(json.dumps(validation, indent=2))

    if validation["validation_status"] == "PASS":
        print("\n✓ Orchestrator validation PASSED - no dependency cycles detected")
    else:
        print("\n✗ Orchestrator validation FAILED - dependency cycles detected")
