#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPRECATED: Legacy Analytical Orchestrator
==========================================

⚠️  DEPRECATION WARNING ⚠️
This orchestrator is DEPRECATED and will be removed in a future version.

Please migrate to:
    from orchestration.deterministic_unified_orchestrator import (
        DeterministicUnifiedOrchestrator,
        create_unified_orchestrator
    )

This legacy file is maintained for backward compatibility only.
All new code should use DeterministicUnifiedOrchestrator.

Original Design Principles (now superseded):
- Sequential phase execution with enforced dependencies
- Deterministic mathematical calibration (no drift)
- Complete audit trail with immutable logs
- Structured data contracts for all phases
- Error handling with fallback mechanisms
"""

import warnings

# Issue deprecation warning when module is imported
warnings.warn(
    "orchestrator.py (AnalyticalOrchestrator) is DEPRECATED. "
    "Use orchestration.deterministic_unified_orchestrator.DeterministicUnifiedOrchestrator instead. "
    "This legacy orchestrator will be removed in FARFAN 3.0.",
    DeprecationWarning,
    stacklevel=2
)

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# SIN_CARRETA Compliance: Use centralized calibration constants
from infrastructure.calibration_constants import CALIBRATION
from infrastructure.metrics_collector import MetricsCollector
from infrastructure.audit_logger import ImmutableAuditLogger


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
    """Standardized return signature for all analytical phases"""
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
        calibration: Any = None
    ):
        """
        Initialize orchestrator with centralized calibration constants.
        
        SIN_CARRETA Compliance:
        - Uses CALIBRATION singleton by default
        - Accepts override only for testing (with explicit markers)
        
        Args:
            log_dir: Directory for audit logs (default: logs/orchestrator)
            calibration: Override calibration (TESTING ONLY, default: CALIBRATION)
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
                "execution_end": None
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
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def orchestrate_analysis(
        self,
        text: str,
        plan_name: str = "PDM",
        dimension: str = "estratégico"
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
        self._global_report["orchestration_metadata"]["execution_start"] = start_time.isoformat()
        
        # SHA-256 source hash for audit trail
        sha256_source = ImmutableAuditLogger.hash_string(text)
        
        try:
            # Record pipeline start
            self.metrics.record("pipeline.start", 1.0)
            # Phase 1: Extract Statements
            self.metrics.record("phase.extract_statements.start", 1.0)
            statements_result = self._extract_statements(text, plan_name, dimension)
            self._append_audit_log(statements_result)
            self.metrics.record("extraction.statements_count", len(statements_result.outputs["statements"]))
            
            # Phase 2: Detect Contradictions
            self.metrics.record("phase.detect_contradictions.start", 1.0)
            contradictions_result = self._detect_contradictions(
                statements_result.outputs["statements"],
                text,
                plan_name,
                dimension
            )
            self._append_audit_log(contradictions_result)
            self.metrics.record("contradictions.total_count", len(contradictions_result.outputs["contradictions"]))
            
            # Phase 3: Analyze Regulatory Constraints
            regulatory_result = self._analyze_regulatory_constraints(
                statements_result.outputs["statements"],
                text,
                contradictions_result.outputs.get("temporal_conflicts", [])
            )
            self._append_audit_log(regulatory_result)
            
            # Phase 4: Calculate Coherence Metrics
            coherence_result = self._calculate_coherence_metrics(
                contradictions_result.outputs["contradictions"],
                statements_result.outputs["statements"],
                text
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
                contradictions_count=len(contradictions_result.outputs["contradictions"])
            )
            
            self._global_report["orchestration_metadata"]["execution_end"] = datetime.now().isoformat()
            
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
                contradictions_count=len(contradictions_result.outputs["contradictions"]),
                final_score=final_report.get("total_contradictions", 0)
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
                error=str(e)
            )
            
            return self._generate_error_report(str(e))
    
    def _extract_statements(
        self,
        text: str,
        plan_name: str,
        dimension: str
    ) -> PhaseResult:
        """
        Phase 1: Extract policy statements from text.
        
        This is a placeholder implementation. In production, this would call
        the actual statement extraction logic from contradiction_deteccion.py
        """
        timestamp = datetime.now().isoformat()
        
        # Placeholder: In real implementation, call actual extraction logic
        statements = []  # Would be extracted from text
        
        return PhaseResult(
            phase_name="extract_statements",
            inputs={
                "text_length": len(text),
                "plan_name": plan_name,
                "dimension": dimension
            },
            outputs={
                "statements": statements
            },
            metrics={
                "statements_count": len(statements),
                "avg_statement_length": 0  # Would be calculated
            },
            timestamp=timestamp
        )
    
    def _detect_contradictions(
        self,
        statements: List[Any],
        text: str
    ) -> PhaseResult:
        """
        Phase 2: Detect contradictions across statements.
        
        This is a placeholder implementation. In production, this would call
        the actual contradiction detection logic from contradiction_deteccion.py
        """
        timestamp = datetime.now().isoformat()
        
        # Placeholder: In real implementation, call actual detection logic
        contradictions = []  # Would be detected
        temporal_conflicts = []  # Would be extracted
        
        return PhaseResult(
            phase_name="detect_contradictions",
            inputs={
                "statements_count": len(statements),
                "text_length": len(text)
            },
            outputs={
                "contradictions": contradictions,
                "temporal_conflicts": temporal_conflicts
            },
            metrics={
                "total_contradictions": len(contradictions),
                "critical_severity_count": 0,
                "high_severity_count": 0,
                "medium_severity_count": 0
            },
            timestamp=timestamp
        )
    
    def _analyze_regulatory_constraints(
        self,
        statements: List[Any],
        temporal_conflicts: List[Any]
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
            "d1_q5_quality": "insuficiente"
        }
        
        return PhaseResult(
            phase_name="analyze_regulatory_constraints",
            inputs={
                "statements_count": len(statements),
                "temporal_conflicts_count": len(temporal_conflicts),
                "regulatory_depth_factor": self.calibration.REGULATORY_DEPTH_FACTOR
            },
            outputs={
                "d1_q5_regulatory_analysis": regulatory_analysis
            },
            metrics={
                "regulatory_references": regulatory_analysis["regulatory_references_count"],
                "constraint_types": regulatory_analysis["constraint_types_mentioned"]
            },
            timestamp=timestamp
        )
    
    def _calculate_coherence_metrics(
        self,
        contradictions: List[Any],
        statements: List[Any]
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
            "quality_grade": "insuficiente"
        }
        
        return PhaseResult(
            phase_name="calculate_coherence_metrics",
            inputs={
                "contradictions_count": len(contradictions),
                "statements_count": len(statements),
                "coherence_threshold": self.calibration.COHERENCE_THRESHOLD
            },
            outputs={
                "coherence_metrics": coherence_metrics
            },
            metrics={
                "overall_score": coherence_metrics["overall_coherence_score"],
                "meets_threshold": coherence_metrics["overall_coherence_score"] >= self.calibration.COHERENCE_THRESHOLD
            },
            timestamp=timestamp
        )
    
    def _generate_audit_summary(
        self,
        contradictions: List[Any]
    ) -> PhaseResult:
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
            "meets_causal_limit": causal_incoherence_count < self.calibration.CAUSAL_INCOHERENCE_LIMIT
        }
        
        return PhaseResult(
            phase_name="generate_audit_summary",
            inputs={
                "contradictions_count": total_contradictions,
                "causal_incoherence_limit": self.calibration.CAUSAL_INCOHERENCE_LIMIT
            },
            outputs={
                "harmonic_front_4_audit": audit_summary
            },
            metrics={
                "quality_grade": quality_grade,
                "causal_flags": causal_incoherence_count
            },
            timestamp=timestamp
        )
    
    def _compile_final_report(
        self,
        plan_name: str,
        dimension: str,
        statements_count: int,
        contradictions_count: int
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
                "status": phase_result.status
            }
        
        # Add top-level summary
        self._global_report.update({
            "plan_name": plan_name,
            "dimension": dimension,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_statements": statements_count,
            "total_contradictions": contradictions_count
        })
        
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
                    "outputs": {k: str(v)[:100] if isinstance(v, (list, dict)) else v 
                               for k, v in phase.outputs.items()},  # Truncate for readability
                    "metrics": phase.metrics,
                    "timestamp": phase.timestamp,
                    "status": phase.status,
                    "error": phase.error
                }
                for phase in self._audit_log
            ]
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
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
                phase.phase_name: phase.outputs
                for phase in self._audit_log
            }
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
                AnalyticalPhase.DETECT_CONTRADICTIONS
            },
            AnalyticalPhase.CALCULATE_COHERENCE_METRICS: {
                AnalyticalPhase.EXTRACT_STATEMENTS,
                AnalyticalPhase.DETECT_CONTRADICTIONS
            },
            AnalyticalPhase.GENERATE_AUDIT_SUMMARY: {
                AnalyticalPhase.DETECT_CONTRADICTIONS
            },
            AnalyticalPhase.COMPILE_FINAL_REPORT: {
                AnalyticalPhase.EXTRACT_STATEMENTS,
                AnalyticalPhase.DETECT_CONTRADICTIONS,
                AnalyticalPhase.ANALYZE_REGULATORY_CONSTRAINTS,
                AnalyticalPhase.CALCULATE_COHERENCE_METRICS,
                AnalyticalPhase.GENERATE_AUDIT_SUMMARY
            }
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
            "validation_status": "PASS" if not has_cycle else "FAIL"
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_orchestrator(
    log_dir: Optional[Path] = None,
    **calibration_overrides
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
