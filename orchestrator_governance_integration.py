#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator Integration with Governance Standards
===================================================

Integrates industrial governance standards (Part 5) with the analytical orchestrator.

This module demonstrates:
- Execution isolation with worker timeout enforcement
- Immutable audit logging with hash chains
- Explainability payloads for Bayesian evaluations
- Human-in-the-loop gates for quality control
- CI contract enforcement integration
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from governance_standards import (
    ExplainabilityPayload,
    ExecutionIsolationConfig,
    HumanInTheLoopGate,
    ImmutableAuditLog,
    IsolationMetrics,
    IsolationMode,
    QualityGrade,
    compute_document_hash,
)
from orchestrator import (
    COHERENCE_THRESHOLD,
    EXCELLENT_CONTRADICTION_LIMIT,
    GOOD_CONTRADICTION_LIMIT,
    AnalyticalOrchestrator,
    PhaseResult,
)


class GovernanceEnhancedOrchestrator(AnalyticalOrchestrator):
    """
    Analytical orchestrator enhanced with governance standards.
    
    Extends base orchestrator with:
    - Execution isolation tracking
    - Immutable audit log with hash chains
    - Human-in-the-loop quality gates
    - Explainability payloads
    """
    
    def __init__(
        self,
        log_dir: Path = None,
        coherence_threshold: float = COHERENCE_THRESHOLD,
        causal_incoherence_limit: int = 5,
        regulatory_depth_factor: float = 1.3,
        enable_governance: bool = True
    ):
        """
        Initialize governance-enhanced orchestrator.
        
        Args:
            log_dir: Directory for audit logs
            coherence_threshold: Minimum coherence score
            causal_incoherence_limit: Maximum causal incoherence count
            regulatory_depth_factor: Regulatory analysis depth multiplier
            enable_governance: Enable governance features
        """
        super().__init__(
            log_dir=log_dir,
            coherence_threshold=coherence_threshold,
            causal_incoherence_limit=causal_incoherence_limit,
            regulatory_depth_factor=regulatory_depth_factor
        )
        
        self.enable_governance = enable_governance
        
        # Initialize governance components
        if self.enable_governance:
            self.governance_audit_log = ImmutableAuditLog(
                log_dir=self.log_dir / "governance"
            )
            
            self.isolation_config = ExecutionIsolationConfig(
                mode=IsolationMode.DOCKER,
                worker_timeout_secs=300,
                fail_open_on_timeout=True
            )
            
            self.isolation_metrics = IsolationMetrics()
            
            self.logger.info("Governance standards enabled")
        else:
            self.governance_audit_log = None
            self.isolation_config = None
            self.isolation_metrics = None
    
    def orchestrate_analysis_with_governance(
        self,
        text: str,
        plan_name: str = "PDM",
        dimension: str = "estratégico",
        source_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Execute analytical pipeline with full governance compliance.
        
        Args:
            text: Full policy document text
            plan_name: Policy plan identifier
            dimension: Analytical dimension
            source_file: Optional source file path for hash computation
            
        Returns:
            Unified structured report with governance metadata
        """
        # Compute source document hash for traceability
        if source_file and source_file.exists():
            sha256_source = compute_document_hash(source_file)
        else:
            # Use text hash if no file provided
            sha256_source = hashlib.sha256(text.encode('utf-8')).hexdigest()
        
        # Generate unique run ID
        run_id = f"{plan_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Track execution start
        execution_start = datetime.now()
        
        try:
            # Run standard orchestration
            report = self.orchestrate_analysis(text, plan_name, dimension)
            
            # Track execution metrics
            execution_time = (datetime.now() - execution_start).total_seconds()
            
            if self.enable_governance:
                self._update_isolation_metrics(
                    success=True,
                    execution_time=execution_time
                )
                
                # Append phases to governance audit log
                for phase_result in self._audit_log:
                    self.governance_audit_log.append(
                        run_id=run_id,
                        sha256_source=sha256_source,
                        phase=phase_result.phase_name,
                        status=phase_result.status,
                        metrics=phase_result.metrics,
                        outputs=phase_result.outputs
                    )
                
                # Create human-in-the-loop gate
                hitl_gate = self._create_hitl_gate(report)
                
                # Add governance metadata to report
                report["governance"] = {
                    "run_id": run_id,
                    "sha256_source": sha256_source,
                    "execution_time_secs": round(execution_time, 2),
                    "isolation_metrics": self.isolation_metrics.to_dict(),
                    "human_in_the_loop_gate": hitl_gate.to_dict(),
                    "audit_log_entries": len(self.governance_audit_log._entries),
                    "audit_chain_valid": self.governance_audit_log.verify_chain()[0]
                }
                
                # Persist governance audit log
                self.governance_audit_log.persist(run_id)
            
            return report
            
        except Exception as e:
            # Track failure
            execution_time = (datetime.now() - execution_start).total_seconds()
            
            if self.enable_governance:
                self._update_isolation_metrics(
                    success=False,
                    execution_time=execution_time
                )
                
                # Log error to governance audit
                self.governance_audit_log.append(
                    run_id=run_id,
                    sha256_source=sha256_source,
                    phase="error_handler",
                    status="error",
                    metrics={"execution_time_secs": execution_time},
                    outputs={"error_message": str(e)}
                )
            
            raise
    
    def _update_isolation_metrics(
        self,
        success: bool,
        execution_time: float
    ) -> None:
        """
        Update isolation metrics after execution.
        
        Args:
            success: Whether execution succeeded
            execution_time: Execution time in seconds
        """
        if not self.enable_governance:
            return
        
        self.isolation_metrics.total_executions += 1
        
        if not success:
            self.isolation_metrics.failure_count += 1
        
        # Check for timeout
        if execution_time >= self.isolation_config.worker_timeout_secs:
            self.isolation_metrics.timeout_count += 1
            if self.isolation_config.fail_open_on_timeout:
                self.isolation_metrics.fallback_count += 1
        
        # Update average execution time
        total_time = (
            self.isolation_metrics.avg_execution_time_secs * 
            (self.isolation_metrics.total_executions - 1) +
            execution_time
        )
        self.isolation_metrics.avg_execution_time_secs = (
            total_time / self.isolation_metrics.total_executions
        )
        
        # Update uptime percentage
        self.isolation_metrics.update_uptime()
    
    def _create_hitl_gate(self, report: Dict[str, Any]) -> HumanInTheLoopGate:
        """
        Create human-in-the-loop gate based on report quality.
        
        Args:
            report: Analysis report
            
        Returns:
            Configured HumanInTheLoopGate instance
        """
        # Extract quality metrics from report
        total_contradictions = report.get("total_contradictions", 0)
        
        # Determine quality grade
        if total_contradictions < EXCELLENT_CONTRADICTION_LIMIT:
            quality_grade = QualityGrade.EXCELENTE
        elif total_contradictions < GOOD_CONTRADICTION_LIMIT:
            quality_grade = QualityGrade.BUENO
        else:
            quality_grade = QualityGrade.REGULAR
        
        # Extract critical severity count from detect_contradictions phase
        critical_severity_count = 0
        if "detect_contradictions" in report:
            metrics = report["detect_contradictions"].get("metrics", {})
            critical_severity_count = metrics.get("critical_severity_count", 0)
        
        # Extract coherence score
        coherence_score = 0.0
        if "calculate_coherence_metrics" in report:
            outputs = report["calculate_coherence_metrics"].get("outputs", {})
            coherence_metrics = outputs.get("coherence_metrics", {})
            coherence_score = coherence_metrics.get("overall_coherence_score", 0.0)
        
        return HumanInTheLoopGate(
            quality_grade=quality_grade,
            critical_severity_count=critical_severity_count,
            total_contradictions=total_contradictions,
            coherence_score=coherence_score
        )
    
    def create_explainability_payload(
        self,
        link_id: str,
        posterior_mean: float,
        posterior_std: float,
        confidence_interval: tuple,
        necessity_test_passed: bool,
        necessity_test_missing: List[str],
        evidence_snippets: List[str]
    ) -> ExplainabilityPayload:
        """
        Create explainability payload for a Bayesian evaluation.
        
        Args:
            link_id: Unique identifier for causal link
            posterior_mean: Posterior distribution mean
            posterior_std: Posterior distribution std
            confidence_interval: 95% credible interval
            necessity_test_passed: Whether necessity test passed
            necessity_test_missing: Missing necessity components
            evidence_snippets: Supporting evidence text snippets
            
        Returns:
            Configured ExplainabilityPayload instance
        """
        return ExplainabilityPayload(
            link_id=link_id,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            confidence_interval=confidence_interval,
            necessity_test_passed=necessity_test_passed,
            necessity_test_missing=necessity_test_missing,
            evidence_snippets=evidence_snippets,
            sha256_evidence=ExplainabilityPayload.compute_evidence_hash(
                evidence_snippets
            )
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def create_governance_orchestrator(
    log_dir: Optional[Path] = None,
    enable_governance: bool = True,
    **calibration_overrides
) -> GovernanceEnhancedOrchestrator:
    """
    Factory function to create governance-enhanced orchestrator.
    
    Args:
        log_dir: Directory for audit logs
        enable_governance: Enable governance features
        **calibration_overrides: Optional overrides for calibration constants
        
    Returns:
        Configured GovernanceEnhancedOrchestrator instance
    """
    return GovernanceEnhancedOrchestrator(
        log_dir=log_dir,
        enable_governance=enable_governance,
        **calibration_overrides
    )


# ============================================================================
# Main - Demonstration
# ============================================================================


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("Governance-Enhanced Orchestrator - Demonstration")
    print("=" * 70)
    print()
    
    # Create governance-enhanced orchestrator
    orchestrator = create_governance_orchestrator(enable_governance=True)
    
    print("✓ Orchestrator created with governance enabled")
    print(f"  - Isolation mode: {orchestrator.isolation_config.mode.value}")
    print(f"  - Worker timeout: {orchestrator.isolation_config.worker_timeout_secs}s")
    print(f"  - Audit log enabled: {orchestrator.governance_audit_log is not None}")
    print()
    
    # Simulate analysis
    sample_text = """
    Plan de Desarrollo Municipal 2024-2027
    
    Objetivo: Mejorar la calidad de vida de los habitantes mediante
    inversiones en infraestructura social y productiva.
    
    Meta: Construir 5 escuelas nuevas en zonas rurales.
    Meta: Pavimentar 20 km de vías terciarias.
    """
    
    print("Running analysis with governance compliance...")
    result = orchestrator.orchestrate_analysis_with_governance(
        text=sample_text,
        plan_name="PDM_Demo",
        dimension="estratégico"
    )
    
    print("✓ Analysis completed")
    print()
    
    # Show governance metadata
    if "governance" in result:
        gov = result["governance"]
        print("Governance Metrics:")
        print(f"  - Run ID: {gov['run_id']}")
        print(f"  - Source hash: {gov['sha256_source'][:16]}...")
        print(f"  - Execution time: {gov['execution_time_secs']}s")
        print(f"  - Audit entries: {gov['audit_log_entries']}")
        print(f"  - Chain valid: {gov['audit_chain_valid']}")
        print()
        
        # Show isolation metrics
        iso_metrics = gov['isolation_metrics']
        print("Isolation Metrics:")
        print(f"  - Total executions: {iso_metrics['total_executions']}")
        print(f"  - Uptime: {iso_metrics['uptime_percentage']}%")
        print(f"  - Meets SOTA: {iso_metrics['meets_sota_standard']}")
        print()
        
        # Show HITL gate
        hitl = gov['human_in_the_loop_gate']
        print("Human-in-the-Loop Gate:")
        print(f"  - Quality grade: {hitl['quality_grade']}")
        print(f"  - Hold for review: {hitl['hold_for_manual_review']}")
        if hitl['hold_for_manual_review']:
            print(f"  - Approver role: {hitl['approver_role']}")
            print(f"  - Trigger reason: {hitl['trigger_reason']}")
    
    print()
    print("=" * 70)
    print("✓ Demonstration completed successfully")
    print("=" * 70)
