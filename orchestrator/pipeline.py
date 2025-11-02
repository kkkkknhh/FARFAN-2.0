#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator Pipeline
=====================

Pure orchestration logic for FARFAN 2.0 policy analysis pipeline.

This module composes all analytical steps:
1. Statement Extraction
2. Contradiction Detection
3. Coherence Metrics Calculation
4. Regulatory Analysis
5. Audit Summary Generation

Design Principles:
- Pure orchestration: No I/O, only coordination
- Accept contracts, return contracts
- All I/O through ports
- Composable and testable
- Fail-safe with meaningful error messages
"""

from typing import Dict, Any, List
from datetime import datetime

from core_contracts import (
    PipelineInput,
    PipelineOutput,
    StatementExtractionOutput,
    ContradictionDetectionOutput,
    CoherenceMetricsOutput,
    RegulatoryAnalysisOutput,
    AuditSummaryOutput,
    PolicyStatement,
    Contradiction,
    CURRENT_VERSIONS,
)
from ports import LogPort, FilePort, ClockPort


# ============================================================================
# Calibration Constants
# ============================================================================

# These constants define quality thresholds for the analysis
COHERENCE_THRESHOLD = 0.7
CAUSAL_INCOHERENCE_LIMIT = 5
EXCELLENT_CONTRADICTION_LIMIT = 5
GOOD_CONTRADICTION_LIMIT = 10
CRITICAL_SEVERITY_THRESHOLD = 0.85
HIGH_SEVERITY_THRESHOLD = 0.70
MEDIUM_SEVERITY_THRESHOLD = 0.50


# ============================================================================
# Pipeline Orchestrator
# ============================================================================


class PolicyAnalysisPipeline:
    """
    Orchestrates the complete FARFAN 2.0 policy analysis pipeline.
    
    This is a pure orchestration layer - it coordinates different analytical
    modules but does not perform I/O directly. All I/O is done through ports.
    """
    
    def __init__(
        self,
        log_port: LogPort,
        file_port: FilePort = None,
        clock_port: ClockPort = None,
    ):
        """
        Initialize pipeline with required ports.
        
        Args:
            log_port: Logging port for structured logging
            file_port: Optional file port for persisting intermediate results
            clock_port: Optional clock port for timestamps
        """
        self.log_port = log_port
        self.file_port = file_port
        self.clock_port = clock_port
        
        # Calibration constants
        self.calibration = {
            "coherence_threshold": COHERENCE_THRESHOLD,
            "causal_incoherence_limit": CAUSAL_INCOHERENCE_LIMIT,
            "excellent_contradiction_limit": EXCELLENT_CONTRADICTION_LIMIT,
            "good_contradiction_limit": GOOD_CONTRADICTION_LIMIT,
            "critical_severity_threshold": CRITICAL_SEVERITY_THRESHOLD,
            "high_severity_threshold": HIGH_SEVERITY_THRESHOLD,
            "medium_severity_threshold": MEDIUM_SEVERITY_THRESHOLD,
        }
    
    def orchestrate(self, pipeline_input: PipelineInput) -> PipelineOutput:
        """
        Execute complete policy analysis pipeline.
        
        Args:
            pipeline_input: Pipeline input contract with text, plan_name, dimension
            
        Returns:
            PipelineOutput contract with all analysis results
        """
        self.log_port.info(
            "Starting policy analysis pipeline",
            plan_name=pipeline_input["plan_name"],
            dimension=pipeline_input["dimension"],
        )
        
        # Phase 1: Extract policy statements
        self.log_port.info("Phase 1: Extracting policy statements")
        statements = self._extract_statements(
            pipeline_input["text"],
            pipeline_input["plan_name"]
        )
        
        # Phase 2: Detect contradictions
        self.log_port.info("Phase 2: Detecting contradictions")
        contradictions = self._detect_contradictions(
            statements,
            pipeline_input["text"],
            pipeline_input["plan_name"],
            pipeline_input["dimension"]
        )
        
        # Phase 3: Calculate coherence metrics
        self.log_port.info("Phase 3: Calculating coherence metrics")
        coherence_metrics = self._calculate_coherence_metrics(
            contradictions,
            statements,
            pipeline_input["text"]
        )
        
        # Phase 4: Analyze regulatory constraints
        self.log_port.info("Phase 4: Analyzing regulatory constraints")
        regulatory_analysis = self._analyze_regulatory_constraints(
            statements,
            pipeline_input["text"],
            pipeline_input["plan_name"]
        )
        
        # Phase 5: Generate audit summary
        self.log_port.info("Phase 5: Generating audit summary")
        audit_summary = self._generate_audit_summary(
            coherence_metrics,
            contradictions,
            regulatory_analysis,
            pipeline_input["plan_name"]
        )
        
        # Compile final report
        output: PipelineOutput = {
            "statements": statements,
            "contradictions": contradictions,
            "coherence_metrics": coherence_metrics,
            "regulatory_analysis": regulatory_analysis,
            "audit_summary": audit_summary,
            "schema_version": pipeline_input["schema_version"],
        }
        
        self.log_port.info(
            "Pipeline completed successfully",
            total_statements=statements["total_count"],
            total_contradictions=contradictions["total_count"],
            coherence_score=coherence_metrics["coherence_score"],
        )
        
        return output
    
    def _extract_statements(
        self,
        text: str,
        plan_name: str
    ) -> StatementExtractionOutput:
        """
        Extract policy statements from text.
        
        This is a simplified implementation. In production, this would call
        the actual statement extraction module.
        """
        # Placeholder: Simple sentence splitting
        # TODO: Replace with actual semantic statement extraction
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        statements: List[PolicyStatement] = [
            {
                "text": sentence,
                "section": "general",
                "confidence": 0.8,
                "causal_dimension": "general",
                "metadata": {},
            }
            for sentence in sentences[:50]  # Limit to 50 for demo
        ]
        
        return {
            "statements": statements,
            "total_count": len(statements),
            "schema_version": CURRENT_VERSIONS["statement"],
        }
    
    def _detect_contradictions(
        self,
        statements: StatementExtractionOutput,
        text: str,
        plan_name: str,
        dimension: str
    ) -> ContradictionDetectionOutput:
        """
        Detect contradictions between policy statements.
        
        This is a simplified implementation. In production, this would call
        the contradiction detection module.
        """
        # Placeholder: No contradictions detected
        # TODO: Replace with actual contradiction detection
        contradictions: List[Contradiction] = []
        
        total_count = len(contradictions)
        
        # Determine quality grade based on count
        if total_count < self.calibration["excellent_contradiction_limit"]:
            quality_grade = "Excelente"
        elif total_count < self.calibration["good_contradiction_limit"]:
            quality_grade = "Bueno"
        else:
            quality_grade = "Regular"
        
        return {
            "contradictions": contradictions,
            "total_count": total_count,
            "quality_grade": quality_grade,
            "schema_version": CURRENT_VERSIONS["contradiction"],
        }
    
    def _calculate_coherence_metrics(
        self,
        contradictions: ContradictionDetectionOutput,
        statements: StatementExtractionOutput,
        text: str
    ) -> CoherenceMetricsOutput:
        """
        Calculate coherence metrics for the policy document.
        
        This is a simplified implementation. In production, this would call
        the coherence calculation module.
        """
        # Placeholder: Simple coherence calculation
        # TODO: Replace with actual coherence metrics
        
        total_contradictions = contradictions["total_count"]
        total_statements = statements["total_count"]
        
        # Simple coherence score: inverse of contradiction rate
        if total_statements > 0:
            contradiction_rate = total_contradictions / total_statements
            coherence_score = max(0.0, 1.0 - contradiction_rate)
        else:
            coherence_score = 1.0
        
        # Causal incoherence count (placeholder)
        causal_incoherence_count = 0
        
        # Determine quality status
        if causal_incoherence_count < self.calibration["causal_incoherence_limit"]:
            quality_status = "Aceptable"
        else:
            quality_status = "Requiere revisión"
        
        return {
            "coherence_score": coherence_score,
            "causal_incoherence_count": causal_incoherence_count,
            "quality_status": quality_status,
            "detailed_metrics": {
                "semantic_coherence": coherence_score,
                "temporal_consistency": 0.9,
                "causal_coherence": 0.85,
            },
            "schema_version": CURRENT_VERSIONS["coherence"],
        }
    
    def _analyze_regulatory_constraints(
        self,
        statements: StatementExtractionOutput,
        text: str,
        plan_name: str
    ) -> RegulatoryAnalysisOutput:
        """
        Analyze regulatory compliance and constraints.
        
        This is a simplified implementation. In production, this would call
        the regulatory analysis module.
        """
        # Placeholder: No violations
        # TODO: Replace with actual regulatory analysis
        
        return {
            "constraints": [],
            "compliance_score": 1.0,
            "critical_violations": 0,
            "schema_version": CURRENT_VERSIONS["regulatory"],
        }
    
    def _generate_audit_summary(
        self,
        coherence_metrics: CoherenceMetricsOutput,
        contradictions: ContradictionDetectionOutput,
        regulatory_analysis: RegulatoryAnalysisOutput,
        plan_name: str
    ) -> AuditSummaryOutput:
        """
        Generate comprehensive audit summary.
        
        This is a simplified implementation. In production, this would call
        the audit summary generation module.
        """
        # Determine overall grade based on multiple factors
        coherence_score = coherence_metrics["coherence_score"]
        total_contradictions = contradictions["total_count"]
        critical_violations = regulatory_analysis["critical_violations"]
        
        # Grading logic
        if (coherence_score >= 0.9 and 
            total_contradictions < 5 and 
            critical_violations == 0):
            overall_grade = "Excelente"
        elif (coherence_score >= 0.7 and 
              total_contradictions < 10 and 
              critical_violations == 0):
            overall_grade = "Bueno"
        elif coherence_score >= 0.5:
            overall_grade = "Regular"
        else:
            overall_grade = "Malo"
        
        key_findings = [
            f"Coherence score: {coherence_score:.2f}",
            f"Total contradictions: {total_contradictions}",
            f"Critical regulatory violations: {critical_violations}",
        ]
        
        recommendations = []
        if total_contradictions > 5:
            recommendations.append(
                "Review and resolve contradictions in policy statements"
            )
        if coherence_score < 0.7:
            recommendations.append(
                "Improve coherence between different policy sections"
            )
        if critical_violations > 0:
            recommendations.append(
                "Address critical regulatory compliance issues"
            )
        
        executive_summary = (
            f"Analysis of {plan_name} completed. "
            f"Overall quality grade: {overall_grade}. "
            f"Coherence score: {coherence_score:.2f}. "
            f"Contradictions detected: {total_contradictions}."
        )
        
        return {
            "overall_grade": overall_grade,
            "key_findings": key_findings,
            "recommendations": recommendations,
            "executive_summary": executive_summary,
            "schema_version": CURRENT_VERSIONS["audit"],
        }


# ============================================================================
# Factory Function
# ============================================================================


def create_pipeline(
    log_port: LogPort,
    file_port: FilePort = None,
    clock_port: ClockPort = None,
) -> PolicyAnalysisPipeline:
    """
    Create a policy analysis pipeline with injected dependencies.
    
    Args:
        log_port: Logging port for structured logging
        file_port: Optional file port for persisting intermediate results
        clock_port: Optional clock port for timestamps
        
    Returns:
        Configured PolicyAnalysisPipeline instance
    """
    return PolicyAnalysisPipeline(
        log_port=log_port,
        file_port=file_port,
        clock_port=clock_port,
    )
