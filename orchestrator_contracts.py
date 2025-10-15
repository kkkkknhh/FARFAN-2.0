#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator Phase Contracts
============================

Immutable dataclass contracts for all orchestrator phase inputs and outputs.
SIN_CARRETA doctrine: Explicit, contract-rich, deterministic data structures.

All contracts are frozen dataclasses with complete type hints and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================================
# ENUMERATIONS
# ============================================================================


class ContractViolationType(Enum):
    """Types of contract violations"""
    MISSING_REQUIRED_FIELD = auto()
    INVALID_TYPE = auto()
    VALUE_OUT_OF_RANGE = auto()
    SCHEMA_MISMATCH = auto()
    INVALID_LENGTH = auto()
    INVALID_FORMAT = auto()


class PhaseStatus(Enum):
    """Phase execution status"""
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


# ============================================================================
# EXCEPTIONS
# ============================================================================


class ContractViolationError(Exception):
    """Raised when a phase contract is violated"""
    
    def __init__(
        self,
        phase_name: str,
        violation_type: ContractViolationType,
        field_name: str,
        expected: Any,
        actual: Any,
        message: Optional[str] = None
    ):
        self.phase_name = phase_name
        self.violation_type = violation_type
        self.field_name = field_name
        self.expected = expected
        self.actual = actual
        
        default_message = (
            f"Contract violation in {phase_name}: "
            f"{violation_type.name} for field '{field_name}'. "
            f"Expected: {expected}, Actual: {actual}"
        )
        super().__init__(message or default_message)


# ============================================================================
# PHASE INPUT CONTRACTS
# ============================================================================


@dataclass(frozen=True)
class ExtractStatementsInput:
    """
    Intent: Extract policy statements from raw text
    Mechanism: Pass full document text with metadata
    Constraint: text must be non-empty string, plan_name and dimension are required
    """
    text: str
    plan_name: str
    dimension: str
    trace_id: str
    timestamp: str
    
    def validate(self) -> None:
        """Validate input contract"""
        if not self.text or len(self.text) < 10:
            raise ContractViolationError(
                "extract_statements",
                ContractViolationType.INVALID_LENGTH,
                "text",
                "non-empty string (>= 10 chars)",
                f"{len(self.text)} chars",
                "Text must be at least 10 characters"
            )
        if not self.plan_name:
            raise ContractViolationError(
                "extract_statements",
                ContractViolationType.MISSING_REQUIRED_FIELD,
                "plan_name",
                "non-empty string",
                self.plan_name
            )
        if not self.dimension:
            raise ContractViolationError(
                "extract_statements",
                ContractViolationType.MISSING_REQUIRED_FIELD,
                "dimension",
                "non-empty string",
                self.dimension
            )


@dataclass(frozen=True)
class DetectContradictionsInput:
    """
    Intent: Detect contradictions across policy statements
    Mechanism: Analyze statements using NLI and semantic similarity
    Constraint: statements must be non-empty list, text for context
    """
    statements: Tuple[Any, ...]  # Frozen tuple of PolicyStatement objects
    text: str
    plan_name: str
    dimension: str
    trace_id: str
    timestamp: str
    
    def validate(self) -> None:
        """Validate input contract"""
        if not self.statements:
            raise ContractViolationError(
                "detect_contradictions",
                ContractViolationType.INVALID_LENGTH,
                "statements",
                "non-empty tuple",
                f"{len(self.statements)} statements"
            )
        if not self.text:
            raise ContractViolationError(
                "detect_contradictions",
                ContractViolationType.MISSING_REQUIRED_FIELD,
                "text",
                "non-empty string",
                self.text
            )


@dataclass(frozen=True)
class AnalyzeRegulatoryInput:
    """
    Intent: Analyze regulatory constraints and compliance
    Mechanism: Cross-reference statements against regulatory frameworks
    Constraint: statements required, temporal_conflicts can be empty
    """
    statements: Tuple[Any, ...]
    text: str
    temporal_conflicts: Tuple[Any, ...]
    trace_id: str
    timestamp: str
    
    def validate(self) -> None:
        """Validate input contract"""
        if not isinstance(self.statements, tuple):
            raise ContractViolationError(
                "analyze_regulatory",
                ContractViolationType.INVALID_TYPE,
                "statements",
                "tuple",
                type(self.statements).__name__
            )
        if not isinstance(self.temporal_conflicts, tuple):
            raise ContractViolationError(
                "analyze_regulatory",
                ContractViolationType.INVALID_TYPE,
                "temporal_conflicts",
                "tuple",
                type(self.temporal_conflicts).__name__
            )


@dataclass(frozen=True)
class ValidateRegulatoryInput:
    """
    Intent: Validate regulatory compliance using TeoriaCambio
    Mechanism: Execute theory of change validation
    Constraint: causal_graph required for DAG validation
    """
    causal_graph: Dict[str, Any]
    statements: Tuple[Any, ...]
    trace_id: str
    timestamp: str
    
    def validate(self) -> None:
        """Validate input contract"""
        if not self.causal_graph:
            raise ContractViolationError(
                "validate_regulatory",
                ContractViolationType.MISSING_REQUIRED_FIELD,
                "causal_graph",
                "non-empty dict",
                self.causal_graph
            )


@dataclass(frozen=True)
class CalculateCoherenceInput:
    """
    Intent: Calculate coherence metrics using Bayesian inference
    Mechanism: Apply statistical tests and Bayesian posterior calculation
    Constraint: contradictions and statements required for metrics
    """
    contradictions: Tuple[Any, ...]
    statements: Tuple[Any, ...]
    text: str
    trace_id: str
    timestamp: str
    
    def validate(self) -> None:
        """Validate input contract"""
        if not isinstance(self.contradictions, tuple):
            raise ContractViolationError(
                "calculate_coherence",
                ContractViolationType.INVALID_TYPE,
                "contradictions",
                "tuple",
                type(self.contradictions).__name__
            )
        if not isinstance(self.statements, tuple):
            raise ContractViolationError(
                "calculate_coherence",
                ContractViolationType.INVALID_TYPE,
                "statements",
                "tuple",
                type(self.statements).__name__
            )


@dataclass(frozen=True)
class GenerateRecommendationsInput:
    """
    Intent: Generate SMART recommendations for improvements
    Mechanism: Prioritize contradictions and create actionable recommendations
    Constraint: contradictions and coherence_metrics required
    """
    contradictions: Tuple[Any, ...]
    coherence_metrics: Dict[str, Any]
    regulatory_analysis: Dict[str, Any]
    trace_id: str
    timestamp: str
    
    def validate(self) -> None:
        """Validate input contract"""
        if not isinstance(self.coherence_metrics, dict):
            raise ContractViolationError(
                "generate_recommendations",
                ContractViolationType.INVALID_TYPE,
                "coherence_metrics",
                "dict",
                type(self.coherence_metrics).__name__
            )


# ============================================================================
# PHASE OUTPUT CONTRACTS
# ============================================================================


@dataclass(frozen=True)
class ExtractStatementsOutput:
    """
    Intent: Extracted policy statements with metadata
    Mechanism: List of PolicyStatement objects
    Constraint: statements must be immutable tuple
    """
    statements: Tuple[Any, ...]
    statement_count: int
    avg_statement_length: float
    dimensions_covered: Tuple[str, ...]
    trace_id: str
    timestamp: str
    
    def validate(self) -> None:
        """Validate output contract"""
        if self.statement_count < 0:
            raise ContractViolationError(
                "extract_statements",
                ContractViolationType.VALUE_OUT_OF_RANGE,
                "statement_count",
                ">= 0",
                self.statement_count
            )
        if self.statement_count != len(self.statements):
            raise ContractViolationError(
                "extract_statements",
                ContractViolationType.SCHEMA_MISMATCH,
                "statement_count",
                len(self.statements),
                self.statement_count,
                "Count mismatch between statement_count and actual statements"
            )


@dataclass(frozen=True)
class DetectContradictionsOutput:
    """
    Intent: Detected contradictions with evidence
    Mechanism: Tuple of ContradictionEvidence objects
    Constraint: all counts must match actual data
    """
    contradictions: Tuple[Any, ...]
    temporal_conflicts: Tuple[Any, ...]
    total_contradictions: int
    critical_severity_count: int
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int
    trace_id: str
    timestamp: str
    
    def validate(self) -> None:
        """Validate output contract"""
        if self.total_contradictions != len(self.contradictions):
            raise ContractViolationError(
                "detect_contradictions",
                ContractViolationType.SCHEMA_MISMATCH,
                "total_contradictions",
                len(self.contradictions),
                self.total_contradictions
            )
        
        severity_sum = (
            self.critical_severity_count +
            self.high_severity_count +
            self.medium_severity_count +
            self.low_severity_count
        )
        if severity_sum > self.total_contradictions:
            raise ContractViolationError(
                "detect_contradictions",
                ContractViolationType.SCHEMA_MISMATCH,
                "severity_counts",
                f"<= {self.total_contradictions}",
                severity_sum,
                "Severity counts exceed total contradictions"
            )


@dataclass(frozen=True)
class AnalyzeRegulatoryOutput:
    """
    Intent: Regulatory analysis results
    Mechanism: DNP compliance validation
    Constraint: all boolean fields must be explicit
    """
    regulatory_references_count: int
    constraint_types_mentioned: int
    is_consistent: bool
    d1_q5_quality: str
    competencias_validadas: Tuple[str, ...]
    cumple_competencias: bool
    cumple_mga: bool
    trace_id: str
    timestamp: str
    
    def validate(self) -> None:
        """Validate output contract"""
        if self.regulatory_references_count < 0:
            raise ContractViolationError(
                "analyze_regulatory",
                ContractViolationType.VALUE_OUT_OF_RANGE,
                "regulatory_references_count",
                ">= 0",
                self.regulatory_references_count
            )
        
        valid_qualities = {"excelente", "bueno", "aceptable", "insuficiente"}
        if self.d1_q5_quality not in valid_qualities:
            raise ContractViolationError(
                "analyze_regulatory",
                ContractViolationType.INVALID_FORMAT,
                "d1_q5_quality",
                f"one of {valid_qualities}",
                self.d1_q5_quality
            )


@dataclass(frozen=True)
class ValidateRegulatoryOutput:
    """
    Intent: Theory of change validation results
    Mechanism: DAG validation and causal inference
    Constraint: validation status must be boolean
    """
    dag_is_valid: bool
    has_cycles: bool
    causal_coherence_score: float
    violations: Tuple[str, ...]
    trace_id: str
    timestamp: str
    
    def validate(self) -> None:
        """Validate output contract"""
        if not (0.0 <= self.causal_coherence_score <= 1.0):
            raise ContractViolationError(
                "validate_regulatory",
                ContractViolationType.VALUE_OUT_OF_RANGE,
                "causal_coherence_score",
                "[0.0, 1.0]",
                self.causal_coherence_score
            )


@dataclass(frozen=True)
class CalculateCoherenceOutput:
    """
    Intent: Coherence metrics with Bayesian analysis
    Mechanism: Statistical coherence calculation
    Constraint: all scores must be in [0.0, 1.0] range
    """
    overall_coherence_score: float
    temporal_consistency: float
    causal_coherence: float
    quality_grade: str
    meets_threshold: bool
    bayesian_posterior: float
    credible_interval: Tuple[float, float]
    trace_id: str
    timestamp: str
    
    def validate(self) -> None:
        """Validate output contract"""
        for field_name, value in [
            ("overall_coherence_score", self.overall_coherence_score),
            ("temporal_consistency", self.temporal_consistency),
            ("causal_coherence", self.causal_coherence),
            ("bayesian_posterior", self.bayesian_posterior)
        ]:
            if not (0.0 <= value <= 1.0):
                raise ContractViolationError(
                    "calculate_coherence",
                    ContractViolationType.VALUE_OUT_OF_RANGE,
                    field_name,
                    "[0.0, 1.0]",
                    value
                )
        
        lower, upper = self.credible_interval
        if not (0.0 <= lower <= upper <= 1.0):
            raise ContractViolationError(
                "calculate_coherence",
                ContractViolationType.VALUE_OUT_OF_RANGE,
                "credible_interval",
                "0.0 <= lower <= upper <= 1.0",
                self.credible_interval
            )


@dataclass(frozen=True)
class GenerateRecommendationsOutput:
    """
    Intent: SMART recommendations for improvement
    Mechanism: AHP-prioritized recommendation list
    Constraint: all recommendations must pass validation
    """
    recommendations: Tuple[Any, ...]  # SMARTRecommendation objects
    recommendation_count: int
    critical_recommendations: int
    high_recommendations: int
    medium_recommendations: int
    low_recommendations: int
    trace_id: str
    timestamp: str
    
    def validate(self) -> None:
        """Validate output contract"""
        if self.recommendation_count != len(self.recommendations):
            raise ContractViolationError(
                "generate_recommendations",
                ContractViolationType.SCHEMA_MISMATCH,
                "recommendation_count",
                len(self.recommendations),
                self.recommendation_count
            )
        
        priority_sum = (
            self.critical_recommendations +
            self.high_recommendations +
            self.medium_recommendations +
            self.low_recommendations
        )
        if priority_sum != self.recommendation_count:
            raise ContractViolationError(
                "generate_recommendations",
                ContractViolationType.SCHEMA_MISMATCH,
                "priority_counts",
                self.recommendation_count,
                priority_sum,
                "Priority counts must sum to total recommendations"
            )


# ============================================================================
# PHASE RESULT CONTRACT
# ============================================================================


@dataclass(frozen=True)
class PhaseResult:
    """
    Intent: Standardized phase execution result
    Mechanism: Immutable result with full trace context
    Constraint: All fields required, status must be valid enum
    """
    phase_name: str
    inputs: Any  # Frozen input contract
    outputs: Any  # Frozen output contract
    metrics: Dict[str, Any]
    trace_id: str
    timestamp: str
    status: PhaseStatus
    error: Optional[str] = None
    duration_ms: float = 0.0
    
    def validate(self) -> None:
        """Validate phase result contract"""
        if not self.phase_name:
            raise ContractViolationError(
                "phase_result",
                ContractViolationType.MISSING_REQUIRED_FIELD,
                "phase_name",
                "non-empty string",
                self.phase_name
            )
        
        if self.status == PhaseStatus.ERROR and not self.error:
            raise ContractViolationError(
                "phase_result",
                ContractViolationType.MISSING_REQUIRED_FIELD,
                "error",
                "non-empty string when status=ERROR",
                self.error
            )
        
        if self.duration_ms < 0:
            raise ContractViolationError(
                "phase_result",
                ContractViolationType.VALUE_OUT_OF_RANGE,
                "duration_ms",
                ">= 0",
                self.duration_ms
            )
        
        # Validate nested contracts
        if hasattr(self.inputs, 'validate'):
            self.inputs.validate()
        
        if self.status == PhaseStatus.SUCCESS and hasattr(self.outputs, 'validate'):
            self.outputs.validate()
