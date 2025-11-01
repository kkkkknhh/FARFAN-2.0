#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Contracts - TypedDict Definitions
=======================================

Single source of truth for all data contracts in FARFAN 2.0.

All contracts follow these rules:
1. Every contract has a schema_version field (e.g., "sem-1.3")
2. Value bounds and invariants are documented in docstrings
3. Contracts are immutable (use TypedDict, not dict)
4. Pydantic runtime validators mirror these in contracts_runtime.py

Design Principles:
- Explicit over implicit
- Type-safe by default
- Version all interfaces
- Document all invariants
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal


# ============================================================================
# Semantic Analysis Contracts
# ============================================================================

class SemanticAnalyzerInput(TypedDict):
    """Input contract for semantic analysis.
    
    Invariants:
    - text: Must be non-empty string (min_length=1)
    - segments: Can be empty list for auto-segmentation
    - ontology_params: Can be empty dict for defaults
    - schema_version: Must match pattern "sem-\\d+\\.\\d+"
    """
    text: str
    segments: List[str]
    ontology_params: Dict[str, Any]
    schema_version: str


class SemanticChunk(TypedDict):
    """A semantic chunk from document segmentation.
    
    Invariants:
    - id: Unique identifier for chunk
    - text: Non-empty chunk text
    - start_pos: >= 0
    - end_pos: > start_pos
    - embedding: Optional 768-dim vector (BGE-M3)
    - metadata: Additional chunk-level metadata
    """
    id: str
    text: str
    start_pos: int
    end_pos: int
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]


class SemanticAnalyzerOutput(TypedDict):
    """Output contract for semantic analysis.
    
    Invariants:
    - chunks: List of semantic chunks (can be empty)
    - coherence_score: Range [0.0, 1.0]
    - quality_metrics: Required metrics dict
    - schema_version: Must match input version
    """
    chunks: List[SemanticChunk]
    coherence_score: float
    quality_metrics: Dict[str, Any]
    schema_version: str


# ============================================================================
# Contradiction Detection Contracts
# ============================================================================

class ContradictionDetectionInput(TypedDict):
    """Input contract for contradiction detection.
    
    Invariants:
    - statements: List of extracted policy statements
    - text: Full document text for context
    - plan_name: Non-empty plan identifier
    - dimension: Valid causal dimension
    - schema_version: Must match pattern "cd-\\d+\\.\\d+"
    """
    statements: List[str]
    text: str
    plan_name: str
    dimension: str
    schema_version: str


class Contradiction(TypedDict):
    """A detected contradiction between statements.
    
    Invariants:
    - statement1: First contradicting statement
    - statement2: Second contradicting statement
    - severity: One of ["critical", "high", "medium", "low"]
    - confidence: Range [0.0, 1.0]
    - explanation: Human-readable explanation
    - metadata: Additional context
    """
    statement1: str
    statement2: str
    severity: Literal["critical", "high", "medium", "low"]
    confidence: float
    explanation: str
    metadata: Dict[str, Any]


class ContradictionDetectionOutput(TypedDict):
    """Output contract for contradiction detection.
    
    Invariants:
    - contradictions: List of detected contradictions
    - total_count: >= 0, must equal len(contradictions)
    - quality_grade: One of ["Excelente", "Bueno", "Regular", "Malo"]
    - schema_version: Must match input version
    """
    contradictions: List[Contradiction]
    total_count: int
    quality_grade: Literal["Excelente", "Bueno", "Regular", "Malo"]
    schema_version: str


# ============================================================================
# Embedding Contracts
# ============================================================================

class EmbeddingInput(TypedDict):
    """Input contract for text embedding.
    
    Invariants:
    - texts: Non-empty list of texts to embed
    - model_name: Valid model identifier (default: "BAAI/bge-m3")
    - batch_size: > 0, typically 32
    - normalize: Whether to L2-normalize embeddings
    - schema_version: Must match pattern "emb-\\d+\\.\\d+"
    """
    texts: List[str]
    model_name: str
    batch_size: int
    normalize: bool
    schema_version: str


class EmbeddingOutput(TypedDict):
    """Output contract for text embedding.
    
    Invariants:
    - embeddings: List of embedding vectors, len = len(texts)
    - dimension: Embedding dimension (typically 768 or 1024)
    - model_name: Model used for embedding
    - schema_version: Must match input version
    """
    embeddings: List[List[float]]
    dimension: int
    model_name: str
    schema_version: str


# ============================================================================
# Policy Statement Extraction Contracts
# ============================================================================

class PolicyStatement(TypedDict):
    """A single extracted policy statement.
    
    Invariants:
    - text: The statement text
    - section: Section where statement was found
    - confidence: Range [0.0, 1.0]
    - causal_dimension: One of valid causal dimensions
    - metadata: Additional extraction metadata
    """
    text: str
    section: str
    confidence: float
    causal_dimension: str
    metadata: Dict[str, Any]


class StatementExtractionInput(TypedDict):
    """Input contract for statement extraction.
    
    Invariants:
    - text: Non-empty document text
    - plan_name: Non-empty plan identifier
    - extract_all: Whether to extract all or filter by confidence
    - schema_version: Must match pattern "stmt-\\d+\\.\\d+"
    """
    text: str
    plan_name: str
    extract_all: bool
    schema_version: str


class StatementExtractionOutput(TypedDict):
    """Output contract for statement extraction.
    
    Invariants:
    - statements: List of extracted statements
    - total_count: >= 0, must equal len(statements)
    - schema_version: Must match input version
    """
    statements: List[PolicyStatement]
    total_count: int
    schema_version: str


# ============================================================================
# Coherence Metrics Contracts
# ============================================================================

class CoherenceMetricsInput(TypedDict):
    """Input contract for coherence calculation.
    
    Invariants:
    - contradictions: List of contradictions to analyze
    - statements: List of policy statements
    - text: Full document text
    - schema_version: Must match pattern "coh-\\d+\\.\\d+"
    """
    contradictions: List[Contradiction]
    statements: List[PolicyStatement]
    text: str
    schema_version: str


class CoherenceMetricsOutput(TypedDict):
    """Output contract for coherence metrics.
    
    Invariants:
    - coherence_score: Range [0.0, 1.0]
    - causal_incoherence_count: >= 0
    - quality_status: One of ["Aceptable", "Requiere revisión"]
    - detailed_metrics: Breakdown of coherence components
    - schema_version: Must match input version
    """
    coherence_score: float
    causal_incoherence_count: int
    quality_status: Literal["Aceptable", "Requiere revisión"]
    detailed_metrics: Dict[str, Any]
    schema_version: str


# ============================================================================
# Regulatory Constraint Analysis Contracts
# ============================================================================

class RegulatoryConstraint(TypedDict):
    """A regulatory constraint violation or compliance item.
    
    Invariants:
    - constraint_type: Type of regulatory requirement
    - description: Human-readable description
    - severity: One of ["critical", "high", "medium", "low"]
    - compliant: Whether constraint is satisfied
    - evidence: Supporting evidence for assessment
    """
    constraint_type: str
    description: str
    severity: Literal["critical", "high", "medium", "low"]
    compliant: bool
    evidence: str


class RegulatoryAnalysisInput(TypedDict):
    """Input contract for regulatory analysis.
    
    Invariants:
    - statements: Policy statements to analyze
    - text: Full document text
    - plan_name: Non-empty plan identifier
    - schema_version: Must match pattern "reg-\\d+\\.\\d+"
    """
    statements: List[PolicyStatement]
    text: str
    plan_name: str
    schema_version: str


class RegulatoryAnalysisOutput(TypedDict):
    """Output contract for regulatory analysis.
    
    Invariants:
    - constraints: List of assessed constraints
    - compliance_score: Range [0.0, 1.0]
    - critical_violations: >= 0
    - schema_version: Must match input version
    """
    constraints: List[RegulatoryConstraint]
    compliance_score: float
    critical_violations: int
    schema_version: str


# ============================================================================
# Audit Summary Contracts
# ============================================================================

class AuditSummaryInput(TypedDict):
    """Input contract for audit summary generation.
    
    Invariants:
    - coherence_metrics: Coherence analysis results
    - contradictions: List of contradictions
    - regulatory_analysis: Regulatory compliance results
    - plan_name: Non-empty plan identifier
    - schema_version: Must match pattern "audit-\\d+\\.\\d+"
    """
    coherence_metrics: CoherenceMetricsOutput
    contradictions: List[Contradiction]
    regulatory_analysis: RegulatoryAnalysisOutput
    plan_name: str
    schema_version: str


class AuditSummaryOutput(TypedDict):
    """Output contract for audit summary.
    
    Invariants:
    - overall_grade: One of ["Excelente", "Bueno", "Regular", "Malo"]
    - key_findings: List of main audit findings
    - recommendations: List of improvement recommendations
    - executive_summary: Brief summary text
    - schema_version: Must match input version
    """
    overall_grade: Literal["Excelente", "Bueno", "Regular", "Malo"]
    key_findings: List[str]
    recommendations: List[str]
    executive_summary: str
    schema_version: str


# ============================================================================
# Orchestrator Pipeline Contracts
# ============================================================================

class PipelineInput(TypedDict):
    """Input contract for full orchestration pipeline.
    
    Invariants:
    - text: Non-empty document text
    - plan_name: Non-empty plan identifier
    - dimension: Valid causal dimension
    - config: Optional configuration overrides
    - schema_version: Must match pattern "pipe-\\d+\\.\\d+"
    """
    text: str
    plan_name: str
    dimension: str
    config: Dict[str, Any]
    schema_version: str


class PipelineOutput(TypedDict):
    """Output contract for full orchestration pipeline.
    
    Invariants:
    - statements: Statement extraction results
    - contradictions: Contradiction detection results
    - coherence_metrics: Coherence calculation results
    - regulatory_analysis: Regulatory analysis results
    - audit_summary: Audit summary results
    - schema_version: Must match input version
    """
    statements: StatementExtractionOutput
    contradictions: ContradictionDetectionOutput
    coherence_metrics: CoherenceMetricsOutput
    regulatory_analysis: RegulatoryAnalysisOutput
    audit_summary: AuditSummaryOutput
    schema_version: str


# ============================================================================
# File I/O Contracts (for Ports & Adapters)
# ============================================================================

class FileReadRequest(TypedDict):
    """Request to read a file.
    
    Invariants:
    - path: Non-empty file path
    - encoding: Valid encoding name (default: "utf-8")
    """
    path: str
    encoding: str


class FileWriteRequest(TypedDict):
    """Request to write a file.
    
    Invariants:
    - path: Non-empty file path
    - content: Content to write (can be empty)
    - encoding: Valid encoding name (default: "utf-8")
    - create_dirs: Whether to create parent directories
    """
    path: str
    content: str
    encoding: str
    create_dirs: bool


class FileReadResult(TypedDict):
    """Result of file read operation.
    
    Invariants:
    - content: File content (can be empty)
    - path: Path that was read
    - success: Whether operation succeeded
    - error: Error message if success=False
    """
    content: str
    path: str
    success: bool
    error: Optional[str]


class FileWriteResult(TypedDict):
    """Result of file write operation.
    
    Invariants:
    - path: Path that was written
    - bytes_written: Number of bytes written (>= 0)
    - success: Whether operation succeeded
    - error: Error message if success=False
    """
    path: str
    bytes_written: int
    success: bool
    error: Optional[str]


# ============================================================================
# Version Constants
# ============================================================================

# Current schema versions for each contract type
CURRENT_VERSIONS = {
    "semantic": "sem-1.3",
    "contradiction": "cd-1.3",
    "embedding": "emb-1.3",
    "statement": "stmt-1.3",
    "coherence": "coh-1.3",
    "regulatory": "reg-1.3",
    "audit": "audit-1.3",
    "pipeline": "pipe-1.3",
}
