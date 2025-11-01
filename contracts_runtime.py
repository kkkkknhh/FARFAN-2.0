#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runtime Contract Validators
============================

Pydantic models that mirror core_contracts.py TypedDict definitions.
Used for runtime validation at architectural boundaries.

Design Principles:
- Mirror every TypedDict in core_contracts.py
- Enforce all invariants with validators
- Provide clear error messages
- Fail fast on invalid data
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import re


# ============================================================================
# Semantic Analysis Validators
# ============================================================================

class SemanticAnalyzerInputModel(BaseModel):
    """Runtime validator for SemanticAnalyzerInput contract."""
    
    text: str = Field(min_length=1, description="Non-empty document text")
    segments: List[str] = Field(default_factory=list, description="Optional pre-segmented chunks")
    ontology_params: Dict[str, Any] = Field(default_factory=dict, description="Optional ontology parameters")
    schema_version: str = Field(pattern=r"^sem-\d+\.\d+$", description="Schema version (e.g., sem-1.3)")
    
    model_config = {
        "extra": "forbid",  # Strict mode: refuse unknown fields
    }


class SemanticChunkModel(BaseModel):
    """Runtime validator for SemanticChunk contract."""
    
    id: str = Field(min_length=1, description="Unique chunk identifier")
    text: str = Field(min_length=1, description="Chunk text content")
    start_pos: int = Field(ge=0, description="Start position in document")
    end_pos: int = Field(gt=0, description="End position in document")
    embedding: Optional[List[float]] = Field(default=None, description="Optional embedding vector")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = {
        "extra": "forbid",
    }
    
    @model_validator(mode='after')
    def validate_positions(self):
        """Ensure end_pos > start_pos."""
        if self.end_pos <= self.start_pos:
            raise ValueError(f"end_pos ({self.end_pos}) must be > start_pos ({self.start_pos})")
        return self
    
    @field_validator('embedding')
    @classmethod
    def validate_embedding_dimension(cls, v):
        """Validate embedding dimension if present."""
        if v is not None and len(v) not in [768, 1024]:
            raise ValueError(f"Embedding dimension must be 768 or 1024, got {len(v)}")
        return v


class SemanticAnalyzerOutputModel(BaseModel):
    """Runtime validator for SemanticAnalyzerOutput contract."""
    
    chunks: List[SemanticChunkModel] = Field(default_factory=list, description="Semantic chunks")
    coherence_score: float = Field(ge=0.0, le=1.0, description="Coherence score [0.0, 1.0]")
    quality_metrics: Dict[str, Any] = Field(description="Quality metrics dictionary")
    schema_version: str = Field(pattern=r"^sem-\d+\.\d+$", description="Schema version")
    
    model_config = {
        "extra": "forbid",
    }


# ============================================================================
# Contradiction Detection Validators
# ============================================================================

class ContradictionDetectionInputModel(BaseModel):
    """Runtime validator for ContradictionDetectionInput contract."""
    
    statements: List[str] = Field(min_length=0, description="Policy statements to analyze")
    text: str = Field(min_length=1, description="Full document text")
    plan_name: str = Field(min_length=1, description="Plan identifier")
    dimension: str = Field(min_length=1, description="Causal dimension")
    schema_version: str = Field(pattern=r"^cd-\d+\.\d+$", description="Schema version")
    
    model_config = {
        "extra": "forbid",
    }


class ContradictionModel(BaseModel):
    """Runtime validator for Contradiction contract."""
    
    statement1: str = Field(min_length=1, description="First contradicting statement")
    statement2: str = Field(min_length=1, description="Second contradicting statement")
    severity: Literal["critical", "high", "medium", "low"] = Field(description="Severity level")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score [0.0, 1.0]")
    explanation: str = Field(min_length=1, description="Explanation of contradiction")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = {
        "extra": "forbid",
    }


class ContradictionDetectionOutputModel(BaseModel):
    """Runtime validator for ContradictionDetectionOutput contract."""
    
    contradictions: List[ContradictionModel] = Field(default_factory=list, description="Detected contradictions")
    total_count: int = Field(ge=0, description="Total contradiction count")
    quality_grade: Literal["Excelente", "Bueno", "Regular", "Malo"] = Field(description="Quality grade")
    schema_version: str = Field(pattern=r"^cd-\d+\.\d+$", description="Schema version")
    
    model_config = {
        "extra": "forbid",
    }
    
    @model_validator(mode='after')
    def validate_count(self):
        """Ensure total_count matches len(contradictions)."""
        if self.total_count != len(self.contradictions):
            raise ValueError(
                f"total_count ({self.total_count}) must equal len(contradictions) ({len(self.contradictions)})"
            )
        return self


# ============================================================================
# Embedding Validators
# ============================================================================

class EmbeddingInputModel(BaseModel):
    """Runtime validator for EmbeddingInput contract."""
    
    texts: List[str] = Field(min_length=1, description="Non-empty list of texts")
    model_name: str = Field(min_length=1, description="Model identifier")
    batch_size: int = Field(gt=0, description="Batch size > 0")
    normalize: bool = Field(description="Whether to normalize embeddings")
    schema_version: str = Field(pattern=r"^emb-\d+\.\d+$", description="Schema version")
    
    model_config = {
        "extra": "forbid",
    }


class EmbeddingOutputModel(BaseModel):
    """Runtime validator for EmbeddingOutput contract."""
    
    embeddings: List[List[float]] = Field(description="List of embedding vectors")
    dimension: int = Field(gt=0, description="Embedding dimension")
    model_name: str = Field(min_length=1, description="Model used")
    schema_version: str = Field(pattern=r"^emb-\d+\.\d+$", description="Schema version")
    
    model_config = {
        "extra": "forbid",
    }


# ============================================================================
# Policy Statement Extraction Validators
# ============================================================================

class PolicyStatementModel(BaseModel):
    """Runtime validator for PolicyStatement contract."""
    
    text: str = Field(min_length=1, description="Statement text")
    section: str = Field(min_length=1, description="Document section")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    causal_dimension: str = Field(min_length=1, description="Causal dimension")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    
    model_config = {
        "extra": "forbid",
    }


class StatementExtractionInputModel(BaseModel):
    """Runtime validator for StatementExtractionInput contract."""
    
    text: str = Field(min_length=1, description="Document text")
    plan_name: str = Field(min_length=1, description="Plan identifier")
    extract_all: bool = Field(description="Extract all or filter by confidence")
    schema_version: str = Field(pattern=r"^stmt-\d+\.\d+$", description="Schema version")
    
    model_config = {
        "extra": "forbid",
    }


class StatementExtractionOutputModel(BaseModel):
    """Runtime validator for StatementExtractionOutput contract."""
    
    statements: List[PolicyStatementModel] = Field(default_factory=list, description="Extracted statements")
    total_count: int = Field(ge=0, description="Total statement count")
    schema_version: str = Field(pattern=r"^stmt-\d+\.\d+$", description="Schema version")
    
    model_config = {
        "extra": "forbid",
    }
    
    @model_validator(mode='after')
    def validate_count(self):
        """Ensure total_count matches len(statements)."""
        if self.total_count != len(self.statements):
            raise ValueError(
                f"total_count ({self.total_count}) must equal len(statements) ({len(self.statements)})"
            )
        return self


# ============================================================================
# Coherence Metrics Validators
# ============================================================================

class CoherenceMetricsInputModel(BaseModel):
    """Runtime validator for CoherenceMetricsInput contract."""
    
    contradictions: List[ContradictionModel] = Field(default_factory=list, description="Contradictions")
    statements: List[PolicyStatementModel] = Field(default_factory=list, description="Statements")
    text: str = Field(min_length=1, description="Document text")
    schema_version: str = Field(pattern=r"^coh-\d+\.\d+$", description="Schema version")
    
    model_config = {
        "extra": "forbid",
    }


class CoherenceMetricsOutputModel(BaseModel):
    """Runtime validator for CoherenceMetricsOutput contract."""
    
    coherence_score: float = Field(ge=0.0, le=1.0, description="Coherence score")
    causal_incoherence_count: int = Field(ge=0, description="Incoherence count")
    quality_status: Literal["Aceptable", "Requiere revisión"] = Field(description="Quality status")
    detailed_metrics: Dict[str, Any] = Field(default_factory=dict, description="Detailed metrics")
    schema_version: str = Field(pattern=r"^coh-\d+\.\d+$", description="Schema version")
    
    model_config = {
        "extra": "forbid",
    }


# ============================================================================
# Regulatory Constraint Analysis Validators
# ============================================================================

class RegulatoryConstraintModel(BaseModel):
    """Runtime validator for RegulatoryConstraint contract."""
    
    constraint_type: str = Field(min_length=1, description="Constraint type")
    description: str = Field(min_length=1, description="Description")
    severity: Literal["critical", "high", "medium", "low"] = Field(description="Severity")
    compliant: bool = Field(description="Compliance status")
    evidence: str = Field(description="Supporting evidence")
    
    model_config = {
        "extra": "forbid",
    }


class RegulatoryAnalysisInputModel(BaseModel):
    """Runtime validator for RegulatoryAnalysisInput contract."""
    
    statements: List[PolicyStatementModel] = Field(default_factory=list, description="Statements")
    text: str = Field(min_length=1, description="Document text")
    plan_name: str = Field(min_length=1, description="Plan identifier")
    schema_version: str = Field(pattern=r"^reg-\d+\.\d+$", description="Schema version")
    
    model_config = {
        "extra": "forbid",
    }


class RegulatoryAnalysisOutputModel(BaseModel):
    """Runtime validator for RegulatoryAnalysisOutput contract."""
    
    constraints: List[RegulatoryConstraintModel] = Field(default_factory=list, description="Constraints")
    compliance_score: float = Field(ge=0.0, le=1.0, description="Compliance score")
    critical_violations: int = Field(ge=0, description="Critical violation count")
    schema_version: str = Field(pattern=r"^reg-\d+\.\d+$", description="Schema version")
    
    model_config = {
        "extra": "forbid",
    }


# ============================================================================
# Audit Summary Validators
# ============================================================================

class AuditSummaryInputModel(BaseModel):
    """Runtime validator for AuditSummaryInput contract."""
    
    coherence_metrics: CoherenceMetricsOutputModel = Field(description="Coherence metrics")
    contradictions: List[ContradictionModel] = Field(default_factory=list, description="Contradictions")
    regulatory_analysis: RegulatoryAnalysisOutputModel = Field(description="Regulatory analysis")
    plan_name: str = Field(min_length=1, description="Plan identifier")
    schema_version: str = Field(pattern=r"^audit-\d+\.\d+$", description="Schema version")
    
    model_config = {
        "extra": "forbid",
    }


class AuditSummaryOutputModel(BaseModel):
    """Runtime validator for AuditSummaryOutput contract."""
    
    overall_grade: Literal["Excelente", "Bueno", "Regular", "Malo"] = Field(description="Overall grade")
    key_findings: List[str] = Field(default_factory=list, description="Key findings")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    executive_summary: str = Field(min_length=1, description="Executive summary")
    schema_version: str = Field(pattern=r"^audit-\d+\.\d+$", description="Schema version")
    
    model_config = {
        "extra": "forbid",
    }


# ============================================================================
# Orchestrator Pipeline Validators
# ============================================================================

class PipelineInputModel(BaseModel):
    """Runtime validator for PipelineInput contract."""
    
    text: str = Field(min_length=1, description="Document text")
    plan_name: str = Field(min_length=1, description="Plan identifier")
    dimension: str = Field(min_length=1, description="Causal dimension")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration")
    schema_version: str = Field(pattern=r"^pipe-\d+\.\d+$", description="Schema version")
    
    model_config = {
        "extra": "forbid",
    }


class PipelineOutputModel(BaseModel):
    """Runtime validator for PipelineOutput contract."""
    
    statements: StatementExtractionOutputModel = Field(description="Statement extraction results")
    contradictions: ContradictionDetectionOutputModel = Field(description="Contradiction detection results")
    coherence_metrics: CoherenceMetricsOutputModel = Field(description="Coherence metrics results")
    regulatory_analysis: RegulatoryAnalysisOutputModel = Field(description="Regulatory analysis results")
    audit_summary: AuditSummaryOutputModel = Field(description="Audit summary results")
    schema_version: str = Field(pattern=r"^pipe-\d+\.\d+$", description="Schema version")
    
    model_config = {
        "extra": "forbid",
    }


# ============================================================================
# File I/O Validators
# ============================================================================

class FileReadRequestModel(BaseModel):
    """Runtime validator for FileReadRequest contract."""
    
    path: str = Field(min_length=1, description="File path")
    encoding: str = Field(default="utf-8", description="File encoding")
    
    model_config = {
        "extra": "forbid",
    }


class FileWriteRequestModel(BaseModel):
    """Runtime validator for FileWriteRequest contract."""
    
    path: str = Field(min_length=1, description="File path")
    content: str = Field(description="Content to write")
    encoding: str = Field(default="utf-8", description="File encoding")
    create_dirs: bool = Field(default=False, description="Create parent directories")
    
    model_config = {
        "extra": "forbid",
    }


class FileReadResultModel(BaseModel):
    """Runtime validator for FileReadResult contract."""
    
    content: str = Field(description="File content")
    path: str = Field(min_length=1, description="File path")
    success: bool = Field(description="Success status")
    error: Optional[str] = Field(default=None, description="Error message")
    
    model_config = {
        "extra": "forbid",
    }


class FileWriteResultModel(BaseModel):
    """Runtime validator for FileWriteResult contract."""
    
    path: str = Field(min_length=1, description="File path")
    bytes_written: int = Field(ge=0, description="Bytes written")
    success: bool = Field(description="Success status")
    error: Optional[str] = Field(default=None, description="Error message")
    
    model_config = {
        "extra": "forbid",
    }
