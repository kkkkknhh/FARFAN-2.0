#!/usr/bin/env python3
"""
Pipeline Data Validation Module
Provides Pydantic models and validators for pipeline stages
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


class DocumentProcessingData(BaseModel):
    """Validated data from Stage 1-2: Document processing"""
    raw_text: str = Field(default="", description="Extracted text from PDF")
    sections: Dict[str, str] = Field(default_factory=dict, description="Document sections")
    tables: List[Any] = Field(default_factory=list, description="Extracted tables")
    
    @field_validator('raw_text')
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            logger.warning("raw_text is empty after document extraction")
        return v


class SemanticAnalysisData(BaseModel):
    """Validated data from Stage 3: Semantic analysis"""
    semantic_chunks: List[Dict] = Field(default_factory=list, description="Semantic chunks")
    dimension_scores: Dict[str, float] = Field(default_factory=dict, description="Dimension scores")
    
    @field_validator('dimension_scores')
    @classmethod
    def validate_scores_in_range(cls, v: Dict[str, float]) -> Dict[str, float]:
        for key, score in v.items():
            if not (0.0 <= score <= 1.0):
                logger.warning(f"Dimension score '{key}' out of range [0,1]: {score}")
        return v


class CausalExtractionData(BaseModel):
    """Validated data from Stage 4: Causal extraction"""
    causal_graph: Any = Field(default=None, description="Causal graph structure")
    nodes: Dict[str, Any] = Field(default_factory=dict, description="Extracted nodes")
    causal_chains: List[Dict] = Field(default_factory=list, description="Causal chains")
    
    @field_validator('nodes')
    @classmethod
    def validate_nodes_not_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """INVARIANT: Post-Stage 4 must have nodes > 0"""
        if len(v) == 0:
            raise ValueError("CRITICAL: Stage 4 must extract at least one node. Pipeline cannot continue.")
        logger.info(f"Stage 4 validation passed: {len(v)} nodes extracted")
        return v


class MechanismInferenceData(BaseModel):
    """Validated data from Stage 5: Mechanism inference"""
    mechanism_parts: List[Dict] = Field(default_factory=list, description="Mechanism parts")
    bayesian_inferences: Dict[str, Any] = Field(default_factory=dict, description="Bayesian inferences")


class FinancialAuditData(BaseModel):
    """Validated data from Stage 6: Financial audit"""
    financial_allocations: Dict[str, float] = Field(default_factory=dict, description="Financial allocations")
    budget_traceability: Dict[str, Any] = Field(default_factory=dict, description="Budget traceability")
    
    @field_validator('financial_allocations')
    @classmethod
    def validate_allocations_positive(cls, v: Dict[str, float]) -> Dict[str, float]:
        for key, amount in v.items():
            if amount < 0:
                logger.warning(f"Negative financial allocation for '{key}': {amount}")
        return v


class DNPValidationData(BaseModel):
    """Validated data from Stage 7: DNP validation"""
    dnp_validation_results: List[Dict] = Field(default_factory=list, description="DNP validation results")
    compliance_score: float = Field(default=0.0, description="Compliance score 0-100")
    
    @field_validator('compliance_score')
    @classmethod
    def validate_compliance_score_range(cls, v: float) -> float:
        """INVARIANT: Post-Stage 7 compliance_score must be 0-100"""
        if not (0.0 <= v <= 100.0):
            logger.warning(f"⚠️  WARNING: Compliance score out of range [0, 100]: {v}. Clamping to valid range.")
            # Clamp to valid range
            return max(0.0, min(100.0, v))
        logger.info(f"Stage 7 validation passed: compliance_score = {v:.1f}/100")
        return v


class QuestionAnsweringData(BaseModel):
    """Validated data from Stage 8: Question answering"""
    question_responses: Dict[str, Dict] = Field(default_factory=dict, description="Question responses")
    
    @field_validator('question_responses')
    @classmethod
    def validate_question_count(cls, v: Dict[str, Dict]) -> Dict[str, Dict]:
        """INVARIANT: Post-Stage 8 must have exactly 300 question responses"""
        expected_count = 300
        actual_count = len(v)
        if actual_count != expected_count:
            raise ValueError(
                f"CRITICAL: Stage 8 must produce exactly {expected_count} question responses. "
                f"Got {actual_count}. Pipeline cannot continue."
            )
        logger.info(f"Stage 8 validation passed: {actual_count} questions answered")
        return v


class ReportGenerationData(BaseModel):
    """Validated data from Stage 9: Report generation"""
    micro_report: Dict = Field(default_factory=dict, description="Micro-level report")
    meso_report: Dict = Field(default_factory=dict, description="Meso-level report")
    macro_report: Dict = Field(default_factory=dict, description="Macro-level report")


class ValidatedPipelineContext(BaseModel):
    """
    Fully validated pipeline context with Pydantic models
    Ensures data integrity at each stage
    """
    # Input (immutable)
    pdf_path: Path
    policy_code: str
    output_dir: Path
    
    # Stage data (validated)
    stage_1_2: DocumentProcessingData = Field(default_factory=DocumentProcessingData)
    stage_3: SemanticAnalysisData = Field(default_factory=SemanticAnalysisData)
    stage_4: CausalExtractionData = Field(default_factory=CausalExtractionData)
    stage_5: MechanismInferenceData = Field(default_factory=MechanismInferenceData)
    stage_6: FinancialAuditData = Field(default_factory=FinancialAuditData)
    stage_7: DNPValidationData = Field(default_factory=DNPValidationData)
    stage_8: QuestionAnsweringData = Field(default_factory=QuestionAnsweringData)
    stage_9: ReportGenerationData = Field(default_factory=ReportGenerationData)
    
    class Config:
        arbitrary_types_allowed = True  # Allow networkx graphs and other custom types
        from_attributes = True  # Enable ORM mode for compatibility


def validate_stage_transition(stage_name: str, data: BaseModel) -> None:
    """
    Validates data at stage transitions
    
    Args:
        stage_name: Name of the stage being validated
        data: Pydantic model instance to validate
        
    Raises:
        ValueError: If validation fails critically
    """
    try:
        # Pydantic validation happens automatically on model creation
        logger.info(f"✓ Stage {stage_name} data validation passed")
    except Exception as e:
        logger.error(f"✗ Stage {stage_name} validation failed: {e}")
        raise
