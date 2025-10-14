#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Pipeline Validators
Validates Pydantic models and invariant checks
"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from pipeline_validators import (
    DocumentProcessingData,
    SemanticAnalysisData,
    CausalExtractionData,
    MechanismInferenceData,
    FinancialAuditData,
    DNPValidationData,
    QuestionAnsweringData,
    ReportGenerationData,
    ValidatedPipelineContext
)


class TestDocumentProcessingData:
    """Test Stage 1-2 validation"""
    
    def test_valid_document_data(self):
        """Test valid document processing data"""
        data = DocumentProcessingData(
            raw_text="Sample text content",
            sections={"intro": "Introduction text"},
            tables=[{"col1": "value1"}]
        )
        assert data.raw_text == "Sample text content"
        assert len(data.sections) == 1
        assert len(data.tables) == 1
    
    def test_empty_text_warning(self, caplog):
        """Test that empty text generates warning"""
        data = DocumentProcessingData(
            raw_text="",
            sections={},
            tables=[]
        )
        # Should not raise error, but may log warning
        assert data.raw_text == ""


class TestCausalExtractionData:
    """Test Stage 4 validation - Critical invariant"""
    
    def test_valid_nodes(self):
        """Test valid node extraction"""
        data = CausalExtractionData(
            causal_graph=None,
            nodes={"node1": {"type": "outcome"}, "node2": {"type": "output"}},
            causal_chains=[]
        )
        assert len(data.nodes) == 2
    
    def test_empty_nodes_raises_error(self):
        """Test that empty nodes raises ValueError (INVARIANT)"""
        with pytest.raises(ValidationError) as exc_info:
            CausalExtractionData(
                causal_graph=None,
                nodes={},  # Empty nodes - should fail
                causal_chains=[]
            )
        assert "at least one node" in str(exc_info.value).lower()
    
    def test_single_node_valid(self):
        """Test that single node is valid"""
        data = CausalExtractionData(
            causal_graph=None,
            nodes={"node1": {"type": "outcome"}},
            causal_chains=[]
        )
        assert len(data.nodes) == 1


class TestDNPValidationData:
    """Test Stage 7 validation - Compliance score range"""
    
    def test_valid_compliance_score(self):
        """Test valid compliance scores"""
        for score in [0.0, 50.0, 100.0]:
            data = DNPValidationData(
                dnp_validation_results=[],
                compliance_score=score
            )
            assert data.compliance_score == score
    
    def test_out_of_range_score_clamped(self):
        """Test that out-of-range scores are clamped"""
        # Score above 100
        data_high = DNPValidationData(
            dnp_validation_results=[],
            compliance_score=150.0
        )
        assert data_high.compliance_score == 100.0
        
        # Score below 0
        data_low = DNPValidationData(
            dnp_validation_results=[],
            compliance_score=-50.0
        )
        assert data_low.compliance_score == 0.0
    
    def test_edge_cases(self):
        """Test edge cases for compliance score"""
        data = DNPValidationData(
            dnp_validation_results=[],
            compliance_score=99.999
        )
        assert 0.0 <= data.compliance_score <= 100.0


class TestQuestionAnsweringData:
    """Test Stage 8 validation - 300 questions invariant"""
    
    def test_exact_300_questions(self):
        """Test that exactly 300 questions is valid"""
        questions = {f"Q{i}": {"answer": "test"} for i in range(300)}
        data = QuestionAnsweringData(question_responses=questions)
        assert len(data.question_responses) == 300
    
    def test_less_than_300_raises_error(self):
        """Test that less than 300 questions raises error"""
        questions = {f"Q{i}": {"answer": "test"} for i in range(299)}
        with pytest.raises(ValidationError) as exc_info:
            QuestionAnsweringData(question_responses=questions)
        assert "exactly 300" in str(exc_info.value).lower()
    
    def test_more_than_300_raises_error(self):
        """Test that more than 300 questions raises error"""
        questions = {f"Q{i}": {"answer": "test"} for i in range(301)}
        with pytest.raises(ValidationError) as exc_info:
            QuestionAnsweringData(question_responses=questions)
        assert "exactly 300" in str(exc_info.value).lower()
    
    def test_empty_questions_raises_error(self):
        """Test that empty questions raises error"""
        with pytest.raises(ValidationError) as exc_info:
            QuestionAnsweringData(question_responses={})
        assert "exactly 300" in str(exc_info.value).lower()


class TestFinancialAuditData:
    """Test Stage 6 validation"""
    
    def test_positive_allocations(self):
        """Test valid positive allocations"""
        data = FinancialAuditData(
            financial_allocations={"project1": 1000.0, "project2": 2000.0},
            budget_traceability={"total": 3000.0}
        )
        assert data.financial_allocations["project1"] == 1000.0
    
    def test_negative_allocation_warning(self, caplog):
        """Test that negative allocations generate warning"""
        data = FinancialAuditData(
            financial_allocations={"project1": -500.0},
            budget_traceability={}
        )
        # Should not raise error, but may log warning
        assert data.financial_allocations["project1"] == -500.0


class TestSemanticAnalysisData:
    """Test Stage 3 validation"""
    
    def test_dimension_scores_in_range(self):
        """Test valid dimension scores"""
        data = SemanticAnalysisData(
            semantic_chunks=[],
            dimension_scores={"D1": 0.5, "D2": 0.8, "D3": 1.0}
        )
        assert all(0.0 <= score <= 1.0 for score in data.dimension_scores.values())
    
    def test_out_of_range_scores_warning(self, caplog):
        """Test that out-of-range scores generate warning"""
        data = SemanticAnalysisData(
            semantic_chunks=[],
            dimension_scores={"D1": 1.5, "D2": -0.5}
        )
        # Should not raise error, but may log warning
        assert data.dimension_scores["D1"] == 1.5


class TestValidatedPipelineContext:
    """Test full pipeline context validation"""
    
    def test_create_empty_context(self):
        """Test creating empty validated context"""
        ctx = ValidatedPipelineContext(
            pdf_path=Path("/test/path.pdf"),
            policy_code="TEST-001",
            output_dir=Path("/output")
        )
        assert ctx.policy_code == "TEST-001"
        assert isinstance(ctx.stage_1_2, DocumentProcessingData)
    
    def test_stage_data_validation(self):
        """Test that stage data is validated"""
        ctx = ValidatedPipelineContext(
            pdf_path=Path("/test/path.pdf"),
            policy_code="TEST-001",
            output_dir=Path("/output")
        )
        
        # Update stage 1-2 data
        ctx.stage_1_2 = DocumentProcessingData(
            raw_text="Test content",
            sections={"intro": "test"},
            tables=[]
        )
        assert ctx.stage_1_2.raw_text == "Test content"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
