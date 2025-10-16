#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Questionnaire Parser Integration
Tests canonical source of truth integration across orchestration components
"""

from pathlib import Path

import pytest

from orchestrator import create_orchestrator
from questionnaire_parser import create_questionnaire_parser


def test_questionnaire_parser_initialization():
    """Test that questionnaire parser initializes correctly"""
    parser = create_questionnaire_parser()

    # Validate structure
    validation = parser.validate_structure()
    assert validation["valid"], f"Validation errors: {validation['errors']}"
    assert validation["policy_count"] == 10
    assert validation["question_count"] == 300


def test_questionnaire_parser_policies():
    """Test policy parsing"""
    parser = create_questionnaire_parser()

    policies = parser.get_all_policies()
    assert len(policies) == 10

    # Check first policy
    p1 = parser.get_policy("P1")
    assert p1 is not None
    assert p1.policy_id == "P1"
    assert "género" in p1.policy_name.lower()
    assert len(p1.dimensions) == 6


def test_questionnaire_parser_dimensions():
    """Test dimension parsing"""
    parser = create_questionnaire_parser()

    # Get dimension names
    dim_names = parser.get_dimension_names()
    assert len(dim_names) == 6
    assert "D1" in dim_names
    assert "D6" in dim_names

    # Check specific dimension
    dim = parser.get_dimension("P1", "D1")
    assert dim is not None
    assert dim.dimension_id == "D1"
    assert len(dim.questions) >= 5


def test_questionnaire_parser_questions():
    """Test question parsing"""
    parser = create_questionnaire_parser()

    # Test specific question
    q = parser.get_question("P1-D1-Q1")
    assert q is not None
    assert q.policy_id == "P1"
    assert q.dimension_id == "D1"
    assert q.question_id == "Q1"
    assert q.full_id == "P1-D1-Q1"
    assert len(q.text) > 0

    # Test questions by dimension
    questions = parser.get_questions_by_dimension("P1", "D1")
    assert len(questions) >= 5


def test_orchestrator_integration():
    """Test orchestrator uses questionnaire parser"""
    orchestrator = create_orchestrator()

    # Verify parser is initialized
    assert orchestrator.questionnaire_parser is not None

    # Test dimension lookup
    dim_name = orchestrator.get_dimension_description("D1")
    assert dim_name is not None
    assert len(dim_name) > 0

    # Test policy lookup
    policy_name = orchestrator.get_policy_description("P1")
    assert policy_name is not None
    assert "género" in policy_name.lower()

    # Test question lookup
    question = orchestrator.get_question("P1-D1-Q1")
    assert question is not None
    assert question.full_id == "P1-D1-Q1"


def test_canonical_path_tracking():
    """Test that canonical path is tracked in orchestrator metadata"""
    orchestrator = create_orchestrator()

    # Check metadata includes canonical path
    metadata = orchestrator._global_report["orchestration_metadata"]
    assert "canonical_questionnaire_path" in metadata

    canonical_path = Path(metadata["canonical_questionnaire_path"])
    assert canonical_path.exists()
    assert canonical_path.name == "cuestionario_canonico"


def test_no_legacy_sources():
    """Test that no alternative questionnaire sources exist"""
    repo_root = Path(__file__).parent

    # Should not find alternative cuestionario files
    json_files = list(repo_root.glob("**/cuestionario*.json"))
    txt_files = list(repo_root.glob("**/cuestionario*.txt"))

    # Only cuestionario_canonico should exist
    canonico_files = [f for f in txt_files if f.name == "cuestionario_canonico"]
    assert len(canonico_files) == 1, (
        "Should have exactly one cuestionario_canonico file"
    )


def test_deterministic_parsing():
    """Test that parser produces deterministic results"""
    parser1 = create_questionnaire_parser()
    parser2 = create_questionnaire_parser()

    # Get all questions from both parsers
    questions1 = parser1.get_all_questions()
    questions2 = parser2.get_all_questions()

    # Should have same count
    assert len(questions1) == len(questions2)

    # Should have same question IDs
    ids1 = sorted([q.full_id for q in questions1])
    ids2 = sorted([q.full_id for q in questions2])
    assert ids1 == ids2

    # Should have same text for same question
    q1 = parser1.get_question("P1-D1-Q1")
    q2 = parser2.get_question("P1-D1-Q1")
    assert q1.text == q2.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
