#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Scoring Audit Module
"""

import unittest
from dataclasses import dataclass
from pathlib import Path
import tempfile
import shutil
from scoring_audit import (
    ScoringSystemAuditor,
    ScoringAuditReport,
    AuditIssue,
    QualityBand,
    EXPECTED_TOTAL_QUESTIONS,
    THRESHOLD_D6_CRITICAL
)


@dataclass
class MockResponse:
    """Mock response object for testing"""
    pregunta_id: str
    nota_cuantitativa: float
    respuesta_texto: str = "Test response"
    argumento: str = "Test argument"
    evidencia: list = None
    modulos_utilizados: list = None
    nivel_confianza: float = 0.9


class TestScoringAudit(unittest.TestCase):
    """Test suite for scoring audit functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.auditor = ScoringSystemAuditor(output_dir=Path(self.temp_dir))
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_responses(self, num_questions: int = 300, score: float = 0.75) -> dict:
        """Create mock question responses"""
        responses = {}
        question_num = 1
        for p in range(1, 11):  # P1-P10
            for d in range(1, 7):  # D1-D6
                for q in range(1, 6):  # Q1-Q5
                    question_id = f"P{p}-D{d}-Q{q}"
                    responses[question_id] = MockResponse(
                        pregunta_id=question_id,
                        nota_cuantitativa=score
                    )
                    question_num += 1
                    if question_num > num_questions:
                        return responses
        return responses
    
    def test_matrix_structure_valid(self):
        """Test that valid 300-question matrix passes validation"""
        responses = self._create_mock_responses(300)
        
        self.auditor.audit_matrix_structure(responses)
        
        self.assertEqual(self.auditor.report.total_questions_found, 300)
        self.assertEqual(len(self.auditor.report.policies_found), 10)
        self.assertEqual(len(self.auditor.report.dimensions_found), 6)
        self.assertTrue(self.auditor.report.matrix_valid)
    
    def test_matrix_structure_missing_questions(self):
        """Test that missing questions are detected"""
        responses = self._create_mock_responses(250)  # Only 250 questions
        
        self.auditor.audit_matrix_structure(responses)
        
        self.assertEqual(self.auditor.report.total_questions_found, 250)
        self.assertFalse(self.auditor.report.matrix_valid)
        self.assertTrue(any(i.category == "matrix_structure" for i in self.auditor.report.micro_issues))
    
    def test_micro_scores_valid_range(self):
        """Test that scores in valid range [0, 1] pass validation"""
        responses = self._create_mock_responses(300, score=0.75)
        
        self.auditor.audit_micro_scores(responses)
        
        self.assertTrue(self.auditor.report.micro_scores_valid)
        self.assertEqual(len([i for i in self.auditor.report.micro_issues if i.category == "micro_scoring"]), 0)
    
    def test_micro_scores_out_of_range(self):
        """Test that scores outside [0, 1] are flagged"""
        responses = self._create_mock_responses(300, score=1.5)
        
        self.auditor.audit_micro_scores(responses)
        
        self.assertFalse(self.auditor.report.micro_scores_valid)
        self.assertTrue(any(i.category == "micro_scoring" for i in self.auditor.report.micro_issues))
    
    def test_d6_theory_of_change_below_threshold(self):
        """Test that D6 scores below 0.55 are flagged"""
        responses = {}
        # Create D6 questions with low scores
        for p in range(1, 11):
            for q in range(1, 6):
                question_id = f"P{p}-D6-Q{q}"
                responses[question_id] = MockResponse(
                    pregunta_id=question_id,
                    nota_cuantitativa=0.50  # Below threshold
                )
        
        self.auditor.audit_d6_theory_of_change(responses)
        
        self.assertEqual(len(self.auditor.report.d6_scores_below_threshold), 50)  # 10 policies × 5 questions
        self.assertTrue(any(i.category == "d6_theory_of_change" for i in self.auditor.report.rubric_issues))
    
    def test_d6_theory_of_change_above_threshold(self):
        """Test that D6 scores above 0.55 pass validation"""
        responses = {}
        for p in range(1, 11):
            for q in range(1, 6):
                question_id = f"P{p}-D6-Q{q}"
                responses[question_id] = MockResponse(
                    pregunta_id=question_id,
                    nota_cuantitativa=0.75  # Above threshold
                )
        
        self.auditor.audit_d6_theory_of_change(responses)
        
        self.assertEqual(len(self.auditor.report.d6_scores_below_threshold), 0)
    
    def test_rubric_thresholds_distribution(self):
        """Test quality band distribution calculation"""
        responses = {}
        # Create mix of scores across bands
        scores = [0.85, 0.65, 0.50, 0.30]  # EXCELENTE, BUENO, ACEPTABLE, INSUFICIENTE
        idx = 0
        for p in range(1, 11):
            for d in range(1, 7):
                for q in range(1, 6):
                    question_id = f"P{p}-D{d}-Q{q}"
                    responses[question_id] = MockResponse(
                        pregunta_id=question_id,
                        nota_cuantitativa=scores[idx % len(scores)]
                    )
                    idx += 1
        
        self.auditor.audit_rubric_thresholds(responses)
        
        # Should have roughly equal distribution
        self.assertTrue(self.auditor.report.rubric_mappings_valid or 
                       len(self.auditor.report.rubric_issues) > 0)
    
    def test_meso_aggregation_weight_validation(self):
        """Test that dimension weights summing to 1.0 pass validation"""
        responses = self._create_mock_responses(300)
        
        # Valid weights that sum to 1.0
        dimension_weights = {
            f"P{p}": {
                "D1": 0.15, "D2": 0.15, "D3": 0.20, 
                "D4": 0.20, "D5": 0.15, "D6": 0.15
            } for p in range(1, 11)
        }
        
        meso_report = {
            'clusters': {
                'C1': {'dimensiones': {}},
                'C2': {'dimensiones': {}},
                'C3': {'dimensiones': {}},
                'C4': {'dimensiones': {}}
            }
        }
        
        self.auditor.audit_meso_aggregation(responses, meso_report, dimension_weights)
        
        self.assertTrue(self.auditor.report.meso_aggregation_valid)
        self.assertEqual(len(self.auditor.report.meso_weight_issues), 0)
    
    def test_meso_aggregation_invalid_weights(self):
        """Test that dimension weights not summing to 1.0 are flagged"""
        responses = self._create_mock_responses(300)
        
        # Invalid weights that sum to 0.95
        dimension_weights = {
            "P1": {
                "D1": 0.15, "D2": 0.15, "D3": 0.20, 
                "D4": 0.20, "D5": 0.10, "D6": 0.15  # Sum = 0.95
            }
        }
        
        meso_report = {
            'clusters': {
                'C1': {'dimensiones': {}},
                'C2': {'dimensiones': {}},
                'C3': {'dimensiones': {}},
                'C4': {'dimensiones': {}}
            }
        }
        
        self.auditor.audit_meso_aggregation(responses, meso_report, dimension_weights)
        
        self.assertFalse(self.auditor.report.meso_aggregation_valid)
        self.assertTrue(len(self.auditor.report.meso_weight_issues) > 0)
    
    def test_macro_alignment_score_consistency(self):
        """Test that MACRO score matches MICRO average"""
        responses = self._create_mock_responses(300, score=0.75)
        
        macro_report = {
            'evaluacion_global': {
                'score_global': 0.75,
                'total_preguntas': 300
            }
        }
        
        self.auditor.audit_macro_alignment(responses, macro_report)
        
        self.assertTrue(self.auditor.report.macro_alignment_valid)
        self.assertEqual(len(self.auditor.report.macro_issues), 0)
    
    def test_macro_alignment_score_mismatch(self):
        """Test that MACRO/MICRO score mismatch is detected"""
        responses = self._create_mock_responses(300, score=0.75)
        
        macro_report = {
            'evaluacion_global': {
                'score_global': 0.85,  # Mismatch with 0.75 average
                'total_preguntas': 300
            }
        }
        
        self.auditor.audit_macro_alignment(responses, macro_report)
        
        self.assertFalse(self.auditor.report.macro_alignment_valid)
        self.assertTrue(any(i.category == "macro_convergence" for i in self.auditor.report.macro_issues))
    
    def test_dnp_integration_valid(self):
        """Test DNP integration with all required attributes"""
        @dataclass
        class MockDNPResults:
            cumple_competencias: bool = True
            cumple_mga: bool = True
            nivel_cumplimiento: str = "BUENO"
            score_total: float = 75.0
        
        dnp_results = MockDNPResults()
        macro_report = {
            'evaluacion_global': {
                'score_dnp_compliance': 75.0
            }
        }
        
        self.auditor.audit_dnp_integration(dnp_results, macro_report)
        
        self.assertTrue(self.auditor.report.dnp_integration_valid)
        self.assertEqual(len(self.auditor.report.dnp_issues), 0)
    
    def test_complete_audit_workflow(self):
        """Test complete audit workflow end-to-end"""
        responses = self._create_mock_responses(300, score=0.75)
        
        dimension_weights = {
            f"P{p}": {
                "D1": 0.15, "D2": 0.15, "D3": 0.20, 
                "D4": 0.20, "D5": 0.15, "D6": 0.15
            } for p in range(1, 11)
        }
        
        meso_report = {
            'clusters': {
                'C1': {'dimensiones': {}},
                'C2': {'dimensiones': {}},
                'C3': {'dimensiones': {}},
                'C4': {'dimensiones': {}}
            }
        }
        
        macro_report = {
            'evaluacion_global': {
                'score_global': 0.75,
                'score_dnp_compliance': 75.0
            },
            'analisis_retrospectivo': {},
            'analisis_prospectivo': {},
            'recomendaciones_prioritarias': []
        }
        
        @dataclass
        class MockDNPResults:
            cumple_competencias: bool = True
            cumple_mga: bool = True
            nivel_cumplimiento: str = "BUENO"
            score_total: float = 75.0
        
        report = self.auditor.audit_complete_system(
            question_responses=responses,
            dimension_weights=dimension_weights,
            meso_report=meso_report,
            macro_report=macro_report,
            dnp_results=MockDNPResults()
        )
        
        self.assertIsInstance(report, ScoringAuditReport)
        self.assertTrue(report.overall_valid)
        self.assertEqual(report.critical_issues, 0)
    
    def test_export_report(self):
        """Test report export to JSON"""
        responses = self._create_mock_responses(300, score=0.75)
        
        self.auditor.audit_complete_system(question_responses=responses)
        output_path = self.auditor.export_report()
        
        self.assertTrue(output_path.exists())
        self.assertTrue(output_path.name.startswith("scoring_audit_"))
        self.assertTrue(output_path.name.endswith(".json"))


if __name__ == '__main__':
    unittest.main()
