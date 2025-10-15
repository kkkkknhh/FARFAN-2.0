#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for convergence verification
===================================

Tests the convergence verification between cuestionario_canonico, 
questions_config.json, and guia_cuestionario.json.

Author: AI Systems Architect
Version: 1.0.0
"""

import unittest
import json
from pathlib import Path
from verify_convergence import ConvergenceVerifier, ConvergenceIssue


class TestConvergenceVerifier(unittest.TestCase):
    """Test suite for ConvergenceVerifier"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.verifier = ConvergenceVerifier()
    
    def test_verifier_initialization(self):
        """Test that verifier initializes correctly"""
        self.assertIsNotNone(self.verifier.questions_config)
        self.assertIsNotNone(self.verifier.guia_cuestionario)
        self.assertIsNotNone(self.verifier.cuestionario_canonico_text)
        
        # Check expected structure
        self.assertEqual(self.verifier.expected_policies, 10)
        self.assertEqual(self.verifier.expected_dimensions, 6)
        self.assertEqual(self.verifier.expected_questions_per_dim, 5)
        self.assertEqual(self.verifier.expected_total_questions, 300)
    
    def test_questions_config_loaded(self):
        """Test that questions_config.json is loaded correctly"""
        config = self.verifier.questions_config
        
        # Should have metadata
        self.assertIn('metadata', config)
        metadata = config['metadata']
        self.assertIn('total_questions', metadata)
        self.assertEqual(metadata['total_questions'], 300)
        
        # Should have dimensions
        self.assertIn('dimensiones', config)
        dimensions = config['dimensiones']
        self.assertEqual(len(dimensions), 6)
        
        # Should have base questions
        self.assertIn('preguntas_base', config)
        base_questions = config['preguntas_base']
        self.assertEqual(len(base_questions), 30)
    
    def test_guia_cuestionario_loaded(self):
        """Test that guia_cuestionario is loaded correctly"""
        guia = self.verifier.guia_cuestionario
        
        # Should have metadata
        self.assertIn('metadata', guia)
        
        # Should have decalogo_dimension_mapping
        self.assertIn('decalogo_dimension_mapping', guia)
        mapping = guia['decalogo_dimension_mapping']
        self.assertEqual(len(mapping), 10, "Should have mapping for all 10 policies")
        
        # Check all policies are present
        for i in range(1, 11):
            policy_id = f"P{i}"
            self.assertIn(policy_id, mapping, f"Policy {policy_id} should be in mapping")
        
        # Should have causal_verification_templates
        self.assertIn('causal_verification_templates', guia)
        templates = guia['causal_verification_templates']
        self.assertEqual(len(templates), 6, "Should have templates for all 6 dimensions")
        
        # Should have scoring_system
        self.assertIn('scoring_system', guia)
    
    def test_cuestionario_canonico_loaded(self):
        """Test that cuestionario_canonico is loaded correctly"""
        text = self.verifier.cuestionario_canonico_text
        
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        
        # Should contain canonical notation
        import re
        pattern = r'P\d{1,2}-D\d-Q\d{1,2}'
        matches = re.findall(pattern, text)
        self.assertGreater(len(matches), 0, "Should contain canonical notation")
    
    def test_canonical_notation_validation(self):
        """Test canonical notation validation"""
        # Run the verification
        self.verifier.verify_canonical_notation_usage()
        
        # Check for critical issues
        critical_issues = [i for i in self.verifier.issues if i.severity == "CRITICAL"]
        
        # Should have no critical canonical notation issues
        notation_issues = [i for i in critical_issues 
                          if i.issue_type == "invalid_canonical_notation"]
        self.assertEqual(len(notation_issues), 0, 
                        "Should have no invalid canonical notation")
    
    def test_scoring_consistency(self):
        """Test scoring consistency verification"""
        # Run the verification
        self.verifier.verify_scoring_consistency()
        
        # Check that we don't have missing scoring issues
        scoring_issues = [i for i in self.verifier.issues 
                         if i.issue_type in ['missing_score_range', 'incomplete_scoring']]
        
        # It's OK to have some scoring issues but not too many
        self.assertLess(len(scoring_issues), 50, 
                       "Should not have excessive scoring issues")
    
    def test_dimension_mapping_validation(self):
        """Test dimension mapping validation"""
        # Run the verification
        self.verifier.verify_dimension_mapping()
        
        # Check for policy ID issues
        policy_issues = [i for i in self.verifier.issues 
                        if i.issue_type == "invalid_policy_id"]
        self.assertEqual(len(policy_issues), 0, 
                        "Should have no invalid policy IDs")
        
        # Check for weight sum issues
        weight_issues = [i for i in self.verifier.issues 
                        if i.issue_type == "invalid_weight_sum"]
        self.assertEqual(len(weight_issues), 0, 
                        "All dimension weights should sum to 1.0")
    
    def test_no_legacy_mappings(self):
        """Test that there are no legacy file mappings"""
        # Run the verification
        self.verifier.verify_no_legacy_mapping()
        
        # Check for legacy mapping issues
        legacy_issues = [i for i in self.verifier.issues 
                        if i.issue_type == "legacy_mapping_found"]
        self.assertEqual(len(legacy_issues), 0, 
                        "Should have no legacy file mappings")
    
    def test_module_references(self):
        """Test module reference validation"""
        # Run the verification
        self.verifier.verify_module_references()
        
        # Check for unknown module issues
        module_issues = [i for i in self.verifier.issues 
                        if i.issue_type == "unknown_module_reference"]
        self.assertEqual(len(module_issues), 0, 
                        "Should only reference known modules")
    
    def test_full_verification(self):
        """Test full verification run"""
        # Create a fresh verifier to avoid accumulated issues
        verifier = ConvergenceVerifier()
        
        # Run full verification
        report = verifier.run_full_verification()
        
        # Check report structure
        self.assertIn('convergence_issues', report)
        self.assertIn('recommendations', report)
        self.assertIn('verification_summary', report)
        
        summary = report['verification_summary']
        self.assertIn('percent_questions_converged', summary)
        self.assertIn('issues_detected', summary)
        self.assertIn('critical_issues', summary)
        self.assertIn('total_questions_expected', summary)
        
        # Check expected values
        self.assertEqual(summary['total_questions_expected'], 300)
        
        # Should have high convergence
        self.assertGreaterEqual(summary['percent_questions_converged'], 95.0,
                               "Should have at least 95% convergence")
        
        # Should not have critical issues
        self.assertEqual(summary['critical_issues'], 0,
                        "Should have zero critical issues")
    
    def test_convergence_issue_dataclass(self):
        """Test ConvergenceIssue dataclass"""
        issue = ConvergenceIssue(
            question_id="P1-D1-Q1",
            issue_type="test_issue",
            description="Test description",
            suggested_fix="Test fix",
            severity="MEDIUM"
        )
        
        # Check to_dict conversion
        issue_dict = issue.to_dict()
        self.assertIn('question_id', issue_dict)
        self.assertIn('issue_type', issue_dict)
        self.assertIn('description', issue_dict)
        self.assertIn('suggested_fix', issue_dict)
        self.assertIn('severity', issue_dict)
        
        self.assertEqual(issue_dict['question_id'], "P1-D1-Q1")
        self.assertEqual(issue_dict['severity'], "MEDIUM")
    
    def test_report_saved(self):
        """Test that convergence report is saved"""
        report_path = Path("/home/runner/work/FARFAN-2.0/FARFAN-2.0/convergence_report.json")
        self.assertTrue(report_path.exists(), "Convergence report should be saved")
        
        # Load and verify report
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        self.assertIn('convergence_issues', report)
        self.assertIn('recommendations', report)
        self.assertIn('verification_summary', report)


if __name__ == '__main__':
    unittest.main()
