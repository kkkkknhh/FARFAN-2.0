#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal structure tests for Axiomatic Validator
Tests basic structure without requiring heavy dependencies
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest


class TestValidatorStructure(unittest.TestCase):
    """Test the basic structure of the validator module"""
    
    def test_module_exists(self):
        """Test that the validators module exists"""
        import validators
        self.assertIsNotNone(validators)
    
    def test_imports_available(self):
        """Test that expected classes can be imported"""
        from validators import (
            AxiomaticValidator,
            AxiomaticValidationResult,
            ValidationConfig,
            PDMOntology,
            SemanticChunk,
            ExtractedTable,
        )
        
        # Verify they are importable
        self.assertIsNotNone(AxiomaticValidator)
        self.assertIsNotNone(AxiomaticValidationResult)
        self.assertIsNotNone(ValidationConfig)
        self.assertIsNotNone(PDMOntology)
        self.assertIsNotNone(SemanticChunk)
        self.assertIsNotNone(ExtractedTable)
    
    def test_validation_config_creation(self):
        """Test creating a ValidationConfig"""
        from validators import ValidationConfig
        
        config = ValidationConfig(
            dnp_lexicon_version="2025",
            es_municipio_pdet=False,
            contradiction_threshold=0.05
        )
        
        self.assertEqual(config.dnp_lexicon_version, "2025")
        self.assertFalse(config.es_municipio_pdet)
        self.assertEqual(config.contradiction_threshold, 0.05)
        self.assertTrue(config.enable_structural_penalty)
        self.assertTrue(config.enable_human_gating)
    
    def test_pdm_ontology_creation(self):
        """Test creating a PDMOntology"""
        from validators import PDMOntology
        
        ontology = PDMOntology()
        
        # Should have default canonical chain
        self.assertIsNotNone(ontology.canonical_chain)
        self.assertEqual(len(ontology.canonical_chain), 5)
        self.assertIn('INSUMOS', ontology.canonical_chain)
        self.assertIn('PROCESOS', ontology.canonical_chain)
        self.assertIn('PRODUCTOS', ontology.canonical_chain)
        self.assertIn('RESULTADOS', ontology.canonical_chain)
        self.assertIn('CAUSALIDAD', ontology.canonical_chain)
    
    def test_semantic_chunk_creation(self):
        """Test creating a SemanticChunk"""
        from validators import SemanticChunk
        
        chunk = SemanticChunk(
            text="Test text",
            dimension="ESTRATEGICO",
            position=(10, 20)
        )
        
        self.assertEqual(chunk.text, "Test text")
        self.assertEqual(chunk.dimension, "ESTRATEGICO")
        self.assertEqual(chunk.position, (10, 20))
        self.assertIsInstance(chunk.metadata, dict)
    
    def test_extracted_table_creation(self):
        """Test creating an ExtractedTable"""
        from validators import ExtractedTable
        
        table = ExtractedTable(
            title="Test Table",
            headers=["Column A", "Column B"],
            rows=[["Value 1", "Value 2"]]
        )
        
        self.assertEqual(table.title, "Test Table")
        self.assertEqual(len(table.headers), 2)
        self.assertEqual(len(table.rows), 1)
        self.assertIsInstance(table.metadata, dict)
    
    def test_validation_result_creation(self):
        """Test creating an AxiomaticValidationResult"""
        from validators import AxiomaticValidationResult
        
        result = AxiomaticValidationResult()
        
        # Check default values
        self.assertTrue(result.is_valid)
        self.assertTrue(result.structural_valid)
        self.assertEqual(result.contradiction_density, 0.0)
        self.assertEqual(result.regulatory_score, 0.0)
        self.assertFalse(result.requires_manual_review)
        self.assertIsNone(result.hold_reason)
        self.assertEqual(len(result.failures), 0)
    
    def test_validation_result_summary(self):
        """Test getting a summary from validation result"""
        from validators import AxiomaticValidationResult
        
        result = AxiomaticValidationResult()
        result.structural_valid = False
        result.contradiction_density = 0.08
        result.regulatory_score = 65.0
        
        summary = result.get_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('is_valid', summary)
        self.assertIn('structural_valid', summary)
        self.assertIn('contradiction_density', summary)
        self.assertIn('regulatory_score', summary)
        self.assertIn('critical_failures', summary)
        self.assertIn('requires_manual_review', summary)
        
        self.assertEqual(summary['structural_valid'], False)
        self.assertEqual(summary['contradiction_density'], 0.08)
        self.assertEqual(summary['regulatory_score'], 65.0)
    
    def test_add_critical_failure(self):
        """Test adding a critical failure"""
        from validators import AxiomaticValidationResult, ValidationSeverity
        
        result = AxiomaticValidationResult()
        
        # Initially valid with no failures
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.failures), 0)
        
        # Add a critical failure
        result.add_critical_failure(
            dimension='D6',
            question='Q2',
            evidence=[('Node1', 'Node2')],
            impact='Structural violation detected',
            recommendations=['Review causal order']
        )
        
        # Should now be invalid with one failure
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.failures), 1)
        
        # Verify failure details
        failure = result.failures[0]
        self.assertEqual(failure.dimension, 'D6')
        self.assertEqual(failure.question, 'Q2')
        self.assertEqual(failure.severity, ValidationSeverity.CRITICAL)
        self.assertEqual(failure.impact, 'Structural violation detected')
        self.assertEqual(len(failure.recommendations), 1)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
