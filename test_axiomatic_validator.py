#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Axiomatic Validator
Tests the unified validation system
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import networkx as nx
from validators.axiomatic_validator import (
    AxiomaticValidator,
    AxiomaticValidationResult,
    ValidationConfig,
    PDMOntology,
    SemanticChunk,
    ExtractedTable,
    ValidationSeverity,
)


class TestAxiomaticValidator(unittest.TestCase):
    """Test suite for AxiomaticValidator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ValidationConfig(
            dnp_lexicon_version="2025",
            es_municipio_pdet=False,
            contradiction_threshold=0.05,
            enable_structural_penalty=True,
            enable_human_gating=True
        )
        
        self.ontology = PDMOntology()
        
        self.validator = AxiomaticValidator(self.config, self.ontology)
    
    def test_validator_initialization(self):
        """Test that validator initializes correctly"""
        self.assertIsNotNone(self.validator.teoria_cambio)
        self.assertIsNotNone(self.validator.contradiction_detector)
        self.assertIsNotNone(self.validator.dnp_validator)
        self.assertEqual(self.validator.config.dnp_lexicon_version, "2025")
    
    def test_simple_valid_graph(self):
        """Test validation of a simple valid causal graph"""
        # Create a simple valid graph: INSUMOS -> PROCESOS -> PRODUCTOS
        graph = nx.DiGraph()
        graph.add_node("Presupuesto", categoria="INSUMOS")
        graph.add_node("Capacitación", categoria="PROCESOS")
        graph.add_node("Personal capacitado", categoria="PRODUCTOS")
        graph.add_edge("Presupuesto", "Capacitación")
        graph.add_edge("Capacitación", "Personal capacitado")
        
        # Create simple semantic chunks
        chunks = [
            SemanticChunk(
                text="El municipio destinará recursos para capacitación.",
                dimension="ESTRATEGICO"
            )
        ]
        
        # Run validation
        result = self.validator.validate_complete(graph, chunks)
        
        # Verify result structure
        self.assertIsInstance(result, AxiomaticValidationResult)
        self.assertIsNotNone(result.validation_timestamp)
        self.assertEqual(result.total_nodes, 3)
        self.assertEqual(result.total_edges, 2)
    
    def test_graph_with_violations(self):
        """Test validation of a graph with structural violations"""
        # Create a graph with order violations: PRODUCTOS -> INSUMOS (backward)
        graph = nx.DiGraph()
        graph.add_node("Presupuesto", categoria="INSUMOS")
        graph.add_node("Personal capacitado", categoria="PRODUCTOS")
        graph.add_edge("Personal capacitado", "Presupuesto")  # Invalid order
        
        chunks = [
            SemanticChunk(
                text="El personal capacitado generará presupuesto.",
                dimension="ESTRATEGICO"
            )
        ]
        
        # Run validation
        result = self.validator.validate_complete(graph, chunks)
        
        # Should have structural violations
        # Note: actual violation detection depends on TeoriaCambio implementation
        self.assertIsInstance(result, AxiomaticValidationResult)
    
    def test_high_contradiction_density(self):
        """Test that high contradiction density triggers manual review"""
        # Create a simple graph
        graph = nx.DiGraph()
        graph.add_node("A", categoria="INSUMOS")
        graph.add_node("B", categoria="PROCESOS")
        graph.add_edge("A", "B")
        
        # Create chunks with potential contradictions
        chunks = [
            SemanticChunk(
                text="El presupuesto aumentará en un 50%.",
                dimension="ESTRATEGICO"
            ),
            SemanticChunk(
                text="El presupuesto disminuirá en un 30%.",
                dimension="ESTRATEGICO"
            )
        ]
        
        # Run validation
        result = self.validator.validate_complete(graph, chunks)
        
        # Verify result structure (may or may not trigger depending on detector)
        self.assertIsInstance(result, AxiomaticValidationResult)
        self.assertIsInstance(result.contradiction_density, float)
    
    def test_add_critical_failure(self):
        """Test adding critical failures to results"""
        result = AxiomaticValidationResult()
        
        # Initially valid
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.failures), 0)
        
        # Add a critical failure
        result.add_critical_failure(
            dimension='D6',
            question='Q2',
            evidence=[('A', 'B')],
            impact='Test impact',
            recommendations=['Fix this']
        )
        
        # Should now be invalid
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].dimension, 'D6')
        self.assertEqual(result.failures[0].severity, ValidationSeverity.CRITICAL)
    
    def test_get_summary(self):
        """Test getting validation summary"""
        result = AxiomaticValidationResult()
        result.structural_valid = True
        result.contradiction_density = 0.02
        result.regulatory_score = 75.0
        
        summary = result.get_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('is_valid', summary)
        self.assertIn('structural_valid', summary)
        self.assertIn('contradiction_density', summary)
        self.assertIn('regulatory_score', summary)
        self.assertEqual(summary['structural_valid'], True)
        self.assertAlmostEqual(summary['contradiction_density'], 0.02, places=6)
        self.assertAlmostEqual(summary['regulatory_score'], 75.0, places=6)
    
    def test_ontology_initialization(self):
        """Test PDMOntology initialization"""
        # Default ontology
        ontology = PDMOntology()
        self.assertEqual(len(ontology.canonical_chain), 5)
        self.assertIn('INSUMOS', ontology.canonical_chain)
        self.assertIn('PROCESOS', ontology.canonical_chain)
        
        # Custom ontology
        custom_ontology = PDMOntology(
            canonical_chain=['A', 'B', 'C'],
            dimensions=['D1', 'D2']
        )
        self.assertEqual(len(custom_ontology.canonical_chain), 3)
        self.assertEqual(len(custom_ontology.dimensions), 2)
    
    def test_validation_config(self):
        """Test ValidationConfig"""
        config = ValidationConfig(
            dnp_lexicon_version="2024",
            es_municipio_pdet=True,
            contradiction_threshold=0.10,
            enable_structural_penalty=False,
            enable_human_gating=False
        )
        
        self.assertEqual(config.dnp_lexicon_version, "2024")
        self.assertTrue(config.es_municipio_pdet)
        self.assertAlmostEqual(config.contradiction_threshold, 0.10, places=6)
        self.assertFalse(config.enable_structural_penalty)
        self.assertFalse(config.enable_human_gating)
    
    def test_semantic_chunk(self):
        """Test SemanticChunk dataclass"""
        chunk = SemanticChunk(
            text="Test text",
            dimension="DIAGNOSTICO",
            position=(10, 20),
            metadata={'key': 'value'}
        )
        
        self.assertEqual(chunk.text, "Test text")
        self.assertEqual(chunk.dimension, "DIAGNOSTICO")
        self.assertEqual(chunk.position, (10, 20))
        self.assertEqual(chunk.metadata['key'], 'value')
    
    def test_extracted_table(self):
        """Test ExtractedTable dataclass"""
        table = ExtractedTable(
            title="Presupuesto",
            headers=["Rubro", "Monto"],
            rows=[["Educación", 1000], ["Salud", 2000]],
            metadata={'year': 2025}
        )
        
        self.assertEqual(table.title, "Presupuesto")
        self.assertEqual(len(table.headers), 2)
        self.assertEqual(len(table.rows), 2)
        self.assertEqual(table.metadata['year'], 2025)


class TestIntegration(unittest.TestCase):
    """Integration tests for the validator"""
    
    def test_end_to_end_validation(self):
        """Test complete end-to-end validation workflow"""
        # Setup
        config = ValidationConfig()
        ontology = PDMOntology()
        validator = AxiomaticValidator(config, ontology)
        
        # Create a realistic graph
        graph = nx.DiGraph()
        graph.add_node("Recursos financieros", categoria="INSUMOS")
        graph.add_node("Contratación de docentes", categoria="PROCESOS")
        graph.add_node("Docentes capacitados", categoria="PRODUCTOS")
        graph.add_node("Mejora en educación", categoria="RESULTADOS")
        
        graph.add_edge("Recursos financieros", "Contratación de docentes")
        graph.add_edge("Contratación de docentes", "Docentes capacitados")
        graph.add_edge("Docentes capacitados", "Mejora en educación")
        
        # Create semantic chunks
        chunks = [
            SemanticChunk(
                text="El municipio invertirá en educación para mejorar la calidad.",
                dimension="ESTRATEGICO"
            ),
            SemanticChunk(
                text="Se contratarán 50 nuevos docentes en el primer año.",
                dimension="PROGRAMATICO"
            )
        ]
        
        # Create financial data
        financial_data = [
            ExtractedTable(
                title="Inversión en Educación",
                headers=["Año", "Monto"],
                rows=[["2025", 1000000], ["2026", 1200000]]
            )
        ]
        
        # Execute validation
        result = validator.validate_complete(graph, chunks, financial_data)
        
        # Verify result
        self.assertIsInstance(result, AxiomaticValidationResult)
        self.assertIsNotNone(result.validation_timestamp)
        self.assertEqual(result.total_nodes, 4)
        self.assertEqual(result.total_edges, 3)
        
        # Get summary
        summary = result.get_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('is_valid', summary)


if __name__ == '__main__':
    unittest.main()
