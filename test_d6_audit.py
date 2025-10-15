#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for D6 Audit Module
Tests the structural coherence and adaptive learning audit
"""

import os
import sys
import unittest
from pathlib import Path

import networkx as nx

from validators.d6_audit import (
    D1Q5D6Q5RestrictionsResult,
    D6AuditOrchestrator,
    D6AuditReport,
    D6Q1AxiomaticResult,
    D6Q3InconsistencyResult,
    D6Q4AdaptiveMEResult,
    execute_d6_audit,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestD6AuditModule(unittest.TestCase):
    """Test suite for D6 Audit Module"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary log directory
        self.log_dir = Path("./test_logs/d6_audit")
        self.orchestrator = D6AuditOrchestrator(log_dir=self.log_dir)

        # Sample PDM text for testing
        self.sample_text = """
        PLAN DE DESARROLLO MUNICIPAL 2024-2027
        
        COMPONENTE ESTRATÉGICO
        
        Eje 1: Educación de Calidad
        Objetivo: Aumentar la cobertura educativa al 95% para el año 2027.
        
        El municipio destinará recursos para capacitación de docentes.
        Se construirán 5 nuevas instituciones educativas.
        
        Este plan está sujeto a la Ley 152 de 1994 y el Decreto 1082 de 2015.
        Las restricciones presupuestales limitan la ejecución a los recursos del SGP.
        El horizonte temporal del cuatrienio establece plazos claros.
        La competencia municipal está definida en la Ley 1551 de 2012.
        
        Se implementará un plan piloto en 2025 para validar el modelo educativo.
        
        PLAN PLURIANUAL DE INVERSIONES
        Recursos SGP Educación: $1,500 millones anuales.
        Capacidad financiera municipal: límite fiscal establecido.
        """

    def test_orchestrator_initialization(self):
        """Test that orchestrator initializes correctly"""
        self.assertIsNotNone(self.orchestrator)
        self.assertTrue(self.log_dir.exists())

    def test_d6_q1_axiomatic_validation_complete_graph(self):
        """Test D6-Q1 with a complete valid graph"""
        # Create a complete valid graph with all 5 elements
        graph = nx.DiGraph()

        # Add nodes for all 5 categories
        from teoria_cambio import CategoriaCausal

        graph.add_node("Presupuesto", categoria=CategoriaCausal.INSUMOS)
        graph.add_node("Capacitación", categoria=CategoriaCausal.PROCESOS)
        graph.add_node("Docentes capacitados", categoria=CategoriaCausal.PRODUCTOS)
        graph.add_node("Mejora educativa", categoria=CategoriaCausal.RESULTADOS)
        graph.add_node("Teoría validada", categoria=CategoriaCausal.CAUSALIDAD)

        # Add edges in correct order
        graph.add_edge("Presupuesto", "Capacitación")
        graph.add_edge("Capacitación", "Docentes capacitados")
        graph.add_edge("Docentes capacitados", "Mejora educativa")
        graph.add_edge("Mejora educativa", "Teoría validada")

        # Test D6-Q1
        result = self.orchestrator._audit_d6_q1_axiomatic_validation(graph)

        # Verify structure
        self.assertIsInstance(result, D6Q1AxiomaticResult)

        # Check criteria
        self.assertTrue(result.has_five_elements, "Should have all 5 elements")
        self.assertEqual(
            len(result.elements_present), 5, "Should have 5 elements present"
        )
        self.assertEqual(
            len(result.elements_missing), 0, "Should have no missing elements"
        )
        self.assertTrue(
            result.violaciones_orden_empty, "Should have no order violations"
        )
        self.assertTrue(
            result.caminos_completos_exist, "Should have at least one complete path"
        )

        # Quality should be Excelente
        self.assertEqual(
            result.quality_grade,
            "Excelente",
            "Complete graph should get Excelente grade",
        )

    def test_d6_q1_axiomatic_validation_incomplete_graph(self):
        """Test D6-Q1 with incomplete graph (missing elements)"""
        # Create incomplete graph (missing CAUSALIDAD)
        graph = nx.DiGraph()

        from teoria_cambio import CategoriaCausal

        graph.add_node("Presupuesto", categoria=CategoriaCausal.INSUMOS)
        graph.add_node("Capacitación", categoria=CategoriaCausal.PROCESOS)
        graph.add_node("Docentes capacitados", categoria=CategoriaCausal.PRODUCTOS)

        graph.add_edge("Presupuesto", "Capacitación")
        graph.add_edge("Capacitación", "Docentes capacitados")

        # Test D6-Q1
        result = self.orchestrator._audit_d6_q1_axiomatic_validation(graph)

        # Should not have all 5 elements
        self.assertFalse(result.has_five_elements, "Should not have all 5 elements")
        self.assertTrue(
            len(result.elements_missing) > 0, "Should have missing elements"
        )

        # Quality should not be Excelente
        self.assertNotEqual(
            result.quality_grade,
            "Excelente",
            "Incomplete graph should not get Excelente grade",
        )

    def test_d6_q3_inconsistency_recognition(self):
        """Test D6-Q3 inconsistency recognition"""
        # Create mock contradiction results
        contradiction_results = {
            "total_contradictions": 3,
            "harmonic_front_4_audit": {
                "causal_incoherence_flags": 2,
                "total_contradictions": 3,
                "quality_grade": "Excelente",
            },
            "contradictions": [
                {"contradiction_type": "CAUSAL_INCOHERENCE"},
                {"contradiction_type": "CAUSAL_INCOHERENCE"},
                {"contradiction_type": "NUMERICAL_INCONSISTENCY"},
            ],
            "coherence_metrics": {"contradiction_density": 0.03},
        }

        # Test D6-Q3
        result = self.orchestrator._audit_d6_q3_inconsistency_recognition(
            text=self.sample_text,
            plan_name="PDM_Test",
            dimension="estratégico",
            contradiction_results=contradiction_results,
        )

        # Verify structure
        self.assertIsInstance(result, D6Q3InconsistencyResult)

        # Check criteria
        self.assertEqual(
            result.causal_incoherence_count,
            2,
            "Should detect 2 causal incoherence flags",
        )
        self.assertTrue(result.flags_below_limit, "Should be below limit of 5")
        self.assertTrue(result.has_pilot_testing, "Should detect pilot testing mention")

        # Quality should be Excelente (< 5 flags + pilot testing)
        self.assertEqual(
            result.quality_grade,
            "Excelente",
            "Should get Excelente with <5 flags and pilot testing",
        )

    def test_d6_q3_high_incoherence(self):
        """Test D6-Q3 with high incoherence count"""
        # Create mock contradiction results with high incoherence
        contradiction_results = {
            "total_contradictions": 10,
            "harmonic_front_4_audit": {
                "causal_incoherence_flags": 7,
                "total_contradictions": 10,
            },
        }

        # Test D6-Q3
        result = self.orchestrator._audit_d6_q3_inconsistency_recognition(
            text="No pilot testing mentioned",
            plan_name="PDM_Test",
            dimension="estratégico",
            contradiction_results=contradiction_results,
        )

        # Should not be below limit
        self.assertFalse(
            result.flags_below_limit, "Should not be below limit with 7 flags"
        )

        # Quality should be Regular
        self.assertEqual(
            result.quality_grade, "Regular", "Should get Regular with >= 5 flags"
        )

    def test_d6_q4_adaptive_me_with_learning(self):
        """Test D6-Q4 with adaptive learning evidence"""
        # Create mock contradiction results with correction mechanism
        contradiction_results = {
            "recommendations": [
                {"priority": "crítica", "description": "Fix causal loop"},
                {"priority": "alta", "description": "Reduce contradictions"},
            ],
            "harmonic_front_4_audit": {"total_contradictions": 3},
        }

        # Create mock prior history showing learning
        prior_history = [
            {
                "timestamp": "2024-01-01",
                "mechanism_type_priors": {
                    "politico": 0.4,
                    "economico": 0.3,
                    "social": 0.3,
                },
            },
            {
                "timestamp": "2024-02-01",
                "mechanism_type_priors": {
                    "politico": 0.35,  # Reduced due to failures
                    "economico": 0.35,
                    "social": 0.30,
                },
            },
            {
                "timestamp": "2024-03-01",
                "mechanism_type_priors": {
                    "politico": 0.30,  # Further reduced
                    "economico": 0.40,
                    "social": 0.30,
                },
            },
        ]

        # Test D6-Q4
        result = self.orchestrator._audit_d6_q4_adaptive_me_system(
            contradiction_results=contradiction_results, prior_history=prior_history
        )

        # Verify structure
        self.assertIsInstance(result, D6Q4AdaptiveMEResult)

        # Check criteria
        self.assertTrue(
            result.has_correction_mechanism, "Should detect correction mechanism"
        )
        self.assertTrue(
            result.has_feedback_mechanism, "Should detect feedback mechanism"
        )
        self.assertTrue(result.prior_updates_detected, "Should detect prior updates")
        self.assertIsNotNone(
            result.uncertainty_reduction, "Should calculate uncertainty reduction"
        )

        # Uncertainty reduction should be positive
        if result.uncertainty_reduction is not None:
            self.assertGreaterEqual(
                result.uncertainty_reduction,
                0,
                "Uncertainty reduction should be non-negative",
            )

    def test_d1_q5_d6_q5_restrictions_analysis(self):
        """Test D1-Q5, D6-Q5 contextual restrictions analysis"""
        # Test with sample text containing multiple restriction types
        result = self.orchestrator._audit_d1_q5_d6_q5_contextual_restrictions(
            text=self.sample_text, contradiction_results=None
        )

        # Verify structure
        self.assertIsInstance(result, D1Q5D6Q5RestrictionsResult)

        # Check criteria
        self.assertGreaterEqual(
            result.restriction_count, 3, "Should detect at least 3 restriction types"
        )
        self.assertTrue(
            result.meets_minimum_threshold,
            "Should meet minimum threshold of 3 restrictions",
        )

        # Should detect legal constraints (Ley 152, Decreto 1082, Ley 1551)
        self.assertTrue(
            len(result.legal_constraints) > 0, "Should detect legal constraints"
        )

        # Quality should be Excelente (≥3 restrictions + temporal consistency)
        self.assertIn(
            result.quality_grade,
            ["Excelente", "Bueno"],
            "Should get Excelente or Bueno with ≥3 restrictions",
        )

    def test_full_audit_execution(self):
        """Test complete D6 audit execution"""
        # Create a complete causal graph
        graph = nx.DiGraph()

        from teoria_cambio import CategoriaCausal

        graph.add_node("Presupuesto", categoria=CategoriaCausal.INSUMOS)
        graph.add_node("Capacitación", categoria=CategoriaCausal.PROCESOS)
        graph.add_node("Docentes capacitados", categoria=CategoriaCausal.PRODUCTOS)
        graph.add_node("Mejora educativa", categoria=CategoriaCausal.RESULTADOS)
        graph.add_node("Teoría validada", categoria=CategoriaCausal.CAUSALIDAD)

        graph.add_edge("Presupuesto", "Capacitación")
        graph.add_edge("Capacitación", "Docentes capacitados")
        graph.add_edge("Docentes capacitados", "Mejora educativa")
        graph.add_edge("Mejora educativa", "Teoría validada")

        # Create mock contradiction results
        contradiction_results = {
            "total_contradictions": 2,
            "harmonic_front_4_audit": {
                "causal_incoherence_flags": 1,
                "total_contradictions": 2,
            },
            "recommendations": [
                {"priority": "alta", "description": "Test recommendation"}
            ],
            "coherence_metrics": {"contradiction_density": 0.02},
            "d1_q5_regulatory_analysis": {
                "constraint_types_detected": {
                    "Legal": 2,
                    "Budgetary": 1,
                    "Temporal": 1,
                },
                "is_consistent": True,
                "regulatory_references": ["Ley 152 de 1994", "Decreto 1082"],
            },
        }

        # Execute full audit
        report = self.orchestrator.execute_full_audit(
            causal_graph=graph,
            text=self.sample_text,
            plan_name="PDM_Test",
            dimension="estratégico",
            contradiction_results=contradiction_results,
            prior_history=None,
        )

        # Verify report structure
        self.assertIsInstance(report, D6AuditReport)

        # Verify all components are present
        self.assertIsInstance(report.d6_q1_axiomatic, D6Q1AxiomaticResult)
        self.assertIsInstance(report.d6_q3_inconsistency, D6Q3InconsistencyResult)
        self.assertIsInstance(report.d6_q4_adaptive_me, D6Q4AdaptiveMEResult)
        self.assertIsInstance(
            report.d1_q5_d6_q5_restrictions, D1Q5D6Q5RestrictionsResult
        )

        # Verify metadata
        self.assertEqual(report.plan_name, "PDM_Test")
        self.assertEqual(report.dimension, "estratégico")
        self.assertIsNotNone(report.timestamp)
        self.assertIsNotNone(report.overall_quality)

        # Verify overall assessment
        self.assertIn(report.overall_quality, ["Excelente", "Bueno", "Regular"])
        self.assertIsInstance(report.meets_sota_standards, bool)
        self.assertIsInstance(report.critical_issues, list)
        self.assertIsInstance(report.actionable_recommendations, list)

    def test_convenience_function(self):
        """Test convenience function execute_d6_audit"""
        # Create a simple graph
        graph = nx.DiGraph()

        from teoria_cambio import CategoriaCausal

        graph.add_node("Input", categoria=CategoriaCausal.INSUMOS)
        graph.add_node("Process", categoria=CategoriaCausal.PROCESOS)

        graph.add_edge("Input", "Process")

        # Execute audit via convenience function
        report = execute_d6_audit(
            causal_graph=graph,
            text=self.sample_text,
            plan_name="Test_PDM",
            dimension="estratégico",
        )

        # Verify it returns a valid report
        self.assertIsInstance(report, D6AuditReport)
        self.assertEqual(report.plan_name, "Test_PDM")

    def test_overall_assessment_calculation(self):
        """Test overall assessment calculation logic"""
        # Create mock results
        d6_q1 = D6Q1AxiomaticResult(
            has_five_elements=True,
            elements_present=[
                "INSUMOS",
                "PROCESOS",
                "PRODUCTOS",
                "RESULTADOS",
                "CAUSALIDAD",
            ],
            elements_missing=[],
            violaciones_orden_empty=True,
            violaciones_orden_count=0,
            caminos_completos_exist=True,
            caminos_completos_count=1,
            quality_grade="Excelente",
            evidence={},
            recommendations=[],
        )

        d6_q3 = D6Q3InconsistencyResult(
            causal_incoherence_count=2,
            total_contradictions=3,
            flags_below_limit=True,
            has_pilot_testing=True,
            pilot_testing_mentions=[],
            quality_grade="Excelente",
            evidence={},
            recommendations=[],
        )

        d6_q4 = D6Q4AdaptiveMEResult(
            has_correction_mechanism=True,
            has_feedback_mechanism=True,
            mechanism_types_tracked=["politico", "economico"],
            prior_updates_detected=True,
            learning_loop_evidence={},
            uncertainty_reduction=0.10,
            quality_grade="Excelente",
            evidence={},
            recommendations=[],
        )

        d1_q5_d6_q5 = D1Q5D6Q5RestrictionsResult(
            restriction_types_detected=["Legal", "Budgetary", "Temporal"],
            restriction_count=3,
            meets_minimum_threshold=True,
            temporal_consistency=True,
            legal_constraints=[],
            budgetary_constraints=[],
            temporal_constraints=[],
            competency_constraints=[],
            quality_grade="Excelente",
            evidence={},
            recommendations=[],
        )

        # Calculate overall assessment
        overall_quality, meets_sota, critical_issues, recommendations = (
            self.orchestrator._calculate_overall_assessment(
                d6_q1, d6_q3, d6_q4, d1_q5_d6_q5
            )
        )

        # All excellent should give overall excellent
        self.assertEqual(overall_quality, "Excelente")

        # Should meet SOTA standards
        self.assertTrue(meets_sota)

        # Should have no critical issues
        self.assertEqual(len(critical_issues), 0)

    def tearDown(self):
        """Clean up test artifacts"""
        # Clean up test logs if needed
        import shutil

        if self.log_dir.exists():
            try:
                shutil.rmtree(self.log_dir)
            except Exception:
                pass  # Ignore cleanup errors


class TestD6AuditQualityCriteria(unittest.TestCase):
    """Test suite specifically for quality criteria validation"""

    def test_d6_q1_criteria_excelente(self):
        """Test D6-Q1 Excelente criteria"""
        # For Excelente: 5 elements + no violations + complete paths
        graph = nx.DiGraph()
        from teoria_cambio import CategoriaCausal

        for i, cat in enumerate(CategoriaCausal):
            graph.add_node(f"Node_{i}", categoria=cat)

        # Create complete path
        nodes = list(graph.nodes())
        for i in range(len(nodes) - 1):
            graph.add_edge(nodes[i], nodes[i + 1])

        orchestrator = D6AuditOrchestrator()
        result = orchestrator._audit_d6_q1_axiomatic_validation(graph)

        self.assertEqual(result.quality_grade, "Excelente")

    def test_d6_q3_criteria_excelente(self):
        """Test D6-Q3 Excelente criteria"""
        # For Excelente: <5 causal_incoherence + pilot testing
        contradiction_results = {
            "total_contradictions": 3,
            "harmonic_front_4_audit": {"causal_incoherence_flags": 2},
        }

        text_with_pilot = "Implementaremos un plan piloto en 2025"

        orchestrator = D6AuditOrchestrator()
        result = orchestrator._audit_d6_q3_inconsistency_recognition(
            text=text_with_pilot,
            plan_name="Test",
            dimension="estratégico",
            contradiction_results=contradiction_results,
        )

        self.assertTrue(result.flags_below_limit)
        self.assertTrue(result.has_pilot_testing)
        self.assertEqual(result.quality_grade, "Excelente")

    def test_d6_q4_criteria_excelente(self):
        """Test D6-Q4 Excelente criteria"""
        # For Excelente: correction + feedback + prior updates + ≥5% reduction
        contradiction_results = {
            "recommendations": [{"description": "test"}],
            "harmonic_front_4_audit": {"total_contradictions": 2},
        }

        prior_history = [
            {"mechanism_type_priors": {"a": 0.5, "b": 0.5}},
            {"mechanism_type_priors": {"a": 0.4, "b": 0.6}},  # Entropy reduced
        ]

        orchestrator = D6AuditOrchestrator()
        result = orchestrator._audit_d6_q4_adaptive_me_system(
            contradiction_results=contradiction_results, prior_history=prior_history
        )

        self.assertTrue(result.has_correction_mechanism)
        self.assertTrue(result.has_feedback_mechanism)
        self.assertTrue(result.prior_updates_detected)
        # Note: actual uncertainty reduction depends on entropy calculation

    def test_d1_q5_d6_q5_criteria_excelente(self):
        """Test D1-Q5, D6-Q5 Excelente criteria"""
        # For Excelente: ≥3 restriction types + temporal consistency
        text_with_restrictions = """
        Ley 152 de 1994 establece el marco normativo.
        Restricción presupuestal por límite fiscal.
        Horizonte temporal del cuatrienio.
        Competencia administrativa municipal.
        """

        orchestrator = D6AuditOrchestrator()
        result = orchestrator._audit_d1_q5_d6_q5_contextual_restrictions(
            text=text_with_restrictions, contradiction_results=None
        )

        self.assertGreaterEqual(result.restriction_count, 3)
        self.assertTrue(result.meets_minimum_threshold)
        # Temporal consistency is True by default if no conflicts


if __name__ == "__main__":
    unittest.main()
