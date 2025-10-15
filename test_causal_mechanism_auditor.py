#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for Causal Mechanism Auditor
Part 2: Analytical D3, D6 Audit
"""

import unittest
from unittest.mock import Mock, patch

import networkx as nx
import numpy as np

from audits.causal_mechanism_auditor import (
    ActivityLogicResult,
    CausalMechanismAuditor,
    CausalProportionalityResult,
    MechanismNecessityResult,
    QualityGrade,
    RootCauseMappingResult,
)


class TestCausalMechanismAuditor(unittest.TestCase):
    """Test cases for CausalMechanismAuditor"""

    def setUp(self):
        """Set up test fixtures"""
        self.auditor = CausalMechanismAuditor()

        # Create a sample graph
        self.graph = nx.DiGraph()

        # Add nodes with different types
        self.graph.add_node(
            "PROG-001",
            type="programa",
            text="Diagnóstico de necesidades",
            responsible_entity="Secretaría de Planeación",
        )
        self.graph.add_node(
            "PROD-001",
            type="producto",
            text="Construcción de infraestructura",
            responsible_entity="Secretaría de Obras",
            financial_allocation=1000000,
            entity_activity={
                "entity": "Secretaría de Obras",
                "activity": "construcción",
                "verb_lemma": "construir",
            },
        )
        self.graph.add_node(
            "RES-001", type="resultado", text="Mejora en cobertura de servicios"
        )
        self.graph.add_node(
            "IMP-001", type="impacto", text="Desarrollo sostenible regional"
        )

        # Add edges
        self.graph.add_edge(
            "PROG-001",
            "PROD-001",
            logic="para atender",
            keyword="para",
            posterior_mean=0.85,
            strength=0.85,
        )
        self.graph.add_edge(
            "PROD-001",
            "RES-001",
            logic="mediante",
            keyword="mediante",
            posterior_mean=0.75,
            strength=0.75,
        )
        self.graph.add_edge(
            "RES-001",
            "IMP-001",
            logic="contribuye a",
            keyword="contribuye",
            posterior_mean=0.65,
            strength=0.65,
        )

        # Impossible transition (logical jump)
        self.graph.add_edge(
            "PROD-001",
            "IMP-001",
            logic="genera",
            keyword="genera",
            posterior_mean=0.90,
            strength=0.90,
        )

        # Sample PDM text
        self.sample_text = """
        PROG-001 diagnóstico de necesidades realizado por la Secretaría de Planeación en 2024.
        PROD-001 construcción de infraestructura mediante la Secretaría de Obras, con presupuesto de $1,000,000.
        La entidad responsable es la Secretaría de Obras, que realizará actividades de construcción.
        Cronograma: enero 2024 a diciembre 2024.
        
        PROD-001 para abordar la causa PROG-001, dirigido a población rural, 
        porque genera mejora en calidad de vida mediante el mecanismo de acceso a servicios.
        
        RES-001 mejora en cobertura de servicios contribuye a IMP-001 desarrollo sostenible regional.
        """

    def test_audit_mechanism_necessity_complete(self):
        """Test mechanism necessity audit with complete components"""
        results = self.auditor.audit_mechanism_necessity(self.graph, self.sample_text)

        # Check that results exist for all edges
        self.assertEqual(len(results), 4)

        # Check PROD-001 -> RES-001 (should have most components)
        key = "PROD-001→RES-001"
        self.assertIn(key, results)
        result = results[key]

        # Should have entity and activity
        self.assertTrue(result.has_entity)
        self.assertTrue(result.has_activity)
        self.assertTrue(result.has_budget)
        # Timeline might be detected

        # Necessity score should be high
        self.assertGreaterEqual(result.necessity_score, 0.75)

    def test_audit_mechanism_necessity_missing_components(self):
        """Test necessity audit detects missing components"""
        # Create node without budget
        graph = nx.DiGraph()
        graph.add_node("N1", type="producto", text="Test")
        graph.add_node("N2", type="resultado", text="Test")
        graph.add_edge("N1", "N2", logic="test")

        results = self.auditor.audit_mechanism_necessity(graph, "N1 N2 test")

        result = results["N1→N2"]
        self.assertFalse(result.has_budget)
        self.assertIn("budget", result.missing_components)
        self.assertLess(result.necessity_score, 1.0)

    def test_audit_root_cause_mapping(self):
        """Test root cause mapping audit"""
        results = self.auditor.audit_root_cause_mapping(self.graph, self.sample_text)

        # Should find PROD-001 (producto node)
        self.assertIn("PROD-001", results)

        result = results["PROD-001"]

        # Should find linkage to PROG-001
        self.assertIn("PROG-001", result.root_causes)

        # Should detect linguistic marker
        self.assertTrue(len(result.linguistic_markers) > 0)

        # Coherence score should be calculated
        self.assertGreaterEqual(result.coherence_score, 0.0)
        self.assertLessEqual(result.coherence_score, 1.0)

    def test_audit_causal_proportionality_logical_jump(self):
        """Test proportionality audit detects logical jumps"""
        results = self.auditor.audit_causal_proportionality(self.graph)

        # Check impossible transition PROD-001 -> IMP-001
        key = "PROD-001→IMP-001"
        self.assertIn(key, results)

        result = results[key]

        # Should detect logical jump
        self.assertTrue(result.has_logical_jump)

        # Should not be proportional
        self.assertFalse(result.is_proportional)

        # Should cap posterior
        self.assertTrue(result.posterior_capped)
        self.assertLessEqual(result.adjusted_posterior, 0.6)

        # Original was 0.90
        self.assertEqual(result.original_posterior, 0.90)

    def test_audit_causal_proportionality_valid_transition(self):
        """Test proportionality audit allows valid transitions"""
        results = self.auditor.audit_causal_proportionality(self.graph)

        # Check valid transition PROD-001 -> RES-001
        key = "PROD-001→RES-001"
        self.assertIn(key, results)

        result = results[key]

        # Should be proportional
        self.assertTrue(result.is_proportional)

        # Should not have logical jump
        self.assertFalse(result.has_logical_jump)

        # Should not cap posterior
        self.assertFalse(result.posterior_capped)

        # Posterior should remain unchanged
        self.assertEqual(result.original_posterior, result.adjusted_posterior)

    def test_audit_activity_logic(self):
        """Test activity logic extraction"""
        results = self.auditor.audit_activity_logic(self.graph, self.sample_text)

        # Should find PROD-001
        self.assertIn("PROD-001", results)

        result = results["PROD-001"]

        # Should extract at least some components
        self.assertIsNotNone(result.causal_logic)

        # Extraction accuracy should be calculated
        self.assertGreaterEqual(result.extraction_accuracy, 0.0)
        self.assertLessEqual(result.extraction_accuracy, 1.0)

    def test_comprehensive_audit(self):
        """Test comprehensive audit generates all reports"""
        audit_report = self.auditor.generate_comprehensive_audit(
            self.graph, self.sample_text
        )

        # Check structure
        self.assertIn("summary", audit_report)
        self.assertIn("audit_point_2_1_necessity", audit_report)
        self.assertIn("audit_point_2_2_root_cause_mapping", audit_report)
        self.assertIn("audit_point_2_3_proportionality", audit_report)
        self.assertIn("audit_point_2_4_activity_logic", audit_report)

        # Check summary statistics
        summary = audit_report["summary"]
        self.assertIn("audit_point_2_1_mechanism_necessity", summary)
        self.assertIn("audit_point_2_2_root_cause_mapping", summary)
        self.assertIn("audit_point_2_3_causal_proportionality", summary)
        self.assertIn("audit_point_2_4_activity_logic", summary)

        # Check necessity summary
        necessity_summary = summary["audit_point_2_1_mechanism_necessity"]
        self.assertIn("total_links", necessity_summary)
        self.assertIn("necessity_rate", necessity_summary)
        self.assertIn("average_score", necessity_summary)

        # Total links should be 4
        self.assertEqual(necessity_summary["total_links"], 4)

    def test_quality_grade_thresholds(self):
        """Test quality grade determination"""
        # Test necessity grade thresholds
        self.assertEqual(
            self.auditor._get_necessity_quality_grade(1.0), QualityGrade.EXCELENTE
        )
        self.assertEqual(
            self.auditor._get_necessity_quality_grade(0.8), QualityGrade.BUENO
        )
        self.assertEqual(
            self.auditor._get_necessity_quality_grade(0.6), QualityGrade.REGULAR
        )
        self.assertEqual(
            self.auditor._get_necessity_quality_grade(0.3), QualityGrade.INSUFICIENTE
        )

    def test_remediation_generation(self):
        """Test remediation text generation"""
        remediation = self.auditor._generate_necessity_remediation(
            "TEST→LINK", ["entity", "budget"]
        )

        self.assertIsInstance(remediation, str)
        self.assertIn("TEST→LINK", remediation)
        self.assertIn("entidad responsable", remediation)
        self.assertIn("presupuesto asignado", remediation)
        self.assertIn("Beach & Pedersen 2019", remediation)


class TestDataStructures(unittest.TestCase):
    """Test audit result data structures"""

    def test_mechanism_necessity_result_to_dict(self):
        """Test MechanismNecessityResult serialization"""
        result = MechanismNecessityResult(
            link_id="A→B",
            is_necessary=True,
            necessity_score=0.95,
            missing_components=[],
            quality_grade=QualityGrade.EXCELENTE,
            evidence={"test": "data"},
            has_entity=True,
            has_activity=True,
            has_budget=True,
            has_timeline=True,
        )

        result_dict = result.to_dict()

        self.assertEqual(result_dict["link_id"], "A→B")
        self.assertTrue(result_dict["is_necessary"])
        self.assertEqual(result_dict["necessity_score"], 0.95)
        self.assertEqual(result_dict["quality_grade"], "Excelente")
        self.assertIn("micro_foundations", result_dict)
        self.assertTrue(result_dict["micro_foundations"]["has_entity"])

    def test_root_cause_mapping_result_to_dict(self):
        """Test RootCauseMappingResult serialization"""
        result = RootCauseMappingResult(
            activity_id="PROD-001",
            root_causes=["PROG-001"],
            linguistic_markers=["para abordar"],
            mapping_confidence=0.8,
            coherence_score=0.95,
            quality_grade=QualityGrade.EXCELENTE,
            linkage_phrases=[("para abordar", "PROD-001", "PROG-001")],
        )

        result_dict = result.to_dict()

        self.assertEqual(result_dict["activity_id"], "PROD-001")
        self.assertEqual(result_dict["coherence_score"], 0.95)
        self.assertEqual(len(result_dict["linkage_phrases"]), 1)
        self.assertEqual(result_dict["linkage_phrases"][0]["marker"], "para abordar")

    def test_causal_proportionality_result_to_dict(self):
        """Test CausalProportionalityResult serialization"""
        result = CausalProportionalityResult(
            link_id="PROD→IMP",
            source_type="producto",
            target_type="impacto",
            is_proportional=False,
            has_logical_jump=True,
            posterior_capped=True,
            original_posterior=0.9,
            adjusted_posterior=0.6,
            quality_grade=QualityGrade.REGULAR,
            violation_details="Logical jump",
        )

        result_dict = result.to_dict()

        self.assertEqual(result_dict["link_id"], "PROD→IMP")
        self.assertTrue(result_dict["has_logical_jump"])
        self.assertTrue(result_dict["posterior_capped"])
        self.assertEqual(result_dict["original_posterior"], 0.9)
        self.assertEqual(result_dict["adjusted_posterior"], 0.6)

    def test_activity_logic_result_to_dict(self):
        """Test ActivityLogicResult serialization"""
        result = ActivityLogicResult(
            activity_id="PROD-001",
            instrument="infraestructura",
            target_population="población rural",
            causal_logic="genera mejora",
            extraction_accuracy=1.0,
            quality_grade=QualityGrade.EXCELENTE,
            matched_rationale="Sample rationale",
        )

        result_dict = result.to_dict()

        self.assertEqual(result_dict["activity_id"], "PROD-001")
        self.assertEqual(result_dict["instrument"], "infraestructura")
        self.assertEqual(result_dict["target_population"], "población rural")
        self.assertEqual(result_dict["extraction_accuracy"], 1.0)
        self.assertEqual(result_dict["quality_grade"], "Excelente")


if __name__ == "__main__":
    unittest.main()
