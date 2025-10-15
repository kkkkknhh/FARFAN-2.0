#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Tests for Axiomatic Validator Requirements
=======================================================

Tests:
1. Structural validation runs before semantic and regulatory
2. TeoriaCambio → D6-Q1/Q2 score mappings
3. PolicyContradictionDetectorV2 → D2-Q5, D6-Q3 score mappings
4. ValidadorDNP → D1-Q5, D4-Q5 score mappings with BPIN
5. Three governance triggers with correct thresholds:
   - contradiction_density > 0.05 → manual review
   - D6 scores < 0.55 → block progression
   - structural violations → penalty factors to Bayesian posteriors
6. ValidationFailure instances properly handled
7. AxiomaticValidationResult correctly aggregates results
8. Cross-validation between GNN and Bayesian contradictions
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAxiomaticValidatorIntegration(unittest.TestCase):
    """Integration tests for complete validator system"""

    def setUp(self):
        """Set up test fixtures"""
        # Import after path setup
        from validators.axiomatic_validator import (
            AxiomaticValidationResult,
            AxiomaticValidator,
            PDMOntology,
            SemanticChunk,
            ValidationConfig,
            ValidationSeverity,
        )

        self.AxiomaticValidator = AxiomaticValidator
        self.ValidationConfig = ValidationConfig
        self.PDMOntology = PDMOntology
        self.SemanticChunk = SemanticChunk
        self.AxiomaticValidationResult = AxiomaticValidationResult
        self.ValidationSeverity = ValidationSeverity

        self.config = ValidationConfig(
            contradiction_threshold=0.05,
            enable_structural_penalty=True,
            enable_human_gating=True,
        )
        self.ontology = PDMOntology()

    def test_validation_execution_order(self):
        """Test 1: Structural validation runs before semantic and regulatory"""
        validator = self.AxiomaticValidator(self.config, self.ontology)

        # Create mock graph and chunks
        mock_graph = Mock()
        mock_graph.number_of_edges.return_value = 10
        mock_graph.number_of_nodes.return_value = 8

        chunks = [self.SemanticChunk(text="Test", dimension="ESTRATEGICO")]

        # Mock validators to track execution order
        execution_order = []

        def track_structural(*args):
            execution_order.append("structural")
            result = Mock()
            result.es_valida = True
            result.violaciones_orden = []
            result.categorias_faltantes = []
            result.caminos_completos = []
            result.sugerencias = []
            return result

        def track_semantic(*args):
            execution_order.append("semantic")
            return []

        def track_regulatory(*args):
            execution_order.append("regulatory")
            result = Mock()
            result.score_total = 70.0
            result.cumple_competencias = True
            result.cumple_mga = True
            result.indicadores_mga_usados = ["EDU-001"]
            return result

        validator._validate_structural = track_structural
        validator._validate_semantic = track_semantic
        validator._validate_regulatory = track_regulatory

        result = validator.validate_complete(mock_graph, chunks)

        # Verify execution order
        self.assertEqual(
            execution_order,
            ["structural", "semantic", "regulatory"],
            "Validators must execute in order: structural → semantic → regulatory",
        )
        print("✓ Test 1 PASSED: Execution order verified")

    def test_teoria_cambio_score_mappings(self):
        """Test 2: TeoriaCambio outputs map to D6-Q1/Q2 scores"""
        validator = self.AxiomaticValidator(self.config, self.ontology)

        # Mock structural result with specific values
        structural_result = Mock()
        structural_result.es_valida = False
        structural_result.violaciones_orden = [("A", "B"), ("C", "D")]  # 2 violations
        structural_result.categorias_faltantes = []  # All categories present
        structural_result.caminos_completos = [
            ["I", "P", "Pr", "R", "C"]
        ]  # 1 complete path
        structural_result.sugerencias = []

        # Calculate D6-Q1 (completeness)
        d6_q1 = validator._calculate_d6_q1_score(structural_result)

        # Calculate D6-Q2 (causal order)
        d6_q2 = validator._calculate_d6_q2_score(structural_result)

        # Verify scores are in valid range
        self.assertGreaterEqual(d6_q1, 0.0)
        self.assertLessEqual(d6_q1, 1.0)
        self.assertGreaterEqual(d6_q2, 0.0)
        self.assertLessEqual(d6_q2, 1.0)

        # D6-Q1 should be high (no missing categories, has path)
        self.assertGreater(d6_q1, 0.6, "D6-Q1 should be >0.6 with complete categories")

        # D6-Q2 should be lower (has violations)
        self.assertLess(d6_q2, 0.8, "D6-Q2 should be <0.8 with 2 violations")

        print(f"✓ Test 2 PASSED: TeoriaCambio → D6-Q1={d6_q1:.3f}, D6-Q2={d6_q2:.3f}")

    def test_contradiction_detector_score_mappings(self):
        """Test 3: PolicyContradictionDetectorV2 → D2-Q5, D6-Q3 scores"""
        validator = self.AxiomaticValidator(self.config, self.ontology)

        # Mock contradictions
        contradictions = [
            {"dimension": "ESTRATEGICO", "type": "semantic"},
            {"dimension": "PROGRAMATICO", "type": "logical"},
            {"dimension": "FINANCIERO", "type": "numerical"},
        ]

        # Mock chunks
        chunks = [
            self.SemanticChunk(text="Chunk 1", dimension="ESTRATEGICO"),
            self.SemanticChunk(text="Chunk 2", dimension="PROGRAMATICO"),
            self.SemanticChunk(text="Chunk 3", dimension="ESTRATEGICO"),
            self.SemanticChunk(text="Chunk 4", dimension="FINANCIERO"),
        ]

        # Calculate D2-Q5 (design contradictions)
        d2_q5 = validator._calculate_d2_q5_score(contradictions, chunks)

        # Calculate D6-Q3 (causal coherence)
        contradiction_density = 0.03  # 3%
        d6_q3 = validator._calculate_d6_q3_score(contradiction_density, contradictions)

        # Verify scores are in valid range
        self.assertGreaterEqual(d2_q5, 0.0)
        self.assertLessEqual(d2_q5, 1.0)
        self.assertGreaterEqual(d6_q3, 0.0)
        self.assertLessEqual(d6_q3, 1.0)

        # D6-Q3 should be high with low density (0.03 < 0.05)
        self.assertGreater(d6_q3, 0.7, "D6-Q3 should be >0.7 with density=0.03")

        print(f"✓ Test 3 PASSED: Contradictions → D2-Q5={d2_q5:.3f}, D6-Q3={d6_q3:.3f}")

    def test_dnp_validator_score_mappings(self):
        """Test 4: ValidadorDNP → D1-Q5, D4-Q5 scores with BPIN"""
        validator = self.AxiomaticValidator(self.config, self.ontology)

        # Mock DNP result
        dnp_result = Mock()
        dnp_result.score_total = 80.0
        dnp_result.cumple_competencias = True
        dnp_result.cumple_mga = True
        dnp_result.indicadores_mga_usados = [
            "EDU-001",
            "EDU-002",
            "SAL-001",
            "INF-001",
            "AMB-001",
        ]

        # Calculate D1-Q5 (diagnostic compliance)
        d1_q5 = validator._calculate_d1_q5_score(dnp_result)

        # Calculate D4-Q5 (results validation with BPIN)
        d4_q5 = validator._calculate_d4_q5_score(dnp_result)

        # Verify scores are in valid range
        self.assertGreaterEqual(d1_q5, 0.0)
        self.assertLessEqual(d1_q5, 1.0)
        self.assertGreaterEqual(d4_q5, 0.0)
        self.assertLessEqual(d4_q5, 1.0)

        # Both should be high with good DNP compliance
        self.assertGreater(
            d1_q5, 0.7, "D1-Q5 should be >0.7 with competency compliance"
        )
        self.assertGreater(
            d4_q5, 0.7, "D4-Q5 should be >0.7 with MGA compliance and 5 indicators"
        )

        print(
            f"✓ Test 4 PASSED: DNP → D1-Q5={d1_q5:.3f}, D4-Q5={d4_q5:.3f} (BPIN: 5 indicators)"
        )

    def test_governance_trigger_1_contradiction_density(self):
        """Test 5a: Governance Trigger 1 - contradiction_density > 0.05 flags manual review"""
        validator = self.AxiomaticValidator(self.config, self.ontology)

        # Create mock graph
        mock_graph = Mock()
        mock_graph.number_of_edges.return_value = 100
        mock_graph.number_of_nodes.return_value = 50

        # Mock high contradiction count (7 contradictions / 100 edges = 0.07 > 0.05)
        contradictions = [{"type": f"c{i}"} for i in range(7)]

        validator._validate_structural = lambda g: Mock(
            es_valida=True,
            violaciones_orden=[],
            categorias_faltantes=[],
            caminos_completos=[[]],
            sugerencias=[],
        )
        validator._validate_semantic = lambda g, c: contradictions
        validator._validate_regulatory = lambda c, f: Mock(
            score_total=70.0,
            cumple_competencias=True,
            cumple_mga=True,
            indicadores_mga_usados=[],
        )

        chunks = [self.SemanticChunk(text="Test", dimension="ESTRATEGICO")]
        result = validator.validate_complete(mock_graph, chunks)

        # Verify governance trigger
        self.assertGreater(
            result.contradiction_density,
            0.05,
            "Contradiction density should exceed threshold",
        )
        self.assertTrue(
            result.requires_manual_review, "Manual review should be required"
        )
        self.assertEqual(
            result.hold_reason,
            "HIGH_CONTRADICTION_DENSITY",
            "Hold reason should be HIGH_CONTRADICTION_DENSITY",
        )

        print(
            f"✓ Test 5a PASSED: Governance Trigger 1 activated (density={result.contradiction_density:.4f} > 0.05)"
        )

    def test_governance_trigger_2_d6_threshold(self):
        """Test 5b: Governance Trigger 2 - D6 scores < 0.55 block progression"""
        validator = self.AxiomaticValidator(self.config, self.ontology)

        # Create mock graph
        mock_graph = Mock()
        mock_graph.number_of_edges.return_value = 10
        mock_graph.number_of_nodes.return_value = 8
        mock_graph.has_edge.return_value = True
        mock_graph.edges = {
            ("A", "B"): {},
            ("C", "D"): {},
            ("E", "F"): {},
        }

        # Mock poor structural result (low D6 scores)
        structural_result = Mock()
        structural_result.es_valida = False
        structural_result.violaciones_orden = [
            ("A", "B"),
            ("C", "D"),
            ("E", "F"),
        ]  # 3 violations
        structural_result.categorias_faltantes = [
            "INSUMOS",
            "CAUSALIDAD",
        ]  # Missing 2 categories
        structural_result.caminos_completos = []  # No complete paths
        structural_result.sugerencias = []

        validator._validate_structural = lambda g: structural_result
        validator._validate_semantic = lambda g, c: []
        validator._validate_regulatory = lambda c, f: Mock(
            score_total=70.0,
            cumple_competencias=True,
            cumple_mga=True,
            indicadores_mga_usados=[],
        )

        chunks = [self.SemanticChunk(text="Test", dimension="ESTRATEGICO")]
        result = validator.validate_complete(mock_graph, chunks)

        # Verify D6 scores are below threshold
        d6_q1 = result.score_mappings.get("D6-Q1", 1.0)
        d6_q2 = result.score_mappings.get("D6-Q2", 1.0)

        self.assertTrue(
            d6_q1 < 0.55 or d6_q2 < 0.55,
            f"At least one D6 score should be <0.55 (D6-Q1={d6_q1:.3f}, D6-Q2={d6_q2:.3f})",
        )
        self.assertTrue(
            result.requires_manual_review,
            "Manual review should be required for low D6 scores",
        )
        self.assertEqual(
            result.hold_reason,
            "D6_SCORE_BELOW_THRESHOLD",
            "Hold reason should be D6_SCORE_BELOW_THRESHOLD",
        )

        print(
            f"✓ Test 5b PASSED: Governance Trigger 2 activated (D6-Q1={d6_q1:.3f}, D6-Q2={d6_q2:.3f} < 0.55)"
        )

    def test_governance_trigger_3_structural_penalties(self):
        """Test 5c: Governance Trigger 3 - structural violations apply penalty to Bayesian posteriors"""
        validator = self.AxiomaticValidator(self.config, self.ontology)

        # Create mock prior builder
        mock_prior_builder = Mock()
        mock_prior_builder.structural_penalty_factor = 1.0

        # Create mock graph with edges
        mock_graph = Mock()
        mock_graph.number_of_edges.return_value = 10
        mock_graph.number_of_nodes.return_value = 8
        mock_graph.has_edge.return_value = True
        mock_graph.edges = {
            ("A", "B"): {},
            ("C", "D"): {},
        }

        # Mock structural result with violations
        violations = [("A", "B"), ("C", "D")]
        structural_result = Mock()
        structural_result.es_valida = False
        structural_result.violaciones_orden = violations
        structural_result.categorias_faltantes = []
        structural_result.caminos_completos = []
        structural_result.sugerencias = []

        validator._validate_structural = lambda g: structural_result
        validator._validate_semantic = lambda g, c: []
        validator._validate_regulatory = lambda c, f: Mock(
            score_total=70.0,
            cumple_competencias=True,
            cumple_mga=True,
            indicadores_mga_usados=[],
        )

        chunks = [self.SemanticChunk(text="Test", dimension="ESTRATEGICO")]
        result = validator.validate_complete(
            mock_graph, chunks, prior_builder=mock_prior_builder
        )

        # Verify penalty was applied
        self.assertLess(
            result.structural_penalty_factor,
            1.0,
            "Penalty factor should be <1.0 with violations",
        )

        # Verify penalty matches expected values for 2 violations
        self.assertAlmostEqual(
            result.structural_penalty_factor,
            0.8,
            places=2,
            msg="2 violations should result in 0.8 penalty factor",
        )

        print(
            f"✓ Test 5c PASSED: Governance Trigger 3 activated (penalty_factor={result.structural_penalty_factor:.2f} for 2 violations)"
        )

    def test_validation_failure_handling(self):
        """Test 6: ValidationFailure instances properly handled"""
        result = self.AxiomaticValidationResult()

        # Initially no failures
        self.assertEqual(len(result.failures), 0)
        self.assertTrue(result.is_valid)

        # Add critical failure
        result.add_critical_failure(
            dimension="D6",
            question="Q2",
            evidence={"violations": 3},
            impact="Critical structural issues",
            recommendations=["Fix structural violations"],
        )

        # Verify failure was added
        self.assertEqual(len(result.failures), 1)
        self.assertFalse(result.is_valid, "Should be invalid after critical failure")

        # Verify failure details
        failure = result.failures[0]
        self.assertEqual(failure.dimension, "D6")
        self.assertEqual(failure.question, "Q2")
        self.assertEqual(failure.severity, self.ValidationSeverity.CRITICAL)
        self.assertEqual(failure.impact, "Critical structural issues")
        self.assertEqual(len(failure.recommendations), 1)

        print("✓ Test 6 PASSED: ValidationFailure handling verified")

    def test_result_aggregation(self):
        """Test 7: AxiomaticValidationResult correctly aggregates results"""
        validator = self.AxiomaticValidator(self.config, self.ontology)

        # Create mock graph
        mock_graph = Mock()
        mock_graph.number_of_edges.return_value = 20
        mock_graph.number_of_nodes.return_value = 15

        # Mock all validators
        structural = Mock()
        structural.es_valida = True
        structural.violaciones_orden = []
        structural.categorias_faltantes = []
        structural.caminos_completos = [["I", "P", "Pr", "R", "C"]]
        structural.sugerencias = []

        contradictions = [{"type": "c1"}]  # 1/20 = 0.05 (at threshold)

        dnp = Mock()
        dnp.score_total = 85.0
        dnp.cumple_competencias = True
        dnp.cumple_mga = True
        dnp.indicadores_mga_usados = ["IND-001", "IND-002"]

        validator._validate_structural = lambda g: structural
        validator._validate_semantic = lambda g, c: contradictions
        validator._validate_regulatory = lambda c, f: dnp

        chunks = [self.SemanticChunk(text="Test", dimension="ESTRATEGICO")]
        result = validator.validate_complete(mock_graph, chunks)

        # Verify aggregation
        self.assertEqual(result.total_edges, 20, "Should aggregate edge count")
        self.assertEqual(result.total_nodes, 15, "Should aggregate node count")
        self.assertTrue(result.structural_valid, "Should aggregate structural validity")
        self.assertEqual(
            result.contradiction_density, 0.05, "Should aggregate contradiction density"
        )
        self.assertEqual(
            result.regulatory_score, 85.0, "Should aggregate regulatory score"
        )

        # Verify score mappings exist
        self.assertIn("D6-Q1", result.score_mappings)
        self.assertIn("D6-Q2", result.score_mappings)
        self.assertIn("D2-Q5", result.score_mappings)
        self.assertIn("D6-Q3", result.score_mappings)
        self.assertIn("D1-Q5", result.score_mappings)
        self.assertIn("D4-Q5", result.score_mappings)

        # Verify BPIN integration
        self.assertEqual(
            len(result.bpin_indicators), 2, "Should aggregate BPIN indicators"
        )

        print(
            "✓ Test 7 PASSED: Result aggregation verified (6 score mappings, 2 BPIN indicators)"
        )

    def test_cross_validation_gnn_bayesian(self):
        """Test 8: Cross-validation between GNN and Bayesian contradictions"""
        validator = self.AxiomaticValidator(self.config, self.ontology)

        # Mock GNN contradictions (tuple format)
        gnn_contradictions = [
            (
                Mock(text="Statement A about education"),
                Mock(text="Statement B about education"),
                0.85,
                None,
            ),
            (
                Mock(text="Statement C about health"),
                Mock(text="Statement D about health"),
                0.78,
                None,
            ),
            (
                Mock(text="Statement E about budget"),
                Mock(text="Statement F about budget"),
                0.92,
                None,
            ),
        ]

        # Mock Bayesian contradictions (dict format, some overlap)
        bayesian_contradictions = [
            {
                "statement_a": {"text": "Statement A about education"},
                "statement_b": {"text": "Statement B about education"},
            },
            {
                "statement_a": {"text": "Statement X about infrastructure"},
                "statement_b": {"text": "Statement Y about infrastructure"},
            },
        ]

        chunks = [self.SemanticChunk(text="Test", dimension="ESTRATEGICO")]

        # Run cross-validation
        cross_validation = validator.validate_contradiction_consistency(
            gnn_contradictions, bayesian_contradictions, chunks
        )

        # Verify cross-validation results
        self.assertIn("consistency_rate", cross_validation)
        self.assertIn("total_unique_contradictions", cross_validation)
        self.assertIn("high_confidence", cross_validation)
        self.assertIn("gnn_explicit", cross_validation)
        self.assertIn("bayesian_implicit", cross_validation)
        self.assertIn("recommendations", cross_validation)

        # Verify counts
        self.assertGreater(cross_validation["total_unique_contradictions"], 0)
        self.assertGreaterEqual(len(cross_validation["high_confidence"]), 0)
        self.assertGreaterEqual(len(cross_validation["recommendations"]), 1)

        print(f"✓ Test 8 PASSED: Cross-validation completed")
        print(f"  - Consistency rate: {cross_validation['consistency_rate']:.2f}")
        print(f"  - Total unique: {cross_validation['total_unique_contradictions']}")
        print(f"  - High confidence: {len(cross_validation['high_confidence'])}")
        print(f"  - GNN explicit: {cross_validation['gnn_only_count']}")
        print(f"  - Bayesian implicit: {cross_validation['bayesian_only_count']}")


if __name__ == "__main__":
    print("=" * 80)
    print("AXIOMATIC VALIDATOR INTEGRATION TESTS")
    print("=" * 80)
    unittest.main(verbosity=2)
