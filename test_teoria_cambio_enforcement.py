#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test TeoriaCambio Canonical Chain Enforcement
==============================================

Tests that TeoriaCambio enforces INSUMOS→PROCESOS→PRODUCTOS→RESULTADOS→CAUSALIDAD
canonical chain at graph construction time and identifies violation points.

SIN_CARRETA Compliance:
- Deterministic test execution with fixed graph structures
- Contract verification via assertions
- Explicit violation detection validation
"""

import unittest

import networkx as nx

from teoria_cambio import CategoriaCausal, TeoriaCambio, ValidacionResultado


class TestTeoriaCambioEnforcement(unittest.TestCase):
    """
    Test suite for canonical chain enforcement.

    SIN_CARRETA-RATIONALE: These tests verify deterministic enforcement
    of causal ordering rules, ensuring graph validation is auditable and
    reproducible across all executions.
    """

    def setUp(self):
        """Initialize TeoriaCambio engine"""
        self.tc = TeoriaCambio()

    def test_canonical_chain_valid_sequence(self):
        """
        Test that valid INSUMOS→PROCESOS→PRODUCTOS→RESULTADOS→CAUSALIDAD
        sequence is accepted.

        CONTRACT: Valid canonical chain should pass validation with no violations.
        """
        # Create valid graph
        G = nx.DiGraph()
        G.add_node("N1", categoria=CategoriaCausal.INSUMOS)
        G.add_node("N2", categoria=CategoriaCausal.PROCESOS)
        G.add_node("N3", categoria=CategoriaCausal.PRODUCTOS)
        G.add_node("N4", categoria=CategoriaCausal.RESULTADOS)
        G.add_node("N5", categoria=CategoriaCausal.CAUSALIDAD)

        # Add edges following canonical order
        G.add_edge("N1", "N2")  # INSUMOS → PROCESOS
        G.add_edge("N2", "N3")  # PROCESOS → PRODUCTOS
        G.add_edge("N3", "N4")  # PRODUCTOS → RESULTADOS
        G.add_edge("N4", "N5")  # RESULTADOS → CAUSALIDAD

        # Validate
        resultado = self.tc.validacion_completa(G)

        # Assert: Should be valid with no violations
        self.assertTrue(
            resultado.es_valida, "Valid canonical chain should pass validation"
        )
        self.assertEqual(
            len(resultado.violaciones_orden),
            0,
            "Valid chain should have no order violations",
        )
        self.assertEqual(
            len(resultado.categorias_faltantes), 0, "All categories should be present"
        )

    def test_canonical_chain_skip_violation(self):
        """
        Test that skipping intermediate categories is detected as violation.

        CONTRACT: INSUMOS→RESULTADOS (skipping PROCESOS, PRODUCTOS) must be flagged.
        VIOLATION POINT: Edge ("N1", "N4")
        """
        # Create graph with violation
        G = nx.DiGraph()
        G.add_node("N1", categoria=CategoriaCausal.INSUMOS)
        G.add_node("N4", categoria=CategoriaCausal.RESULTADOS)

        # Add edge that skips intermediate categories
        G.add_edge("N1", "N4")  # INSUMOS → RESULTADOS (INVALID)

        # Validate
        resultado = self.tc.validacion_completa(G)

        # Assert: Should detect violation
        self.assertFalse(resultado.es_valida, "Graph with skip should be invalid")
        self.assertIn(
            ("N1", "N4"),
            resultado.violaciones_orden,
            "Should detect INSUMOS→RESULTADOS skip violation",
        )
        self.assertEqual(
            len(resultado.violaciones_orden), 1, "Should detect exactly one violation"
        )

    def test_canonical_chain_backward_violation(self):
        """
        Test that backward edges are detected as violations.

        CONTRACT: PRODUCTOS→PROCESOS (backward) must be flagged.
        VIOLATION POINT: Edge ("N3", "N2")
        """
        # Create graph with backward edge
        G = nx.DiGraph()
        G.add_node("N2", categoria=CategoriaCausal.PROCESOS)
        G.add_node("N3", categoria=CategoriaCausal.PRODUCTOS)

        # Add backward edge
        G.add_edge("N3", "N2")  # PRODUCTOS → PROCESOS (INVALID)

        # Validate
        resultado = self.tc.validacion_completa(G)

        # Assert: Should detect violation
        self.assertFalse(
            resultado.es_valida, "Graph with backward edge should be invalid"
        )
        self.assertIn(
            ("N3", "N2"),
            resultado.violaciones_orden,
            "Should detect backward edge violation",
        )

    def test_canonical_chain_same_level_allowed(self):
        """
        Test that same-level connections are allowed.

        CONTRACT: PRODUCTOS→PRODUCTOS should be valid (same level).
        """
        # Create graph with same-level edge
        G = nx.DiGraph()
        G.add_node("P1", categoria=CategoriaCausal.PRODUCTOS)
        G.add_node("P2", categoria=CategoriaCausal.PRODUCTOS)

        # Add same-level edge
        G.add_edge("P1", "P2")  # PRODUCTOS → PRODUCTOS (VALID)

        # Validate
        resultado = self.tc.validacion_completa(G)

        # Assert: Should be valid
        self.assertEqual(
            len(resultado.violaciones_orden),
            0,
            "Same-level connections should be allowed",
        )

    def test_canonical_chain_multiple_violations(self):
        """
        Test that multiple violations are all detected.

        CONTRACT: All invalid edges must be flagged.
        VIOLATION POINTS: Multiple edges
        """
        # Create graph with multiple violations
        G = nx.DiGraph()
        G.add_node("N1", categoria=CategoriaCausal.INSUMOS)
        G.add_node("N2", categoria=CategoriaCausal.PROCESOS)
        G.add_node("N3", categoria=CategoriaCausal.PRODUCTOS)
        G.add_node("N4", categoria=CategoriaCausal.RESULTADOS)
        G.add_node("N5", categoria=CategoriaCausal.CAUSALIDAD)

        # Add multiple invalid edges
        G.add_edge("N1", "N3")  # INSUMOS → PRODUCTOS (skip PROCESOS)
        G.add_edge("N2", "N5")  # PROCESOS → CAUSALIDAD (skip PRODUCTOS, RESULTADOS)
        G.add_edge("N4", "N2")  # RESULTADOS → PROCESOS (backward)

        # Validate
        resultado = self.tc.validacion_completa(G)

        # Assert: Should detect all violations
        self.assertGreaterEqual(
            len(resultado.violaciones_orden), 3, "Should detect at least 3 violations"
        )
        self.assertIn(("N1", "N3"), resultado.violaciones_orden)
        self.assertIn(("N2", "N5"), resultado.violaciones_orden)
        self.assertIn(("N4", "N2"), resultado.violaciones_orden)

    def test_canonical_chain_missing_categories(self):
        """
        Test that missing categories are detected.

        CONTRACT: All 5 categories must be present for complete theory of change.
        """
        # Create graph with missing categories
        G = nx.DiGraph()
        G.add_node("N1", categoria=CategoriaCausal.INSUMOS)
        G.add_node("N3", categoria=CategoriaCausal.PRODUCTOS)
        # Missing: PROCESOS, RESULTADOS, CAUSALIDAD

        G.add_edge("N1", "N3")  # INSUMOS → PRODUCTOS

        # Validate
        resultado = self.tc.validacion_completa(G)

        # Assert: Should detect missing categories
        self.assertFalse(
            resultado.es_valida, "Graph with missing categories should be invalid"
        )
        self.assertGreater(
            len(resultado.categorias_faltantes), 0, "Should detect missing categories"
        )

        # Check specific missing categories
        missing_names = [cat.name for cat in resultado.categorias_faltantes]
        self.assertIn("PROCESOS", missing_names)
        self.assertIn("RESULTADOS", missing_names)
        self.assertIn("CAUSALIDAD", missing_names)

    def test_enforcement_at_construction_time(self):
        """
        Test that enforcement happens at validation time (not construction).

        CONTRACT: Graph construction is permissive, validation is strict.
        NOTE: NetworkX allows any edges; TeoriaCambio validates post-construction.
        """
        # Create invalid graph (construction should succeed)
        G = nx.DiGraph()
        G.add_node("N1", categoria=CategoriaCausal.INSUMOS)
        G.add_node("N5", categoria=CategoriaCausal.CAUSALIDAD)
        G.add_edge("N1", "N5")  # Invalid edge

        # Construction should succeed (no exception)
        self.assertEqual(
            G.number_of_edges(), 1, "Graph construction should be permissive"
        )

        # But validation should fail
        resultado = self.tc.validacion_completa(G)
        self.assertFalse(
            resultado.es_valida, "Validation should detect invalid structure"
        )

    def test_complete_path_detection(self):
        """
        Test that complete INSUMOS→CAUSALIDAD paths are detected.

        CONTRACT: At least one complete path required for valid theory of change.
        """
        # Create graph with complete path
        G = nx.DiGraph()
        G.add_node("N1", categoria=CategoriaCausal.INSUMOS)
        G.add_node("N2", categoria=CategoriaCausal.PROCESOS)
        G.add_node("N3", categoria=CategoriaCausal.PRODUCTOS)
        G.add_node("N4", categoria=CategoriaCausal.RESULTADOS)
        G.add_node("N5", categoria=CategoriaCausal.CAUSALIDAD)

        G.add_edge("N1", "N2")
        G.add_edge("N2", "N3")
        G.add_edge("N3", "N4")
        G.add_edge("N4", "N5")

        # Validate
        resultado = self.tc.validacion_completa(G)

        # Assert: Should find complete path
        self.assertGreater(
            len(resultado.caminos_completos),
            0,
            "Should find at least one complete path",
        )
        self.assertIn(["N1", "N2", "N3", "N4", "N5"], resultado.caminos_completos)

    def test_deterministic_validation(self):
        """
        Test that validation is deterministic.

        SIN_CARRETA CONTRACT: Same input must produce same output.
        """
        # Create test graph
        G = nx.DiGraph()
        G.add_node("N1", categoria=CategoriaCausal.INSUMOS)
        G.add_node("N4", categoria=CategoriaCausal.RESULTADOS)
        G.add_edge("N1", "N4")

        # Run validation multiple times
        resultado1 = self.tc.validacion_completa(G)
        resultado2 = self.tc.validacion_completa(G)
        resultado3 = self.tc.validacion_completa(G)

        # Assert: All results should be identical
        self.assertEqual(resultado1.es_valida, resultado2.es_valida)
        self.assertEqual(resultado2.es_valida, resultado3.es_valida)
        self.assertEqual(resultado1.violaciones_orden, resultado2.violaciones_orden)
        self.assertEqual(resultado2.violaciones_orden, resultado3.violaciones_orden)


if __name__ == "__main__":
    unittest.main()
