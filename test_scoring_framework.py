#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Scoring Framework
SIN_CARRETA Compliant: Deterministic validation, contract enforcement
"""

import unittest
import logging
from scoring_framework import (
    ScoringEngine,
    QuestionScore,
    DimensionScore,
    PolicyScore,
    ClusterScore,
    MacroScore,
    DIMENSION_WEIGHTS,
    RUBRIC_THRESHOLDS,
    POLICY_CLUSTERS,
    CLUSTER_WEIGHTS,
    D6_MANUAL_REVIEW_THRESHOLD,
    validate_scoring_framework
)

logging.basicConfig(level=logging.WARNING)


class TestDimensionWeights(unittest.TestCase):
    """Test dimension weights sum to exactly 1.0 for each policy"""

    def test_all_policies_have_six_dimensions(self):
        for policy in [f"P{i}" for i in range(1, 11)]:
            self.assertIn(policy, DIMENSION_WEIGHTS)
            self.assertEqual(len(DIMENSION_WEIGHTS[policy]), 6)
            self.assertEqual(set(DIMENSION_WEIGHTS[policy].keys()), 
                           {"D1", "D2", "D3", "D4", "D5", "D6"})

    def test_dimension_weights_sum_to_one(self):
        for policy, weights in DIMENSION_WEIGHTS.items():
            total = sum(weights.values())
            self.assertAlmostEqual(total, 1.0, places=9, 
                                 msg=f"{policy} weights sum to {total}, not 1.0")

    def test_all_weights_positive(self):
        for policy, weights in DIMENSION_WEIGHTS.items():
            for dimension, weight in weights.items():
                self.assertGreater(weight, 0.0, 
                                 msg=f"{policy}-{dimension} has non-positive weight")
                self.assertLessEqual(weight, 1.0,
                                   msg=f"{policy}-{dimension} weight exceeds 1.0")


class TestRubricThresholds(unittest.TestCase):
    """Test rubric threshold mappings are consistent"""

    def test_all_categories_present(self):
        expected_categories = {"excelente", "bueno", "aceptable", "insuficiente"}
        self.assertEqual(set(RUBRIC_THRESHOLDS.keys()), expected_categories)

    def test_threshold_ranges_valid(self):
        for category, (low, high) in RUBRIC_THRESHOLDS.items():
            self.assertGreaterEqual(low, 0.0, 
                                  msg=f"{category} lower bound < 0")
            self.assertLessEqual(high, 1.0,
                               msg=f"{category} upper bound > 1.0")
            self.assertLess(low, high,
                          msg=f"{category} has inverted range [{low}, {high}]")

    def test_thresholds_cover_full_range(self):
        # Check that thresholds cover [0.0, 1.0] with no gaps
        all_ranges = sorted(RUBRIC_THRESHOLDS.values(), key=lambda x: x[0])
        
        # First range should start at 0.0
        self.assertEqual(all_ranges[0][0], 0.0)
        
        # Last range should end at 1.0
        self.assertEqual(max(r[1] for r in all_ranges), 1.0)

    def test_score_to_rubric_mapping(self):
        engine = ScoringEngine()
        
        # Test boundary values
        self.assertEqual(engine.score_to_rubric_category(0.0), "insuficiente")
        self.assertEqual(engine.score_to_rubric_category(0.54), "insuficiente")
        self.assertEqual(engine.score_to_rubric_category(0.55), "aceptable")
        self.assertEqual(engine.score_to_rubric_category(0.69), "aceptable")
        self.assertEqual(engine.score_to_rubric_category(0.70), "bueno")
        self.assertEqual(engine.score_to_rubric_category(0.84), "bueno")
        self.assertEqual(engine.score_to_rubric_category(0.85), "excelente")
        self.assertEqual(engine.score_to_rubric_category(1.0), "excelente")


class TestCanonicalNotationValidator(unittest.TestCase):
    """Test validation of P1-P10 × D1-D6 × Q1-Q5 = 300 questions"""

    def test_all_300_questions_valid(self):
        engine = ScoringEngine()
        result = engine.validate_all_canonical_questions()
        
        self.assertEqual(result["total_questions"], 300)
        self.assertEqual(result["validation_results"]["total"], 300)
        self.assertEqual(result["validation_results"]["valid"], 300)
        self.assertEqual(len(result["validation_results"]["invalid"]), 0)

    def test_question_format_validation(self):
        engine = ScoringEngine()
        
        # Valid formats
        valid_ids = [
            "P1-D1-Q1", "P5-D3-Q2", "P10-D6-Q5"
        ]
        for qid in valid_ids:
            # Should not raise
            from canonical_notation import CanonicalID
            canonical = CanonicalID.from_string(qid)
            self.assertIsNotNone(canonical)

    def test_question_id_components(self):
        from canonical_notation import CanonicalID
        
        canonical = CanonicalID.from_string("P7-D4-Q3")
        self.assertEqual(canonical.policy, "P7")
        self.assertEqual(canonical.dimension, "D4")
        self.assertEqual(canonical.question, 3)


class TestScoringAggregation(unittest.TestCase):
    """Test MICRO→MESO→MACRO aggregation logic"""

    def setUp(self):
        self.engine = ScoringEngine()

    def test_dimension_score_calculation(self):
        # Create 5 question scores for P1-D1
        question_scores = [
            QuestionScore(f"P1-D1-Q{i}", 0.8, "bueno") 
            for i in range(1, 6)
        ]
        
        dim_score = self.engine.calculate_dimension_score("P1", "D1", question_scores)
        
        self.assertEqual(dim_score.policy, "P1")
        self.assertEqual(dim_score.dimension, "D1")
        self.assertAlmostEqual(dim_score.score, 0.8, places=5)
        self.assertEqual(len(dim_score.question_scores), 5)
        self.assertEqual(dim_score.weight, DIMENSION_WEIGHTS["P1"]["D1"])

    def test_policy_score_calculation(self):
        # Create dimension scores for all 6 dimensions
        dimension_scores = []
        for dim_num in range(1, 7):
            dimension = f"D{dim_num}"
            question_scores = [
                QuestionScore(f"P2-{dimension}-Q{i}", 0.7, "bueno") 
                for i in range(1, 6)
            ]
            dim_score = self.engine.calculate_dimension_score("P2", dimension, question_scores)
            dimension_scores.append(dim_score)
        
        policy_score = self.engine.calculate_policy_score("P2", dimension_scores)
        
        self.assertEqual(policy_score.policy, "P2")
        self.assertAlmostEqual(policy_score.score, 0.7, places=5)
        self.assertEqual(len(policy_score.dimension_scores), 6)

    def test_weighted_policy_score(self):
        # Test with varied dimension scores
        dimension_scores = []
        test_scores = {"D1": 0.6, "D2": 0.8, "D3": 0.7, "D4": 0.9, "D5": 0.75, "D6": 0.65}
        
        for dimension, score in test_scores.items():
            question_scores = [
                QuestionScore(f"P3-{dimension}-Q{i}", score, "bueno") 
                for i in range(1, 6)
            ]
            dim_score = self.engine.calculate_dimension_score("P3", dimension, question_scores)
            dimension_scores.append(dim_score)
        
        policy_score = self.engine.calculate_policy_score("P3", dimension_scores)
        
        # Calculate expected weighted average
        expected = sum(test_scores[dim] * DIMENSION_WEIGHTS["P3"][dim] 
                      for dim in test_scores.keys())
        
        self.assertAlmostEqual(policy_score.score, expected, places=5)

    def test_cluster_score_calculation(self):
        # Create policy scores for derechos_humanos cluster: P1, P2, P8
        policy_scores = []
        for policy in ["P1", "P2", "P8"]:
            dimension_scores = []
            for dim_num in range(1, 7):
                dimension = f"D{dim_num}"
                question_scores = [
                    QuestionScore(f"{policy}-{dimension}-Q{i}", 0.75, "bueno") 
                    for i in range(1, 6)
                ]
                dim_score = self.engine.calculate_dimension_score(policy, dimension, question_scores)
                dimension_scores.append(dim_score)
            
            ps = self.engine.calculate_policy_score(policy, dimension_scores)
            policy_scores.append(ps)
        
        cluster_score = self.engine.calculate_cluster_score("derechos_humanos", policy_scores)
        
        self.assertEqual(cluster_score.cluster_name, "derechos_humanos")
        self.assertEqual(len(cluster_score.policy_scores), 3)
        self.assertAlmostEqual(cluster_score.weight, CLUSTER_WEIGHTS["derechos_humanos"], places=5)


class TestD6ManualReviewFlag(unittest.TestCase):
    """Test D6 < 0.55 triggers manual review"""

    def setUp(self):
        self.engine = ScoringEngine()

    def test_d6_below_threshold_triggers_flag(self):
        # Create a cluster with one policy having D6 < 0.55
        policy_scores = []
        
        for policy in ["P1", "P2", "P8"]:  # Use actual derechos_humanos cluster
            dimension_scores = []
            for dim_num in range(1, 7):
                dimension = f"D{dim_num}"
                # D6 gets low score for P1 only
                if policy == "P1" and dimension == "D6":
                    score = 0.50
                else:
                    score = 0.80
                question_scores = [
                    QuestionScore(f"{policy}-{dimension}-Q{i}", score, "bueno") 
                    for i in range(1, 6)
                ]
                dim_score = self.engine.calculate_dimension_score(policy, dimension, question_scores)
                dimension_scores.append(dim_score)
            
            ps = self.engine.calculate_policy_score(policy, dimension_scores)
            policy_scores.append(ps)
        
        cluster_score = self.engine.calculate_cluster_score("derechos_humanos", policy_scores)
        
        flags = self.engine._check_manual_review_triggers([cluster_score])
        
        self.assertGreater(len(flags), 0)
        self.assertIn("P1-D6", flags[0])
        self.assertIn(str(D6_MANUAL_REVIEW_THRESHOLD), flags[0])

    def test_d6_above_threshold_no_flag(self):
        # Create a cluster with D6 >= 0.55
        policy_scores = []
        
        for policy in ["P3", "P7"]:  # Use actual sostenibilidad cluster
            dimension_scores = []
            for dim_num in range(1, 7):
                dimension = f"D{dim_num}"
                score = 0.70  # All dimensions above threshold
                question_scores = [
                    QuestionScore(f"{policy}-{dimension}-Q{i}", score, "bueno") 
                    for i in range(1, 6)
                ]
                dim_score = self.engine.calculate_dimension_score(policy, dimension, question_scores)
                dimension_scores.append(dim_score)
            
            ps = self.engine.calculate_policy_score(policy, dimension_scores)
            policy_scores.append(ps)
        
        cluster_score = self.engine.calculate_cluster_score("sostenibilidad", policy_scores)
        
        flags = self.engine._check_manual_review_triggers([cluster_score])
        
        self.assertEqual(len(flags), 0)


class TestDNPIntegration(unittest.TestCase):
    """Test DNP compliance integration at D1-Q5 and D4-Q5"""

    def setUp(self):
        self.engine = ScoringEngine()

    def test_d1_q5_detection(self):
        qs = QuestionScore("P3-D1-Q5", 0.75, "bueno")
        
        # Without DNP validator, should not modify score
        result = self.engine.integrate_dnp_compliance(qs, dnp_validator=None)
        
        self.assertIsNotNone(result.dnp_compliance)
        self.assertEqual(result.dnp_compliance["compliance"], "unknown")

    def test_d4_q5_detection(self):
        qs = QuestionScore("P5-D4-Q5", 0.80, "bueno")
        
        # Without DNP validator, should not modify score
        result = self.engine.integrate_dnp_compliance(qs, dnp_validator=None)
        
        self.assertIsNotNone(result.dnp_compliance)
        # D4-Q5 returns "compliance" key not "pnd_alignment" when validator is None
        self.assertIn("compliance", result.dnp_compliance)

    def test_non_dnp_questions_unchanged(self):
        qs = QuestionScore("P1-D2-Q3", 0.85, "excelente")
        
        result = self.engine.integrate_dnp_compliance(qs, dnp_validator=None)
        
        # Should not have DNP compliance data for non-D1-Q5/D4-Q5 questions
        self.assertIsNone(result.dnp_compliance)


class TestProvenanceChain(unittest.TestCase):
    """Test complete provenance documentation"""

    def setUp(self):
        self.engine = ScoringEngine()

    def test_provenance_structure(self):
        # Create minimal cluster scores
        cluster_scores = []
        
        for cluster_name, policies in POLICY_CLUSTERS.items():
            policy_scores = []
            for policy in policies:
                dimension_scores = []
                for dim_num in range(1, 7):
                    dimension = f"D{dim_num}"
                    question_scores = [
                        QuestionScore(f"{policy}-{dimension}-Q{i}", 0.75, "bueno") 
                        for i in range(1, 6)
                    ]
                    dim_score = self.engine.calculate_dimension_score(policy, dimension, question_scores)
                    dimension_scores.append(dim_score)
                
                ps = self.engine.calculate_policy_score(policy, dimension_scores)
                policy_scores.append(ps)
            
            cs = self.engine.calculate_cluster_score(cluster_name, policy_scores)
            cluster_scores.append(cs)
        
        macro_score = self.engine.calculate_macro_score(cluster_scores)
        
        # Check provenance structure
        self.assertIn("provenance", macro_score.__dict__)
        provenance = macro_score.provenance
        
        self.assertIn("aggregation_method", provenance)
        self.assertIn("levels", provenance)
        self.assertIn("dimension_weights", provenance)
        self.assertIn("cluster_weights", provenance)
        self.assertIn("rubric_thresholds", provenance)
        self.assertIn("cluster_breakdown", provenance)
        
        # Check level documentation
        levels = provenance["levels"]
        self.assertIn("MICRO", levels)
        self.assertIn("MESO_dimension", levels)
        self.assertIn("MESO_policy", levels)
        self.assertIn("MACRO_cluster", levels)
        self.assertIn("MACRO_overall", levels)


class TestFrameworkValidation(unittest.TestCase):
    """Test complete framework validation"""

    def test_validate_scoring_framework(self):
        report = validate_scoring_framework()
        
        self.assertTrue(report["configuration_valid"])
        self.assertTrue(report["canonical_questions_valid"])
        self.assertTrue(report["dimension_weights_valid"])
        self.assertTrue(report["rubric_thresholds_valid"])
        self.assertTrue(report["all_valid"])
        self.assertEqual(len(report["errors"]), 0)


if __name__ == "__main__":
    unittest.main()
