#!/usr/bin/env python3
"""
Unit tests for Canonical Notation System
=========================================

Comprehensive test suite for canonical notation validation,
ID parsing, evidence entry creation, and legacy migration.

Author: AI Systems Architect
Version: 2.0.0
"""

import unittest
import json
from canonical_notation import (
    CanonicalID,
    RubricKey,
    EvidenceEntry,
    CanonicalNotationValidator,
    PolicyArea,
    AnalyticalDimension,
    generate_default_questions,
    get_system_structure_summary,
    QUESTION_UNIQUE_ID_PATTERN,
    RUBRIC_KEY_PATTERN,
    POLICY_PATTERN,
    DIMENSION_PATTERN
)


class TestPolicyArea(unittest.TestCase):
    """Test PolicyArea enum"""

    def test_all_policies_exist(self):
        """Test that all 10 policies are defined"""
        self.assertEqual(len(PolicyArea), 10)

    def test_get_title_valid(self):
        """Test getting title for valid policy"""
        self.assertEqual(
            PolicyArea.get_title("P1"),
            "Derechos de las mujeres e igualdad de género"
        )
        self.assertEqual(
            PolicyArea.get_title("P10"),
            "Migración transfronteriza"
        )

    def test_get_title_invalid(self):
        """Test getting title for invalid policy raises error"""
        with self.assertRaises(ValueError):
            PolicyArea.get_title("P11")
        with self.assertRaises(ValueError):
            PolicyArea.get_title("P0")


class TestAnalyticalDimension(unittest.TestCase):
    """Test AnalyticalDimension enum"""

    def test_all_dimensions_exist(self):
        """Test that all 6 dimensions are defined"""
        self.assertEqual(len(AnalyticalDimension), 6)

    def test_get_name_valid(self):
        """Test getting name for valid dimension"""
        self.assertEqual(
            AnalyticalDimension.get_name("D1"),
            "Diagnóstico y Recursos"
        )
        self.assertEqual(
            AnalyticalDimension.get_name("D6"),
            "Teoría de Cambio y Coherencia Causal"
        )

    def test_get_focus_valid(self):
        """Test getting focus for valid dimension"""
        focus = AnalyticalDimension.get_focus("D1")
        self.assertIn("Baseline", focus)
        self.assertIn("resources", focus)


class TestRubricKey(unittest.TestCase):
    """Test RubricKey dataclass"""

    def test_valid_creation(self):
        """Test creating valid rubric keys"""
        rk = RubricKey(dimension="D2", question=3)
        self.assertEqual(rk.dimension, "D2")
        self.assertEqual(rk.question, 3)
        self.assertEqual(str(rk), "D2-Q3")

    def test_invalid_dimension(self):
        """Test invalid dimension raises error"""
        with self.assertRaises(ValueError):
            RubricKey(dimension="D7", question=1)
        with self.assertRaises(ValueError):
            RubricKey(dimension="D0", question=1)
        with self.assertRaises(ValueError):
            RubricKey(dimension="X1", question=1)

    def test_invalid_question(self):
        """Test invalid question number raises error"""
        with self.assertRaises(ValueError):
            RubricKey(dimension="D1", question=0)
        with self.assertRaises(ValueError):
            RubricKey(dimension="D1", question=-1)

    def test_from_string_valid(self):
        """Test parsing valid rubric key strings"""
        rk = RubricKey.from_string("D2-Q3")
        self.assertEqual(rk.dimension, "D2")
        self.assertEqual(rk.question, 3)
        
        rk = RubricKey.from_string("D6-Q100")
        self.assertEqual(rk.dimension, "D6")
        self.assertEqual(rk.question, 100)

    def test_from_string_invalid(self):
        """Test parsing invalid rubric key strings"""
        with self.assertRaises(ValueError):
            RubricKey.from_string("D7-Q3")
        with self.assertRaises(ValueError):
            RubricKey.from_string("D2-Q0")
        with self.assertRaises(ValueError):
            RubricKey.from_string("P4-D2-Q3")

    def test_to_dict(self):
        """Test converting rubric key to dictionary"""
        rk = RubricKey(dimension="D2", question=3)
        d = rk.to_dict()
        self.assertEqual(d["dimension"], "D2")
        self.assertEqual(d["question"], 3)
        self.assertEqual(d["rubric_key"], "D2-Q3")


class TestCanonicalID(unittest.TestCase):
    """Test CanonicalID dataclass"""

    def test_valid_creation(self):
        """Test creating valid canonical IDs"""
        cid = CanonicalID(policy="P4", dimension="D2", question=3)
        self.assertEqual(cid.policy, "P4")
        self.assertEqual(cid.dimension, "D2")
        self.assertEqual(cid.question, 3)
        self.assertEqual(str(cid), "P4-D2-Q3")

    def test_p10_valid(self):
        """Test P10 is valid"""
        cid = CanonicalID(policy="P10", dimension="D6", question=30)
        self.assertEqual(str(cid), "P10-D6-Q30")

    def test_invalid_policy(self):
        """Test invalid policy raises error"""
        with self.assertRaises(ValueError):
            CanonicalID(policy="P11", dimension="D1", question=1)
        with self.assertRaises(ValueError):
            CanonicalID(policy="P0", dimension="D1", question=1)

    def test_invalid_dimension(self):
        """Test invalid dimension raises error"""
        with self.assertRaises(ValueError):
            CanonicalID(policy="P1", dimension="D7", question=1)
        with self.assertRaises(ValueError):
            CanonicalID(policy="P1", dimension="D0", question=1)

    def test_invalid_question(self):
        """Test invalid question raises error"""
        with self.assertRaises(ValueError):
            CanonicalID(policy="P1", dimension="D1", question=0)
        with self.assertRaises(ValueError):
            CanonicalID(policy="P1", dimension="D1", question=-1)

    def test_from_string_valid(self):
        """Test parsing valid canonical ID strings"""
        cid = CanonicalID.from_string("P4-D2-Q3")
        self.assertEqual(cid.policy, "P4")
        self.assertEqual(cid.dimension, "D2")
        self.assertEqual(cid.question, 3)
        
        cid = CanonicalID.from_string("P10-D6-Q30")
        self.assertEqual(cid.policy, "P10")
        self.assertEqual(cid.question, 30)

    def test_from_string_invalid(self):
        """Test parsing invalid canonical ID strings"""
        with self.assertRaises(ValueError):
            CanonicalID.from_string("P11-D2-Q3")
        with self.assertRaises(ValueError):
            CanonicalID.from_string("P4-D7-Q3")
        with self.assertRaises(ValueError):
            CanonicalID.from_string("P4-D2-Q0")
        with self.assertRaises(ValueError):
            CanonicalID.from_string("D2-Q3")

    def test_to_rubric_key(self):
        """Test deriving rubric key from canonical ID"""
        cid = CanonicalID(policy="P4", dimension="D2", question=3)
        rk = cid.to_rubric_key()
        self.assertEqual(str(rk), "D2-Q3")
        self.assertEqual(rk.dimension, "D2")
        self.assertEqual(rk.question, 3)

    def test_get_policy_title(self):
        """Test getting policy title"""
        cid = CanonicalID(policy="P1", dimension="D1", question=1)
        title = cid.get_policy_title()
        self.assertIn("mujeres", title.lower())

    def test_get_dimension_name(self):
        """Test getting dimension name"""
        cid = CanonicalID(policy="P1", dimension="D2", question=1)
        name = cid.get_dimension_name()
        self.assertIn("Diseño", name)

    def test_to_dict(self):
        """Test converting canonical ID to dictionary"""
        cid = CanonicalID(policy="P7", dimension="D3", question=5)
        d = cid.to_dict()
        self.assertEqual(d["policy"], "P7")
        self.assertEqual(d["dimension"], "D3")
        self.assertEqual(d["question"], 5)
        self.assertEqual(d["question_unique_id"], "P7-D3-Q5")
        self.assertEqual(d["rubric_key"], "D3-Q5")
        self.assertIn("policy_title", d)
        self.assertIn("dimension_name", d)


class TestEvidenceEntry(unittest.TestCase):
    """Test EvidenceEntry dataclass"""

    def test_create_valid(self):
        """Test creating valid evidence entry"""
        evidence = EvidenceEntry.create(
            policy="P7",
            dimension="D3",
            question=5,
            score=0.82,
            confidence=0.82,
            stage="teoria_cambio",
            evidence_id_prefix="toc_"
        )
        self.assertEqual(evidence.evidence_id, "toc_P7-D3-Q5")
        self.assertEqual(evidence.question_unique_id, "P7-D3-Q5")
        self.assertEqual(evidence.content["rubric_key"], "D3-Q5")
        self.assertEqual(evidence.confidence, 0.82)
        self.assertEqual(evidence.stage, "teoria_cambio")

    def test_create_without_prefix(self):
        """Test creating evidence without prefix"""
        evidence = EvidenceEntry.create(
            policy="P1",
            dimension="D1",
            question=1,
            score=0.5,
            confidence=0.5,
            stage="test"
        )
        self.assertEqual(evidence.evidence_id, "P1-D1-Q1")

    def test_invalid_confidence(self):
        """Test invalid confidence values"""
        with self.assertRaises(ValueError):
            EvidenceEntry.create(
                policy="P1",
                dimension="D1",
                question=1,
                score=0.5,
                confidence=1.5,
                stage="test"
            )
        with self.assertRaises(ValueError):
            EvidenceEntry.create(
                policy="P1",
                dimension="D1",
                question=1,
                score=0.5,
                confidence=-0.1,
                stage="test"
            )

    def test_invalid_score(self):
        """Test invalid score values"""
        with self.assertRaises(ValueError):
            EvidenceEntry.create(
                policy="P1",
                dimension="D1",
                question=1,
                score=1.5,
                confidence=0.5,
                stage="test"
            )

    def test_to_json(self):
        """Test converting evidence to JSON"""
        evidence = EvidenceEntry.create(
            policy="P1",
            dimension="D1",
            question=1,
            score=0.5,
            confidence=0.5,
            stage="test"
        )
        json_str = evidence.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["evidence_id"], "P1-D1-Q1")
        self.assertEqual(parsed["content"]["score"], 0.5)

    def test_content_consistency(self):
        """Test that content is consistent with question_unique_id"""
        # This should work fine
        evidence = EvidenceEntry.create(
            policy="P5",
            dimension="D4",
            question=10,
            score=0.75,
            confidence=0.8,
            stage="test"
        )
        self.assertEqual(evidence.content["policy"], "P5")
        self.assertEqual(evidence.content["dimension"], "D4")
        self.assertEqual(evidence.content["question"], 10)


class TestCanonicalNotationValidator(unittest.TestCase):
    """Test CanonicalNotationValidator"""

    def setUp(self):
        self.validator = CanonicalNotationValidator()

    def test_validate_question_unique_id(self):
        """Test question unique ID validation"""
        # Valid cases
        self.assertTrue(self.validator.validate_question_unique_id("P1-D1-Q1"))
        self.assertTrue(self.validator.validate_question_unique_id("P4-D2-Q3"))
        self.assertTrue(self.validator.validate_question_unique_id("P10-D6-Q30"))
        self.assertTrue(self.validator.validate_question_unique_id("P9-D3-Q100"))
        
        # Invalid cases
        self.assertFalse(self.validator.validate_question_unique_id("P11-D1-Q1"))
        self.assertFalse(self.validator.validate_question_unique_id("P0-D1-Q1"))
        self.assertFalse(self.validator.validate_question_unique_id("P1-D7-Q1"))
        self.assertFalse(self.validator.validate_question_unique_id("P1-D1-Q0"))
        self.assertFalse(self.validator.validate_question_unique_id("D1-Q1"))

    def test_validate_rubric_key(self):
        """Test rubric key validation"""
        # Valid cases
        self.assertTrue(self.validator.validate_rubric_key("D1-Q1"))
        self.assertTrue(self.validator.validate_rubric_key("D2-Q3"))
        self.assertTrue(self.validator.validate_rubric_key("D6-Q30"))
        self.assertTrue(self.validator.validate_rubric_key("D3-Q100"))
        
        # Invalid cases
        self.assertFalse(self.validator.validate_rubric_key("D7-Q1"))
        self.assertFalse(self.validator.validate_rubric_key("D0-Q1"))
        self.assertFalse(self.validator.validate_rubric_key("D1-Q0"))
        self.assertFalse(self.validator.validate_rubric_key("P1-D1-Q1"))

    def test_validate_policy(self):
        """Test policy validation"""
        # Valid cases
        self.assertTrue(self.validator.validate_policy("P1"))
        self.assertTrue(self.validator.validate_policy("P5"))
        self.assertTrue(self.validator.validate_policy("P10"))
        
        # Invalid cases
        self.assertFalse(self.validator.validate_policy("P0"))
        self.assertFalse(self.validator.validate_policy("P11"))
        self.assertFalse(self.validator.validate_policy("D1"))

    def test_validate_dimension(self):
        """Test dimension validation"""
        # Valid cases
        self.assertTrue(self.validator.validate_dimension("D1"))
        self.assertTrue(self.validator.validate_dimension("D6"))
        
        # Invalid cases
        self.assertFalse(self.validator.validate_dimension("D0"))
        self.assertFalse(self.validator.validate_dimension("D7"))
        self.assertFalse(self.validator.validate_dimension("P1"))

    def test_extract_rubric_key_from_question_id(self):
        """Test extracting rubric key from question ID"""
        self.assertEqual(
            self.validator.extract_rubric_key_from_question_id("P4-D2-Q3"),
            "D2-Q3"
        )
        self.assertEqual(
            self.validator.extract_rubric_key_from_question_id("P10-D6-Q30"),
            "D6-Q30"
        )

    def test_migrate_legacy_id_with_policy(self):
        """Test migrating legacy ID with inferred policy"""
        result = self.validator.migrate_legacy_id("D2-Q3", inferred_policy="P4")
        self.assertEqual(result, "P4-D2-Q3")

    def test_migrate_legacy_id_without_policy(self):
        """Test migrating legacy ID without policy raises error"""
        with self.assertRaises(ValueError):
            self.validator.migrate_legacy_id("D2-Q3")

    def test_migrate_legacy_id_invalid_policy(self):
        """Test migrating with invalid inferred policy"""
        with self.assertRaises(ValueError):
            self.validator.migrate_legacy_id("D2-Q3", inferred_policy="P11")

    def test_migrate_legacy_id_already_canonical(self):
        """Test migrating ID that's already canonical"""
        result = self.validator.migrate_legacy_id("P4-D2-Q3")
        self.assertEqual(result, "P4-D2-Q3")

    def test_migrate_legacy_id_invalid_format(self):
        """Test migrating invalid format returns None"""
        result = self.validator.migrate_legacy_id("INVALID-FORMAT")
        self.assertIsNone(result)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""

    def test_generate_default_questions(self):
        """Test generating default question structure"""
        questions = generate_default_questions(max_questions_per_dimension=5)
        self.assertEqual(len(questions), 300)  # 10 * 6 * 5
        
        # Check first question
        self.assertEqual(str(questions[0]), "P1-D1-Q1")
        
        # Check last question
        self.assertEqual(str(questions[-1]), "P10-D6-Q5")

    def test_generate_custom_questions(self):
        """Test generating with custom number of questions"""
        questions = generate_default_questions(max_questions_per_dimension=3)
        self.assertEqual(len(questions), 180)  # 10 * 6 * 3

    def test_get_system_structure_summary(self):
        """Test getting system structure summary"""
        summary = get_system_structure_summary()
        self.assertEqual(summary["total_policies"], 10)
        self.assertEqual(summary["total_dimensions"], 6)
        self.assertEqual(summary["default_questions_per_dimension"], 5)
        self.assertEqual(summary["default_total_questions"], 300)
        self.assertIn("policies", summary)
        self.assertIn("dimensions", summary)
        self.assertIn("patterns", summary)


class TestRegexPatterns(unittest.TestCase):
    """Test regex patterns"""

    def test_question_unique_id_pattern(self):
        """Test question unique ID regex pattern"""
        # Valid
        self.assertTrue(QUESTION_UNIQUE_ID_PATTERN.match("P1-D1-Q1"))
        self.assertTrue(QUESTION_UNIQUE_ID_PATTERN.match("P10-D6-Q30"))
        self.assertTrue(QUESTION_UNIQUE_ID_PATTERN.match("P5-D3-Q999"))
        
        # Invalid
        self.assertFalse(QUESTION_UNIQUE_ID_PATTERN.match("P0-D1-Q1"))
        self.assertFalse(QUESTION_UNIQUE_ID_PATTERN.match("P11-D1-Q1"))
        self.assertFalse(QUESTION_UNIQUE_ID_PATTERN.match("P1-D0-Q1"))
        self.assertFalse(QUESTION_UNIQUE_ID_PATTERN.match("P1-D7-Q1"))
        self.assertFalse(QUESTION_UNIQUE_ID_PATTERN.match("P1-D1-Q0"))

    def test_rubric_key_pattern(self):
        """Test rubric key regex pattern"""
        # Valid
        self.assertTrue(RUBRIC_KEY_PATTERN.match("D1-Q1"))
        self.assertTrue(RUBRIC_KEY_PATTERN.match("D6-Q30"))
        self.assertTrue(RUBRIC_KEY_PATTERN.match("D3-Q999"))
        
        # Invalid
        self.assertFalse(RUBRIC_KEY_PATTERN.match("D0-Q1"))
        self.assertFalse(RUBRIC_KEY_PATTERN.match("D7-Q1"))
        self.assertFalse(RUBRIC_KEY_PATTERN.match("D1-Q0"))

    def test_policy_pattern(self):
        """Test policy regex pattern"""
        # Valid
        self.assertTrue(POLICY_PATTERN.match("P1"))
        self.assertTrue(POLICY_PATTERN.match("P10"))
        
        # Invalid
        self.assertFalse(POLICY_PATTERN.match("P0"))
        self.assertFalse(POLICY_PATTERN.match("P11"))

    def test_dimension_pattern(self):
        """Test dimension regex pattern"""
        # Valid
        self.assertTrue(DIMENSION_PATTERN.match("D1"))
        self.assertTrue(DIMENSION_PATTERN.match("D6"))
        
        # Invalid
        self.assertFalse(DIMENSION_PATTERN.match("D0"))
        self.assertFalse(DIMENSION_PATTERN.match("D7"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
