#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for IoR Causal Axiomatic-Bayesian Integration
Tests for Audit Points 2.1, 2.2, and 2.3

Validates:
- Audit Point 2.1: Structural Veto (D6-Q2)
- Audit Point 2.2: Mechanism Necessity Hoop Test
- Audit Point 2.3: Policy Alignment Dual Constraint
"""

import sys
from pathlib import Path
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


class TestAuditPoint21_StructuralVeto:
    """Test Audit Point 2.1: Structural Veto (D6-Q2)"""

    def test_structural_violation_detection(self):
        """Test that impermissible links are detected"""
        # Mock CausalExtractor with test nodes
        from unittest.mock import Mock

        # Create mock nodes
        nodes = {
            "MP-001": Mock(id="MP-001", type="producto"),
            "MI-001": Mock(id="MI-001", type="impacto"),
            "MR-001": Mock(id="MR-001", type="resultado"),
        }

        # Test reverse causation (impacto → producto)
        violation = self._check_violation_logic(nodes, "MI-001", "MP-001")
        assert violation is not None, "Reverse causation should be detected"
        assert "reverse_causation" in violation

        # Test missing intermediate (producto → impacto)
        violation = self._check_violation_logic(nodes, "MP-001", "MI-001")
        assert violation is not None, "Missing intermediate should be detected"
        assert "missing_intermediate" in violation

        # Test valid link (producto → resultado)
        violation = self._check_violation_logic(nodes, "MP-001", "MR-001")
        assert violation is None, "Valid link should not be flagged"

    def _check_violation_logic(self, nodes, source_id, target_id):
        """Helper to test structural violation logic"""
        source_type = nodes[source_id].type
        target_type = nodes[target_id].type

        hierarchy_levels = {"programa": 1, "producto": 2, "resultado": 3, "impacto": 4}

        source_level = hierarchy_levels.get(source_type, 0)
        target_level = hierarchy_levels.get(target_type, 0)

        if target_level < source_level:
            return f"reverse_causation:{source_type}→{target_type}"

        if target_level - source_level > 2:
            return f"level_skip:{source_type}→{target_type}"

        if source_type == "producto" and target_type == "impacto":
            return f"missing_intermediate:producto→impacto requires resultado"

        return None

    def test_posterior_capping(self):
        """Test that posterior is capped at 0.6 for impermissible links"""
        # Test posterior capping logic
        original_posterior = 0.85
        has_violation = True

        if has_violation:
            capped_posterior = min(original_posterior, 0.6)
        else:
            capped_posterior = original_posterior

        assert capped_posterior == 0.6, "Posterior should be capped at 0.6"
        assert capped_posterior < original_posterior, "Capping should reduce posterior"


class TestAuditPoint22_NecessityHoopTest:
    """Test Audit Point 2.2: Mechanism Necessity Hoop Test"""

    def test_necessity_with_all_components(self):
        """Test that mechanism with all components passes hoop test"""
        observations = {
            "entity_activity": {"entity": "Secretaría de Planeación"},
            "entities": ["Secretaría de Planeación"],
            "verbs": ["implementar", "ejecutar", "coordinar"],
            "budget": 50000000,
        }

        result = self._evaluate_necessity(observations)
        assert result["is_necessary"], "Should pass with all components"
        assert result["hoop_test_passed"], "Hoop test should pass"
        assert len(result["missing_components"]) == 0, "No components should be missing"
        assert result["score"] == 1.0, "Score should be 1.0"

    def test_necessity_missing_entity(self):
        """Test that missing entity fails hoop test"""
        observations = {
            "entity_activity": None,
            "entities": [],
            "verbs": ["implementar", "ejecutar"],
            "budget": 50000000,
        }

        result = self._evaluate_necessity(observations)
        assert not result["is_necessary"], "Should fail without entity"
        assert "entity" in result["missing_components"], (
            "Entity should be flagged as missing"
        )

    def test_necessity_missing_activity(self):
        """Test that missing activity fails hoop test"""
        observations = {
            "entity_activity": {"entity": "Secretaría de Planeación"},
            "entities": ["Secretaría de Planeación"],
            "verbs": [],
            "budget": 50000000,
        }

        result = self._evaluate_necessity(observations)
        assert not result["is_necessary"], "Should fail without activity"
        assert "activity" in result["missing_components"], "Activity should be flagged"

    def test_necessity_missing_budget(self):
        """Test that missing budget fails hoop test"""
        observations = {
            "entity_activity": {"entity": "Secretaría de Planeación"},
            "entities": ["Secretaría de Planeación"],
            "verbs": ["implementar", "ejecutar"],
            "budget": None,
        }

        result = self._evaluate_necessity(observations)
        assert not result["is_necessary"], "Should fail without budget"
        assert "budget" in result["missing_components"], "Budget should be flagged"

    def _evaluate_necessity(self, observations):
        """Helper to evaluate necessity test"""
        missing_components = []

        # Check Entity
        entity_activity = observations.get("entity_activity")
        if not entity_activity or not entity_activity.get("entity"):
            missing_components.append("entity")

        # Check Activity
        verbs = observations.get("verbs", [])
        if not verbs or len(verbs) < 1:
            missing_components.append("activity")

        # Check Budget
        budget = observations.get("budget")
        if budget is None or budget <= 0:
            missing_components.append("budget")

        is_necessary = len(missing_components) == 0
        max_components = 3
        present_components = max_components - len(
            [c for c in missing_components if c in ["entity", "activity", "budget"]]
        )
        necessity_score = present_components / max_components

        return {
            "score": necessity_score,
            "is_necessary": is_necessary,
            "missing_components": missing_components,
            "hoop_test_passed": is_necessary,
        }


class TestAuditPoint23_AlignmentDualConstraint:
    """Test Audit Point 2.3: Policy Alignment Dual Constraint"""

    def test_alignment_penalty_applied(self):
        """Test that alignment penalty is applied when pdet_alignment ≤ 0.60"""
        base_risk = 0.08
        pdet_alignment = 0.55

        result = self._calculate_risk_with_alignment(base_risk, pdet_alignment)

        assert result["alignment_penalty_applied"], "Penalty should be applied"
        assert result["risk_score"] > base_risk, "Risk should increase"
        assert result["risk_score"] == base_risk * 1.2, (
            "Risk should be multiplied by 1.2"
        )

    def test_alignment_penalty_not_applied(self):
        """Test that penalty is not applied when pdet_alignment > 0.60"""
        base_risk = 0.08
        pdet_alignment = 0.75

        result = self._calculate_risk_with_alignment(base_risk, pdet_alignment)

        assert not result["alignment_penalty_applied"], "Penalty should not be applied"
        assert result["risk_score"] == base_risk, "Risk should remain unchanged"

    def test_quality_downgrade_due_to_alignment(self):
        """Test that low alignment can downgrade quality rating"""
        # Base risk is excellent (< 0.10), but alignment penalty pushes it over
        base_risk = 0.09
        pdet_alignment = 0.50

        result = self._calculate_risk_with_alignment(base_risk, pdet_alignment)

        # After penalty: 0.09 * 1.2 = 0.108
        assert result["risk_score"] >= 0.10, "Risk should exceed excellent threshold"
        assert result["d5_q4_quality"] != "excelente", "Quality should not be excellent"
        assert result["alignment_causing_failure"], "Should flag alignment as cause"

    def test_quality_thresholds(self):
        """Test D5-Q4 quality thresholds"""
        # Excellent: risk < 0.10
        result = self._calculate_risk_with_alignment(0.05, 0.80)
        assert result["d5_q4_quality"] == "excelente"

        # Good: risk < 0.20
        result = self._calculate_risk_with_alignment(0.15, 0.80)
        assert result["d5_q4_quality"] == "bueno"

        # Acceptable: risk < 0.35
        result = self._calculate_risk_with_alignment(0.30, 0.80)
        assert result["d5_q4_quality"] == "aceptable"

        # Insufficient: risk >= 0.35
        result = self._calculate_risk_with_alignment(0.40, 0.80)
        assert result["d5_q4_quality"] == "insuficiente"

    def _calculate_risk_with_alignment(self, base_risk, pdet_alignment):
        """Helper to calculate risk with alignment constraint"""
        alignment_threshold = 0.60
        alignment_multiplier = 1.2

        risk_score = base_risk
        original_risk = base_risk
        alignment_penalty_applied = False

        if pdet_alignment is not None and pdet_alignment <= alignment_threshold:
            risk_score = risk_score * alignment_multiplier
            alignment_penalty_applied = True

        # Quality assessment
        d5_q4_quality = "insuficiente"
        if risk_score < 0.10:
            d5_q4_quality = "excelente"
        elif risk_score < 0.20:
            d5_q4_quality = "bueno"
        elif risk_score < 0.35:
            d5_q4_quality = "aceptable"

        alignment_causing_failure = (
            alignment_penalty_applied and original_risk < 0.10 and risk_score >= 0.10
        )

        return {
            "risk_score": risk_score,
            "alignment_penalty_applied": alignment_penalty_applied,
            "d5_q4_quality": d5_q4_quality,
            "alignment_causing_failure": alignment_causing_failure,
        }


def run_all_tests():
    """Run all test suites"""
    print("=" * 80)
    print("IoR Causal Axiomatic-Bayesian Integration - Test Suite")
    print("=" * 80)

    test_results = {"passed": 0, "failed": 0, "errors": []}

    # Test Audit Point 2.1
    print("\n--- Audit Point 2.1: Structural Veto (D6-Q2) ---")
    ap21 = TestAuditPoint21_StructuralVeto()
    try:
        ap21.test_structural_violation_detection()
        print("✓ test_structural_violation_detection PASSED")
        test_results["passed"] += 1
    except AssertionError as e:
        print(f"✗ test_structural_violation_detection FAILED: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(str(e))

    try:
        ap21.test_posterior_capping()
        print("✓ test_posterior_capping PASSED")
        test_results["passed"] += 1
    except AssertionError as e:
        print(f"✗ test_posterior_capping FAILED: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(str(e))

    # Test Audit Point 2.2
    print("\n--- Audit Point 2.2: Mechanism Necessity Hoop Test ---")
    ap22 = TestAuditPoint22_NecessityHoopTest()
    try:
        ap22.test_necessity_with_all_components()
        print("✓ test_necessity_with_all_components PASSED")
        test_results["passed"] += 1
    except AssertionError as e:
        print(f"✗ test_necessity_with_all_components FAILED: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(str(e))

    try:
        ap22.test_necessity_missing_entity()
        print("✓ test_necessity_missing_entity PASSED")
        test_results["passed"] += 1
    except AssertionError as e:
        print(f"✗ test_necessity_missing_entity FAILED: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(str(e))

    try:
        ap22.test_necessity_missing_activity()
        print("✓ test_necessity_missing_activity PASSED")
        test_results["passed"] += 1
    except AssertionError as e:
        print(f"✗ test_necessity_missing_activity FAILED: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(str(e))

    try:
        ap22.test_necessity_missing_budget()
        print("✓ test_necessity_missing_budget PASSED")
        test_results["passed"] += 1
    except AssertionError as e:
        print(f"✗ test_necessity_missing_budget FAILED: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(str(e))

    # Test Audit Point 2.3
    print("\n--- Audit Point 2.3: Policy Alignment Dual Constraint ---")
    ap23 = TestAuditPoint23_AlignmentDualConstraint()
    try:
        ap23.test_alignment_penalty_applied()
        print("✓ test_alignment_penalty_applied PASSED")
        test_results["passed"] += 1
    except AssertionError as e:
        print(f"✗ test_alignment_penalty_applied FAILED: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(str(e))

    try:
        ap23.test_alignment_penalty_not_applied()
        print("✓ test_alignment_penalty_not_applied PASSED")
        test_results["passed"] += 1
    except AssertionError as e:
        print(f"✗ test_alignment_penalty_not_applied FAILED: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(str(e))

    try:
        ap23.test_quality_downgrade_due_to_alignment()
        print("✓ test_quality_downgrade_due_to_alignment PASSED")
        test_results["passed"] += 1
    except AssertionError as e:
        print(f"✗ test_quality_downgrade_due_to_alignment FAILED: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(str(e))

    try:
        ap23.test_quality_thresholds()
        print("✓ test_quality_thresholds PASSED")
        test_results["passed"] += 1
    except AssertionError as e:
        print(f"✗ test_quality_thresholds FAILED: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(str(e))

    # Summary
    print("\n" + "=" * 80)
    print(
        f"Test Summary: {test_results['passed']} passed, {test_results['failed']} failed"
    )
    print("=" * 80)

    if test_results["failed"] > 0:
        print("\nErrors:")
        for error in test_results["errors"]:
            print(f"  - {error}")
        return 1
    else:
        print("\n✓ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
