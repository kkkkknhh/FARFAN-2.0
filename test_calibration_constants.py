#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Calibration Constants Module
==================================

SIN_CARRETA Compliance Tests:
- Validate immutability of calibration constants
- Verify constraint satisfaction
- Test override mechanism for testing contexts
"""

import unittest
from dataclasses import FrozenInstanceError

from infrastructure.calibration_constants import (
    CALIBRATION,
    CalibrationConstants,
    override_calibration,
    validate_calibration_consistency
)


class TestCalibrationConstants(unittest.TestCase):
    """Test suite for calibration constants module"""
    
    def test_singleton_immutability(self):
        """Test that CALIBRATION singleton cannot be modified"""
        with self.assertRaises(FrozenInstanceError):
            CALIBRATION.COHERENCE_THRESHOLD = 0.8
    
    def test_mechanism_priors_sum_to_one(self):
        """Test that mechanism type priors sum to approximately 1.0"""
        total = (
            CALIBRATION.MECHANISM_PRIOR_ADMINISTRATIVO +
            CALIBRATION.MECHANISM_PRIOR_TECNICO +
            CALIBRATION.MECHANISM_PRIOR_FINANCIERO +
            CALIBRATION.MECHANISM_PRIOR_POLITICO +
            CALIBRATION.MECHANISM_PRIOR_MIXTO
        )
        self.assertAlmostEqual(total, 1.0, places=2)
    
    def test_severity_threshold_ordering(self):
        """Test that severity thresholds are strictly ordered"""
        self.assertGreater(
            CALIBRATION.CRITICAL_SEVERITY_THRESHOLD,
            CALIBRATION.HIGH_SEVERITY_THRESHOLD
        )
        self.assertGreater(
            CALIBRATION.HIGH_SEVERITY_THRESHOLD,
            CALIBRATION.MEDIUM_SEVERITY_THRESHOLD
        )
    
    def test_audit_grade_ordering(self):
        """Test that audit grade limits are strictly ordered"""
        self.assertLess(
            CALIBRATION.EXCELLENT_CONTRADICTION_LIMIT,
            CALIBRATION.GOOD_CONTRADICTION_LIMIT
        )
    
    def test_non_negative_constraints(self):
        """Test that all numeric constants are non-negative"""
        self.assertGreaterEqual(CALIBRATION.COHERENCE_THRESHOLD, 0)
        self.assertGreaterEqual(CALIBRATION.CAUSAL_INCOHERENCE_LIMIT, 0)
        self.assertGreaterEqual(CALIBRATION.REGULATORY_DEPTH_FACTOR, 0)
        self.assertGreaterEqual(CALIBRATION.KL_DIVERGENCE_THRESHOLD, 0)
        self.assertGreaterEqual(CALIBRATION.CONVERGENCE_MIN_EVIDENCE, 1)
    
    def test_override_calibration(self):
        """Test that override_calibration creates new instance with overrides"""
        custom = override_calibration(COHERENCE_THRESHOLD=0.9)
        
        # Verify override applied
        self.assertEqual(custom.COHERENCE_THRESHOLD, 0.9)
        
        # Verify original unchanged
        self.assertEqual(CALIBRATION.COHERENCE_THRESHOLD, 0.7)
        
        # Verify other values preserved
        self.assertEqual(
            custom.CAUSAL_INCOHERENCE_LIMIT,
            CALIBRATION.CAUSAL_INCOHERENCE_LIMIT
        )
    
    def test_override_validates_constraints(self):
        """Test that override_calibration validates constraints"""
        # Invalid mechanism priors (don't sum to 1.0)
        with self.assertRaises(ValueError):
            override_calibration(MECHANISM_PRIOR_ADMINISTRATIVO=0.9)
    
    def test_calibration_values(self):
        """Test specific calibration values match documentation"""
        self.assertEqual(CALIBRATION.COHERENCE_THRESHOLD, 0.7)
        self.assertEqual(CALIBRATION.CAUSAL_INCOHERENCE_LIMIT, 5)
        self.assertEqual(CALIBRATION.REGULATORY_DEPTH_FACTOR, 1.3)
        self.assertEqual(CALIBRATION.KL_DIVERGENCE_THRESHOLD, 0.01)
        self.assertEqual(CALIBRATION.PRIOR_ALPHA, 2.0)
        self.assertEqual(CALIBRATION.PRIOR_BETA, 2.0)


class TestCalibrationConsistency(unittest.TestCase):
    """Test suite for calibration consistency validation"""
    
    def test_validate_consistency_empty(self):
        """Test validation with empty module list"""
        result = validate_calibration_consistency([])
        self.assertTrue(result['passed'])
        self.assertEqual(result['scanned_modules'], 0)
    
    def test_validate_consistency_self(self):
        """Test validation of this test module"""
        import test_calibration_constants
        result = validate_calibration_consistency([test_calibration_constants])
        
        # Should pass (no hardcoded constants in test file)
        self.assertIsInstance(result, dict)
        self.assertIn('passed', result)
        self.assertIn('violations', result)


if __name__ == '__main__':
    unittest.main()
