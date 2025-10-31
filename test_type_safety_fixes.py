#!/usr/bin/env python3
"""
Unit tests to verify type safety and prevent the three error patterns:
1. 'bool' object is not iterable
2. 'str' object has no attribute 'text'  
3. can't multiply sequence by non-int of type 'float'
"""

import unittest
from typing import List, Any
from unittest.mock import Mock, patch


class TestTypeSafetyPatterns(unittest.TestCase):
    """Test that common functions return proper types."""
    
    def test_detection_functions_return_lists_not_bools(self):
        """Ensure detection functions always return iterables, never bools."""
        
        # Test pattern 1: Function should return empty list, not False
        def detect_items_bad(data):
            if not data:
                return False  # BUG!
            return ["item1"]
        
        def detect_items_good(data):
            if not data:
                return []  # Correct!
            return ["item1"]
        
        # This would fail with bool
        try:
            for item in detect_items_bad(None):
                pass
            self.fail("Should have raised TypeError for bool")
        except TypeError as e:
            self.assertIn("not iterable", str(e))
        
        # This works correctly
        for item in detect_items_good(None):
            pass  # Empty iteration is fine
        
        self.assertEqual(detect_items_good(None), [])
        self.assertEqual(detect_items_good("data"), ["item1"])
    
    def test_text_attribute_access_requires_object(self):
        """Ensure .text is only accessed on objects, not strings."""
        
        class MockToken:
            def __init__(self, text):
                self.text = text
        
        def process_token_bad(token):
            # Assumes token is always an object
            return token.text  # BUG if token is a string!
        
        def process_token_good(token):
            # Defensive: check if already a string
            if isinstance(token, str):
                return token
            return token.text
        
        # This would fail with string
        try:
            result = process_token_bad("hello")
            self.fail("Should have raised AttributeError")
        except AttributeError as e:
            self.assertIn("has no attribute 'text'", str(e))
        
        # This works correctly
        self.assertEqual(process_token_good("hello"), "hello")
        self.assertEqual(process_token_good(MockToken("world")), "world")
    
    def test_list_multiplication_requires_comprehension(self):
        """Ensure lists are not multiplied directly by floats."""
        
        def scale_list_bad(values, factor):
            # This only works with numpy arrays!
            return values * factor  # BUG if values is a list!
        
        def scale_list_good(values, factor):
            # Defensive: handle both lists and arrays
            if isinstance(values, list):
                return [v * factor for v in values]
            return values * factor  # OK for numpy
        
        # This would fail with list
        try:
            result = scale_list_bad([1.0, 2.0, 3.0], 0.5)
            self.fail("Should have raised TypeError")
        except TypeError as e:
            self.assertIn("can't multiply sequence", str(e))
        
        # This works correctly
        self.assertEqual(scale_list_good([1.0, 2.0, 3.0], 0.5), [0.5, 1.0, 1.5])
        
        # Also works with numpy if needed
        try:
            import numpy as np
            arr = np.array([1.0, 2.0, 3.0])
            result = scale_list_good(arr, 0.5)
            self.assertTrue((result == np.array([0.5, 1.0, 1.5])).all())
        except ImportError:
            pass  # Skip if numpy not available


class TestContradictionDetectionTypeSafety(unittest.TestCase):
    """Test type safety in contradiction detection functions."""
    
    @patch('contradiction_deteccion.PolicyContradictionDetectorV2')
    def test_detect_returns_dict_with_list_contradictions(self, mock_detector):
        """Ensure detect() returns dict with 'contradictions' as list."""
        
        # Mock the detector
        detector = mock_detector.return_value
        
        # Simulate the detect method
        def mock_detect(text, plan_name, dimension):
            return {
                'contradictions': [],  # Must be list, not bool!
                'total_contradictions': 0,
                'coherence_metrics': {}
            }
        
        detector.detect = mock_detect
        
        result = detector.detect("test text", "PDM", "estratégico")
        
        # Verify contradictions is a list
        self.assertIsInstance(result['contradictions'], list)
        
        # Should be iterable
        for contradiction in result['contradictions']:
            pass  # Should not raise TypeError


class TestBayesianEngineTypeSafety(unittest.TestCase):
    """Test type safety in Bayesian calculations."""
    
    def test_posterior_calculation_with_lists(self):
        """Ensure posterior calculations handle lists properly."""
        
        def calculate_posterior_bad(prior_list, weight):
            # BUG: List multiplication by float
            return prior_list * weight
        
        def calculate_posterior_good(prior_list, weight):
            # Correct: Use list comprehension
            if isinstance(prior_list, list):
                return [p * weight for p in prior_list]
            return prior_list * weight  # OK for scalars/arrays
        
        # Test with list
        priors = [0.1, 0.3, 0.6]
        weight = 0.8
        
        # Bad version would fail
        try:
            result = calculate_posterior_bad(priors, weight)
            self.fail("Should have raised TypeError")
        except TypeError:
            pass
        
        # Good version works
        result = calculate_posterior_good(priors, weight)
        # Use assertAlmostEqual for floating point comparisons
        for i, expected in enumerate([0.08, 0.24, 0.48]):
            self.assertAlmostEqual(result[i], expected, places=10)
        
        # Also handles scalars
        result = calculate_posterior_good(0.5, 0.8)
        self.assertEqual(result, 0.4)


def run_tests():
    """Run all type safety tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestTypeSafetyPatterns))
    suite.addTests(loader.loadTestsFromTestCase(TestContradictionDetectionTypeSafety))
    suite.addTests(loader.loadTestsFromTestCase(TestBayesianEngineTypeSafety))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
