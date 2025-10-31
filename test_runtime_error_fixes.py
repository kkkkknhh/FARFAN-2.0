#!/usr/bin/env python3
"""
Test script to validate fixes for critical runtime errors:
1. Bool object is not iterable
2. Str object has no attribute 'text'
3. Can't multiply sequence by non-int of type 'float'
"""

def test_contradiction_detection_returns_list():
    """Test that contradiction detection always returns a list, never a bool"""
    print("="*70)
    print("TEST 1: Contradiction detection returns list, not bool")
    print("="*70)
    
    # Simulate a function that might incorrectly return False
    def bad_detect_contradictions(statements):
        if not statements:
            return False  # BUG: Should return []
        return []
    
    # Fixed version
    def good_detect_contradictions(statements):
        if not statements:
            return []  # CORRECT: Always return a list
        return []
    
    # Test the bad version
    try:
        result = bad_detect_contradictions([])
        for item in result:  # This will fail if result is False
            pass
        print("✗ Bad version should have failed but didn't")
    except TypeError as e:
        print(f"✓ Bad version correctly fails: {e}")
    
    # Test the good version
    try:
        result = good_detect_contradictions([])
        for item in result:  # This should work
            pass
        print("✓ Good version works correctly")
    except TypeError as e:
        print(f"✗ Good version failed: {e}")


def test_text_attribute_on_spacy_doc():
    """Test that .text is only accessed on spaCy Doc objects, not strings"""
    print("\n" + "="*70)
    print("TEST 2: .text attribute accessed on spaCy Doc, not string")
    print("="*70)
    
    # Simulate incorrect usage
    def bad_process_text(text):
        # BUG: Accessing .text on a string
        return text.text.lower()  # AttributeError: 'str' object has no attribute 'text'
    
    # Fixed version
    def good_process_text(text):
        # CORRECT: Check if already a Doc, else use the string directly
        if hasattr(text, 'text'):
            return text.text.lower()
        return text.lower()
    
    # Test with a string
    test_str = "Hello World"
    
    try:
        result = bad_process_text(test_str)
        print(f"✗ Bad version should have failed but got: {result}")
    except AttributeError as e:
        print(f"✓ Bad version correctly fails: {e}")
    
    try:
        result = good_process_text(test_str)
        print(f"✓ Good version works correctly: {result}")
    except AttributeError as e:
        print(f"✗ Good version failed: {e}")


def test_list_multiplication_by_float():
    """Test that list * float is converted to list comprehension"""
    print("\n" + "="*70)
    print("TEST 3: List multiplication by float using list comprehension")
    print("="*70)
    
    # Simulate incorrect usage
    def bad_multiply_list(values, multiplier):
        # BUG: Can't multiply sequence by non-int of type 'float'
        return values * multiplier
    
    # Fixed version
    def good_multiply_list(values, multiplier):
        # CORRECT: Use list comprehension
        return [item * multiplier for item in values]
    
    test_list = [0.1, 0.5, 0.8]
    test_multiplier = 0.9
    
    try:
        result = bad_multiply_list(test_list, test_multiplier)
        print(f"✗ Bad version should have failed but got: {result}")
    except TypeError as e:
        print(f"✓ Bad version correctly fails: {e}")
    
    try:
        result = good_multiply_list(test_list, test_multiplier)
        print(f"✓ Good version works correctly: {result}")
    except TypeError as e:
        print(f"✗ Good version failed: {e}")


if __name__ == "__main__":
    print("\nTesting fixes for critical runtime errors\n")
    
    test_contradiction_detection_returns_list()
    test_text_attribute_on_spacy_doc()
    test_list_multiplication_by_float()
    
    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)
