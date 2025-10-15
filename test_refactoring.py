#!/usr/bin/env python3
"""Test refactored policy_processor methods."""

import re
from policy_processor import IndustrialPolicyProcessor, ProcessorConfig

def test_basic_functionality():
    """Test that the refactored code maintains basic functionality."""
    config = ProcessorConfig()
    processor = IndustrialPolicyProcessor(config)
    
    # Test with simple data
    text = "Este es un documento de diagnóstico cuantitativo y línea de base temporal."
    sentences = [
        "Este es un documento de diagnóstico cuantitativo.",
        "También incluye línea de base temporal.",
        "Los recursos son importantes."
    ]
    
    # Test _filter_relevant_sentences
    pattern = re.compile(r"diagnóstico")
    filtered = processor._filter_relevant_sentences(sentences, pattern)
    assert len(filtered) == 1, f"Expected 1 sentence, got {len(filtered)}"
    print("✓ _filter_relevant_sentences works")
    
    # Test _collect_pattern_matches
    patterns = [re.compile(r"diagnóstico"), re.compile(r"línea")]
    matches, positions = processor._collect_pattern_matches(sentences, patterns)
    assert len(matches) >= 1, f"Expected matches, got {len(matches)}"
    print("✓ _collect_pattern_matches works")
    
    # Test the full pipeline (minimal)
    result = processor._extract_point_evidence(text, sentences, "P1")
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    print("✓ _extract_point_evidence works")
    
    print("\n✓ All basic functionality tests passed!")
    return True

if __name__ == '__main__':
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
