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
        "Los recursos son importantes.",
    ]

    # Test _match_patterns_in_sentences
    patterns = [re.compile(r"diagnóstico"), re.compile(r"línea")]
    matches, positions = processor._match_patterns_in_sentences(patterns, sentences)
    assert len(matches) >= 1, f"Expected matches, got {len(matches)}"
    print("✓ _match_patterns_in_sentences works")

    # Test _compute_evidence_confidence
    confidence = processor._compute_evidence_confidence(matches, len(text), 0.85)
    assert isinstance(confidence, float), f"Expected float, got {type(confidence)}"
    print("✓ _compute_evidence_confidence works")

    # Test _construct_evidence_bundle (requires proper setup)
    # This is tested as part of _extract_point_evidence
    print("✓ _construct_evidence_bundle tested via integration")

    # Test the full pipeline (minimal)
    result = processor._extract_point_evidence(text, sentences, "P1")
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    print("✓ _extract_point_evidence works")

    print("\n✓ All basic functionality tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
