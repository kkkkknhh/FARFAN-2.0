#!/usr/bin/env python3
"""
Canonical Notation System Examples
===================================

Comprehensive examples demonstrating the canonical notation system
for PDM evaluation in FARFAN-2.0.

This script showcases:
1. Creating and validating canonical IDs
2. Working with rubric keys
3. Creating evidence entries
4. Migrating legacy formats
5. Integration with DNP validation
6. System structure and metadata

Author: AI Systems Architect
Version: 2.0.0
"""

from canonical_notation import (
    CanonicalID,
    RubricKey,
    EvidenceEntry,
    CanonicalNotationValidator,
    PolicyArea,
    AnalyticalDimension,
    generate_default_questions,
    get_system_structure_summary
)
import json


def example_1_basic_canonical_ids():
    """Example 1: Creating and using canonical IDs"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Canonical IDs")
    print("=" * 70)
    
    # Create canonical IDs for different policy areas
    examples = [
        ("P1", "D1", 1, "Baseline assessment for women's rights"),
        ("P4", "D2", 3, "Intervention design for economic rights"),
        ("P7", "D3", 5, "Product verification for land rights"),
        ("P10", "D6", 30, "Theory of change for migration")
    ]
    
    for policy, dimension, question, description in examples:
        canonical_id = CanonicalID(policy=policy, dimension=dimension, question=question)
        print(f"\n{description}")
        print(f"  Canonical ID: {canonical_id}")
        print(f"  Policy: {canonical_id.get_policy_title()}")
        print(f"  Dimension: {canonical_id.get_dimension_name()}")
        print(f"  Focus: {canonical_id.get_dimension_focus()}")
        print(f"  Rubric Key: {canonical_id.to_rubric_key()}")
    
    print()


def example_2_parsing_and_validation():
    """Example 2: Parsing strings and validation"""
    print("=" * 70)
    print("EXAMPLE 2: Parsing and Validation")
    print("=" * 70)
    
    validator = CanonicalNotationValidator()
    
    # Test various ID formats
    test_ids = [
        ("P4-D2-Q3", "Valid question ID"),
        ("P10-D6-Q30", "Valid with P10"),
        ("P11-D2-Q3", "Invalid - P11 doesn't exist"),
        ("P4-D7-Q3", "Invalid - D7 doesn't exist"),
        ("P4-D2-Q0", "Invalid - Q must be positive"),
        ("D2-Q3", "Valid rubric key (not question ID)"),
    ]
    
    print("\nValidation Results:")
    print(f"{'ID':<15} {'Type':<20} {'Valid Q-ID':<12} {'Valid RK':<12} {'Description'}")
    print("-" * 90)
    
    for test_id, description in test_ids:
        is_valid_q = validator.validate_question_unique_id(test_id)
        is_valid_r = validator.validate_rubric_key(test_id)
        
        if is_valid_q:
            id_type = "Question ID"
        elif is_valid_r:
            id_type = "Rubric Key"
        else:
            id_type = "Invalid"
        
        print(f"{test_id:<15} {id_type:<20} {str(is_valid_q):<12} {str(is_valid_r):<12} {description}")
    
    print("\nParsing valid IDs:")
    valid_question_id = "P7-D3-Q5"
    canonical_id = CanonicalID.from_string(valid_question_id)
    print(f"  Parsed {valid_question_id}:")
    print(f"    Policy: {canonical_id.policy}")
    print(f"    Dimension: {canonical_id.dimension}")
    print(f"    Question: {canonical_id.question}")
    
    valid_rubric_key = "D3-Q5"
    rubric_key = RubricKey.from_string(valid_rubric_key)
    print(f"\n  Parsed {valid_rubric_key}:")
    print(f"    Dimension: {rubric_key.dimension}")
    print(f"    Question: {rubric_key.question}")
    
    print()


def example_3_evidence_entries():
    """Example 3: Creating standardized evidence entries"""
    print("=" * 70)
    print("EXAMPLE 3: Evidence Entries")
    print("=" * 70)
    
    # Create evidence entries for different stages
    evidence_examples = [
        {
            "policy": "P1", "dimension": "D1", "question": 1,
            "score": 0.85, "confidence": 0.90,
            "stage": "diagnostic", "prefix": "diag_"
        },
        {
            "policy": "P4", "dimension": "D2", "question": 3,
            "score": 0.72, "confidence": 0.78,
            "stage": "design", "prefix": "design_"
        },
        {
            "policy": "P7", "dimension": "D3", "question": 5,
            "score": 0.82, "confidence": 0.82,
            "stage": "teoria_cambio", "prefix": "toc_"
        }
    ]
    
    print("\nCreated Evidence Entries:\n")
    
    for ex in evidence_examples:
        evidence = EvidenceEntry.create(
            policy=ex["policy"],
            dimension=ex["dimension"],
            question=ex["question"],
            score=ex["score"],
            confidence=ex["confidence"],
            stage=ex["stage"],
            evidence_id_prefix=ex["prefix"]
        )
        
        print(f"Evidence ID: {evidence.evidence_id}")
        print(f"Question ID: {evidence.question_unique_id}")
        print(f"Rubric Key: {evidence.content['rubric_key']}")
        print(f"Score: {evidence.content['score']:.2f}")
        print(f"Confidence: {evidence.confidence:.2f}")
        print(f"Stage: {evidence.stage}")
        print()
    
    # Show JSON format
    print("JSON Format Example:")
    evidence = evidence_examples[0]
    evidence_entry = EvidenceEntry.create(
        policy=evidence["policy"],
        dimension=evidence["dimension"],
        question=evidence["question"],
        score=evidence["score"],
        confidence=evidence["confidence"],
        stage=evidence["stage"],
        evidence_id_prefix=evidence["prefix"]
    )
    print(evidence_entry.to_json())
    print()


def example_4_legacy_migration():
    """Example 4: Migrating legacy ID formats"""
    print("=" * 70)
    print("EXAMPLE 4: Legacy Format Migration")
    print("=" * 70)
    
    validator = CanonicalNotationValidator()
    
    # Simulate legacy IDs that need migration
    legacy_ids = [
        ("D1-Q1", "P1", "Gender baseline question"),
        ("D2-Q3", "P4", "Economic intervention design"),
        ("D3-Q5", "P7", "Land rights product verification"),
        ("D6-Q10", "P5", "Peace theory of change")
    ]
    
    print("\nMigrating Legacy IDs to Canonical Format:\n")
    print(f"{'Legacy ID':<12} {'Context Policy':<15} {'Canonical ID':<15} {'Description'}")
    print("-" * 80)
    
    for legacy_id, inferred_policy, description in legacy_ids:
        canonical_id = validator.migrate_legacy_id(legacy_id, inferred_policy=inferred_policy)
        print(f"{legacy_id:<12} {inferred_policy:<15} {canonical_id:<15} {description}")
    
    # Show that already canonical IDs pass through unchanged
    print("\nAlready Canonical (No Change Needed):")
    canonical_test = "P4-D2-Q3"
    result = validator.migrate_legacy_id(canonical_test)
    print(f"  {canonical_test} → {result}")
    
    print()


def example_5_system_structure():
    """Example 5: System structure and metadata"""
    print("=" * 70)
    print("EXAMPLE 5: System Structure and Metadata")
    print("=" * 70)
    
    # Get system structure
    structure = get_system_structure_summary()
    
    print("\nSystem Configuration:")
    print(f"  Total Policies: {structure['total_policies']}")
    print(f"  Total Dimensions: {structure['total_dimensions']}")
    print(f"  Questions per Dimension (default): {structure['default_questions_per_dimension']}")
    print(f"  Total Questions (default): {structure['default_total_questions']}")
    
    print("\nPolicy Areas (P1-P10):")
    for policy_id, title in structure['policies'].items():
        print(f"  {policy_id}: {title}")
    
    print("\nAnalytical Dimensions (D1-D6):")
    for dim_id, name in structure['dimensions'].items():
        focus = structure['dimension_focus'][dim_id]
        print(f"  {dim_id}: {name}")
        print(f"      Focus: {focus}")
    
    print("\nValidation Patterns:")
    for pattern_name, pattern in structure['patterns'].items():
        print(f"  {pattern_name}: {pattern}")
    
    print()


def example_6_question_generation():
    """Example 6: Generating question structures"""
    print("=" * 70)
    print("EXAMPLE 6: Question Generation")
    print("=" * 70)
    
    # Generate default questions
    questions = generate_default_questions(max_questions_per_dimension=5)
    
    print(f"\nGenerated {len(questions)} default questions")
    print("\nFirst 10 questions:")
    for i, q in enumerate(questions[:10], 1):
        print(f"  {i}. {q} - {q.get_policy_title()[:40]}... / {q.get_dimension_name()}")
    
    print("\nLast 10 questions:")
    for i, q in enumerate(questions[-10:], len(questions)-9):
        print(f"  {i}. {q} - {q.get_policy_title()[:40]}... / {q.get_dimension_name()}")
    
    # Custom generation
    custom_questions = generate_default_questions(max_questions_per_dimension=3)
    print(f"\nCustom generation with 3 questions per dimension:")
    print(f"  Total: {len(custom_questions)} questions (10 policies × 6 dimensions × 3 questions)")
    
    print()


def example_7_dnp_integration():
    """Example 7: Integration with DNP validation"""
    print("=" * 70)
    print("EXAMPLE 7: DNP Integration Example")
    print("=" * 70)
    
    print("\nSimulating DNP validation results with canonical notation:\n")
    
    # Simulate evaluation results for a subset of questions
    dnp_results = []
    
    # Policy 4 (Economic rights) evaluations
    for dimension in ["D1", "D2", "D3"]:
        for question in range(1, 4):
            canonical_id = CanonicalID(policy="P4", dimension=dimension, question=question)
            
            # Simulate scoring (in real use, this comes from actual validation)
            import random
            score = round(random.uniform(0.6, 0.95), 2)
            confidence = round(random.uniform(0.7, 0.9), 2)
            
            evidence = EvidenceEntry.create(
                policy="P4",
                dimension=dimension,
                question=question,
                score=score,
                confidence=confidence,
                stage="dnp_validation",
                evidence_id_prefix="dnp_"
            )
            
            dnp_results.append({
                "canonical_id": str(canonical_id),
                "evidence": evidence,
                "policy_title": canonical_id.get_policy_title(),
                "dimension": canonical_id.get_dimension_name()
            })
    
    # Display results
    print(f"{'Question ID':<15} {'Policy/Dimension':<50} {'Score':<8} {'Confidence'}")
    print("-" * 100)
    
    for result in dnp_results[:10]:  # Show first 10
        policy_dim = f"{result['policy_title'][:25]}... / {result['dimension'][:20]}"
        print(f"{result['canonical_id']:<15} {policy_dim:<50} "
              f"{result['evidence'].content['score']:<8.2f} {result['evidence'].confidence:.2f}")
    
    # Summary statistics
    avg_score = sum(r['evidence'].content['score'] for r in dnp_results) / len(dnp_results)
    avg_confidence = sum(r['evidence'].confidence for r in dnp_results) / len(dnp_results)
    
    print(f"\nSummary Statistics:")
    print(f"  Total Evaluations: {len(dnp_results)}")
    print(f"  Average Score: {avg_score:.2f}")
    print(f"  Average Confidence: {avg_confidence:.2f}")
    
    print()


def example_8_error_handling():
    """Example 8: Error handling and validation"""
    print("=" * 70)
    print("EXAMPLE 8: Error Handling")
    print("=" * 70)
    
    print("\nDemonstrating error messages for invalid inputs:\n")
    
    # Test invalid policy
    print("1. Invalid Policy (P11):")
    try:
        CanonicalID(policy="P11", dimension="D1", question=1)
    except ValueError as e:
        print(f"   ERROR: {e}\n")
    
    # Test invalid dimension
    print("2. Invalid Dimension (D7):")
    try:
        CanonicalID(policy="P1", dimension="D7", question=1)
    except ValueError as e:
        print(f"   ERROR: {e}\n")
    
    # Test invalid question number
    print("3. Invalid Question Number (Q0):")
    try:
        CanonicalID(policy="P1", dimension="D1", question=0)
    except ValueError as e:
        print(f"   ERROR: {e}\n")
    
    # Test invalid score in evidence
    print("4. Invalid Score (1.5):")
    try:
        EvidenceEntry.create(
            policy="P1", dimension="D1", question=1,
            score=1.5, confidence=0.8, stage="test"
        )
    except ValueError as e:
        print(f"   ERROR: {e}\n")
    
    # Test invalid confidence
    print("5. Invalid Confidence (-0.1):")
    try:
        EvidenceEntry.create(
            policy="P1", dimension="D1", question=1,
            score=0.8, confidence=-0.1, stage="test"
        )
    except ValueError as e:
        print(f"   ERROR: {e}\n")
    
    print("All validation errors are caught with descriptive messages.")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("CANONICAL NOTATION SYSTEM - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates the canonical notation system for")
    print("standardized PDM evaluation in FARFAN-2.0.")
    print()
    
    examples = [
        example_1_basic_canonical_ids,
        example_2_parsing_and_validation,
        example_3_evidence_entries,
        example_4_legacy_migration,
        example_5_system_structure,
        example_6_question_generation,
        example_7_dnp_integration,
        example_8_error_handling
    ]
    
    for example_func in examples:
        example_func()
        input("Press Enter to continue to next example...")
        print("\n")
    
    print("=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)
    print("\nFor more information, see CANONICAL_NOTATION_DOCS.md")
    print()


if __name__ == "__main__":
    main()
