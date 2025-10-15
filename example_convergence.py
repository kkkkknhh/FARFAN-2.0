#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Usage of Convergence Verification System
================================================

This script demonstrates various use cases of the convergence verification system.

Author: AI Systems Architect
Version: 1.0.0
"""

from pathlib import Path
from verify_convergence import ConvergenceVerifier
from canonical_notation import CanonicalID, CanonicalNotationValidator


def example_1_basic_verification():
    """Example 1: Basic convergence verification"""
    print("=" * 70)
    print("Example 1: Basic Convergence Verification")
    print("=" * 70)
    print()
    
    verifier = ConvergenceVerifier()
    report = verifier.run_full_verification()
    
    print("\nVerification Results:")
    print(f"  Convergence: {report['verification_summary']['percent_questions_converged']}%")
    print(f"  Total Issues: {report['verification_summary']['issues_detected']}")
    print(f"  Critical Issues: {report['verification_summary']['critical_issues']}")
    
    if report['verification_summary']['critical_issues'] == 0:
        print("\n✓ System is ready for production use")
    else:
        print("\n✗ Critical issues must be resolved before production")
    
    print()


def example_2_validate_question_id():
    """Example 2: Validate a specific question ID"""
    print("=" * 70)
    print("Example 2: Validate Specific Question IDs")
    print("=" * 70)
    print()
    
    validator = CanonicalNotationValidator()
    
    test_ids = [
        "P1-D1-Q1",      # Valid
        "P10-D6-Q5",     # Valid
        "P4-D3-Q2",      # Valid
        "P11-D1-Q1",     # Invalid - P11 doesn't exist
        "P1-D7-Q1",      # Invalid - D7 doesn't exist
        "D1-Q1",         # Invalid - missing policy
    ]
    
    print("Question ID Validation:")
    for qid in test_ids:
        is_valid = validator.validate_question_unique_id(qid)
        status = "✓" if is_valid else "✗"
        print(f"  {status} {qid:15} -> {'Valid' if is_valid else 'Invalid'}")
    
    print()


def example_3_parse_canonical_id():
    """Example 3: Parse and extract canonical ID components"""
    print("=" * 70)
    print("Example 3: Parse Canonical ID Components")
    print("=" * 70)
    print()
    
    question_id = "P7-D3-Q5"
    
    print(f"Parsing question ID: {question_id}")
    
    canonical = CanonicalID.from_string(question_id)
    
    print("\nExtracted Components:")
    print(f"  Policy: {canonical.policy}")
    print(f"  Policy Title: {canonical.get_policy_title()}")
    print(f"  Dimension: {canonical.dimension}")
    print(f"  Dimension Name: {canonical.get_dimension_name()}")
    print(f"  Question Number: {canonical.question}")
    print(f"  Rubric Key: {canonical.to_rubric_key()}")
    
    print()


def example_4_check_specific_policy():
    """Example 4: Check convergence for a specific policy"""
    print("=" * 70)
    print("Example 4: Check Convergence for Specific Policy")
    print("=" * 70)
    print()
    
    verifier = ConvergenceVerifier()
    
    # Load guia_cuestionario to check specific policy
    policy_id = "P1"
    
    if 'decalogo_dimension_mapping' in verifier.guia_cuestionario:
        mapping = verifier.guia_cuestionario['decalogo_dimension_mapping']
        
        if policy_id in mapping:
            policy_map = mapping[policy_id]
            
            print(f"Policy: {policy_id}")
            
            # Get policy name from questions_config
            if 'puntos_decalogo' in verifier.questions_config:
                if policy_id in verifier.questions_config['puntos_decalogo']:
                    policy_name = verifier.questions_config['puntos_decalogo'][policy_id].get('nombre')
                    print(f"Name: {policy_name}")
            
            print("\nDimension Weights:")
            for i in range(1, 7):
                dim_key = f"D{i}_weight"
                if dim_key in policy_map:
                    weight = policy_map[dim_key]
                    print(f"  D{i}: {weight:.2f}")
            
            print("\nCritical Dimensions:")
            if 'critical_dimensions' in policy_map:
                for dim in policy_map['critical_dimensions']:
                    print(f"  - {dim}")
    
    print()


def example_5_get_scoring_levels():
    """Example 5: Get scoring levels for questions"""
    print("=" * 70)
    print("Example 5: Get Scoring Levels")
    print("=" * 70)
    print()
    
    verifier = ConvergenceVerifier()
    
    if 'scoring_system' in verifier.guia_cuestionario:
        scoring = verifier.guia_cuestionario['scoring_system']
        
        if 'response_scale' in scoring:
            print("Response Scale Levels:")
            print()
            
            for level, config in scoring['response_scale'].items():
                label = config.get('label', 'N/A')
                score_range = config.get('range', [0, 0])
                print(f"  Level {level}: {label}")
                print(f"    Range: {score_range[0]:.2f} - {score_range[1]:.2f}")
                print()
    
    print()


def example_6_list_all_questions():
    """Example 6: List all questions for a policy"""
    print("=" * 70)
    print("Example 6: List All Questions for a Policy")
    print("=" * 70)
    print()
    
    policy_id = "P1"
    
    print(f"All questions for {policy_id}:")
    print()
    
    for d in range(1, 7):  # D1-D6
        print(f"  Dimension D{d}:")
        for q in range(1, 6):  # Q1-Q5
            question_id = f"{policy_id}-D{d}-Q{q}"
            print(f"    - {question_id}")
        print()
    
    total_questions = 6 * 5
    print(f"Total questions for {policy_id}: {total_questions}")
    print()


def example_7_check_dimension_templates():
    """Example 7: Check causal verification templates"""
    print("=" * 70)
    print("Example 7: Causal Verification Templates")
    print("=" * 70)
    print()
    
    verifier = ConvergenceVerifier()
    
    if 'causal_verification_templates' in verifier.guia_cuestionario:
        templates = verifier.guia_cuestionario['causal_verification_templates']
        
        print("Available Causal Verification Templates:")
        print()
        
        for dim_id, template in templates.items():
            dim_name = template.get('dimension_name', 'N/A')
            required = template.get('required_elements', [])
            
            print(f"  {dim_id}: {dim_name}")
            print(f"    Required Elements: {len(required)}")
            for element in required[:3]:  # Show first 3
                print(f"      - {element}")
            if len(required) > 3:
                print(f"      ... and {len(required) - 3} more")
            print()
    
    print()


def example_8_custom_verification():
    """Example 8: Custom verification workflow"""
    print("=" * 70)
    print("Example 8: Custom Verification Workflow")
    print("=" * 70)
    print()
    
    verifier = ConvergenceVerifier()
    
    # Step 1: Verify canonical notation
    print("Step 1: Verifying canonical notation...")
    verifier.verify_canonical_notation_usage()
    notation_issues = [i for i in verifier.issues if i.issue_type == 'invalid_canonical_notation']
    print(f"  Found {len(notation_issues)} notation issues")
    
    # Step 2: Verify scoring
    print("\nStep 2: Verifying scoring consistency...")
    verifier.verify_scoring_consistency()
    scoring_issues = [i for i in verifier.issues if 'scoring' in i.issue_type]
    print(f"  Found {len(scoring_issues)} scoring issues")
    
    # Step 3: Verify mappings
    print("\nStep 3: Verifying dimension mappings...")
    verifier.verify_dimension_mapping()
    mapping_issues = [i for i in verifier.issues if 'weight' in i.issue_type or 'policy' in i.issue_type]
    print(f"  Found {len(mapping_issues)} mapping issues")
    
    # Step 4: Generate report
    print("\nStep 4: Generating report...")
    report = verifier.generate_report()
    
    print("\nFinal Results:")
    print(f"  Total Issues: {len(verifier.issues)}")
    print(f"  Convergence: {report['verification_summary']['percent_questions_converged']}%")
    
    print()


def main():
    """Run all examples"""
    examples = [
        example_1_basic_verification,
        example_2_validate_question_id,
        example_3_parse_canonical_id,
        example_4_check_specific_policy,
        example_5_get_scoring_levels,
        example_6_list_all_questions,
        example_7_check_dimension_templates,
        example_8_custom_verification,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            print()
    
    print("=" * 70)
    print("All Examples Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
