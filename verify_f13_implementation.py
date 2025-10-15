#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification Script for F1.3: Axiomatic Validator

This script verifies that the implementation is complete and working correctly.
It performs comprehensive checks on:
- Module structure
- Import availability
- Data structure creation
- Validation workflow
- Test execution
"""

import sys
import os

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")

def verify_module_structure():
    """Verify the module structure is correct"""
    print_header("1. MODULE STRUCTURE VERIFICATION")
    
    # Check validators directory exists
    if os.path.exists('validators'):
        print_success("validators/ directory exists")
    else:
        print("❌ validators/ directory missing")
        return False
    
    # Check required files
    required_files = [
        'validators/__init__.py',
        'validators/axiomatic_validator.py',
        'validators/README.md'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print_success(f"{file} exists ({size} bytes)")
        else:
            print(f"❌ {file} missing")
            return False
    
    return True

def verify_imports():
    """Verify all imports work correctly"""
    print_header("2. IMPORT VERIFICATION")
    
    try:
        from validators import (
            AxiomaticValidator,
            AxiomaticValidationResult,
            ValidationConfig,
            PDMOntology,
            SemanticChunk,
            ExtractedTable,
            ValidationSeverity,
            ValidationDimension,
            ValidationFailure,
        )
        print_success("All required classes imported successfully")
        
        # Verify enums
        print_info(f"ValidationSeverity levels: {[s.value for s in ValidationSeverity]}")
        print_info(f"ValidationDimension values: {[d.value for d in ValidationDimension]}")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def verify_data_structures():
    """Verify data structures can be created"""
    print_header("3. DATA STRUCTURE VERIFICATION")
    
    try:
        from validators import (
            ValidationConfig,
            PDMOntology,
            SemanticChunk,
            ExtractedTable,
            AxiomaticValidationResult
        )
        
        # Test ValidationConfig
        config = ValidationConfig(
            dnp_lexicon_version="2025",
            es_municipio_pdet=False,
            contradiction_threshold=0.05
        )
        print_success(f"ValidationConfig created: {config.dnp_lexicon_version}")
        
        # Test PDMOntology
        ontology = PDMOntology()
        print_success(f"PDMOntology created with {len(ontology.canonical_chain)} categories")
        
        # Test SemanticChunk
        chunk = SemanticChunk(
            text="Test text",
            dimension="ESTRATEGICO"
        )
        print_success(f"SemanticChunk created: {chunk.dimension}")
        
        # Test ExtractedTable
        table = ExtractedTable(
            title="Test Table",
            headers=["A", "B"],
            rows=[["1", "2"]]
        )
        print_success(f"ExtractedTable created: {table.title}")
        
        # Test AxiomaticValidationResult
        result = AxiomaticValidationResult()
        print_success(f"AxiomaticValidationResult created: valid={result.is_valid}")
        
        return True
    except Exception as e:
        print(f"❌ Data structure creation failed: {e}")
        return False

def verify_validator_initialization():
    """Verify validator can be initialized"""
    print_header("4. VALIDATOR INITIALIZATION VERIFICATION")
    
    try:
        from validators import AxiomaticValidator, ValidationConfig, PDMOntology
        
        config = ValidationConfig()
        ontology = PDMOntology()
        validator = AxiomaticValidator(config, ontology)
        
        print_success("AxiomaticValidator initialized successfully")
        print_info(f"Config: {config.dnp_lexicon_version}")
        print_info(f"Ontology chain: {' → '.join(ontology.canonical_chain)}")
        
        return True
    except Exception as e:
        print(f"❌ Validator initialization failed: {e}")
        return False

def verify_validation_result_methods():
    """Verify validation result methods work"""
    print_header("5. VALIDATION RESULT METHODS VERIFICATION")
    
    try:
        from validators import AxiomaticValidationResult, ValidationSeverity
        
        result = AxiomaticValidationResult()
        
        # Test add_critical_failure
        result.add_critical_failure(
            dimension='D6',
            question='Q2',
            evidence=[('A', 'B')],
            impact='Test impact'
        )
        print_success(f"add_critical_failure works: {len(result.failures)} failures")
        
        # Test get_summary
        summary = result.get_summary()
        print_success(f"get_summary works: {len(summary)} keys")
        print_info(f"Summary keys: {list(summary.keys())}")
        
        # Verify failure details
        if result.failures:
            failure = result.failures[0]
            print_success(f"Failure details accessible: {failure.dimension}-{failure.question}")
        
        return True
    except Exception as e:
        print(f"❌ Validation result methods failed: {e}")
        return False

def verify_tests():
    """Verify tests can run"""
    print_header("6. TEST EXECUTION VERIFICATION")
    
    try:
        import unittest
        
        # Load tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName('test_validator_structure')
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print_success(f"All tests passed: {result.testsRun} tests")
            return True
        else:
            print(f"❌ Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
            return False
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def verify_documentation():
    """Verify documentation files exist"""
    print_header("7. DOCUMENTATION VERIFICATION")
    
    docs = {
        'validators/README.md': 'Module documentation',
        'INTEGRATION_GUIDE.md': 'Integration guide',
        'example_axiomatic_validator.py': 'Usage examples',
        'F1.3_IMPLEMENTATION_SUMMARY.md': 'Implementation summary'
    }
    
    all_exist = True
    for file, description in docs.items():
        if os.path.exists(file):
            size = os.path.getsize(file)
            print_success(f"{description}: {file} ({size} bytes)")
        else:
            print(f"❌ {description} missing: {file}")
            all_exist = False
    
    return all_exist

def verify_example_runs():
    """Verify example script runs"""
    print_header("8. EXAMPLE EXECUTION VERIFICATION")
    
    try:
        import subprocess
        result = subprocess.run(
            ['python3', 'example_axiomatic_validator.py'],
            capture_output=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print_success("Example script runs successfully")
            output_lines = len(result.stdout.decode().split('\n'))
            print_info(f"Example output: {output_lines} lines")
            return True
        else:
            print(f"❌ Example script failed with code {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ Example execution failed: {e}")
        return False

def generate_final_report(results):
    """Generate final verification report"""
    print_header("FINAL VERIFICATION REPORT")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal Checks: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    print("\nDetailed Results:")
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {check}")
    
    if passed == total:
        print("\n🎉 ALL VERIFICATIONS PASSED!")
        print("\nThe F1.3 implementation is complete and working correctly.")
        return True
    else:
        print("\n⚠️  Some verifications failed.")
        print("Please review the output above for details.")
        return False

def main():
    """Main verification workflow"""
    print("\n" + "=" * 80)
    print("  F1.3: AXIOMATIC VALIDATOR - VERIFICATION SCRIPT")
    print("=" * 80)
    print("\nThis script verifies the complete implementation of the Axiomatic Validator.")
    print("It will check module structure, imports, data structures, and tests.\n")
    
    results = {}
    
    # Run all verifications
    results['Module Structure'] = verify_module_structure()
    results['Imports'] = verify_imports()
    results['Data Structures'] = verify_data_structures()
    results['Validator Initialization'] = verify_validator_initialization()
    results['Validation Result Methods'] = verify_validation_result_methods()
    results['Tests'] = verify_tests()
    results['Documentation'] = verify_documentation()
    results['Example Execution'] = verify_example_runs()
    
    # Generate final report
    success = generate_final_report(results)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
