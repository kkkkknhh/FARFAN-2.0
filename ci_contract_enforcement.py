#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CI Contract Enforcement Runner
===============================

Runs critical methodological gates that BLOCK MERGE on failure.

This script is designed to be run in CI/CD pipelines to enforce:
- Hoop test failure detection (test_hoop_test_failure)
- Posterior distribution bounds (test_posterior_cap_enforced)
- Mechanism prior decay validation (test_mechanism_prior_decay)

Exit codes:
- 0: All critical tests passed
- 1: One or more critical tests failed (BLOCK MERGE)
"""

import sys
import unittest
from io import StringIO


def run_ci_contracts() -> int:
    """
    Run CI contract enforcement tests.

    Returns:
        Exit code (0 = pass, 1 = fail)
    """
    print("=" * 70)
    print("CI CONTRACT ENFORCEMENT - Methodological Gates")
    print("=" * 70)
    print()

    # Try to import test module
    try:
        import test_governance_standards

        # Check if Bayesian tests are available
        if not test_governance_standards.BAYESIAN_AVAILABLE:
            print("⚠ WARNING: NumPy/SciPy not available")
            print("   Bayesian contract tests will be skipped")
            print()
            print("✓ Non-Bayesian governance tests will run")
            print()
    except ImportError as e:
        print(f"✗ ERROR: Cannot import test module: {e}")
        return 1

    # Create test suite with critical tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all governance standard tests
    suite.addTests(loader.loadTestsFromName("test_governance_standards"))

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print()

    if result.wasSuccessful():
        print("✓ ALL CI CONTRACTS PASSED - Merge allowed")
        return 0
    else:
        print("✗ CI CONTRACTS FAILED - BLOCK MERGE")
        print()

        if result.failures:
            print("Failed tests:")
            for test, traceback in result.failures:
                print(f"  - {test}")

        if result.errors:
            print("Errors:")
            for test, traceback in result.errors:
                print(f"  - {test}")

        return 1


if __name__ == "__main__":
    exit_code = run_ci_contracts()
    sys.exit(exit_code)
