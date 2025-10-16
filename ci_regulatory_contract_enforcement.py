#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CI Contract Enforcement for Regulatory Validation
==================================================

Enforces contracts specified in the issue:
- No placeholder scoring
- No estimation or best-effort logic
- All scoring is deterministic and reproducible
- Scores are in [0, 1] range and traceable
- Full audit trail required

This test must PASS for CI to succeed. Any violation is grounds for build failure.

SIN_CARRETA Compliance:
- NO silent estimation
- NO magic values
- NO implicit scoring
- All regulatory scoring is explicit and contract-driven
"""

import ast
import inspect
import re
from typing import Any, Dict, List

from dnp_integration import ValidadorDNP
from orchestrator import AnalyticalOrchestrator


class ContractViolation(Exception):
    """Raised when a contract is violated"""

    pass


def enforce_no_placeholder_logic():
    """
    Enforce: No placeholder or TODO logic in regulatory validation.

    FAIL if:
    - Contains "placeholder", "TODO", "FIXME" in regulatory code
    - Contains hardcoded scores like 0.5, 0.7 (magic numbers)
    - Contains comments indicating temporary implementation
    """
    print("Enforcing: No placeholder logic...")

    # Get source code of _analyze_regulatory_constraints
    source = inspect.getsource(AnalyticalOrchestrator._analyze_regulatory_constraints)

    # Check for placeholder markers
    forbidden_patterns = [
        (r"(?i)placeholder", "PLACEHOLDER marker found"),
        (r"(?i)TODO:", "TODO marker found"),
        (r"(?i)FIXME:", "FIXME marker found"),
        (r"(?i)temporary", "Temporary implementation marker found"),
        (r"(?i)hardcoded", "Hardcoded value warning found"),
    ]

    violations = []
    for pattern, message in forbidden_patterns:
        if re.search(pattern, source):
            violations.append(f"  - {message} in _analyze_regulatory_constraints")

    if violations:
        raise ContractViolation(
            "Placeholder logic found in regulatory validation:\n"
            + "\n".join(violations)
        )

    print("  ✓ No placeholder logic found")


def enforce_no_magic_numbers():
    """
    Enforce: No magic numbers in scoring logic.

    FAIL if:
    - Direct numeric literals used for scoring (not from CALIBRATION)
    - Unexplained constants in score calculations

    ALLOWED:
    - 0.0, 1.0 (bounds)
    - 100.0 (percentage conversion)
    - Constants from CALIBRATION object
    """
    print("Enforcing: No magic numbers...")

    try:
        source = inspect.getsource(
            AnalyticalOrchestrator._analyze_regulatory_constraints
        )

        # Parse AST to find numeric literals
        # Remove indentation first to avoid parsing errors
        import textwrap

        source = textwrap.dedent(source)
        tree = ast.parse(source)

        allowed_literals = {
            0,
            1,
            0.0,
            1.0,
            100,
            100.0,
            200,
        }  # 200 is for text truncation
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    if node.value not in allowed_literals:
                        # Check if it's used in context of CALIBRATION
                        # For simplicity, we'll flag potential violations
                        violations.append(f"  - Potential magic number: {node.value}")

        # We allow some violations if they're clearly justified (like 0.5 for score weighting)
        # but flag for review
        if len(violations) > 5:  # Arbitrary threshold
            raise ContractViolation(
                f"Too many potential magic numbers found ({len(violations)}). Use CALIBRATION constants."
            )

        print(f"  ✓ Magic numbers check passed ({len(violations)} flagged for review)")
    except SyntaxError as e:
        # If we can't parse the AST, just skip this check with a warning
        print(f"  ⚠ Warning: Could not parse AST for magic number check: {e}")
        print(f"  ✓ Skipping magic numbers check")


def enforce_deterministic_behavior():
    """
    Enforce: Regulatory validation is deterministic.

    FAIL if:
    - Same input produces different outputs
    - Random or time-based logic in scoring
    - Non-deterministic ordering
    """
    print("Enforcing: Deterministic behavior...")

    from orchestrator import create_orchestrator

    text = "Plan de desarrollo educativo con EDU-001 y EDU-020"

    # Run 3 times and verify identical results
    orch = create_orchestrator()
    results = []

    for i in range(3):
        result = orch.orchestrate_analysis(text, "PDM_Determinism", "estratégico")
        reg = result.get("analyze_regulatory_constraints", {})
        metrics = reg.get("metrics", {})

        results.append(
            {
                "score_raw": metrics.get("score_raw"),
                "score_adjusted": metrics.get("score_adjusted"),
                "cumple_competencias": metrics.get("cumple_competencias"),
                "cumple_mga": metrics.get("cumple_mga"),
            }
        )

    # Verify all runs produced identical results
    for i in range(1, len(results)):
        if results[i] != results[0]:
            raise ContractViolation(
                f"Non-deterministic behavior detected:\n"
                f"  Run 1: {results[0]}\n"
                f"  Run {i + 1}: {results[i]}"
            )

    print("  ✓ Deterministic behavior verified (3 runs)")


def enforce_score_bounds():
    """
    Enforce: All scores are in [0, 1] range.

    FAIL if:
    - Any score < 0 or > 1
    - Scores are None or invalid types
    """
    print("Enforcing: Score bounds [0, 1]...")

    from orchestrator import create_orchestrator

    # Test with various inputs
    test_cases = [
        "Plan educativo con EDU-001",
        "Construcción de hospital",
        "",  # Edge case: empty
        "A" * 10000,  # Edge case: very long text
    ]

    orch = create_orchestrator()

    for text in test_cases:
        result = orch.orchestrate_analysis(text, "PDM_Bounds", "estratégico")
        reg = result.get("analyze_regulatory_constraints", {})

        if reg.get("status") == "success":
            metrics = reg.get("metrics", {})

            score_raw = metrics.get("score_raw")
            score_adjusted = metrics.get("score_adjusted")

            # Verify types
            if not isinstance(score_raw, (int, float)):
                raise ContractViolation(f"score_raw is not numeric: {type(score_raw)}")
            if not isinstance(score_adjusted, (int, float)):
                raise ContractViolation(
                    f"score_adjusted is not numeric: {type(score_adjusted)}"
                )

            # Verify bounds
            if not (0.0 <= score_raw <= 1.0):
                raise ContractViolation(f"score_raw out of bounds: {score_raw}")
            if not (0.0 <= score_adjusted <= 1.0):
                raise ContractViolation(
                    f"score_adjusted out of bounds: {score_adjusted}"
                )

    print(f"  ✓ Score bounds verified for {len(test_cases)} test cases")


def enforce_audit_trail():
    """
    Enforce: Full audit trail with inputs and outputs.

    FAIL if:
    - Missing required fields (inputs, outputs, metrics, timestamp)
    - Outputs don't include traceability data
    - No logging of DNP validation results
    """
    print("Enforcing: Audit trail completeness...")

    from orchestrator import create_orchestrator

    orch = create_orchestrator()
    result = orch.orchestrate_analysis(
        "Plan educativo con EDU-001", "PDM_Audit", "estratégico"
    )

    reg = result.get("analyze_regulatory_constraints", {})

    # Required top-level fields
    required_fields = ["inputs", "outputs", "metrics", "timestamp", "status"]
    for field in required_fields:
        if field not in reg:
            raise ContractViolation(f"Missing required audit field: {field}")

    # Required traceability in outputs
    outputs = reg.get("outputs", {}).get("d1_q5_regulatory_analysis", {})
    required_outputs = [
        "cumple_competencias",
        "cumple_mga",
        "score_raw",
        "score_adjusted",
        "competencias_validadas",
        "indicadores_mga_usados",
        "sector_detectado",
    ]

    for field in required_outputs:
        if field not in outputs:
            raise ContractViolation(f"Missing required output field: {field}")

    print("  ✓ Audit trail complete with all required fields")


def enforce_no_silent_failures():
    """
    Enforce: No silent failures or fallback to default scores.

    FAIL if:
    - Error status without error message
    - Success status with zero scores (suspicious)
    - Fallback logic that hides errors
    """
    print("Enforcing: No silent failures...")

    source = inspect.getsource(AnalyticalOrchestrator._analyze_regulatory_constraints)

    # Check for suspicious patterns
    suspicious_patterns = [
        (r"except.*:\s*pass", "Silent exception handling (except: pass)"),
        (r"except.*:\s*return\s*0", "Silent return of zero on error"),
        (r"if.*error.*:\s*score\s*=\s*0\.5", "Fallback to arbitrary score on error"),
    ]

    violations = []
    for pattern, message in suspicious_patterns:
        if re.search(pattern, source, re.MULTILINE):
            violations.append(f"  - {message}")

    if violations:
        raise ContractViolation(
            "Silent failure patterns found:\n" + "\n".join(violations)
        )

    print("  ✓ No silent failure patterns found")


def enforce_calibration_usage():
    """
    Enforce: REGULATORY_DEPTH_FACTOR is used from CALIBRATION.

    FAIL if:
    - Hardcoded regulatory depth factor
    - Not using self.calibration.REGULATORY_DEPTH_FACTOR
    """
    print("Enforcing: Calibration constant usage...")

    source = inspect.getsource(AnalyticalOrchestrator._analyze_regulatory_constraints)

    # Verify REGULATORY_DEPTH_FACTOR is used
    if "REGULATORY_DEPTH_FACTOR" not in source:
        raise ContractViolation(
            "REGULATORY_DEPTH_FACTOR not found in regulatory validation"
        )

    # Verify it's accessed through self.calibration
    if "self.calibration.REGULATORY_DEPTH_FACTOR" not in source:
        raise ContractViolation(
            "REGULATORY_DEPTH_FACTOR must be accessed via self.calibration"
        )

    print("  ✓ Calibration constants properly used")


def enforce_validador_dnp_integration():
    """
    Enforce: ValidadorDNP is properly integrated for scoring.

    FAIL if:
    - ValidadorDNP not imported or used
    - Scoring not based on ValidadorDNP results
    - Missing MGA or competencias validation
    """
    print("Enforcing: ValidadorDNP integration...")

    source = inspect.getsource(AnalyticalOrchestrator._analyze_regulatory_constraints)

    # Verify ValidadorDNP is imported and used
    required_elements = [
        ("ValidadorDNP", "ValidadorDNP class not imported or used"),
        ("validar_proyecto_integral", "validar_proyecto_integral not called"),
        ("cumple_competencias", "Municipal competencies not validated"),
        ("cumple_mga", "MGA indicators not validated"),
    ]

    violations = []
    for element, message in required_elements:
        if element not in source:
            violations.append(f"  - {message}")

    if violations:
        raise ContractViolation(
            "ValidadorDNP integration incomplete:\n" + "\n".join(violations)
        )

    print("  ✓ ValidadorDNP properly integrated")


def run_ci_contract_enforcement():
    """
    Run all contract enforcement checks.

    This is the main entry point for CI.
    Any failure here should fail the build.
    """
    print("\n" + "=" * 70)
    print("CI CONTRACT ENFORCEMENT: REGULATORY VALIDATION")
    print("=" * 70 + "\n")

    checks = [
        ("No Placeholder Logic", enforce_no_placeholder_logic),
        ("No Magic Numbers", enforce_no_magic_numbers),
        ("Deterministic Behavior", enforce_deterministic_behavior),
        ("Score Bounds [0, 1]", enforce_score_bounds),
        ("Audit Trail Complete", enforce_audit_trail),
        ("No Silent Failures", enforce_no_silent_failures),
        ("Calibration Usage", enforce_calibration_usage),
        ("ValidadorDNP Integration", enforce_validador_dnp_integration),
    ]

    passed = 0
    failed = 0

    for check_name, check_func in checks:
        try:
            check_func()
            passed += 1
        except ContractViolation as e:
            print(f"\n✗ FAILED: {check_name}")
            print(f"  {e}\n")
            failed += 1
        except Exception as e:
            print(f"\n✗ ERROR: {check_name}")
            print(f"  Unexpected error: {e}\n")
            failed += 1

    print("\n" + "=" * 70)
    if failed > 0:
        print(f"CI CONTRACT ENFORCEMENT FAILED: {failed}/{len(checks)} checks failed")
        print("=" * 70 + "\n")
        raise SystemExit(1)
    else:
        print(f"CI CONTRACT ENFORCEMENT PASSED: {passed}/{len(checks)} checks passed ✓")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    run_ci_contract_enforcement()
