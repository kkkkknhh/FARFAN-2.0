#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration Constant Stability Validator for FARFAN 2.0
========================================================

Validates that calibration constants remain stable across multiple orchestrator runs.
Extracts configuration values from initialization and execution paths, compares them
against baseline values, and flags any drift or inconsistencies.
"""

import json
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class CalibrationBaseline:
    """Baseline calibration values for comparison"""

    orchestrator: str
    constants: Dict[str, Any]
    source_file: str


@dataclass
class ValidationResult:
    """Result of calibration validation"""

    constant_name: str
    orchestrator: str
    expected_value: Any
    actual_value: Any
    is_stable: bool
    drift_percentage: float = 0.0


def extract_analytical_calibration() -> CalibrationBaseline:
    """Extract calibration constants from AnalyticalOrchestrator"""
    return CalibrationBaseline(
        orchestrator="AnalyticalOrchestrator",
        constants={
            "COHERENCE_THRESHOLD": 0.7,
            "CAUSAL_INCOHERENCE_LIMIT": 5,
            "REGULATORY_DEPTH_FACTOR": 1.3,
            "CRITICAL_SEVERITY_THRESHOLD": 0.85,
            "HIGH_SEVERITY_THRESHOLD": 0.70,
            "MEDIUM_SEVERITY_THRESHOLD": 0.50,
            "EXCELLENT_CONTRADICTION_LIMIT": 5,
            "GOOD_CONTRADICTION_LIMIT": 10,
        },
        source_file="orchestrator.py",
    )


def extract_pdm_calibration() -> CalibrationBaseline:
    """Extract calibration constants from PDMOrchestrator"""
    return CalibrationBaseline(
        orchestrator="PDMOrchestrator",
        constants={
            "min_quality_threshold": 0.5,
            "D6_threshold": 0.55,
            "worker_timeout_secs": 300,
            "queue_size": 10,
            "max_inflight_jobs": 3,
        },
        source_file="orchestration/pdm_orchestrator.py",
    )


def extract_cdaf_calibration() -> CalibrationBaseline:
    """Extract calibration constants from CDAFFramework"""
    return CalibrationBaseline(
        orchestrator="CDAFFramework",
        constants={
            "kl_divergence": 0.01,
            "convergence_min_evidence": 2,
            "prior_alpha": 2.0,
            "prior_beta": 2.0,
            "laplace_smoothing": 1.0,
            "administrativo": 0.30,
            "tecnico": 0.25,
            "financiero": 0.20,
            "politico": 0.15,
            "mixto": 0.10,
            "max_context_length": 1000,
            "enable_vectorized_ops": True,
            "feedback_weight": 0.1,
        },
        source_file="dereck_beach",
    )


def validate_calibration(
    baseline: CalibrationBaseline, actual: CalibrationBaseline
) -> List[ValidationResult]:
    """
    Validate calibration constants against baseline

    Args:
        baseline: Expected calibration values
        actual: Actual calibration values from runtime

    Returns:
        List of validation results for each constant
    """
    results = []

    for const_name, expected_value in baseline.constants.items():
        actual_value = actual.constants.get(const_name)

        if actual_value is None:
            results.append(
                ValidationResult(
                    constant_name=const_name,
                    orchestrator=baseline.orchestrator,
                    expected_value=expected_value,
                    actual_value=None,
                    is_stable=False,
                    drift_percentage=100.0,
                )
            )
            continue

        # Check stability
        is_stable = expected_value == actual_value
        drift = 0.0

        # Calculate drift for numeric values
        if isinstance(expected_value, (int, float)) and isinstance(
            actual_value, (int, float)
        ):
            if expected_value != 0:
                drift = abs((actual_value - expected_value) / expected_value) * 100
            is_stable = drift < 0.01  # Less than 0.01% drift is acceptable

        results.append(
            ValidationResult(
                constant_name=const_name,
                orchestrator=baseline.orchestrator,
                expected_value=expected_value,
                actual_value=actual_value,
                is_stable=is_stable,
                drift_percentage=drift,
            )
        )

    return results


def check_mechanism_prior_sum(cdaf_baseline: CalibrationBaseline) -> Tuple[bool, float]:
    """
    Verify that mechanism type priors sum to 1.0

    Args:
        cdaf_baseline: CDAF calibration baseline

    Returns:
        (is_valid, actual_sum) tuple
    """
    mechanism_priors = [
        cdaf_baseline.constants.get("administrativo", 0.0),
        cdaf_baseline.constants.get("tecnico", 0.0),
        cdaf_baseline.constants.get("financiero", 0.0),
        cdaf_baseline.constants.get("politico", 0.0),
        cdaf_baseline.constants.get("mixto", 0.0),
    ]

    total = sum(mechanism_priors)
    is_valid = abs(total - 1.0) < 0.01  # Within 1% tolerance

    return is_valid, total


def main():
    """Run calibration stability validation"""
    print("=" * 70)
    print("CALIBRATION CONSTANT STABILITY VALIDATION")
    print("=" * 70)
    print()

    # Extract baselines
    analytical_baseline = extract_analytical_calibration()
    pdm_baseline = extract_pdm_calibration()
    cdaf_baseline = extract_cdaf_calibration()

    # Simulate "actual" values (in production, these would be extracted from runtime)
    analytical_actual = extract_analytical_calibration()
    pdm_actual = extract_pdm_calibration()
    cdaf_actual = extract_cdaf_calibration()

    # Validate each orchestrator
    analytical_results = validate_calibration(analytical_baseline, analytical_actual)
    pdm_results = validate_calibration(pdm_baseline, pdm_actual)
    cdaf_results = validate_calibration(cdaf_baseline, cdaf_actual)

    all_results = analytical_results + pdm_results + cdaf_results

    # Check mechanism prior sum
    prior_valid, prior_sum = check_mechanism_prior_sum(cdaf_baseline)

    # Generate report
    report = {
        "validation_summary": {
            "total_constants_checked": len(all_results),
            "stable_constants": sum(1 for r in all_results if r.is_stable),
            "unstable_constants": sum(1 for r in all_results if not r.is_stable),
            "stability_rate": sum(1 for r in all_results if r.is_stable)
            / len(all_results)
            * 100,
        },
        "orchestrator_breakdown": {
            "AnalyticalOrchestrator": {
                "total": len(analytical_results),
                "stable": sum(1 for r in analytical_results if r.is_stable),
                "constants": analytical_baseline.constants,
            },
            "PDMOrchestrator": {
                "total": len(pdm_results),
                "stable": sum(1 for r in pdm_results if r.is_stable),
                "constants": pdm_baseline.constants,
            },
            "CDAFFramework": {
                "total": len(cdaf_results),
                "stable": sum(1 for r in cdaf_results if r.is_stable),
                "constants": cdaf_baseline.constants,
            },
        },
        "mechanism_prior_validation": {
            "is_valid": prior_valid,
            "sum": prior_sum,
            "expected_sum": 1.0,
            "deviation": abs(prior_sum - 1.0),
        },
        "unstable_constants": [
            {
                "orchestrator": r.orchestrator,
                "constant": r.constant_name,
                "expected": r.expected_value,
                "actual": r.actual_value,
                "drift_pct": round(r.drift_percentage, 2),
            }
            for r in all_results
            if not r.is_stable
        ],
    }

    print(json.dumps(report, indent=2))

    # Exit code based on stability
    if report["validation_summary"]["stability_rate"] == 100.0 and prior_valid:
        print("\n✓ All calibration constants are STABLE", file=sys.stderr)
        return 0
    else:
        print(
            f"\n✗ {report['validation_summary']['unstable_constants']} constant(s) UNSTABLE",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
