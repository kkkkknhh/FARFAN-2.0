#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theory of Change Demo
=====================

Demonstrates causal validation and acyclicity testing for Colombian
Municipal Development Plans using Bayesian DAG inference.

Run with:
    python -m examples.demo_teoria_cambio industrial-check
    python -m examples.demo_teoria_cambio stochastic-validation "PDM_EJEMPLO_2024"
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from teoria_cambio import (
    IndustrialGradeValidator,
    create_policy_theory_of_change_graph,
    LOGGER,
)


def main() -> None:
    """Command-line interface for causal validation framework."""
    parser = argparse.ArgumentParser(
        description="Unified Framework for Causal Validation of Public Policies.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command: industrial-check
    subparsers.add_parser(
        "industrial-check",
        help="Execute industrial certification suite on validation engines.",
    )

    # Command: stochastic-validation
    parser_stochastic = subparsers.add_parser(
        "stochastic-validation",
        help="Execute stochastic validation on a policy causal model.",
    )
    parser_stochastic.add_argument(
        "plan_name",
        type=str,
        help="Name of the plan or policy to validate (used as seed).",
    )
    parser_stochastic.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=10000,
        help="Number of iterations for Monte Carlo simulation.",
    )

    args = parser.parse_args()

    if args.command == "industrial-check":
        validator = IndustrialGradeValidator()
        success = validator.execute_suite()
        sys.exit(0 if success else 1)

    elif args.command == "stochastic-validation":
        LOGGER.info("Starting stochastic validation for plan: %s", args.plan_name)
        # Could load graph from file, but for demo we use constructor
        dag_validator = create_policy_theory_of_change_graph()
        result = dag_validator.calculate_acyclicity_pvalue(
            args.plan_name, args.iterations
        )

        LOGGER.info("\n" + "=" * 80)
        LOGGER.info(
            f"STOCHASTIC VALIDATION RESULTS FOR '{result.plan_name}'"
        )
        LOGGER.info("=" * 80)
        LOGGER.info(f"  - Number of iterations: {result.num_iterations}")
        LOGGER.info(f"  - Acyclicity p-value: {result.acyclicity_pvalue:.6f}")
        LOGGER.info(
            f"  - Bayesian Posterior of Acyclicity: {result.bayesian_posterior:.4f}"
        )
        LOGGER.info(
            f"  - Confidence Interval (95%): [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]"
        )
        LOGGER.info(
            f"  - Statistical Power: {result.statistical_power:.4f} {'(ADEQUATE)' if result.adequate_power else '(INSUFFICIENT)'}"
        )
        LOGGER.info(f"  - Structural Robustness Score: {result.robustness_score:.4f}")
        LOGGER.info(f"  - Computation Time: {result.computation_time:.3f}s")
        LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
