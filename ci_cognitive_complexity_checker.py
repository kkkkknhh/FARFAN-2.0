#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cognitive Complexity Checker
=============================

Checks cognitive complexity increases in orchestrator changes.
Flags PRs that increase complexity for sin-carreta/approver review.

Exit codes:
- 0: Complexity acceptable or not increased
- 1: Complexity increased significantly (requires approver review)
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List


class CognitiveComplexityChecker:
    """
    Measures cognitive complexity of functions.

    Based on simplified cognitive complexity metric:
    - +1 for each if/elif/else
    - +1 for each while/for loop
    - +1 for each try/except
    - +1 for each boolean operator (and/or)
    - +1 for nested structures (additional per nesting level)
    """

    # Complexity thresholds
    ACCEPTABLE_COMPLEXITY = 15
    HIGH_COMPLEXITY = 25
    CRITICAL_COMPLEXITY = 40

    def __init__(self, repo_path: Path):
        """Initialize checker"""
        self.repo_path = Path(repo_path)
        self.results: List[Dict] = []

    def calculate_complexity(self, node: ast.AST, nesting_level: int = 0) -> int:
        """
        Calculate cognitive complexity of an AST node.

        Args:
            node: AST node to analyze
            nesting_level: Current nesting depth

        Returns:
            Complexity score
        """
        complexity = 0

        # Control flow structures
        if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1 + nesting_level

        # Boolean operators
        if isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1

        # Recursively analyze children
        for child in ast.iter_child_nodes(node):
            # Increase nesting for control structures
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.FunctionDef)):
                complexity += self.calculate_complexity(child, nesting_level + 1)
            else:
                complexity += self.calculate_complexity(child, nesting_level)

        return complexity

    def analyze_file(self, file_path: Path) -> List[Dict]:
        """Analyze complexity of functions in file"""
        results = []

        if not file_path.exists():
            return results

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError):
            return results

        # Find all function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self.calculate_complexity(node)

                if complexity > self.ACCEPTABLE_COMPLEXITY:
                    results.append(
                        {
                            "file": str(file_path.relative_to(self.repo_path)),
                            "function": node.name,
                            "line": node.lineno,
                            "complexity": complexity,
                            "threshold": self._get_threshold_category(complexity),
                        }
                    )

        return results

    def _get_threshold_category(self, complexity: int) -> str:
        """Get threshold category for complexity score"""
        if complexity >= self.CRITICAL_COMPLEXITY:
            return "CRITICAL"
        elif complexity >= self.HIGH_COMPLEXITY:
            return "HIGH"
        else:
            return "ACCEPTABLE"

    def check_repository(self) -> bool:
        """Check repository for high complexity functions"""
        print("=" * 70)
        print("COGNITIVE COMPLEXITY CHECKER")
        print("=" * 70)
        print()

        # Check orchestrator files
        orchestrator_files = [
            "orchestrator.py",
            "infrastructure/async_orchestrator.py",
            "orchestrator_with_checkpoints.py",
        ]

        for file_pattern in orchestrator_files:
            file_path = self.repo_path / file_pattern

            if not file_path.exists():
                continue

            print(f"Checking: {file_pattern}")
            results = self.analyze_file(file_path)

            if results:
                self.results.extend(results)
                print(f"  ⚠ {len(results)} function(s) exceed complexity threshold")
            else:
                print(f"  ✓ All functions within acceptable complexity")

        return len(self.results) == 0

    def report_results(self):
        """Print complexity report"""
        if not self.results:
            return

        print()
        print("=" * 70)
        print("COMPLEXITY REPORT")
        print("=" * 70)
        print()

        # Group by threshold
        by_threshold = {}
        for r in self.results:
            threshold = r["threshold"]
            by_threshold.setdefault(threshold, []).append(r)

        for threshold in ["CRITICAL", "HIGH", "ACCEPTABLE"]:
            if threshold not in by_threshold:
                continue

            results = by_threshold[threshold]
            print(f"{threshold} Complexity ({len(results)} functions):")
            print("-" * 70)

            for r in results:
                print(f"\n{r['file']}:{r['line']} - {r['function']}()")
                print(f"  Complexity: {r['complexity']}")
            print()


def main():
    """Main entry point"""
    repo_path = Path(__file__).parent

    checker = CognitiveComplexityChecker(repo_path)

    # Run checks
    all_acceptable = checker.check_repository()

    # Report results
    checker.report_results()

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_acceptable:
        print("✓ ALL FUNCTIONS WITHIN ACCEPTABLE COMPLEXITY")
        print()
        print("No high-complexity functions detected.")
        return 0
    else:
        print(f"⚠ FOUND {len(checker.results)} HIGH-COMPLEXITY FUNCTION(S)")
        print()
        print("Functions exceed acceptable complexity threshold.")
        print("This is not necessarily a failure, but requires review.")
        print()
        print("If complexity increase is intentional:")
        print("1. Document rationale in CODE_FIX_REPORT.md")
        print("2. Request sin-carreta/approver review")
        print("3. Add label 'intentional-complexity' to PR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
