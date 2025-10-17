#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Git Diff Contract Analyzer
===========================

Analyzes git diff to detect removal of critical patterns:
- Contract checks/assertions
- Telemetry/metrics
- Audit logging
- PhaseResult structures

Requires SIN_CARRETA-RATIONALE comment for any removal.

Exit codes:
- 0: No forbidden removals detected
- 1: Removals detected without rationale (BLOCK MERGE)
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set


class GitDiffContractAnalyzer:
    """
    Analyzes git diff for forbidden code removals.

    Forbidden patterns (require SIN_CARRETA-RATIONALE):
    - assert statements
    - raise statements
    - Contract validation calls
    - Audit logging calls
    - Telemetry/metrics calls
    - PhaseResult returns
    """

    # Patterns that cannot be removed without rationale
    FORBIDDEN_REMOVAL_PATTERNS = {
        "assertion": [
            r"-\s*assert\s+",
            r"-\s*if\s+not\s+.*:\s*raise",
        ],
        "exception_handling": [
            r"-\s*raise\s+\w+Error",
            r"-\s*except\s+\w+Error",
        ],
        "contract_validation": [
            r"-\s*\.validate\(",
            r"-\s*\.verify\(",
            r"-\s*PhaseResult\(",
        ],
        "audit_logging": [
            r"-\s*self\.audit_logger\.",
            r"-\s*self\._append_audit_log\(",
            r"-\s*ImmutableAuditLogger",
            r"-\s*audit_logger\.append",
        ],
        "telemetry": [
            r"-\s*self\.metrics\.record\(",
            r"-\s*MetricsCollector",
            r"-\s*metrics\.increment\(",
        ],
    }

    # Rationale marker in code or commit message
    RATIONALE_PATTERN = r"SIN_CARRETA[-_]RATIONALE"

    def __init__(self, repo_path: Path, base_ref: str = "HEAD~1"):
        """
        Initialize analyzer.

        Args:
            repo_path: Repository root path
            base_ref: Base git ref for diff (default: HEAD~1)
        """
        self.repo_path = Path(repo_path)
        self.base_ref = base_ref
        self.violations: List[Dict] = []

    def get_diff(self) -> str:
        """Get git diff output"""
        try:
            result = subprocess.run(
                ["git", "diff", self.base_ref, "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error getting git diff: {e}")
            return ""

    def get_commit_messages(self) -> str:
        """Get commit messages since base_ref"""
        try:
            result = subprocess.run(
                ["git", "log", f"{self.base_ref}..HEAD", "--pretty=format:%B"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError:
            return ""

    def has_rationale(self, diff_content: str, commit_messages: str) -> bool:
        """Check if SIN_CARRETA-RATIONALE is present"""
        # Check diff content
        if re.search(self.RATIONALE_PATTERN, diff_content, re.IGNORECASE):
            return True

        # Check commit messages
        if re.search(self.RATIONALE_PATTERN, commit_messages, re.IGNORECASE):
            return True

        return False

    def analyze_diff(self, diff_content: str) -> List[Dict]:
        """
        Analyze diff for forbidden removals.

        Returns:
            List of violation dictionaries
        """
        violations = []

        # Split diff into file chunks
        file_diffs = re.split(r"\ndiff --git", diff_content)

        for file_diff in file_diffs:
            if not file_diff.strip():
                continue

            # Extract file path
            match = re.search(r"a/(.*?)\s+b/", file_diff)
            if not match:
                continue
            file_path = match.group(1)

            # Only check Python files in orchestrator/infrastructure
            if not (
                file_path.endswith(".py")
                and ("orchestrator" in file_path or "infrastructure" in file_path)
            ):
                continue

            # Check for forbidden pattern removals
            for category, patterns in self.FORBIDDEN_REMOVAL_PATTERNS.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, file_diff, re.MULTILINE)

                    for match in matches:
                        # Get line context
                        line_start = file_diff.rfind("\n", 0, match.start()) + 1
                        line_end = file_diff.find("\n", match.end())
                        if line_end == -1:
                            line_end = len(file_diff)
                        removed_line = file_diff[line_start:line_end]

                        violations.append(
                            {
                                "type": f"FORBIDDEN_REMOVAL_{category.upper()}",
                                "file": file_path,
                                "category": category,
                                "pattern": pattern,
                                "removed_line": removed_line.strip(),
                                "message": (
                                    f"Removed {category} code without SIN_CARRETA-RATIONALE: "
                                    f"{removed_line.strip()}"
                                ),
                                "severity": "CRITICAL",
                            }
                        )

        return violations

    def check_repository(self) -> bool:
        """
        Check repository diff for violations.

        Returns:
            True if checks pass, False if violations found
        """
        print("=" * 70)
        print("GIT DIFF CONTRACT ANALYZER")
        print("=" * 70)
        print()

        # Get diff and commit messages
        diff_content = self.get_diff()
        commit_messages = self.get_commit_messages()

        if not diff_content:
            print("ℹ No diff found (or not in git repository)")
            return True

        print(f"Analyzing diff from {self.base_ref} to HEAD...")
        print()

        # Check for rationale marker
        has_rationale = self.has_rationale(diff_content, commit_messages)

        # Analyze diff
        self.violations = self.analyze_diff(diff_content)

        if self.violations:
            print(f"⚠ Found {len(self.violations)} potential removal(s)")

            if has_rationale:
                print("✓ SIN_CARRETA-RATIONALE found in diff or commit messages")
                print("  Removals are documented and allowed")
                self.violations = []  # Clear violations if rationale present
                return True
            else:
                print("✗ NO SIN_CARRETA-RATIONALE found")
                print("  Removals require explicit rationale")
                return False
        else:
            print("✓ No forbidden removals detected")
            return True

    def report_violations(self):
        """Print detailed violation report"""
        if not self.violations:
            return

        print()
        print("=" * 70)
        print("FORBIDDEN REMOVAL REPORT")
        print("=" * 70)
        print()

        # Group by category
        by_category = {}
        for v in self.violations:
            category = v["category"]
            by_category.setdefault(category, []).append(v)

        for category, violations in sorted(by_category.items()):
            print(f"{category.upper()} Removals ({len(violations)}):")
            print("-" * 70)

            for v in violations:
                print(f"\nFile: {v['file']}")
                print(f"  Removed: {v['removed_line']}")
            print()

        print("=" * 70)
        print("REQUIRED ACTION")
        print("=" * 70)
        print()
        print("Add SIN_CARRETA-RATIONALE to your commit message or code comments:")
        print()
        print("Example in commit message:")
        print("  SIN_CARRETA-RATIONALE: Removed assertion X because Y")
        print("  Replaced with stronger contract Z (see test_foo.py)")
        print()
        print("Example in code comment:")
        print("  # SIN_CARRETA-RATIONALE: Contract moved to base class")
        print("  # See BaseValidator.validate() for new assertion")
        print()


def main():
    """Main entry point"""
    repo_path = Path(__file__).parent

    # Support environment variable for base ref
    import os

    base_ref = os.environ.get("GITHUB_BASE_REF", "HEAD~1")

    # If running in GitHub Actions, use proper base
    if os.environ.get("GITHUB_EVENT_NAME") == "pull_request":
        base_ref = os.environ.get("GITHUB_BASE_SHA", "HEAD~1")

    analyzer = GitDiffContractAnalyzer(repo_path, base_ref)

    # Run checks
    all_passed = analyzer.check_repository()

    # Report violations
    analyzer.report_violations()

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_passed:
        print("✓ GIT DIFF CONTRACT CHECK PASSED")
        print()
        print("No forbidden removals without rationale.")
        return 0
    else:
        print("✗ GIT DIFF CONTRACT CHECK FAILED")
        print()
        print("⛔ BLOCK MERGE - Add SIN_CARRETA-RATIONALE")
        return 1


if __name__ == "__main__":
    sys.exit(main())
