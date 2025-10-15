#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for CI Contract Enforcement Tools
=============================================

Tests the contract enforcement checkers:
- OrchestratorContractChecker
- GitDiffContractAnalyzer
"""

import ast
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent

from ci_git_diff_contract_analyzer import GitDiffContractAnalyzer
from ci_orchestrator_contract_checker import OrchestratorContractChecker


class TestOrchestratorContractChecker(unittest.TestCase):
    """Test suite for orchestrator contract checker"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

    def test_detects_missing_assertions(self):
        """Test detection of missing assertions in phase methods"""
        # Create test file with phase method lacking assertions
        test_file = self.repo_path / "orchestrator.py"
        test_file.write_text(
            dedent("""
            class Orchestrator:
                def _extract_statements(self, text):
                    # No assertions here
                    statements = []
                    return PhaseResult(
                        phase_name="extract",
                        outputs={"statements": statements}
                    )
        """)
        )

        checker = OrchestratorContractChecker(self.repo_path)
        violations = checker.check_file(test_file)

        # Should detect missing assertion
        assertion_violations = [
            v for v in violations if v["type"] == "MISSING_ASSERTION"
        ]
        self.assertGreater(len(assertion_violations), 0)
        self.assertEqual(assertion_violations[0]["method"], "_extract_statements")

    def test_accepts_valid_assertions(self):
        """Test that valid assertions are accepted"""
        test_file = self.repo_path / "orchestrator.py"
        test_file.write_text(
            dedent("""
            class Orchestrator:
                def _extract_statements(self, text):
                    assert isinstance(text, str), "text must be string"
                    if not text:
                        raise ValueError("Empty text")
                    
                    statements = []
                    return PhaseResult(
                        phase_name="extract",
                        outputs={"statements": statements}
                    )
        """)
        )

        checker = OrchestratorContractChecker(self.repo_path)
        violations = checker.check_file(test_file)

        # Should NOT detect missing assertion
        assertion_violations = [
            v for v in violations if v["type"] == "MISSING_ASSERTION"
        ]
        self.assertEqual(len(assertion_violations), 0)

    def test_detects_missing_phase_result(self):
        """Test detection of missing PhaseResult return"""
        test_file = self.repo_path / "orchestrator.py"
        test_file.write_text(
            dedent("""
            class Orchestrator:
                def _compile_final_report(self, data):
                    assert data is not None
                    return {"report": data}  # Not PhaseResult
        """)
        )

        checker = OrchestratorContractChecker(self.repo_path)
        violations = checker.check_file(test_file)

        # Should detect missing PhaseResult
        phase_result_violations = [
            v for v in violations if v["type"] == "MISSING_PHASE_RESULT"
        ]
        self.assertGreater(len(phase_result_violations), 0)

    def test_detects_missing_audit_logging(self):
        """Test detection of missing audit logging in orchestrator files"""
        test_file = self.repo_path / "orchestrator.py"
        test_file.write_text(
            dedent("""
            class Orchestrator:
                def __init__(self):
                    # No audit logger
                    self.data = {}
                    
                def process(self):
                    # No audit logging calls
                    return "done"
        """)
        )

        checker = OrchestratorContractChecker(self.repo_path)
        violations = checker.check_file(test_file)

        # Should detect missing audit logging
        audit_violations = [
            v for v in violations if v["type"] == "MISSING_AUDIT_LOGGING"
        ]
        self.assertGreater(len(audit_violations), 0)

    def test_detects_async_without_test(self):
        """Test detection of async functions without test files"""
        test_file = self.repo_path / "infrastructure" / "async_orchestrator.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(
            dedent("""
            async def submit_job(self, job):
                await process(job)
                return result
        """)
        )

        checker = OrchestratorContractChecker(self.repo_path)
        violations = checker.check_file(test_file)

        # Should detect async without test
        async_violations = [v for v in violations if v["type"] == "ASYNC_WITHOUT_TEST"]
        self.assertGreater(len(async_violations), 0)


class TestGitDiffContractAnalyzer(unittest.TestCase):
    """Test suite for git diff contract analyzer"""

    def test_detects_assertion_removal(self):
        """Test detection of removed assertions"""
        diff_content = dedent("""
            diff --git a/orchestrator.py b/orchestrator.py
            --- a/orchestrator.py
            +++ b/orchestrator.py
            @@ -10,7 +10,6 @@
             def process(self, data):
            -    assert data is not None
                 return data
        """)

        analyzer = GitDiffContractAnalyzer(Path("."))
        violations = analyzer.analyze_diff(diff_content)

        # Should detect assertion removal
        assertion_removals = [v for v in violations if "assertion" in v["category"]]
        self.assertGreater(len(assertion_removals), 0)

    def test_detects_audit_logging_removal(self):
        """Test detection of removed audit logging"""
        diff_content = dedent("""
            diff --git a/orchestrator.py b/orchestrator.py
            --- a/orchestrator.py
            +++ b/orchestrator.py
            @@ -20,7 +20,6 @@
             def process(self):
            -    self.audit_logger.append("process", data)
                 return result
        """)

        analyzer = GitDiffContractAnalyzer(Path("."))
        violations = analyzer.analyze_diff(diff_content)

        # Should detect audit logging removal
        audit_removals = [v for v in violations if "audit_logging" in v["category"]]
        self.assertGreater(len(audit_removals), 0)

    def test_detects_telemetry_removal(self):
        """Test detection of removed telemetry"""
        diff_content = dedent("""
            diff --git a/orchestrator.py b/orchestrator.py
            --- a/orchestrator.py
            +++ b/orchestrator.py
            @@ -15,7 +15,6 @@
             def process(self):
            -    self.metrics.record("process.start", 1.0)
                 return result
        """)

        analyzer = GitDiffContractAnalyzer(Path("."))
        violations = analyzer.analyze_diff(diff_content)

        # Should detect telemetry removal
        telemetry_removals = [v for v in violations if "telemetry" in v["category"]]
        self.assertGreater(len(telemetry_removals), 0)

    def test_accepts_rationale_in_diff(self):
        """Test that SIN_CARRETA-RATIONALE in diff allows removals"""
        diff_content = dedent("""
            diff --git a/orchestrator.py b/orchestrator.py
            --- a/orchestrator.py
            +++ b/orchestrator.py
            @@ -10,7 +10,8 @@
             def process(self, data):
            -    assert data is not None
            +    # SIN_CARRETA-RATIONALE: Moved to validate() method
            +    self.validate(data)
                 return data
        """)

        analyzer = GitDiffContractAnalyzer(Path("."))

        # Simulate has_rationale check
        has_rationale = analyzer.has_rationale(diff_content, "")
        self.assertTrue(has_rationale)

    def test_accepts_rationale_in_commit(self):
        """Test that SIN_CARRETA-RATIONALE in commit message allows removals"""
        diff_content = dedent("""
            diff --git a/orchestrator.py b/orchestrator.py
            --- a/orchestrator.py
            +++ b/orchestrator.py
            @@ -10,7 +10,6 @@
             def process(self, data):
            -    assert data is not None
                 return data
        """)

        commit_messages = dedent("""
            Refactor validation
            
            SIN_CARRETA-RATIONALE: Moved assertion to base class
            All validation now handled by BaseValidator.validate()
            See test_base_validator.py for new tests
        """)

        analyzer = GitDiffContractAnalyzer(Path("."))
        has_rationale = analyzer.has_rationale(diff_content, commit_messages)
        self.assertTrue(has_rationale)


class TestContractEnforcementIntegration(unittest.TestCase):
    """Integration tests for contract enforcement"""

    def test_end_to_end_violation_detection(self):
        """Test end-to-end violation detection workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)

            # Create orchestrator file with violations
            orch_file = repo_path / "orchestrator.py"
            orch_file.write_text(
                dedent("""
                class Orchestrator:
                    def __init__(self):
                        pass  # No audit logger
                    
                    def _extract_statements(self, text):
                        # No assertions
                        # No PhaseResult
                        return []
            """)
            )

            # Run checker
            checker = OrchestratorContractChecker(repo_path)
            violations = checker.check_file(orch_file)

            # Should find multiple violations
            self.assertGreater(len(violations), 0)

            # Should have different types of violations
            violation_types = {v["type"] for v in violations}
            self.assertIn("MISSING_ASSERTION", violation_types)
            self.assertIn("MISSING_AUDIT_LOGGING", violation_types)


if __name__ == "__main__":
    unittest.main()
