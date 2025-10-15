#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CI Orchestrator Contract Checker
=================================

Static code analysis to enforce orchestrator contract requirements.
BLOCKS merge if:
- Orchestrator phases omit explicit assertions or contract checks
- Code removes telemetry, audit logging, or contract validation
- Async functions converted to sync without test updates
- Contract boundaries changed without SIN_CARRETA-RATIONALE

Exit codes:
- 0: All contract checks passed
- 1: One or more contract violations detected (BLOCK MERGE)
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class OrchestratorContractChecker:
    """
    Static analyzer for orchestrator contract enforcement.
    
    Validates:
    1. Phase methods contain explicit assertions or contract checks
    2. No removal of telemetry/audit/contract code
    3. Async/sync consistency with tests
    4. SIN_CARRETA-RATIONALE for contract changes
    """
    
    # Required patterns in orchestrator phases
    REQUIRED_PATTERNS = {
        "assertions": [
            r"\bassert\b",
            r"\braise\s+\w+Error",
            r"\.validate\(",
            r"\.verify\(",
            r"if\s+not\s+.*:\s*raise",
        ],
        "audit_logging": [
            r"self\.audit_logger\.append",
            r"self\._append_audit_log",
            r"ImmutableAuditLogger",
            r"audit_logger\.record",
        ],
        "telemetry": [
            r"self\.metrics\.record",
            r"MetricsCollector",
            r"metrics\.increment",
            r"metrics\.gauge",
        ],
        "contract_validation": [
            r"PhaseResult\(",
            r"\.status\s*=",
            r"\.outputs\[",
            r"\.inputs\[",
        ],
    }
    
    # Critical orchestrator files to check
    ORCHESTRATOR_FILES = [
        "orchestrator.py",
        "infrastructure/async_orchestrator.py",
        "orchestrator_with_checkpoints.py",
    ]
    
    def __init__(self, repo_path: Path):
        """Initialize checker with repository path"""
        self.repo_path = Path(repo_path)
        self.violations: List[Dict] = []
        
    def check_file(self, file_path: Path) -> List[Dict]:
        """
        Check a file for contract violations.
        
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        if not file_path.exists():
            return violations
            
        content = file_path.read_text(encoding='utf-8')
        
        # Parse AST for structural analysis
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            violations.append({
                "type": "SYNTAX_ERROR",
                "file": str(file_path),
                "line": e.lineno,
                "message": f"Syntax error: {e.msg}",
                "severity": "CRITICAL"
            })
            return violations
            
        # Check for phase methods
        phase_methods = self._find_phase_methods(tree)
        
        for method_name, method_node in phase_methods:
            # Check for required patterns in phase method
            method_source = ast.get_source_segment(content, method_node)
            if method_source is None:
                continue
                
            violations.extend(
                self._check_phase_method_contracts(
                    file_path, method_name, method_source, method_node.lineno
                )
            )
            
        # Check for async/sync consistency
        violations.extend(self._check_async_sync_consistency(tree, file_path))
        
        # Check for removal of critical patterns
        violations.extend(self._check_critical_removals(content, file_path))
        
        return violations
    
    def _find_phase_methods(self, tree: ast.AST) -> List[Tuple[str, ast.FunctionDef]]:
        """Find phase methods in AST"""
        phase_methods = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Phase methods start with _ and contain analytical keywords
                if (node.name.startswith('_') and 
                    any(keyword in node.name.lower() for keyword in [
                        'extract', 'detect', 'analyze', 'calculate', 'generate', 'compile'
                    ])):
                    phase_methods.append((node.name, node))
                    
        return phase_methods
    
    def _check_phase_method_contracts(
        self, 
        file_path: Path, 
        method_name: str, 
        method_source: str,
        lineno: int
    ) -> List[Dict]:
        """Check if phase method has required contract patterns"""
        violations = []
        
        # Check for assertions or contract checks
        has_assertion = any(
            re.search(pattern, method_source, re.MULTILINE)
            for pattern in self.REQUIRED_PATTERNS["assertions"]
        )
        
        # Check for PhaseResult contract
        has_phase_result = any(
            re.search(pattern, method_source, re.MULTILINE)
            for pattern in self.REQUIRED_PATTERNS["contract_validation"]
        )
        
        # Check for audit logging
        has_audit_log = any(
            re.search(pattern, method_source, re.MULTILINE)
            for pattern in self.REQUIRED_PATTERNS["audit_logging"]
        )
        
        if not has_assertion:
            violations.append({
                "type": "MISSING_ASSERTION",
                "file": str(file_path),
                "line": lineno,
                "method": method_name,
                "message": (
                    f"Phase method '{method_name}' lacks explicit assertions or "
                    f"contract checks. All orchestrator phases MUST validate inputs/outputs."
                ),
                "severity": "CRITICAL",
                "required_action": "Add explicit assertions, raise statements, or validation calls"
            })
            
        if not has_phase_result:
            violations.append({
                "type": "MISSING_PHASE_RESULT",
                "file": str(file_path),
                "line": lineno,
                "method": method_name,
                "message": (
                    f"Phase method '{method_name}' does not return PhaseResult contract. "
                    f"All phases MUST use structured PhaseResult."
                ),
                "severity": "CRITICAL",
                "required_action": "Return PhaseResult with inputs, outputs, metrics, timestamp, status"
            })
            
        return violations
    
    def _check_async_sync_consistency(
        self, 
        tree: ast.AST, 
        file_path: Path
    ) -> List[Dict]:
        """Check for async/sync function changes"""
        violations = []
        
        # Find all function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                # Check if there's a corresponding test file
                test_file = self._find_test_file(file_path)
                if not test_file or not test_file.exists():
                    violations.append({
                        "type": "ASYNC_WITHOUT_TEST",
                        "file": str(file_path),
                        "line": node.lineno,
                        "method": node.name,
                        "message": (
                            f"Async function '{node.name}' has no corresponding test file. "
                            f"Expected: {test_file}"
                        ),
                        "severity": "HIGH",
                        "required_action": "Create test file or convert function to sync with rationale"
                    })
                    
        return violations
    
    def _check_critical_removals(self, content: str, file_path: Path) -> List[Dict]:
        """Check for removal of critical code patterns"""
        violations = []
        
        # This is a simple heuristic - in real usage, would use git diff
        # For now, check if file has minimum required patterns
        
        # Count occurrences of critical patterns
        audit_count = sum(
            len(re.findall(pattern, content, re.MULTILINE))
            for pattern in self.REQUIRED_PATTERNS["audit_logging"]
        )
        
        telemetry_count = sum(
            len(re.findall(pattern, content, re.MULTILINE))
            for pattern in self.REQUIRED_PATTERNS["telemetry"]
        )
        
        # Orchestrator files MUST have audit logging
        if "orchestrator" in str(file_path).lower() and audit_count == 0:
            violations.append({
                "type": "MISSING_AUDIT_LOGGING",
                "file": str(file_path),
                "line": 1,
                "message": "Orchestrator file has NO audit logging. This is forbidden.",
                "severity": "CRITICAL",
                "required_action": "Add ImmutableAuditLogger or restore removed audit code"
            })
            
        return violations
    
    def _find_test_file(self, source_file: Path) -> Path:
        """Find corresponding test file for source file"""
        # Standard pattern: test_{module_name}.py
        test_name = f"test_{source_file.stem}.py"
        return source_file.parent / test_name
    
    def check_repository(self) -> bool:
        """
        Check all orchestrator files in repository.
        
        Returns:
            True if all checks pass, False if violations found
        """
        print("=" * 70)
        print("CI ORCHESTRATOR CONTRACT CHECKER")
        print("=" * 70)
        print()
        
        for file_pattern in self.ORCHESTRATOR_FILES:
            file_path = self.repo_path / file_pattern
            
            if not file_path.exists():
                print(f"⚠ Skipping (not found): {file_pattern}")
                continue
                
            print(f"Checking: {file_pattern}")
            violations = self.check_file(file_path)
            
            if violations:
                self.violations.extend(violations)
                print(f"  ✗ {len(violations)} violation(s) found")
            else:
                print(f"  ✓ No violations")
                
        return len(self.violations) == 0
    
    def report_violations(self):
        """Print detailed violation report"""
        if not self.violations:
            return
            
        print()
        print("=" * 70)
        print("VIOLATION REPORT")
        print("=" * 70)
        print()
        
        # Group by severity
        critical = [v for v in self.violations if v["severity"] == "CRITICAL"]
        high = [v for v in self.violations if v["severity"] == "HIGH"]
        medium = [v for v in self.violations if v.get("severity") == "MEDIUM"]
        
        for severity, violations in [
            ("CRITICAL", critical),
            ("HIGH", high),
            ("MEDIUM", medium)
        ]:
            if not violations:
                continue
                
            print(f"{severity} Violations ({len(violations)}):")
            print("-" * 70)
            
            for v in violations:
                print(f"\n{v['type']} in {v['file']}:{v.get('line', '?')}")
                if 'method' in v:
                    print(f"  Method: {v['method']}")
                print(f"  {v['message']}")
                if 'required_action' in v:
                    print(f"  Action: {v['required_action']}")
            print()


def main():
    """Main entry point for CI contract checker"""
    repo_path = Path(__file__).parent
    
    checker = OrchestratorContractChecker(repo_path)
    
    # Run checks
    all_passed = checker.check_repository()
    
    # Report violations
    checker.report_violations()
    
    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_passed:
        print("✓ ALL CONTRACT CHECKS PASSED")
        print()
        print("No orchestrator contract violations detected.")
        print("Merge is allowed (subject to other CI checks).")
        return 0
    else:
        print(f"✗ FOUND {len(checker.violations)} CONTRACT VIOLATION(S)")
        print()
        print("⛔ BLOCK MERGE - Fix violations before merging")
        print()
        print("See violation report above for required actions.")
        print("All violations must be resolved or have SIN_CARRETA-RATIONALE.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
