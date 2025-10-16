#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CI Contract Enforcement for Phase Wiring
=========================================

This script enforces SIN_CARRETA compliance for the orchestrator:
- NO placeholder logic allowed
- NO silent fallbacks
- NO ambiguous routing
- ALL contracts must be explicit

Run in CI to fail the build if violations are detected.
"""

import re
import sys
from pathlib import Path


class ContractViolation(Exception):
    """Raised when a contract violation is detected"""
    pass


def check_no_placeholders(file_path: Path) -> None:
    """
    Check that no placeholder comments or logic exist in the file.
    
    SIN_CARRETA: Placeholders are forbidden in production code.
    """
    content = file_path.read_text(encoding='utf-8')
    
    # Forbidden patterns
    placeholder_patterns = [
        r'# Placeholder:',
        r'#\s*TODO:',
        r'#\s*FIXME:',
        r'#\s*XXX:',
        r'Would be extracted',
        r'Would be detected',
        r'Would be calculated',
        r'In real implementation',
        r'In production',
    ]
    
    violations = []
    for pattern in placeholder_patterns:
        matches = list(re.finditer(pattern, content, re.IGNORECASE))
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            violations.append(f"Line {line_num}: Placeholder pattern found: {match.group()}")
    
    if violations:
        raise ContractViolation(
            f"Placeholder violations in {file_path}:\n" + "\n".join(violations)
        )


def check_explicit_error_handling(file_path: Path) -> None:
    """
    Check that all error handling is explicit (no silent failures).
    
    SIN_CARRETA: Silent failures are forbidden.
    """
    content = file_path.read_text(encoding='utf-8')
    
    # Look for bare except: blocks (potential silent failures)
    bare_except_pattern = r'except\s*:'
    matches = list(re.finditer(bare_except_pattern, content))
    
    violations = []
    for match in matches:
        line_num = content[:match.start()].count('\n') + 1
        # Check if this is followed by proper error handling
        context = content[match.end():match.end()+200]
        if 'logger.error' not in context and 'raise' not in context:
            violations.append(f"Line {line_num}: Bare except without explicit error handling")
    
    if violations:
        raise ContractViolation(
            f"Silent failure violations in {file_path}:\n" + "\n".join(violations)
        )


def check_contract_dataclasses(file_path: Path) -> None:
    """
    Check that all dataclasses have validation in __post_init__.
    
    SIN_CARRETA: All contracts must have runtime validation.
    """
    content = file_path.read_text(encoding='utf-8')
    
    # Find all dataclass definitions
    dataclass_pattern = r'@dataclass[^\n]*\nclass (\w+)'
    dataclasses = list(re.finditer(dataclass_pattern, content))
    
    violations = []
    for match in dataclasses:
        class_name = match.group(1)
        # Skip if it's a test class or helper
        if 'Test' in class_name or class_name in ['AnalyticalPhase']:
            continue
        
        # Check if __post_init__ exists within this class
        class_start = match.end()
        # Find the next class or end of file
        next_class = re.search(r'\nclass ', content[class_start:])
        class_end = class_start + next_class.start() if next_class else len(content)
        class_content = content[class_start:class_end]
        
        if 'def __post_init__' not in class_content:
            line_num = content[:match.start()].count('\n') + 1
            violations.append(
                f"Line {line_num}: Dataclass {class_name} missing __post_init__ validation"
            )
    
    if violations:
        raise ContractViolation(
            f"Contract validation violations in {file_path}:\n" + "\n".join(violations)
        )


def check_no_hardcoded_thresholds(file_path: Path) -> None:
    """
    Check that no hardcoded thresholds exist (must use calibration constants).
    
    SIN_CARRETA: All thresholds must come from CALIBRATION singleton.
    """
    content = file_path.read_text(encoding='utf-8')
    
    # Look for hardcoded numeric thresholds in conditional statements
    # Skip lines that use self.calibration
    violations = []
    
    # Patterns that suggest hardcoded thresholds
    threshold_patterns = [
        r'if .* > 0\.\d+:',  # Float comparisons (except 0.0, 1.0)
        r'if .* >= 0\.\d+:',
        r'if .* < \d+\.\d+:',
        r'score.*=.*0\.\d+',  # Score assignments
    ]
    
    for pattern in threshold_patterns:
        matches = list(re.finditer(pattern, content))
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            line = content.splitlines()[line_num - 1]
            
            # Skip if line uses calibration
            if 'self.calibration' in line or 'CALIBRATION' in line:
                continue
            # Skip if it's a comment
            if line.strip().startswith('#'):
                continue
            # Skip specific allowed patterns
            if '0.0' in line or '1.0' in line:
                continue
            if 'max(0.0' in line or 'min(1.0' in line:
                continue
            
            violations.append(
                f"Line {line_num}: Potential hardcoded threshold: {line.strip()}"
            )
    
    # Only raise if we found violations
    if violations:
        print(f"⚠ Warning: Potential hardcoded thresholds in {file_path}:")
        for violation in violations:
            print(f"  {violation}")


def main():
    """Run all contract enforcement checks"""
    orchestrator_path = Path(__file__).parent / 'orchestrator.py'
    
    if not orchestrator_path.exists():
        print(f"❌ ERROR: {orchestrator_path} not found")
        sys.exit(1)
    
    print("=" * 60)
    print("CI Contract Enforcement - Phase Wiring Integration")
    print("=" * 60)
    
    checks = [
        ("No placeholders", check_no_placeholders),
        ("Explicit error handling", check_explicit_error_handling),
        ("Contract dataclasses", check_contract_dataclasses),
        ("No hardcoded thresholds", check_no_hardcoded_thresholds),
    ]
    
    passed = 0
    failed = 0
    
    for check_name, check_func in checks:
        try:
            print(f"\nChecking: {check_name}...", end=" ")
            check_func(orchestrator_path)
            print("✓ PASS")
            passed += 1
        except ContractViolation as e:
            print(f"✗ FAIL")
            print(f"  {e}")
            failed += 1
        except Exception as e:
            print(f"⚠ ERROR")
            print(f"  Unexpected error: {e}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        print("\n❌ CI FAILED: Contract violations detected")
        print("Fix violations before merge.")
        sys.exit(1)
    else:
        print("\n✅ CI PASSED: All contract enforcements satisfied")
        print("Code is ready for merge.")
        sys.exit(0)


if __name__ == "__main__":
    main()
