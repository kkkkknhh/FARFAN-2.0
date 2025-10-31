#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary Scanner for I/O Operations
====================================

AST-based scanner that detects I/O operations and __main__ blocks in Python code.
Designed to enforce architectural boundaries in FARFAN 2.0.

Fails CI on:
- File I/O operations (open, Path.read_text, Path.write_text, etc.)
- Network operations (requests.*, urllib.*, http.client.*)
- Subprocess operations (subprocess.*, os.system, os.popen)
- __main__ blocks outside allowed directories

Exit codes:
- 0: No violations found
- 1: Violations found or error occurred
"""

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Set, Any, Optional


@dataclass
class Violation:
    """Represents a single boundary violation."""
    file: str
    line: int
    column: int
    node_type: str
    violation_type: str
    message: str
    severity: str = "error"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ScanResult:
    """Results from boundary scanning."""
    total_files: int = 0
    scanned_files: int = 0
    violations: List[Violation] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def has_violations(self) -> bool:
        """Check if any violations were found."""
        return len(self.violations) > 0
    
    @property
    def violation_count(self) -> int:
        """Total number of violations."""
        return len(self.violations)


class BoundaryScanner(ast.NodeVisitor):
    """AST visitor that detects boundary violations."""
    
    # I/O-related function calls to detect
    IO_FUNCTIONS = {
        'open', 'read', 'write', 'readline', 'readlines', 'writelines',
        'read_text', 'write_text', 'read_bytes', 'write_bytes',
        'mkdir', 'rmdir', 'remove', 'unlink', 'rename', 'replace',
    }
    
    # I/O-related modules
    IO_MODULES = {
        'pathlib', 'os.path', 'shutil', 'tempfile', 'io',
    }
    
    # Network-related modules
    NETWORK_MODULES = {
        'requests', 'urllib', 'http.client', 'httpx', 'aiohttp',
    }
    
    # Subprocess modules
    SUBPROCESS_MODULES = {
        'subprocess', 'os.system', 'os.popen',
    }
    
    def __init__(self, filepath: str, scan_main: bool = True, scan_io: bool = True, 
                 scan_subprocess: bool = True, scan_requests: bool = True):
        """Initialize scanner.
        
        Args:
            filepath: Path to the file being scanned
            scan_main: Whether to scan for __main__ blocks
            scan_io: Whether to scan for I/O operations
            scan_subprocess: Whether to scan for subprocess operations
            scan_requests: Whether to scan for network operations
        """
        self.filepath = filepath
        self.scan_main = scan_main
        self.scan_io = scan_io
        self.scan_subprocess = scan_subprocess
        self.scan_requests = scan_requests
        self.violations: List[Violation] = []
        
    def visit_If(self, node: ast.If) -> None:
        """Visit If nodes to detect __main__ blocks."""
        if not self.scan_main:
            self.generic_visit(node)
            return
            
        # Check for if __name__ == "__main__":
        if isinstance(node.test, ast.Compare):
            left = node.test.left
            if isinstance(left, ast.Name) and left.id == '__name__':
                for comparator in node.test.comparators:
                    if isinstance(comparator, ast.Constant) and comparator.value == '__main__':
                        self.violations.append(Violation(
                            file=self.filepath,
                            line=node.lineno,
                            column=node.col_offset,
                            node_type='If',
                            violation_type='main_block',
                            message='__main__ block found (should be in examples/ or cli/)',
                            severity='error'
                        ))
        
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Visit Call nodes to detect I/O operations."""
        # Check for direct function calls (e.g., open())
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if self.scan_io and func_name in self.IO_FUNCTIONS:
                self.violations.append(Violation(
                    file=self.filepath,
                    line=node.lineno,
                    column=node.col_offset,
                    node_type='Call',
                    violation_type='io_operation',
                    message=f'Direct I/O call: {func_name}() (use injected port instead)',
                    severity='error'
                ))
        
        # Check for attribute calls (e.g., Path.read_text(), requests.get())
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            
            # Try to get the full module path
            module_path = self._get_module_path(node.func.value)
            
            # Check I/O operations
            if self.scan_io and (attr_name in self.IO_FUNCTIONS or module_path in self.IO_MODULES):
                self.violations.append(Violation(
                    file=self.filepath,
                    line=node.lineno,
                    column=node.col_offset,
                    node_type='Call',
                    violation_type='io_operation',
                    message=f'I/O operation: {module_path}.{attr_name}() (use injected port instead)',
                    severity='error'
                ))
            
            # Check network operations
            elif self.scan_requests and any(module_path.startswith(net) for net in self.NETWORK_MODULES):
                self.violations.append(Violation(
                    file=self.filepath,
                    line=node.lineno,
                    column=node.col_offset,
                    node_type='Call',
                    violation_type='network_operation',
                    message=f'Network operation: {module_path}.{attr_name}() (use injected port instead)',
                    severity='error'
                ))
            
            # Check subprocess operations  
            elif self.scan_subprocess and any(module_path.startswith(sub) for sub in self.SUBPROCESS_MODULES):
                self.violations.append(Violation(
                    file=self.filepath,
                    line=node.lineno,
                    column=node.col_offset,
                    node_type='Call',
                    violation_type='subprocess_operation',
                    message=f'Subprocess operation: {module_path}.{attr_name}() (use injected port instead)',
                    severity='error'
                ))
        
        self.generic_visit(node)
    
    def _get_module_path(self, node: ast.AST) -> str:
        """Extract module path from AST node."""
        parts = []
        current = node
        
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            parts.append(current.id)
        
        return '.'.join(reversed(parts))


def scan_file(filepath: Path, scan_main: bool = True, scan_io: bool = True,
              scan_subprocess: bool = True, scan_requests: bool = True) -> List[Violation]:
    """Scan a single Python file for boundary violations.
    
    Args:
        filepath: Path to Python file
        scan_main: Whether to scan for __main__ blocks
        scan_io: Whether to scan for I/O operations
        scan_subprocess: Whether to scan for subprocess operations
        scan_requests: Whether to scan for network operations
        
    Returns:
        List of violations found
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source, filename=str(filepath))
        scanner = BoundaryScanner(
            str(filepath),
            scan_main=scan_main,
            scan_io=scan_io,
            scan_subprocess=scan_subprocess,
            scan_requests=scan_requests
        )
        scanner.visit(tree)
        return scanner.violations
    
    except SyntaxError as e:
        return [Violation(
            file=str(filepath),
            line=e.lineno or 0,
            column=e.offset or 0,
            node_type='SyntaxError',
            violation_type='parse_error',
            message=f'Syntax error: {e.msg}',
            severity='error'
        )]
    except Exception as e:
        return [Violation(
            file=str(filepath),
            line=0,
            column=0,
            node_type='Exception',
            violation_type='scan_error',
            message=f'Error scanning file: {str(e)}',
            severity='error'
        )]


def scan_directory(root_path: Path, allowed_paths: Set[str] = None,
                  fail_on: Set[str] = None) -> ScanResult:
    """Scan a directory tree for boundary violations.
    
    Args:
        root_path: Root directory to scan
        allowed_paths: Paths where violations are allowed (relative to root)
        fail_on: Types of violations to report ('io', 'subprocess', 'requests', 'main')
        
    Returns:
        ScanResult with all violations found
    """
    if allowed_paths is None:
        allowed_paths = set()
    
    if fail_on is None:
        fail_on = {'io', 'subprocess', 'requests', 'main'}
    
    # Convert to scanning flags
    scan_main = 'main' in fail_on
    scan_io = 'io' in fail_on
    scan_subprocess = 'subprocess' in fail_on
    scan_requests = 'requests' in fail_on
    
    result = ScanResult()
    
    # Find all Python files
    python_files = list(root_path.rglob('*.py'))
    result.total_files = len(python_files)
    
    for filepath in python_files:
        # Check if file is in allowed path
        relative_path = filepath.relative_to(root_path)
        is_allowed = any(str(relative_path).startswith(allowed) for allowed in allowed_paths)
        
        if is_allowed:
            continue
        
        # Scan file
        violations = scan_file(
            filepath,
            scan_main=scan_main,
            scan_io=scan_io,
            scan_subprocess=scan_subprocess,
            scan_requests=scan_requests
        )
        
        result.violations.extend(violations)
        result.scanned_files += 1
    
    return result


def generate_sarif_report(result: ScanResult, output_path: Path) -> None:
    """Generate SARIF report for IDE/CI integration.
    
    Args:
        result: Scan results
        output_path: Path to write SARIF JSON
    """
    sarif = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "FARFAN Boundary Scanner",
                    "version": "1.0.0",
                    "informationUri": "https://github.com/kkkkknhh/FARFAN-2.0",
                }
            },
            "results": []
        }]
    }
    
    for violation in result.violations:
        sarif["runs"][0]["results"].append({
            "ruleId": violation.violation_type,
            "level": violation.severity,
            "message": {
                "text": violation.message
            },
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": violation.file
                    },
                    "region": {
                        "startLine": violation.line,
                        "startColumn": violation.column
                    }
                }
            }]
        })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sarif, f, indent=2)


def generate_json_report(result: ScanResult, output_path: Path) -> None:
    """Generate JSON report of violations.
    
    Args:
        result: Scan results
        output_path: Path to write JSON
    """
    report = {
        "total_files": result.total_files,
        "scanned_files": result.scanned_files,
        "violation_count": result.violation_count,
        "violations": [v.to_dict() for v in result.violations]
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Scan Python code for architectural boundary violations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('.'),
        help='Root directory to scan (default: current directory)'
    )
    
    parser.add_argument(
        '--fail-on',
        type=str,
        default='io,subprocess,requests,main',
        help='Comma-separated list of violation types to report (default: io,subprocess,requests,main)'
    )
    
    parser.add_argument(
        '--allow-path',
        action='append',
        default=[],
        help='Paths where violations are allowed (can be specified multiple times)'
    )
    
    parser.add_argument(
        '--sarif',
        type=Path,
        help='Output SARIF report to this path'
    )
    
    parser.add_argument(
        '--json',
        type=Path,
        help='Output JSON report to this path'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Parse fail-on argument
    fail_on = set(v.strip() for v in args.fail_on.split(','))
    
    # Scan directory
    print(f"Scanning {args.root} for boundary violations...")
    print(f"Fail on: {', '.join(sorted(fail_on))}")
    if args.allow_path:
        print(f"Allowed paths: {', '.join(args.allow_path)}")
    print()
    
    result = scan_directory(
        args.root,
        allowed_paths=set(args.allow_path),
        fail_on=fail_on
    )
    
    # Generate reports
    if args.sarif:
        generate_sarif_report(result, args.sarif)
        print(f"SARIF report written to: {args.sarif}")
    
    if args.json:
        generate_json_report(result, args.json)
        print(f"JSON report written to: {args.json}")
    
    # Print summary
    print(f"\nScanned {result.scanned_files} files (total: {result.total_files})")
    print(f"Found {result.violation_count} violations")
    
    if result.has_violations:
        print("\nViolations by type:")
        violation_types: Dict[str, int] = {}
        for v in result.violations:
            violation_types[v.violation_type] = violation_types.get(v.violation_type, 0) + 1
        
        for vtype, count in sorted(violation_types.items()):
            print(f"  {vtype}: {count}")
        
        if args.verbose:
            print("\nDetailed violations:")
            for v in result.violations:
                print(f"  {v.file}:{v.line}:{v.column} - {v.message}")
        
        print(f"\n✗ Boundary scan FAILED - {result.violation_count} violations found")
        return 1
    else:
        print("\n✓ Boundary scan PASSED - no violations found")
        return 0


if __name__ == '__main__':
    sys.exit(main())
