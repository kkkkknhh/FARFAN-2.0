#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary Scanner Tests
======================

Tests for tools/scan_boundaries.py

Validates:
- Detection of I/O operations
- Detection of __main__ blocks
- Allowed paths are respected
- SARIF and JSON report generation
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json

from tools.scan_boundaries import scan_file, scan_directory, BoundaryScanner


class TestBoundaryScanner:
    """Test AST-based boundary scanner."""
    
    def test_detect_main_block(self):
        """Test detection of __main__ block."""
        code = '''
def main():
    print("hello")

if __name__ == "__main__":
    main()
'''
        temp_file = Path(tempfile.mktemp(suffix='.py'))
        temp_file.write_text(code)
        
        try:
            violations = scan_file(temp_file, scan_main=True, scan_io=False)
            assert len(violations) == 1
            assert violations[0].violation_type == 'main_block'
        finally:
            temp_file.unlink()
    
    def test_detect_open_call(self):
        """Test detection of open() call."""
        code = '''
def read_file():
    with open("file.txt") as f:
        return f.read()
'''
        temp_file = Path(tempfile.mktemp(suffix='.py'))
        temp_file.write_text(code)
        
        try:
            violations = scan_file(temp_file, scan_main=False, scan_io=True)
            # Should detect both open() and read() calls
            assert len(violations) >= 1
            io_violations = [v for v in violations if v.violation_type == 'io_operation']
            assert len(io_violations) >= 1
            assert any('open' in v.message for v in io_violations)
        finally:
            temp_file.unlink()
    
    def test_detect_path_read_text(self):
        """Test detection of Path.read_text() call."""
        code = '''
from pathlib import Path

def read():
    return Path("file.txt").read_text()
'''
        temp_file = Path(tempfile.mktemp(suffix='.py'))
        temp_file.write_text(code)
        
        try:
            violations = scan_file(temp_file, scan_main=False, scan_io=True)
            assert len(violations) >= 1
            io_violations = [v for v in violations if v.violation_type == 'io_operation']
            assert len(io_violations) >= 1
        finally:
            temp_file.unlink()
    
    def test_no_violations_clean_code(self):
        """Test that clean code has no violations."""
        code = '''
def pure_function(x: int, y: int) -> int:
    """A pure function with no I/O."""
    return x + y

def another_pure(data: list) -> list:
    """Another pure function."""
    return sorted(data)
'''
        temp_file = Path(tempfile.mktemp(suffix='.py'))
        temp_file.write_text(code)
        
        try:
            violations = scan_file(temp_file, scan_main=True, scan_io=True)
            assert len(violations) == 0
        finally:
            temp_file.unlink()
    
    def test_scan_directory_with_allowed_paths(self):
        """Test that allowed paths are excluded from scanning."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create structure:
            # /core/module.py - should be scanned
            # /examples/demo.py - should be skipped (allowed)
            core_dir = temp_dir / "core"
            core_dir.mkdir()
            examples_dir = temp_dir / "examples"
            examples_dir.mkdir()
            
            # Core module with violation
            (core_dir / "module.py").write_text('''
if __name__ == "__main__":
    print("main")
''')
            
            # Example with violation (should be ignored)
            (examples_dir / "demo.py").write_text('''
if __name__ == "__main__":
    print("demo")
''')
            
            # Scan with allowed path
            result = scan_directory(
                temp_dir,
                allowed_paths={'examples'},
                fail_on={'main'}
            )
            
            # Should find violation in core but not examples
            assert result.violation_count == 1
            assert 'core/module.py' in result.violations[0].file
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_syntax_error_handling(self):
        """Test that syntax errors are reported as violations."""
        code = '''
def broken():
    if x == :  # Syntax error
        pass
'''
        temp_file = Path(tempfile.mktemp(suffix='.py'))
        temp_file.write_text(code)
        
        try:
            violations = scan_file(temp_file)
            assert len(violations) >= 1
            assert any(v.violation_type == 'parse_error' for v in violations)
        finally:
            temp_file.unlink()


class TestReportGeneration:
    """Test SARIF and JSON report generation."""
    
    def test_json_report_generation(self):
        """Test that JSON reports are generated correctly."""
        from tools.scan_boundaries import ScanResult, Violation, generate_json_report
        
        result = ScanResult(
            total_files=10,
            scanned_files=8,
            violations=[
                Violation(
                    file='test.py',
                    line=10,
                    column=5,
                    node_type='Call',
                    violation_type='io_operation',
                    message='I/O detected',
                    severity='error'
                )
            ]
        )
        
        temp_file = Path(tempfile.mktemp(suffix='.json'))
        
        try:
            generate_json_report(result, temp_file)
            assert temp_file.exists()
            
            with open(temp_file) as f:
                data = json.load(f)
            
            assert data['total_files'] == 10
            assert data['scanned_files'] == 8
            assert data['violation_count'] == 1
            assert len(data['violations']) == 1
            assert data['violations'][0]['file'] == 'test.py'
            
        finally:
            if temp_file.exists():
                temp_file.unlink()
    
    def test_sarif_report_generation(self):
        """Test that SARIF reports are generated correctly."""
        from tools.scan_boundaries import ScanResult, Violation, generate_sarif_report
        
        result = ScanResult(
            total_files=5,
            scanned_files=5,
            violations=[
                Violation(
                    file='module.py',
                    line=20,
                    column=10,
                    node_type='If',
                    violation_type='main_block',
                    message='__main__ block detected',
                    severity='error'
                )
            ]
        )
        
        temp_file = Path(tempfile.mktemp(suffix='.sarif'))
        
        try:
            generate_sarif_report(result, temp_file)
            assert temp_file.exists()
            
            with open(temp_file) as f:
                data = json.load(f)
            
            assert data['version'] == '2.1.0'
            assert 'runs' in data
            assert len(data['runs']) == 1
            assert len(data['runs'][0]['results']) == 1
            
            result_item = data['runs'][0]['results'][0]
            assert result_item['ruleId'] == 'main_block'
            assert result_item['message']['text'] == '__main__ block detected'
            
        finally:
            if temp_file.exists():
                temp_file.unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
