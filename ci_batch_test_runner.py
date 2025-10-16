#!/usr/bin/env python3
"""
CI-integrated batch test runner with detailed failure analysis
Provides enhanced test execution with regression isolation
"""

import json
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback as tb


@dataclass
class ModuleTestResult:
    """Detailed test results for a specific module"""
    module_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    failure_details: List[Dict] = None
    
    def __post_init__(self):
        if self.failure_details is None:
            self.failure_details = []


class BatchTestRunner:
    """Enhanced test runner with detailed output and regression tracking"""
    
    def __init__(self, log_dir: str = "test_batch_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.ci_results_file = self.log_dir / "ci_test_results.json"
        self.regression_file = self.log_dir / "regressions.json"
    
    def run_all_tests(self, verbose: bool = True) -> Tuple[bool, Dict]:
        """Run all unit tests with detailed output"""
        print("="*80)
        print("COMPREHENSIVE TEST SUITE EXECUTION")
        print("="*80)
        print()
        
        start_time = time.time()
        all_results = {}
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        overall_success = True
        
        # Discover all test files
        test_files = sorted(Path('.').glob('test_*.py'))
        
        print(f"Discovered {len(test_files)} test modules\n")
        
        for i, test_file in enumerate(test_files, 1):
            module_name = test_file.stem
            print(f"[{i}/{len(test_files)}] Running {module_name}...")
            
            result = self._run_module_tests(module_name, verbose)
            all_results[module_name] = result
            
            total_passed += result.passed
            total_failed += result.failed + result.errors
            total_skipped += result.skipped
            
            # Print immediate feedback
            status = "✓ PASSED" if result.failed == 0 and result.errors == 0 else "✗ FAILED"
            print(f"  {status} - {result.passed}/{result.total_tests} tests passed " +
                  f"in {result.duration:.2f}s")
            
            if result.failed > 0 or result.errors > 0:
                overall_success = False
                print(f"  ⚠ {result.failed} failures, {result.errors} errors")
                self._print_module_failures(result, indent=4)
            
            print()
        
        duration = time.time() - start_time
        
        # Print summary
        print("="*80)
        print("TEST EXECUTION SUMMARY")
        print("="*80)
        print(f"Total modules: {len(test_files)}")
        print(f"Total tests:   {total_passed + total_failed + total_skipped}")
        print(f"Passed:        {total_passed}")
        print(f"Failed:        {total_failed}")
        print(f"Skipped:       {total_skipped}")
        print(f"Duration:      {duration:.2f}s")
        print(f"Status:        {'✓ ALL PASSED' if overall_success else '✗ FAILURES DETECTED'}")
        print("="*80)
        
        # Save results
        self._save_ci_results(all_results, overall_success, duration)
        
        return overall_success, all_results
    
    def _run_module_tests(self, module_name: str, verbose: bool) -> ModuleTestResult:
        """Run tests for a specific module"""
        start_time = time.time()
        
        cmd = [
            sys.executable,
            '-m', 'unittest',
            module_name,
            '-v' if verbose else ''
        ]
        cmd = [c for c in cmd if c]  # Remove empty strings
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            output = result.stdout + "\n" + result.stderr
            duration = time.time() - start_time
            
            # Parse output
            total, passed, failed, skipped, errors = self._parse_test_output(output)
            failures = self._extract_detailed_failures(output, module_name)
            
            return ModuleTestResult(
                module_name=module_name,
                total_tests=total,
                passed=passed,
                failed=failed,
                skipped=skipped,
                errors=errors,
                duration=duration,
                failure_details=failures
            )
        
        except subprocess.TimeoutExpired:
            return ModuleTestResult(
                module_name=module_name,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                duration=120.0,
                failure_details=[{
                    'test_name': 'module_execution',
                    'error_type': 'TimeoutError',
                    'message': 'Test module execution exceeded 120s timeout',
                    'traceback': ''
                }]
            )
        
        except Exception as e:
            return ModuleTestResult(
                module_name=module_name,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                duration=0.0,
                failure_details=[{
                    'test_name': 'module_execution',
                    'error_type': type(e).__name__,
                    'message': str(e),
                    'traceback': tb.format_exc()
                }]
            )
    
    def _parse_test_output(self, output: str) -> Tuple[int, int, int, int, int]:
        """Parse unittest output for statistics"""
        import re
        
        ran_match = re.search(r'Ran (\d+) tests? in', output)
        total = int(ran_match.group(1)) if ran_match else 0
        
        # Count different failure types
        failures = len(re.findall(r'^FAIL:', output, re.MULTILINE))
        errors = len(re.findall(r'^ERROR:', output, re.MULTILINE))
        skipped = len(re.findall(r'\.\.\.s+', output)) + output.count('skipped')
        
        failed = failures
        passed = total - failed - errors - skipped
        
        return total, passed, failed, skipped, errors
    
    def _extract_detailed_failures(self, output: str, module_name: str) -> List[Dict]:
        """Extract comprehensive failure information"""
        import re
        
        failures = []
        
        # Pattern for FAIL/ERROR blocks with traceback
        pattern = r'(FAIL|ERROR): (test_\w+) \((\S+)\)\n-+\n(.*?)(?=\n(?:FAIL|ERROR|Ran \d+ tests?|\Z))'
        
        for match in re.finditer(pattern, output, re.DOTALL):
            error_type, test_name, test_class, traceback_block = match.groups()
            
            # Extract assertion message
            assertion_match = re.search(r'(AssertionError|Exception|Error): (.+?)(?:\n|$)', traceback_block)
            message = assertion_match.group(2) if assertion_match else "No message"
            
            # Extract file and line number
            file_match = re.search(r'File "([^"]+)", line (\d+)', traceback_block)
            file_info = f"{file_match.group(1)}:{file_match.group(2)}" if file_match else "unknown"
            
            failures.append({
                'test_name': test_name,
                'test_class': test_class,
                'error_type': error_type,
                'message': message.strip(),
                'file_location': file_info,
                'traceback': traceback_block.strip(),
                'module': module_name
            })
        
        return failures
    
    def _print_module_failures(self, result: ModuleTestResult, indent: int = 0):
        """Print detailed failure information for a module"""
        prefix = " " * indent
        
        for failure in result.failure_details:
            print(f"{prefix}• {failure['test_name']}")
            print(f"{prefix}  Type: {failure['error_type']}")
            print(f"{prefix}  Location: {failure.get('file_location', 'unknown')}")
            print(f"{prefix}  Message: {failure['message'][:100]}")
    
    def _save_ci_results(self, results: Dict[str, ModuleTestResult], success: bool, duration: float):
        """Save test results in CI-friendly format"""
        ci_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_success': success,
            'total_duration': duration,
            'modules': {}
        }
        
        for module_name, result in results.items():
            ci_data['modules'][module_name] = {
                'total_tests': result.total_tests,
                'passed': result.passed,
                'failed': result.failed,
                'skipped': result.skipped,
                'errors': result.errors,
                'duration': result.duration,
                'failures': result.failure_details
            }
        
        with open(self.ci_results_file, 'w') as f:
            json.dump(ci_data, f, indent=2)
        
        print(f"\n✓ CI results saved to: {self.ci_results_file}")
    
    def compare_with_baseline(self, baseline_file: Optional[str] = None) -> Dict:
        """Compare current results with baseline to detect regressions"""
        if not baseline_file:
            baseline_file = self.log_dir / "baseline_results.json"
        
        baseline_path = Path(baseline_file)
        if not baseline_path.exists():
            print(f"No baseline found at {baseline_path}")
            return {}
        
        try:
            with open(baseline_path, 'r') as f:
                baseline = json.load(f)
            
            with open(self.ci_results_file, 'r') as f:
                current = json.load(f)
            
            regressions = self._detect_regressions(baseline, current)
            
            if regressions:
                self._save_regressions(regressions)
                print(f"\n⚠ {len(regressions)} regression(s) detected!")
                self._print_regressions(regressions)
            else:
                print("\n✓ No regressions detected")
            
            return regressions
        
        except Exception as e:
            print(f"Error comparing with baseline: {e}")
            return {}
    
    def _detect_regressions(self, baseline: Dict, current: Dict) -> List[Dict]:
        """Detect regressions by comparing baseline and current results"""
        regressions = []
        
        baseline_modules = baseline.get('modules', {})
        current_modules = current.get('modules', {})
        
        for module_name in current_modules:
            if module_name not in baseline_modules:
                continue  # New module, not a regression
            
            baseline_mod = baseline_modules[module_name]
            current_mod = current_modules[module_name]
            
            # Check if previously passing tests now fail
            baseline_failed = baseline_mod.get('failed', 0) + baseline_mod.get('errors', 0)
            current_failed = current_mod.get('failed', 0) + current_mod.get('errors', 0)
            
            if current_failed > baseline_failed:
                regressions.append({
                    'module': module_name,
                    'type': 'new_failures',
                    'baseline_failed': baseline_failed,
                    'current_failed': current_failed,
                    'delta': current_failed - baseline_failed,
                    'failures': current_mod.get('failures', [])
                })
            
            # Check for performance regressions (>50% slower)
            baseline_duration = baseline_mod.get('duration', 0)
            current_duration = current_mod.get('duration', 0)
            
            if baseline_duration > 0 and current_duration > baseline_duration * 1.5:
                regressions.append({
                    'module': module_name,
                    'type': 'performance_degradation',
                    'baseline_duration': baseline_duration,
                    'current_duration': current_duration,
                    'slowdown_factor': current_duration / baseline_duration
                })
        
        return regressions
    
    def _save_regressions(self, regressions: List[Dict]):
        """Save regression data"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'regressions': regressions
        }
        
        with open(self.regression_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _print_regressions(self, regressions: List[Dict]):
        """Print regression details"""
        print("\n" + "="*80)
        print("REGRESSION REPORT")
        print("="*80)
        
        for i, reg in enumerate(regressions, 1):
            print(f"\n{i}. Module: {reg['module']}")
            print(f"   Type: {reg['type']}")
            
            if reg['type'] == 'new_failures':
                print(f"   Failed tests increased: {reg['baseline_failed']} → {reg['current_failed']}")
                print(f"   New failures: {reg['delta']}")
            elif reg['type'] == 'performance_degradation':
                print(f"   Duration increased: {reg['baseline_duration']:.2f}s → {reg['current_duration']:.2f}s")
                print(f"   Slowdown: {reg['slowdown_factor']:.2f}x")
    
    def set_baseline(self):
        """Set current results as baseline for future comparisons"""
        baseline_file = self.log_dir / "baseline_results.json"
        
        if self.ci_results_file.exists():
            with open(self.ci_results_file, 'r') as src:
                data = json.load(src)
            
            with open(baseline_file, 'w') as dst:
                json.dump(data, dst, indent=2)
            
            print(f"✓ Baseline set from current results: {baseline_file}")
        else:
            print("✗ No current results to set as baseline")


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CI Batch Test Runner')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--set-baseline', action='store_true', help='Set current results as baseline')
    parser.add_argument('--compare', action='store_true', help='Compare with baseline')
    parser.add_argument('--baseline-file', help='Path to baseline file')
    
    args = parser.parse_args()
    
    runner = BatchTestRunner()
    
    if args.set_baseline:
        runner.set_baseline()
        return
    
    # Run tests
    success, results = runner.run_all_tests(verbose=args.verbose)
    
    # Compare with baseline if requested
    if args.compare:
        regressions = runner.compare_with_baseline(args.baseline_file)
        if regressions:
            sys.exit(2)  # Exit code 2 for regressions
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
