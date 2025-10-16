#!/usr/bin/env python3
"""
Automated Batch Test Execution Workflow
Integrates edit tracking, testing, and rollback capabilities
for systematic regression isolation during fix implementation.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from test_batch_tracker import BatchTestTracker
from ci_batch_test_runner import BatchTestRunner


class AutomatedWorkflow:
    """
    Orchestrates the complete workflow:
    1. Track code edits in batches of 5
    2. Auto-trigger tests after each batch
    3. Detect regressions and isolate to specific batch
    4. Provide rollback mechanism for problematic batches
    """
    
    def __init__(self, batch_size: int = 5):
        self.tracker = BatchTestTracker()
        self.tracker.BATCH_SIZE = batch_size
        self.runner = BatchTestRunner(log_dir=str(self.tracker.LOG_DIR))
        self.critical_failure_threshold = 0.1  # 10% test failure triggers rollback warning
    
    def record_and_test(self, file_path: str, description: str = "") -> dict:
        """
        Record a file edit and automatically run tests if batch is complete.
        Returns dict with status and test results if tests were run.
        """
        print(f"\n{'='*80}")
        print(f"RECORDING EDIT: {file_path}")
        print(f"{'='*80}\n")
        
        # Create backup before recording edit
        if self.tracker.current_batch:
            current_batch_id = self.tracker.current_batch.batch_id
        else:
            current_batch_id = len(self.tracker.all_batches) + 1
        
        # Only create backup if this will complete the batch
        edits_in_batch = len(self.tracker.current_batch.edits) if self.tracker.current_batch else 0
        will_complete = (edits_in_batch + 1) >= self.tracker.BATCH_SIZE
        
        if will_complete:
            print(f"📦 Batch {current_batch_id} will complete with this edit")
            print(f"Creating pre-test backup...\n")
            self.tracker.create_backup(current_batch_id)
        
        # Record the edit
        should_test = self.tracker.record_edit(file_path, description)
        
        result = {
            'edit_recorded': True,
            'batch_id': current_batch_id,
            'tests_triggered': should_test
        }
        
        if should_test:
            print(f"\n{'='*80}")
            print(f"BATCH {current_batch_id} COMPLETE - RUNNING TEST SUITE")
            print(f"{'='*80}\n")
            
            # Run comprehensive test suite
            test_success, test_results = self.runner.run_all_tests(verbose=True)
            
            # Analyze results
            analysis = self._analyze_test_results(test_results, current_batch_id)
            
            result.update({
                'test_success': test_success,
                'test_results': test_results,
                'analysis': analysis
            })
            
            # Check for critical failures
            if analysis['critical_failure']:
                self._handle_critical_failure(current_batch_id, analysis)
            
            # Set as baseline if all tests passed
            if test_success:
                self.runner.set_baseline()
                print("\n✓ All tests passed - baseline updated")
        
        return result
    
    def _analyze_test_results(self, test_results: dict, batch_id: int) -> dict:
        """Analyze test results to determine severity and affected modules"""
        total_tests = 0
        total_failed = 0
        failed_modules = []
        
        for module_name, result in test_results.items():
            total_tests += result.total_tests
            module_failed = result.failed + result.errors
            total_failed += module_failed
            
            if module_failed > 0:
                failed_modules.append({
                    'module': module_name,
                    'failed': module_failed,
                    'total': result.total_tests,
                    'failure_rate': module_failed / result.total_tests if result.total_tests > 0 else 0
                })
        
        failure_rate = total_failed / total_tests if total_tests > 0 else 0
        critical_failure = failure_rate >= self.critical_failure_threshold
        
        return {
            'batch_id': batch_id,
            'total_tests': total_tests,
            'total_failed': total_failed,
            'failure_rate': failure_rate,
            'critical_failure': critical_failure,
            'failed_modules': sorted(failed_modules, key=lambda x: x['failed'], reverse=True),
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_critical_failure(self, batch_id: int, analysis: dict):
        """Handle critical test failures"""
        print(f"\n{'='*80}")
        print(f"⚠️  CRITICAL FAILURE DETECTED IN BATCH {batch_id}")
        print(f"{'='*80}\n")
        
        print(f"Failure rate: {analysis['failure_rate']*100:.1f}% ({analysis['total_failed']}/{analysis['total_tests']} tests)")
        print(f"\nMost affected modules:")
        
        for mod in analysis['failed_modules'][:5]:
            print(f"  • {mod['module']}: {mod['failed']}/{mod['total']} tests failed ({mod['failure_rate']*100:.1f}%)")
        
        print(f"\n{'='*80}")
        print("RECOMMENDED ACTIONS:")
        print("="*80)
        print(f"1. Review batch {batch_id} edits in test_batch_logs/batch_tracker.json")
        print(f"2. Review detailed failures in test_batch_logs/test_report_batch_{batch_id}.txt")
        print(f"3. Consider rollback: python batch_test_automation.py rollback {batch_id}")
        print("="*80)
        
        # Save critical failure report
        self._save_critical_failure_report(batch_id, analysis)
    
    def _save_critical_failure_report(self, batch_id: int, analysis: dict):
        """Save detailed critical failure report"""
        report_file = self.tracker.LOG_DIR / f"CRITICAL_FAILURE_batch_{batch_id}.json"
        
        report = {
            'batch_id': batch_id,
            'timestamp': analysis['timestamp'],
            'severity': 'CRITICAL',
            'failure_rate': analysis['failure_rate'],
            'total_failed': analysis['total_failed'],
            'total_tests': analysis['total_tests'],
            'failed_modules': analysis['failed_modules'],
            'rollback_available': (self.tracker.BACKUP_DIR / f"batch_{batch_id}").exists(),
            'batch_edits': []
        }
        
        # Include batch edit details
        if batch_id <= len(self.tracker.all_batches):
            batch = self.tracker.all_batches[batch_id - 1]
            report['batch_edits'] = [
                {
                    'file': edit.file_path,
                    'timestamp': edit.timestamp,
                    'description': edit.description
                }
                for edit in batch.edits
            ]
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Critical failure report saved: {report_file}")
    
    def rollback(self, batch_id: int, confirm: bool = False):
        """Rollback a specific batch"""
        if not confirm:
            print(f"\n{'='*80}")
            print(f"⚠️  ROLLBACK CONFIRMATION REQUIRED")
            print(f"{'='*80}\n")
            print(f"This will revert all changes from batch {batch_id}")
            
            if batch_id <= len(self.tracker.all_batches):
                batch = self.tracker.all_batches[batch_id - 1]
                print(f"\nFiles to be reverted ({len(batch.edits)}):")
                for edit in batch.edits:
                    print(f"  • {edit.file_path}")
            
            response = input("\nProceed with rollback? (yes/no): ").lower()
            if response != 'yes':
                print("Rollback cancelled")
                return False
        
        success = self.tracker.rollback_batch(batch_id)
        
        if success:
            print(f"\n✓ Batch {batch_id} successfully rolled back")
            print("\nRECOMMENDED NEXT STEPS:")
            print("1. Re-run tests to verify system state")
            print("2. Review and fix the problematic changes")
            print("3. Re-apply changes incrementally with smaller batches")
        
        return success
    
    def get_status(self):
        """Get comprehensive workflow status"""
        self.tracker.print_status()
        
        # Print recent test history
        if self.tracker.test_results:
            print(f"\n{'='*60}")
            print("RECENT TEST HISTORY")
            print(f"{'='*60}\n")
            
            for result in self.tracker.test_results[-5:]:
                status = "✓ PASS" if result.success else "✗ FAIL"
                print(f"Batch {result.batch_id}: {status} - {result.passed_tests}/{result.total_tests} tests ({result.duration_seconds:.1f}s)")
    
    def continuous_watch(self, watch_paths: List[str]):
        """
        Watch specified paths for changes and automatically track edits.
        Note: This requires watchdog library (pip install watchdog)
        """
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            print("Error: watchdog library required for continuous watch mode")
            print("Install with: pip install watchdog")
            return
        
        class EditHandler(FileSystemEventHandler):
            def __init__(self, workflow):
                self.workflow = workflow
                self.modified_files = {}
            
            def on_modified(self, event):
                if event.is_directory or not event.src_path.endswith('.py'):
                    return
                
                file_path = event.src_path
                
                # Debounce - only record if not modified in last 2 seconds
                import time
                current_time = time.time()
                if file_path in self.modified_files:
                    if current_time - self.modified_files[file_path] < 2:
                        return
                
                self.modified_files[file_path] = current_time
                
                print(f"\n📝 Detected change: {file_path}")
                self.workflow.record_and_test(file_path, "Auto-detected change")
        
        print(f"\n{'='*80}")
        print("CONTINUOUS WATCH MODE")
        print(f"{'='*80}\n")
        print(f"Watching paths: {', '.join(watch_paths)}")
        print("Press Ctrl+C to stop\n")
        
        observer = Observer()
        handler = EditHandler(self)
        
        for path in watch_paths:
            observer.schedule(handler, path, recursive=True)
        
        observer.start()
        
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            print("\n\nWatch mode stopped")
        
        observer.join()


def main():
    """CLI interface for automated workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Automated Batch Test Execution Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record a single edit
  python batch_test_automation.py record myfile.py --desc "Fixed bug in parser"
  
  # Get workflow status
  python batch_test_automation.py status
  
  # Rollback batch 3
  python batch_test_automation.py rollback 3
  
  # Watch directory for changes
  python batch_test_automation.py watch . --recursive
  
  # Run tests manually
  python batch_test_automation.py test
        """
    )
    
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Number of edits per batch (default: 5)')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Record command
    record_parser = subparsers.add_parser('record', help='Record a file edit')
    record_parser.add_argument('file_path', help='Path to edited file')
    record_parser.add_argument('-d', '--desc', default='', help='Edit description')
    
    # Status command
    subparsers.add_parser('status', help='Show workflow status')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run test suite manually')
    test_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback a batch')
    rollback_parser.add_argument('batch_id', type=int, help='Batch ID to rollback')
    rollback_parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation')
    
    # Watch command
    watch_parser = subparsers.add_parser('watch', help='Watch paths for changes')
    watch_parser.add_argument('paths', nargs='+', help='Paths to watch')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze specific batch')
    analyze_parser.add_argument('batch_id', type=int, help='Batch ID to analyze')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    workflow = AutomatedWorkflow(batch_size=args.batch_size)
    
    if args.command == 'record':
        result = workflow.record_and_test(args.file_path, args.desc)
        
        if result['tests_triggered']:
            sys.exit(0 if result['test_success'] else 1)
        else:
            sys.exit(0)
    
    elif args.command == 'status':
        workflow.get_status()
    
    elif args.command == 'test':
        success, results = workflow.runner.run_all_tests(verbose=args.verbose)
        sys.exit(0 if success else 1)
    
    elif args.command == 'rollback':
        success = workflow.rollback(args.batch_id, confirm=args.yes)
        sys.exit(0 if success else 1)
    
    elif args.command == 'watch':
        workflow.continuous_watch(args.paths)
    
    elif args.command == 'analyze':
        if args.batch_id <= len(workflow.tracker.all_batches):
            batch = workflow.tracker.all_batches[args.batch_id - 1]
            
            print(f"\n{'='*80}")
            print(f"BATCH {args.batch_id} ANALYSIS")
            print(f"{'='*80}\n")
            print(f"Period: {batch.start_time} to {batch.end_time}")
            print(f"Edits: {len(batch.edits)}")
            print(f"Git commit: {batch.git_commit_hash or 'N/A'}\n")
            
            print("Files modified:")
            for i, edit in enumerate(batch.edits, 1):
                print(f"{i}. {edit.file_path}")
                print(f"   Time: {edit.timestamp}")
                print(f"   Hash: {edit.file_hash}")
                if edit.description:
                    print(f"   Description: {edit.description}")
                print()
            
            # Find associated test results
            test_result = next((r for r in workflow.tracker.test_results if r.batch_id == args.batch_id), None)
            
            if test_result:
                print(f"{'='*80}")
                print("TEST RESULTS")
                print(f"{'='*80}\n")
                print(f"Status: {'✓ PASSED' if test_result.success else '✗ FAILED'}")
                print(f"Tests: {test_result.passed_tests}/{test_result.total_tests} passed")
                print(f"Duration: {test_result.duration_seconds:.2f}s")
                
                if test_result.failures:
                    print(f"\nFailures ({len(test_result.failures)}):")
                    for failure in test_result.failures:
                        print(f"  • {failure['module']}.{failure['test_name']}")
        else:
            print(f"Batch {args.batch_id} not found")


if __name__ == '__main__':
    main()
