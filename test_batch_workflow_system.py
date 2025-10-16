#!/usr/bin/env python3
"""
Unit tests for the test batch workflow system
"""

import unittest
import json
import os
import shutil
from pathlib import Path
from test_batch_tracker import BatchTestTracker, EditBatch, FileEdit, TestResult
from ci_batch_test_runner import BatchTestRunner, ModuleTestResult


class TestBatchTracker(unittest.TestCase):
    """Test the batch tracking system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_log_dir = Path("test_batch_logs_test")
        self.tracker = BatchTestTracker()
        self.tracker.LOG_DIR = self.test_log_dir
        self.tracker.TRACKER_FILE = self.test_log_dir / "batch_tracker.json"
        self.tracker.RESULTS_FILE = self.test_log_dir / "test_results.json"
        self.tracker.BACKUP_DIR = self.test_log_dir / "backups"
        self.tracker.LOG_DIR.mkdir(exist_ok=True)
        
        # Reset tracker state for isolated tests
        self.tracker.edit_count = 0
        self.tracker.current_batch = None
        self.tracker.all_batches = []
        self.tracker.test_results = []
        
        # Create test files
        self.test_files = []
        for i in range(5):
            test_file = self.test_log_dir / f"test_file_{i}.py"
            test_file.write_text(f"# Test file {i}\n")
            self.test_files.append(str(test_file))
    
    def tearDown(self):
        """Clean up test environment"""
        if self.test_log_dir.exists():
            shutil.rmtree(self.test_log_dir)
    
    def test_record_single_edit(self):
        """Test recording a single edit"""
        result = self.tracker.record_edit(self.test_files[0], "Test edit")
        
        self.assertEqual(self.tracker.edit_count, 1)
        self.assertIsNotNone(self.tracker.current_batch)
        self.assertEqual(len(self.tracker.current_batch.edits), 1)
        self.assertFalse(result)  # Should not trigger tests yet
    
    def test_record_batch_completion(self):
        """Test batch completion after 5 edits"""
        for i, file_path in enumerate(self.test_files):
            result = self.tracker.record_edit(file_path, f"Edit {i+1}")
            
            if i < 4:
                self.assertFalse(result)
            else:
                self.assertTrue(result)  # 5th edit should trigger
        
        self.assertEqual(self.tracker.edit_count, 5)
        self.assertEqual(len(self.tracker.all_batches), 1)
        self.assertIsNone(self.tracker.current_batch)  # Should be finalized
    
    def test_state_persistence(self):
        """Test that state is persisted and can be loaded"""
        # Record some edits
        for i in range(3):
            self.tracker.record_edit(self.test_files[i], f"Edit {i+1}")
        
        self.tracker._save_state()
        
        # Create new tracker and load state
        new_tracker = BatchTestTracker()
        new_tracker.LOG_DIR = self.test_log_dir
        new_tracker.TRACKER_FILE = self.test_log_dir / "batch_tracker.json"
        new_tracker.RESULTS_FILE = self.test_log_dir / "test_results.json"
        new_tracker.BACKUP_DIR = self.test_log_dir / "backups"
        new_tracker._load_state()
        
        self.assertEqual(new_tracker.edit_count, 3)
        self.assertIsNotNone(new_tracker.current_batch)
        self.assertEqual(len(new_tracker.current_batch.edits), 3)
    
    def test_file_hash_calculation(self):
        """Test file hash calculation"""
        file_path = self.test_files[0]
        hash1 = self.tracker._get_file_hash(file_path)
        
        # Hash should be consistent
        hash2 = self.tracker._get_file_hash(file_path)
        self.assertEqual(hash1, hash2)
        
        # Hash should change when file changes
        Path(file_path).write_text("# Modified content\n")
        hash3 = self.tracker._get_file_hash(file_path)
        self.assertNotEqual(hash1, hash3)
    
    def test_backup_creation(self):
        """Test backup creation"""
        # Create a batch
        for file_path in self.test_files:
            self.tracker.record_edit(file_path, "Test edit")
        
        # Create backup
        backup_dir = self.tracker.create_backup(1)
        
        self.assertIsNotNone(backup_dir)
        self.assertTrue(backup_dir.exists())
        
        # Check that files were backed up
        backed_up_files = list(backup_dir.glob("*.py"))
        self.assertEqual(len(backed_up_files), 5)
    
    def test_test_result_creation(self):
        """Test TestResult dataclass"""
        result = TestResult(
            batch_id=1,
            timestamp="2024-01-01T00:00:00",
            success=True,
            total_tests=100,
            passed_tests=95,
            failed_tests=5,
            skipped_tests=0,
            duration_seconds=10.5
        )
        
        self.assertEqual(result.batch_id, 1)
        self.assertTrue(result.success)
        self.assertEqual(result.total_tests, 100)
        self.assertEqual(result.passed_tests, 95)


class TestBatchTestRunner(unittest.TestCase):
    """Test the CI batch test runner"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_log_dir = "test_batch_logs_runner_test"
        self.runner = BatchTestRunner(log_dir=self.test_log_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        if Path(self.test_log_dir).exists():
            shutil.rmtree(self.test_log_dir)
    
    def test_module_result_creation(self):
        """Test ModuleTestResult creation"""
        result = ModuleTestResult(
            module_name="test_example",
            total_tests=50,
            passed=45,
            failed=5,
            skipped=0,
            errors=0,
            duration=5.2
        )
        
        self.assertEqual(result.module_name, "test_example")
        self.assertEqual(result.passed, 45)
        self.assertEqual(result.failed, 5)
        self.assertIsNotNone(result.failure_details)
        self.assertEqual(len(result.failure_details), 0)
    
    def test_parse_test_output(self):
        """Test parsing of unittest output"""
        sample_output = """
FAIL: test_example_2 (test_module.TestClass) ... 
ERROR: test_example_4 (test_module.TestClass) ...

----------------------------------------------------------------------
Ran 4 tests in 0.123s

FAILED (failures=1, errors=1)
"""
        
        total, passed, failed, skipped, errors = self.runner._parse_test_output(sample_output)
        
        self.assertEqual(total, 4)
        self.assertEqual(failed, 1)
        self.assertEqual(errors, 1)
        self.assertEqual(passed, 2)
    
    def test_ci_results_save(self):
        """Test saving CI results"""
        results = {
            'test_module1': ModuleTestResult(
                module_name='test_module1',
                total_tests=10,
                passed=9,
                failed=1,
                skipped=0,
                errors=0,
                duration=1.5
            )
        }
        
        self.runner._save_ci_results(results, False, 10.0)
        
        self.assertTrue(self.runner.ci_results_file.exists())
        
        with open(self.runner.ci_results_file) as f:
            data = json.load(f)
        
        self.assertFalse(data['overall_success'])
        self.assertEqual(data['total_duration'], 10.0)
        self.assertIn('test_module1', data['modules'])
    
    def test_regression_detection(self):
        """Test regression detection logic"""
        baseline = {
            'modules': {
                'test_module1': {
                    'total_tests': 10,
                    'passed': 10,
                    'failed': 0,
                    'errors': 0,
                    'duration': 1.0
                }
            }
        }
        
        current = {
            'modules': {
                'test_module1': {
                    'total_tests': 10,
                    'passed': 8,
                    'failed': 2,
                    'errors': 0,
                    'duration': 1.0,
                    'failures': []
                }
            }
        }
        
        regressions = self.runner._detect_regressions(baseline, current)
        
        self.assertEqual(len(regressions), 1)
        self.assertEqual(regressions[0]['type'], 'new_failures')
        self.assertEqual(regressions[0]['module'], 'test_module1')
        self.assertEqual(regressions[0]['delta'], 2)
    
    def test_performance_regression_detection(self):
        """Test performance regression detection"""
        baseline = {
            'modules': {
                'test_module1': {
                    'total_tests': 10,
                    'passed': 10,
                    'failed': 0,
                    'errors': 0,
                    'duration': 1.0
                }
            }
        }
        
        current = {
            'modules': {
                'test_module1': {
                    'total_tests': 10,
                    'passed': 10,
                    'failed': 0,
                    'errors': 0,
                    'duration': 2.0,  # 2x slower
                    'failures': []
                }
            }
        }
        
        regressions = self.runner._detect_regressions(baseline, current)
        
        self.assertEqual(len(regressions), 1)
        self.assertEqual(regressions[0]['type'], 'performance_degradation')
        self.assertEqual(regressions[0]['slowdown_factor'], 2.0)


class TestWorkflowIntegration(unittest.TestCase):
    """Integration tests for the complete workflow"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_log_dir = Path("test_batch_logs_integration_test")
        self.tracker = BatchTestTracker()
        self.tracker.LOG_DIR = self.test_log_dir
        self.tracker.TRACKER_FILE = self.test_log_dir / "batch_tracker.json"
        self.tracker.RESULTS_FILE = self.test_log_dir / "test_results.json"
        self.tracker.BACKUP_DIR = self.test_log_dir / "backups"
        self.tracker.LOG_DIR.mkdir(exist_ok=True)
        
        # Create test files
        self.test_files = []
        for i in range(10):
            test_file = self.test_log_dir / f"test_file_{i}.py"
            test_file.write_text(f"# Test file {i}\nprint('Hello')\n")
            self.test_files.append(str(test_file))
    
    def tearDown(self):
        """Clean up test environment"""
        if self.test_log_dir.exists():
            shutil.rmtree(self.test_log_dir)
    
    def test_two_batch_workflow(self):
        """Test complete workflow with two batches"""
        # Reset tracker state to ensure clean test
        self.tracker.edit_count = 0
        self.tracker.current_batch = None
        self.tracker.all_batches = []
        
        # First batch
        for i in range(5):
            should_test = self.tracker.record_edit(
                self.test_files[i],
                f"Batch 1 edit {i+1}"
            )
        
        self.assertTrue(should_test)
        self.assertEqual(len(self.tracker.all_batches), 1)
        self.assertTrue(self.tracker.all_batches[0].end_time)
        
        # Second batch
        for i in range(5, 10):
            should_test = self.tracker.record_edit(
                self.test_files[i],
                f"Batch 2 edit {i-4}"
            )
        
        self.assertTrue(should_test)
        self.assertEqual(len(self.tracker.all_batches), 2)
        self.assertTrue(self.tracker.all_batches[1].end_time)
    
    def test_backup_and_rollback_workflow(self):
        """Test backup creation and rollback"""
        # Reset tracker state
        self.tracker.edit_count = 0
        self.tracker.current_batch = None
        self.tracker.all_batches = []
        
        # Create first batch
        for i in range(5):
            self.tracker.record_edit(self.test_files[i], f"Edit {i+1}")
        
        # Store original content before backup
        original_content = {}
        for file_path in self.test_files[:5]:
            original_content[file_path] = Path(file_path).read_text()
        
        # Create backup
        backup_dir = self.tracker.create_backup(1)
        self.assertTrue(backup_dir.exists())
        
        # Modify files
        for file_path in self.test_files[:5]:
            Path(file_path).write_text("# Modified\n")
        
        # Rollback
        success = self.tracker.rollback_batch(1)
        self.assertTrue(success)
        
        # Verify files were restored to backup content
        for file_path in self.test_files[:5]:
            content = Path(file_path).read_text()
            # Content should match what was backed up, which includes the "Test file" text
            self.assertIn("# Test file", content)


if __name__ == '__main__':
    unittest.main()
