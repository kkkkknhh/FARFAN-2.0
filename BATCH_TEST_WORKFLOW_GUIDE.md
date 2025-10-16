# Batch Test Workflow System - Complete Guide

## Overview

The Batch Test Workflow System provides systematic unit test execution that runs after every 5 code edits, capturing test results and isolating regressions to specific batches of changes. This enables rapid identification and rollback of problematic edits during fix implementation.

## Architecture

The system consists of three integrated components:

### 1. **test_batch_tracker.py** - Edit Tracking & Batch Management
- Records file edits with timestamps and content hashes
- Groups edits into batches (default: 5 edits per batch)
- Triggers test execution automatically when batch completes
- Creates backups before test runs
- Provides rollback functionality

### 2. **ci_batch_test_runner.py** - Comprehensive Test Execution
- Runs all unittest modules with detailed output
- Extracts stack traces and failure details per module
- Detects regressions by comparing with baseline results
- Identifies performance degradations (>50% slower)
- Generates CI-friendly JSON reports

### 3. **batch_test_automation.py** - Unified Workflow Orchestration
- Integrates tracking and testing into single workflow
- Analyzes test results for critical failures (>10% failure rate)
- Automatically creates backups before batch testing
- Provides CLI interface for all operations
- Optional continuous watch mode for automatic tracking

## Quick Start

### Installation

All required dependencies are already in `requirements.txt`. The system is ready to use.

### Basic Usage

#### 1. Record an edit and auto-test when batch completes:
```bash
python batch_test_automation.py record myfile.py --desc "Fixed parser bug"
```

#### 2. Check workflow status:
```bash
python batch_test_automation.py status
```

#### 3. Run tests manually:
```bash
python batch_test_automation.py test
```

#### 4. Rollback problematic batch:
```bash
python batch_test_automation.py rollback 3
```

## Detailed Workflow

### Phase 1: Edit Recording (Batches 1-N)

Each time you make a code edit:

```bash
# After editing file1.py
python batch_test_automation.py record file1.py --desc "Added validation"

# After editing file2.py  
python batch_test_automation.py record file2.py --desc "Fixed edge case"

# Continue for 3 more edits...
```

**What happens:**
- Edit is recorded with timestamp and file hash
- Progress is displayed: "Batch 1: 2/5 edits"
- State is persisted to `test_batch_logs/batch_tracker.json`
- No tests run yet (batch incomplete)

### Phase 2: Batch Completion & Auto-Testing

On the 5th edit:

```bash
python batch_test_automation.py record file5.py --desc "Updated docs"
```

**What happens automatically:**

1. **Pre-test Backup Created**
   ```
   📦 Batch 1 will complete with this edit
   Creating pre-test backup...
   ✓ Backup created at: test_batch_logs/backups/batch_1
   ```

2. **Batch Finalized**
   ```
   ============================================================
   BATCH 1 COMPLETE - Running tests...
   ============================================================
   ```

3. **Comprehensive Test Suite Runs**
   ```
   ============================================================
   COMPREHENSIVE TEST SUITE EXECUTION
   ============================================================
   
   Discovered 30 test modules
   
   [1/30] Running test_circuit_breaker...
     ✓ PASSED - 15/15 tests passed in 0.45s
   
   [2/30] Running test_orchestration...
     ✗ FAILED - 12/15 tests passed in 1.23s
     ⚠ 3 failures, 0 errors
       • test_stage_execution
         Type: FAIL
         Location: test_orchestration.py:45
         Message: AssertionError: Expected 'completed' but got 'failed'
   ...
   ```

4. **Test Summary Generated**
   ```
   ============================================================
   TEST EXECUTION SUMMARY
   ============================================================
   Total modules: 30
   Total tests:   450
   Passed:        437
   Failed:        13
   Skipped:       0
   Duration:      45.23s
   Status:        ✗ FAILURES DETECTED
   ============================================================
   ```

5. **Results Analysis**
   - If failure rate < 10%: Warning logged, baseline not updated
   - If failure rate ≥ 10%: **Critical failure procedure activated**

### Phase 3: Critical Failure Handling

When critical failures are detected:

```
============================================================
⚠️  CRITICAL FAILURE DETECTED IN BATCH 1
============================================================

Failure rate: 12.5% (13/450 tests)

Most affected modules:
  • test_orchestration: 3/15 tests failed (20.0%)
  • test_extraction_pipeline: 5/20 tests failed (25.0%)
  • test_convergence: 2/10 tests failed (20.0%)

============================================================
RECOMMENDED ACTIONS:
============================================================
1. Review batch 1 edits in test_batch_logs/batch_tracker.json
2. Review detailed failures in test_batch_logs/test_report_batch_1.txt
3. Consider rollback: python batch_test_automation.py rollback 1
============================================================

✓ Critical failure report saved: test_batch_logs/CRITICAL_FAILURE_batch_1.json
```

### Phase 4: Investigation & Rollback

#### Review Batch Details:
```bash
python batch_test_automation.py analyze 1
```

Output:
```
============================================================
BATCH 1 ANALYSIS
============================================================

Period: 2024-01-15T10:30:00 to 2024-01-15T10:45:00
Edits: 5
Git commit: a3f7b2c9d1e4

Files modified:
1. orchestrator.py
   Time: 2024-01-15T10:31:23
   Hash: 7f3a2b1c4d5e6789
   Description: Refactored stage execution

2. extraction/pipeline.py
   Time: 2024-01-15T10:35:47
   Hash: 9e8d7c6b5a4f3210
   Description: Updated error handling
...

============================================================
TEST RESULTS
============================================================
Status: ✗ FAILED
Tests: 437/450 passed
Duration: 45.23s

Failures (13):
  • test_orchestration.test_stage_execution
  • test_extraction_pipeline.test_error_recovery
  ...
```

#### Review Detailed Failure Report:
```bash
cat test_batch_logs/test_report_batch_1.txt
```

#### Execute Rollback:
```bash
python batch_test_automation.py rollback 1
```

Interactive confirmation:
```
============================================================
⚠️  ROLLBACK CONFIRMATION REQUIRED
============================================================

This will revert all changes from batch 1

Files to be reverted (5):
  • orchestrator.py
  • extraction/pipeline.py
  • extraction/chunking.py
  • validators/convergence.py
  • infrastructure/retry.py

Proceed with rollback? (yes/no): yes

============================================================
ROLLING BACK BATCH 1
============================================================

✓ Restored orchestrator.py
✓ Restored extraction/pipeline.py
✓ Restored extraction/chunking.py
✓ Restored validators/convergence.py
✓ Restored infrastructure/retry.py

✓ Rollback of batch 1 complete

RECOMMENDED NEXT STEPS:
1. Re-run tests to verify system state
2. Review and fix the problematic changes
3. Re-apply changes incrementally with smaller batches
```

#### Verify System State:
```bash
python batch_test_automation.py test
```

## Tracking Files & Logs

All workflow data is stored in `test_batch_logs/`:

```
test_batch_logs/
├── batch_tracker.json          # Main tracking state
├── test_results.json           # All test execution results
├── ci_test_results.json        # Latest CI-format results
├── baseline_results.json       # Baseline for regression detection
├── regressions.json            # Detected regressions
├── test_report_batch_1.txt     # Detailed report for batch 1
├── test_report_batch_2.txt     # Detailed report for batch 2
├── CRITICAL_FAILURE_batch_1.json  # Critical failure analysis
└── backups/
    ├── batch_1/                # Backup of files before batch 1 tests
    │   ├── orchestrator.py
    │   ├── pipeline.py
    │   └── git_commit.txt
    └── batch_2/                # Backup of files before batch 2 tests
        └── ...
```

### batch_tracker.json Structure:
```json
{
  "edit_count": 10,
  "last_updated": "2024-01-15T11:00:00",
  "batches": [
    {
      "batch_id": 1,
      "start_time": "2024-01-15T10:30:00",
      "end_time": "2024-01-15T10:45:00",
      "git_commit_hash": "a3f7b2c9d1e4",
      "edits": [
        {
          "file_path": "orchestrator.py",
          "timestamp": "2024-01-15T10:31:23",
          "description": "Refactored stage execution",
          "file_hash": "7f3a2b1c4d5e6789"
        }
      ]
    }
  ]
}
```

### test_results.json Structure:
```json
{
  "last_updated": "2024-01-15T11:00:00",
  "results": [
    {
      "batch_id": 1,
      "timestamp": "2024-01-15T10:45:00",
      "success": false,
      "total_tests": 450,
      "passed_tests": 437,
      "failed_tests": 13,
      "skipped_tests": 0,
      "duration_seconds": 45.23,
      "git_commit_hash": "a3f7b2c9d1e4",
      "failures": [
        {
          "type": "FAIL",
          "test_name": "test_stage_execution",
          "test_class": "test_orchestration.TestOrchestrator",
          "traceback": "...",
          "module": "test_orchestration"
        }
      ]
    }
  ]
}
```

## Advanced Features

### 1. Continuous Watch Mode

Automatically track edits as files change:

```bash
python batch_test_automation.py watch . --recursive
```

Requires: `pip install watchdog`

Output:
```
============================================================
CONTINUOUS WATCH MODE
============================================================

Watching paths: .
Press Ctrl+C to stop

📝 Detected change: orchestrator.py

============================================================
RECORDING EDIT: orchestrator.py
============================================================

✓ Recorded edit #1 to orchestrator.py
  Batch 1: 1/5 edits
```

### 2. Custom Batch Sizes

Run smaller or larger batches:

```bash
# Test after every 3 edits
python batch_test_automation.py --batch-size 3 record myfile.py

# Test after every 10 edits
python batch_test_automation.py --batch-size 10 record myfile.py
```

### 3. Regression Detection

Compare with baseline to detect new failures:

```bash
# Set current results as baseline
python ci_batch_test_runner.py --set-baseline

# Run tests and compare with baseline
python ci_batch_test_runner.py --compare
```

Output:
```
============================================================
REGRESSION REPORT
============================================================

1. Module: test_orchestration
   Type: new_failures
   Failed tests increased: 0 → 3
   New failures: 3

2. Module: test_extraction_pipeline
   Type: performance_degradation
   Duration increased: 2.34s → 5.67s
   Slowdown: 2.42x
```

### 4. Manual Backup Creation

Create backups before risky changes:

```bash
python test_batch_tracker.py backup 1
```

### 5. Status Monitoring

Get comprehensive workflow status:

```bash
python batch_test_automation.py status
```

Output:
```
============================================================
BATCH TEST TRACKER STATUS
============================================================

Total edits: 23
Completed batches: 4
Test runs: 4

Current batch: 5
Edits in batch: 3/5

Last test run (Batch 4):
  Status: ✓ PASSED
  Tests: 448/450 passed

Log directory: test_batch_logs

============================================================
RECENT TEST HISTORY
============================================================

Batch 1: ✗ FAIL - 437/450 tests (45.2s)
Batch 2: ✓ PASS - 450/450 tests (43.8s)
Batch 3: ✓ PASS - 450/450 tests (44.1s)
Batch 4: ✓ PASS - 448/450 tests (46.3s)
```

## Integration with CI/CD

### GitHub Actions Example:

```yaml
name: Batch Test Validation

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run comprehensive test suite
        run: python ci_batch_test_runner.py --verbose
      
      - name: Compare with baseline
        run: python ci_batch_test_runner.py --compare
        continue-on-error: true
      
      - name: Upload test results
        uses: actions/upload-artifact@v2
        if: always()
        with:
          name: test-results
          path: test_batch_logs/
```

## Best Practices

### 1. Descriptive Edit Messages
Always provide meaningful descriptions:
```bash
python batch_test_automation.py record orchestrator.py --desc "Fixed stage retry logic for timeout scenarios"
```

### 2. Regular Status Checks
Check status before starting new batch:
```bash
python batch_test_automation.py status
```

### 3. Immediate Investigation
When tests fail, investigate immediately:
```bash
python batch_test_automation.py analyze <batch_id>
cat test_batch_logs/test_report_batch_<batch_id>.txt
```

### 4. Baseline Updates
Update baseline after successful major changes:
```bash
python ci_batch_test_runner.py --set-baseline
```

### 5. Pre-Commit Testing
Before committing to git, ensure current batch passes:
```bash
python batch_test_automation.py test
```

### 6. Incremental Rollback
If a batch has multiple unrelated changes, consider:
1. Rollback entire batch
2. Re-apply changes one-by-one with batch-size=1
3. Identify specific problematic edit

## Troubleshooting

### Issue: Tests not triggering after 5 edits

**Solution:** Check if current batch was finalized:
```bash
python batch_test_automation.py status
```

If batch shows incomplete, manually finalize:
```bash
python test_batch_tracker.py test
```

### Issue: Rollback doesn't restore files

**Solution:** Check if backup exists:
```bash
ls test_batch_logs/backups/batch_<id>/
```

Backups are created automatically before tests run. If no backup exists, the batch completed before backup feature was added.

### Issue: High memory usage during test runs

**Solution:** Run tests per module instead of all at once:
```bash
python -m unittest test_specific_module -v
```

### Issue: Git commit hash not captured

**Solution:** Ensure you're in a git repository:
```bash
git rev-parse HEAD
```

Git integration is optional and doesn't affect core functionality.

## Command Reference

### batch_test_automation.py
```
Main workflow orchestrator

Commands:
  record <file> [-d DESC]    Record edit and auto-test
  status                     Show workflow status
  test [-v]                  Run tests manually
  rollback <id> [-y]         Rollback batch
  watch <paths>              Continuous watch mode
  analyze <id>               Analyze batch details

Options:
  --batch-size N             Edits per batch (default: 5)
```

### test_batch_tracker.py
```
Low-level batch tracking

Commands:
  record <file> [-d DESC]    Record single edit
  test [-p PATTERN]          Run test suite
  status                     Show tracker status
  backup <id>                Create backup
  rollback <id>              Rollback batch
  reset                      Reset tracker state
```

### ci_batch_test_runner.py
```
CI-integrated test runner

Options:
  -v, --verbose              Verbose test output
  --set-baseline             Set current as baseline
  --compare                  Compare with baseline
  --baseline-file FILE       Custom baseline file
```

## Performance Considerations

- **Test Execution Time:** Full suite runs in ~45-60s with 30 modules
- **Backup Storage:** Each backup ~100KB-5MB depending on file sizes
- **Log Rotation:** Manually clean old reports periodically
- **Watch Mode:** Minimal CPU overhead (<1% on file changes)

## Exit Codes

- `0` - Success (all tests passed)
- `1` - Test failures detected
- `2` - Regressions detected (with --compare)

## Support & Extension

### Adding Custom Analyzers

Extend `AutomatedWorkflow._analyze_test_results()` to add custom analysis:

```python
def _analyze_test_results(self, test_results, batch_id):
    analysis = super()._analyze_test_results(test_results, batch_id)
    
    # Custom: Check for specific module patterns
    analysis['affected_critical_modules'] = [
        mod for mod in analysis['failed_modules']
        if 'orchestrator' in mod['module'] or 'circuit_breaker' in mod['module']
    ]
    
    return analysis
```

### Custom Failure Thresholds

Modify `AutomatedWorkflow.__init__()`:

```python
def __init__(self, batch_size=5):
    super().__init__(batch_size)
    self.critical_failure_threshold = 0.05  # 5% threshold
```

### Integration with External Tools

The JSON format is designed for easy integration:

```python
import json

# Read test results
with open('test_batch_logs/ci_test_results.json') as f:
    results = json.load(f)

# Send to external system
send_to_monitoring_system(results)
```

---

**Version:** 1.0  
**Last Updated:** 2024-01-15  
**Compatibility:** Python 3.11+, FARFAN 2.0 codebase
