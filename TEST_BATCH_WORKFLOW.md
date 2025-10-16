# Test Batch Workflow System

## Overview

A systematic unit test execution workflow that runs tests after every 5 code edits, captures results, isolates regressions, and provides rollback capabilities.

## Components

### 1. **test_batch_tracker.py** - Core Tracking System

Tracks code edits in batches of 5 and automatically triggers test execution.

**Features:**
- Records each file edit with timestamp, hash, and description
- Groups edits into batches (configurable, default: 5 edits)
- Automatically runs tests when a batch completes
- Creates detailed test reports with failure analysis
- Tracks which batch was active during each test run
- Provides backup and rollback capabilities

**Usage:**

```bash
# Record a file edit
python test_batch_tracker.py record <file_path> -d "Description of change"

# Run tests manually
python test_batch_tracker.py test

# Show current status
python test_batch_tracker.py status

# Create backup before applying changes
python test_batch_tracker.py backup <batch_id>

# Rollback a batch
python test_batch_tracker.py rollback <batch_id>

# Reset tracker
python test_batch_tracker.py reset
```

**Example Workflow:**

```bash
# Make edits and record them
python test_batch_tracker.py record orchestrator.py -d "Fixed stage 3 bug"
python test_batch_tracker.py record dnp_integration.py -d "Updated validation"
python test_batch_tracker.py record circuit_breaker.py -d "Enhanced error handling"
python test_batch_tracker.py record report_generator.py -d "Added new report section"
python test_batch_tracker.py record canonical_notation.py -d "Fixed parsing issue"
# ↑ Tests run automatically after 5th edit

# If tests fail, rollback the batch
python test_batch_tracker.py rollback 1
```

### 2. **ci_batch_test_runner.py** - Enhanced Test Execution

Provides comprehensive test execution with detailed failure analysis and regression detection.

**Features:**
- Runs all test modules with individual tracking
- Extracts detailed failure information (traceback, location, message)
- Generates CI-friendly JSON output
- Compares results with baseline to detect regressions
- Identifies performance degradations
- Module-level isolation of failures

**Usage:**

```bash
# Run all tests with detailed output
python ci_batch_test_runner.py -v

# Set current results as baseline
python ci_batch_test_runner.py --set-baseline

# Run tests and compare with baseline
python ci_batch_test_runner.py --compare -v

# Use custom baseline file
python ci_batch_test_runner.py --compare --baseline-file custom_baseline.json
```

**Exit Codes:**
- `0`: All tests passed
- `1`: Test failures detected
- `2`: Regressions detected (when using --compare)

### 3. **test_batch_integration.sh** - Unified CLI Interface

Bash script providing a convenient interface to all workflow operations.

**Features:**
- Single command interface for all operations
- Colored output for better readability
- Automatic test verification after rollback
- Watch mode for automatic change detection (requires `inotifywait` or `fswatch`)

**Usage:**

```bash
# Record an edit
./test_batch_integration.sh record orchestrator.py "Fixed stage 3"

# Run tests
./test_batch_integration.sh test

# Show status
./test_batch_integration.sh status

# Create backup
./test_batch_integration.sh backup 1

# Rollback with automatic test verification
./test_batch_integration.sh rollback 1

# Set baseline
./test_batch_integration.sh baseline

# Compare with baseline
./test_batch_integration.sh compare

# Watch mode (auto-record changes)
./test_batch_integration.sh watch
```

## Workflow Architecture

### Batch Lifecycle

```
Edit 1 → Edit 2 → Edit 3 → Edit 4 → Edit 5 → [TRIGGER TESTS]
│                                              │
├─ Record timestamp                            ├─ Finalize batch
├─ Calculate file hash                         ├─ Create backup point
├─ Store description                           ├─ Run full test suite
└─ Save to batch                               ├─ Generate detailed report
                                               ├─ Compare with baseline
                                               └─ Detect regressions
```

### Directory Structure

```
test_batch_logs/
├── batch_tracker.json        # Current tracking state
├── test_results.json         # All test results
├── ci_test_results.json      # Latest CI results
├── baseline_results.json     # Baseline for comparison
├── regressions.json          # Detected regressions
├── test_report_batch_N.txt   # Detailed report per batch
└── backups/
    └── batch_N/              # Backup files for batch N
        ├── file1.py
        ├── file2.py
        └── git_commit.txt
```

### Test Result Format

Each test result includes:

```json
{
  "batch_id": 1,
  "timestamp": "2024-01-15T10:30:00",
  "success": false,
  "total_tests": 150,
  "passed_tests": 145,
  "failed_tests": 5,
  "skipped_tests": 0,
  "duration_seconds": 45.2,
  "git_commit_hash": "abc123def456",
  "failures": [
    {
      "type": "FAIL",
      "test_name": "test_stage3_validation",
      "test_class": "test_orchestrator.TestOrchestrator",
      "module": "test_orchestrator",
      "traceback": "...",
      "file_location": "orchestrator.py:245"
    }
  ]
}
```

## Integration with Development Workflow

### Method 1: Manual Integration

```bash
# Before making changes, set baseline
./test_batch_integration.sh baseline

# Make your edits
vim orchestrator.py
./test_batch_integration.sh record orchestrator.py "Fixed stage 3 logic"

vim dnp_integration.py
./test_batch_integration.sh record dnp_integration.py "Updated DNP validation"

# ... continue until batch completes (5 edits)
# Tests run automatically

# If tests fail, check the report
cat test_batch_logs/test_report_batch_1.txt

# Rollback if needed
./test_batch_integration.sh rollback 1
```

### Method 2: Git Hook Integration

Create `.git/hooks/post-commit`:

```bash
#!/bin/bash
# Auto-record edits after commits

for file in $(git diff-tree --no-commit-id --name-only -r HEAD | grep '\.py$'); do
    if [ -f "$file" ]; then
        python test_batch_tracker.py record "$file" -d "$(git log -1 --pretty=%B)"
    fi
done
```

### Method 3: Watch Mode (Continuous)

```bash
# Start watch mode in a terminal
./test_batch_integration.sh watch

# In another terminal, make your changes
# Changes are automatically recorded and tests run after 5 edits
```

## Rollback Procedures

### Standard Rollback

```bash
# Rollback last batch
./test_batch_integration.sh rollback <batch_id>

# Tests are automatically re-run after rollback
```

### Emergency Rollback

```bash
# If tracker is broken, manual rollback from backup
cp test_batch_logs/backups/batch_N/* .

# Restore git state
git reset --hard $(cat test_batch_logs/backups/batch_N/git_commit.txt)

# Re-run tests
./test_batch_integration.sh test
```

## Regression Detection

The system detects two types of regressions:

1. **New Failures**: Tests that passed in baseline but fail now
2. **Performance Degradation**: Tests that run >50% slower than baseline

```bash
# Set baseline before starting fixes
./test_batch_integration.sh baseline

# Make changes...
# After each batch, compare
./test_batch_integration.sh compare

# If regressions detected, review report
cat test_batch_logs/regressions.json
```

## CI/CD Integration

### GitHub Actions Example

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
        run: |
          pip install -r requirements.txt
          python -m spacy download es_core_news_lg
      
      - name: Run batch test suite
        run: python ci_batch_test_runner.py -v
      
      - name: Compare with baseline
        run: python ci_batch_test_runner.py --compare
        continue-on-error: true
      
      - name: Upload test results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: test_batch_logs/
```

### Jenkins Integration

```groovy
pipeline {
    agent any
    
    stages {
        stage('Test') {
            steps {
                sh 'python ci_batch_test_runner.py -v'
            }
        }
        
        stage('Regression Check') {
            steps {
                sh 'python ci_batch_test_runner.py --compare'
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'test_batch_logs/**', fingerprint: true
            junit 'test_batch_logs/ci_test_results.json'
        }
        
        failure {
            script {
                def report = readFile('test_batch_logs/regressions.json')
                echo "Regressions detected: ${report}"
            }
        }
    }
}
```

## Advanced Usage

### Custom Batch Size

Edit `test_batch_tracker.py`:

```python
class BatchTestTracker:
    BATCH_SIZE = 10  # Change from 5 to 10
```

### Custom Test Patterns

```bash
# Run only specific tests
python test_batch_tracker.py test -p "test_orchestrator*.py"

# Run integration tests only
python test_batch_tracker.py test -p "test_*_integration.py"
```

### Programmatic Usage

```python
from test_batch_tracker import BatchTestTracker

tracker = BatchTestTracker()

# Record edits
tracker.record_edit("orchestrator.py", "Fixed bug")
tracker.record_edit("dnp_integration.py", "Updated validation")

# Manual test trigger
if tracker.edit_count >= 5:
    result = tracker.run_tests()
    if not result.success:
        print(f"Tests failed: {result.failed_tests} failures")
        tracker.rollback_batch(result.batch_id)

# Create backup before risky changes
tracker.create_backup(tracker.current_batch.batch_id)
```

## Best Practices

1. **Set Baseline Early**: Before starting a fix session, set a baseline
   ```bash
   ./test_batch_integration.sh baseline
   ```

2. **Descriptive Edit Messages**: Always provide meaningful descriptions
   ```bash
   ./test_batch_integration.sh record file.py "Fixed null pointer in stage 3 validation"
   ```

3. **Review Reports**: After each batch test run, review the detailed report
   ```bash
   cat test_batch_logs/test_report_batch_N.txt
   ```

4. **Immediate Rollback**: If tests fail, rollback immediately
   ```bash
   ./test_batch_integration.sh rollback <batch_id>
   ```

5. **Regular Backups**: Create backups before making complex changes
   ```bash
   ./test_batch_integration.sh backup <batch_id>
   ```

6. **Monitor Regressions**: Always run comparison after implementing fixes
   ```bash
   ./test_batch_integration.sh compare
   ```

## Troubleshooting

### Tests Not Running After 5 Edits

```bash
# Check tracker status
./test_batch_integration.sh status

# Manually trigger tests
./test_batch_integration.sh test
```

### Tracker State Corrupted

```bash
# Reset tracker
python test_batch_tracker.py reset

# Restore from git
git checkout test_batch_logs/
```

### Rollback Failed

```bash
# Manual file restoration
cp test_batch_logs/backups/batch_N/* .

# Reset git
git reset --hard <commit_hash>
```

### Performance Issues

```bash
# Run specific test modules instead of all
python -m unittest test_specific_module.py -v

# Increase timeout in ci_batch_test_runner.py
# Edit timeout parameter in subprocess.run()
```

## Summary

This test batch workflow provides:

✅ **Systematic Testing**: Automatic test execution every 5 edits  
✅ **Regression Isolation**: Pinpoint which batch introduced failures  
✅ **Detailed Reporting**: Stack traces, module identification, failure counts  
✅ **Easy Rollback**: One-command restoration of previous state  
✅ **Baseline Comparison**: Detect new failures and performance regressions  
✅ **CI Integration**: JSON output compatible with CI/CD systems  
✅ **Audit Trail**: Complete history of edits and test results  

Use this workflow to maintain code quality during large refactoring or bug-fixing sessions.
