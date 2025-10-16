#!/usr/bin/env bash
# Integrated test batch workflow script
# Monitors code changes and triggers tests automatically

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="test_batch_logs"
TRACKER_SCRIPT="test_batch_tracker.py"
CI_RUNNER="ci_batch_test_runner.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "Test Batch Integration Workflow"
echo "=================================="
echo ""

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    
    case $status in
        "success")
            echo -e "${GREEN}✓${NC} $message"
            ;;
        "error")
            echo -e "${RED}✗${NC} $message"
            ;;
        "warning")
            echo -e "${YELLOW}⚠${NC} $message"
            ;;
        *)
            echo "$message"
            ;;
    esac
}

# Function to record an edit
record_edit() {
    local file_path=$1
    local description=$2
    
    print_status "info" "Recording edit: $file_path"
    
    if python3 "$TRACKER_SCRIPT" record "$file_path" -d "$description"; then
        print_status "success" "Edit recorded successfully"
        return 0
    else
        # Check if tests were triggered and failed
        if [ $? -eq 1 ]; then
            print_status "error" "Tests failed after batch completion"
            return 1
        fi
        return 0
    fi
}

# Function to run tests manually
run_tests() {
    local pattern=${1:-"test_*.py"}
    
    print_status "info" "Running test suite with pattern: $pattern"
    
    if python3 "$CI_RUNNER" -v; then
        print_status "success" "All tests passed"
        return 0
    else
        print_status "error" "Tests failed"
        return 1
    fi
}

# Function to show status
show_status() {
    print_status "info" "Fetching tracker status..."
    python3 "$TRACKER_SCRIPT" status
}

# Function to create backup
create_backup() {
    local batch_id=$1
    
    if [ -z "$batch_id" ]; then
        print_status "error" "Batch ID required"
        return 1
    fi
    
    print_status "info" "Creating backup for batch $batch_id"
    
    if python3 "$TRACKER_SCRIPT" backup "$batch_id"; then
        print_status "success" "Backup created"
        return 0
    else
        print_status "error" "Backup failed"
        return 1
    fi
}

# Function to rollback
rollback_batch() {
    local batch_id=$1
    
    if [ -z "$batch_id" ]; then
        print_status "error" "Batch ID required"
        return 1
    fi
    
    print_status "warning" "Rolling back batch $batch_id"
    
    if python3 "$TRACKER_SCRIPT" rollback "$batch_id"; then
        print_status "success" "Rollback completed"
        
        # Re-run tests after rollback
        print_status "info" "Verifying tests after rollback..."
        run_tests
        return $?
    else
        print_status "error" "Rollback failed"
        return 1
    fi
}

# Function to set baseline
set_baseline() {
    print_status "info" "Setting baseline from current test results"
    
    if python3 "$CI_RUNNER" --set-baseline; then
        print_status "success" "Baseline set"
        return 0
    else
        print_status "error" "Failed to set baseline"
        return 1
    fi
}

# Function to compare with baseline
compare_baseline() {
    print_status "info" "Comparing current results with baseline"
    
    if python3 "$CI_RUNNER" --compare -v; then
        print_status "success" "No regressions detected"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 2 ]; then
            print_status "warning" "Regressions detected"
        else
            print_status "error" "Tests failed"
        fi
        return $exit_code
    fi
}

# Function to monitor directory for changes
watch_mode() {
    print_status "info" "Starting watch mode (monitoring for file changes)"
    print_status "info" "This requires 'inotifywait' or 'fswatch' to be installed"
    
    # Check if monitoring tools are available
    if command -v inotifywait &> /dev/null; then
        print_status "success" "Using inotifywait for file monitoring"
        
        inotifywait -m -r -e modify -e create --format '%w%f' . | while read file
        do
            # Only track .py files
            if [[ "$file" == *.py ]] && [[ "$file" != *"venv"* ]] && [[ "$file" != *".git"* ]]; then
                print_status "info" "Detected change: $file"
                record_edit "$file" "Auto-detected change"
            fi
        done
    elif command -v fswatch &> /dev/null; then
        print_status "success" "Using fswatch for file monitoring"
        
        fswatch -r -e "venv" -e ".git" --include "\.py$" . | while read file
        do
            print_status "info" "Detected change: $file"
            record_edit "$file" "Auto-detected change"
        done
    else
        print_status "error" "No file monitoring tool found"
        print_status "info" "Install 'inotifywait' (Linux) or 'fswatch' (macOS)"
        return 1
    fi
}

# Main command handler
case "${1:-help}" in
    "record")
        if [ -z "$2" ]; then
            print_status "error" "Usage: $0 record <file_path> [description]"
            exit 1
        fi
        record_edit "$2" "${3:-}"
        ;;
    
    "test")
        run_tests "${2:-test_*.py}"
        ;;
    
    "status")
        show_status
        ;;
    
    "backup")
        create_backup "$2"
        ;;
    
    "rollback")
        rollback_batch "$2"
        ;;
    
    "baseline")
        set_baseline
        ;;
    
    "compare")
        compare_baseline
        ;;
    
    "watch")
        watch_mode
        ;;
    
    "help"|*)
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  record <file> [desc]  - Record a file edit"
        echo "  test [pattern]        - Run test suite"
        echo "  status                - Show tracker status"
        echo "  backup <batch_id>     - Create backup of batch"
        echo "  rollback <batch_id>   - Rollback a batch"
        echo "  baseline              - Set current results as baseline"
        echo "  compare               - Compare with baseline"
        echo "  watch                 - Watch mode (auto-record changes)"
        echo "  help                  - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 record orchestrator.py 'Fixed bug in stage 3'"
        echo "  $0 test"
        echo "  $0 rollback 5"
        echo "  $0 baseline"
        echo "  $0 compare"
        ;;
esac

exit $?
