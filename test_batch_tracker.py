#!/usr/bin/env python3
"""
Systematic Unit Test Execution Workflow
Tracks code edits in batches and runs tests after every 5 edits.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FileEdit:
    """Represents a single file edit"""

    file_path: str
    timestamp: str
    description: str
    file_hash: str


@dataclass
class EditBatch:
    """Represents a batch of edits"""

    batch_id: int
    edits: List[FileEdit] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    git_commit_hash: Optional[str] = None

    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.now().isoformat()


@dataclass
class TestResult:
    """Represents test execution results"""

    batch_id: int
    timestamp: str
    success: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    duration_seconds: float
    failures: List[Dict[str, Any]] = field(default_factory=list)
    output: str = ""
    git_commit_hash: Optional[str] = None


class BatchTestTracker:
    """Tracks code edits and orchestrates batch testing"""

    BATCH_SIZE = 5
    LOG_DIR = Path("test_batch_logs")
    TRACKER_FILE = LOG_DIR / "batch_tracker.json"
    RESULTS_FILE = LOG_DIR / "test_results.json"
    BACKUP_DIR = LOG_DIR / "backups"

    def __init__(self):
        self.LOG_DIR.mkdir(exist_ok=True)
        self.BACKUP_DIR.mkdir(exist_ok=True)
        self.current_batch: Optional[EditBatch] = None
        self.edit_count = 0
        self.all_batches: List[EditBatch] = []
        self.test_results: List[TestResult] = []
        self._load_state()

    def _load_state(self):
        """Load existing tracker state"""
        if self.TRACKER_FILE.exists():
            try:
                with open(self.TRACKER_FILE, "r") as f:
                    data = json.load(f)
                    self.edit_count = data.get("edit_count", 0)
                    batches = data.get("batches", [])
                    self.all_batches = [
                        EditBatch(
                            batch_id=b["batch_id"],
                            edits=[FileEdit(**e) for e in b.get("edits", [])],
                            start_time=b.get("start_time", ""),
                            end_time=b.get("end_time", ""),
                            git_commit_hash=b.get("git_commit_hash"),
                        )
                        for b in batches
                    ]
                    # Resume current batch if incomplete
                    if batches and not batches[-1].get("end_time"):
                        self.current_batch = self.all_batches[-1]
            except Exception as e:
                print(f"Warning: Could not load tracker state: {e}")

        if self.RESULTS_FILE.exists():
            try:
                with open(self.RESULTS_FILE, "r") as f:
                    data = json.load(f)
                    self.test_results = [
                        TestResult(**r) for r in data.get("results", [])
                    ]
            except Exception as e:
                print(f"Warning: Could not load test results: {e}")

    def _save_state(self):
        """Persist tracker state"""
        data = {
            "edit_count": self.edit_count,
            "last_updated": datetime.now().isoformat(),
            "batches": [
                {
                    "batch_id": b.batch_id,
                    "edits": [asdict(e) for e in b.edits],
                    "start_time": b.start_time,
                    "end_time": b.end_time,
                    "git_commit_hash": b.git_commit_hash,
                }
                for b in self.all_batches
            ],
        }
        with open(self.TRACKER_FILE, "w") as f:
            json.dump(data, f, indent=2)

        results_data = {
            "last_updated": datetime.now().isoformat(),
            "results": [asdict(r) for r in self.test_results],
        }
        with open(self.RESULTS_FILE, "w") as f:
            json.dump(results_data, f, indent=2)

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate file content hash"""
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except:
            return "unknown"

    def _get_git_commit_hash(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except:
            pass
        return None

    def record_edit(self, file_path: str, description: str = ""):
        """Record a file edit"""
        if not self.current_batch:
            batch_id = len(self.all_batches) + 1
            self.current_batch = EditBatch(batch_id=batch_id)
            self.all_batches.append(self.current_batch)

        edit = FileEdit(
            file_path=file_path,
            timestamp=datetime.now().isoformat(),
            description=description,
            file_hash=self._get_file_hash(file_path),
        )
        self.current_batch.edits.append(edit)
        self.edit_count += 1

        print(f"✓ Recorded edit #{self.edit_count} to {file_path}")
        print(
            f"  Batch {self.current_batch.batch_id}: {len(self.current_batch.edits)}/{self.BATCH_SIZE} edits"
        )

        self._save_state()

        # Check if batch is complete
        if len(self.current_batch.edits) >= self.BATCH_SIZE:
            self._finalize_batch()
            return True  # Signal to run tests

        return False

    def _finalize_batch(self):
        """Finalize current batch"""
        if self.current_batch:
            self.current_batch.end_time = datetime.now().isoformat()
            self.current_batch.git_commit_hash = self._get_git_commit_hash()
            print(f"\n{'=' * 60}")
            print(f"BATCH {self.current_batch.batch_id} COMPLETE - Running tests...")
            print(f"{'=' * 60}\n")
            self.current_batch = None
            self._save_state()

    def run_tests(self, test_pattern: str = "test_*.py") -> TestResult:
        """Execute test suite and capture results"""
        batch_id = len(self.all_batches)
        start_time = time.time()

        print(f"Running test suite for Batch {batch_id}...")
        print(f"Test pattern: {test_pattern}\n")

        # Run unittest discovery with detailed output
        cmd = [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            ".",
            "-p",
            test_pattern,
            "-v",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        duration = time.time() - start_time
        output = result.stdout + "\n" + result.stderr

        # Parse test results
        success = result.returncode == 0
        total_tests, passed_tests, failed_tests, skipped_tests = (
            self._parse_unittest_output(output)
        )
        failures = self._extract_failures(output)

        test_result = TestResult(
            batch_id=batch_id,
            timestamp=datetime.now().isoformat(),
            success=success,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            duration_seconds=duration,
            failures=failures,
            output=output,
            git_commit_hash=self._get_git_commit_hash(),
        )

        self.test_results.append(test_result)
        self._save_state()
        self._generate_test_report(test_result)

        return test_result

    def _parse_unittest_output(self, output: str) -> tuple:
        """Parse unittest output for statistics"""
        import re

        # Look for unittest summary line
        ran_match = re.search(r"Ran (\d+) tests? in", output)
        total_tests = int(ran_match.group(1)) if ran_match else 0

        # Count failures and errors
        failures = len(re.findall(r"^FAIL:", output, re.MULTILINE))
        errors = len(re.findall(r"^ERROR:", output, re.MULTILINE))
        skipped = len(re.findall(r"^SKIP:", output, re.MULTILINE))

        failed_tests = failures + errors
        passed_tests = total_tests - failed_tests - skipped

        return total_tests, passed_tests, failed_tests, skipped

    def _extract_failures(self, output: str) -> List[Dict[str, Any]]:
        """Extract detailed failure information"""
        import re

        failures = []

        # Pattern to match FAIL/ERROR blocks
        pattern = r"(FAIL|ERROR): (test_\w+) \((\S+)\)\n(.*?)(?=\n(?:FAIL|ERROR|Ran \d+ tests?|\Z))"

        for match in re.finditer(pattern, output, re.DOTALL):
            failure_type, test_name, test_class, traceback = match.groups()

            failures.append(
                {
                    "type": failure_type,
                    "test_name": test_name,
                    "test_class": test_class,
                    "traceback": traceback.strip(),
                    "module": (
                        test_class.split(".")[0] if "." in test_class else test_class
                    ),
                }
            )

        return failures

    def _generate_test_report(self, result: TestResult):
        """Generate detailed test report"""
        report_file = self.LOG_DIR / f"test_report_batch_{result.batch_id}.txt"

        with open(report_file, "w") as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"TEST EXECUTION REPORT - BATCH {result.batch_id}\n")
            f.write(f"{'=' * 80}\n\n")

            f.write(f"Timestamp: {result.timestamp}\n")
            f.write(f"Git Commit: {result.git_commit_hash or 'N/A'}\n")
            f.write(f"Duration: {result.duration_seconds:.2f}s\n")
            f.write(f"Status: {'✓ PASSED' if result.success else '✗ FAILED'}\n\n")

            f.write(f"Test Summary:\n")
            f.write(f"  Total:   {result.total_tests}\n")
            f.write(
                f"  Passed:  {result.passed_tests} ({result.passed_tests / max(result.total_tests, 1) * 100:.1f}%)\n"
            )
            f.write(f"  Failed:  {result.failed_tests}\n")
            f.write(f"  Skipped: {result.skipped_tests}\n\n")

            if result.failures:
                f.write(f"{'=' * 80}\n")
                f.write(f"FAILURES ({len(result.failures)})\n")
                f.write(f"{'=' * 80}\n\n")

                for i, failure in enumerate(result.failures, 1):
                    f.write(
                        f"{i}. {failure['type']}: {failure['test_class']}.{failure['test_name']}\n"
                    )
                    f.write(f"   Module: {failure['module']}\n")
                    f.write(f"   Traceback:\n")
                    for line in failure["traceback"].split("\n"):
                        f.write(f"     {line}\n")
                    f.write("\n")

            f.write(f"\n{'=' * 80}\n")
            f.write(f"BATCH {result.batch_id} EDITS\n")
            f.write(f"{'=' * 80}\n\n")

            if result.batch_id <= len(self.all_batches):
                batch = self.all_batches[result.batch_id - 1]
                for i, edit in enumerate(batch.edits, 1):
                    f.write(f"{i}. {edit.file_path}\n")
                    f.write(f"   Time: {edit.timestamp}\n")
                    f.write(f"   Hash: {edit.file_hash}\n")
                    if edit.description:
                        f.write(f"   Description: {edit.description}\n")
                    f.write("\n")

            f.write(f"\n{'=' * 80}\n")
            f.write(f"FULL TEST OUTPUT\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(result.output)

        print(f"\n✓ Detailed report saved to: {report_file}")

    def create_backup(self, batch_id: int):
        """Create backup of files before batch application"""
        backup_dir = self.BACKUP_DIR / f"batch_{batch_id}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        if batch_id <= len(self.all_batches):
            batch = self.all_batches[batch_id - 1]

            for edit in batch.edits:
                if os.path.exists(edit.file_path):
                    backup_path = backup_dir / Path(edit.file_path).name
                    try:
                        with open(edit.file_path, "rb") as src:
                            with open(backup_path, "wb") as dst:
                                dst.write(src.read())
                    except Exception as e:
                        print(f"Warning: Could not backup {edit.file_path}: {e}")

            # Save git state
            git_hash = self._get_git_commit_hash()
            if git_hash:
                with open(backup_dir / "git_commit.txt", "w") as f:
                    f.write(git_hash)

            print(f"✓ Backup created at: {backup_dir}")
            return backup_dir

        return None

    def rollback_batch(self, batch_id: int):
        """Rollback to state before specified batch"""
        backup_dir = self.BACKUP_DIR / f"batch_{batch_id}"

        if not backup_dir.exists():
            print(f"✗ No backup found for batch {batch_id}")
            return False

        print(f"\n{'=' * 60}")
        print(f"ROLLING BACK BATCH {batch_id}")
        print(f"{'=' * 60}\n")

        if batch_id <= len(self.all_batches):
            batch = self.all_batches[batch_id - 1]

            for edit in batch.edits:
                backup_path = backup_dir / Path(edit.file_path).name
                if backup_path.exists():
                    try:
                        with open(backup_path, "rb") as src:
                            with open(edit.file_path, "wb") as dst:
                                dst.write(src.read())
                        print(f"✓ Restored {edit.file_path}")
                    except Exception as e:
                        print(f"✗ Failed to restore {edit.file_path}: {e}")

            print(f"\n✓ Rollback of batch {batch_id} complete")
            return True

        return False

    def print_status(self):
        """Print current tracking status"""
        print(f"\n{'=' * 60}")
        print(f"BATCH TEST TRACKER STATUS")
        print(f"{'=' * 60}\n")

        print(f"Total edits: {self.edit_count}")
        print(f"Completed batches: {len([b for b in self.all_batches if b.end_time])}")
        print(f"Test runs: {len(self.test_results)}")

        if self.current_batch:
            print(f"\nCurrent batch: {self.current_batch.batch_id}")
            print(f"Edits in batch: {len(self.current_batch.edits)}/{self.BATCH_SIZE}")

        if self.test_results:
            last_result = self.test_results[-1]
            print(f"\nLast test run (Batch {last_result.batch_id}):")
            print(f"  Status: {'✓ PASSED' if last_result.success else '✗ FAILED'}")
            print(
                f"  Tests: {last_result.passed_tests}/{last_result.total_tests} passed"
            )

            if not last_result.success:
                print(f"  Failures: {last_result.failed_tests}")
                if last_result.failures:
                    print(
                        f"  Failed modules: {', '.join(set(f['module'] for f in last_result.failures))}"
                    )

        print(f"\nLog directory: {self.LOG_DIR}")


def main():
    """CLI interface for batch test tracker"""
    import argparse

    parser = argparse.ArgumentParser(description="Systematic Unit Test Batch Tracker")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Record edit command
    record_parser = subparsers.add_parser("record", help="Record a file edit")
    record_parser.add_argument("file_path", help="Path to edited file")
    record_parser.add_argument(
        "-d", "--description", default="", help="Edit description"
    )

    # Run tests command
    test_parser = subparsers.add_parser("test", help="Run test suite")
    test_parser.add_argument(
        "-p", "--pattern", default="test_*.py", help="Test file pattern"
    )

    # Status command
    subparsers.add_parser("status", help="Show tracker status")

    # Backup command
    backup_parser = subparsers.add_parser(
        "backup", help="Create backup of current batch"
    )
    backup_parser.add_argument("batch_id", type=int, help="Batch ID to backup")

    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback a batch")
    rollback_parser.add_argument("batch_id", type=int, help="Batch ID to rollback")

    # Reset command
    subparsers.add_parser("reset", help="Reset tracker state")

    args = parser.parse_args()
    tracker = BatchTestTracker()

    if args.command == "record":
        should_test = tracker.record_edit(args.file_path, args.description)
        if should_test:
            result = tracker.run_tests()
            sys.exit(0 if result.success else 1)

    elif args.command == "test":
        result = tracker.run_tests(args.pattern)
        sys.exit(0 if result.success else 1)

    elif args.command == "status":
        tracker.print_status()

    elif args.command == "backup":
        tracker.create_backup(args.batch_id)

    elif args.command == "rollback":
        success = tracker.rollback_batch(args.batch_id)
        sys.exit(0 if success else 1)

    elif args.command == "reset":
        if input("Reset all tracking data? (yes/no): ").lower() == "yes":
            tracker.LOG_DIR.rmdir() if tracker.LOG_DIR.exists() else None
            print("✓ Tracker reset")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
