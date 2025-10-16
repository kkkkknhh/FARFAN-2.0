#!/usr/bin/env python3
"""
Demonstration of the Batch Test Workflow System

This script demonstrates the complete workflow:
1. Recording edits in batches
2. Automatic test triggering after 5 edits
3. Regression isolation
4. Rollback procedures
"""

import sys
import time
from pathlib import Path

from batch_test_automation import AutomatedWorkflow


def demo_basic_workflow():
    """Demonstrate basic batch workflow"""
    print("=" * 80)
    print("DEMO: BASIC BATCH WORKFLOW")
    print("=" * 80)
    print()

    workflow = AutomatedWorkflow(batch_size=3)  # Smaller batch for demo

    # Create demo files
    demo_dir = Path("demo_workflow_files")
    demo_dir.mkdir(exist_ok=True)

    demo_files = []
    for i in range(6):
        demo_file = demo_dir / f"module_{i}.py"
        demo_file.write_text(f"# Module {i}\ndef function_{i}():\n    return {i}\n")
        demo_files.append(str(demo_file))

    print("📝 Scenario: Implementing 6 sequential code fixes\n")

    # First batch (3 edits)
    print("--- Batch 1: Edits 1-3 ---\n")
    for i in range(3):
        print(f"Making edit {i + 1}...\n")
        workflow.record_and_test(
            demo_files[i], f"Refactored function_{i} for better performance"
        )
        time.sleep(0.5)
        print()

    print("\n✓ First batch complete - tests should have run")
    print()

    # Second batch (3 edits)
    print("--- Batch 2: Edits 4-6 ---\n")
    for i in range(3, 6):
        print(f"Making edit {i + 1}...\n")
        workflow.record_and_test(demo_files[i], f"Added error handling to function_{i}")
        time.sleep(0.5)
        print()

    print("\n✓ Second batch complete - tests should have run")
    print()

    # Show status
    print("\n" + "=" * 80)
    print("FINAL STATUS")
    print("=" * 80)
    workflow.get_status()

    # Cleanup
    import shutil

    shutil.rmtree(demo_dir)


def demo_rollback_scenario():
    """Demonstrate rollback after critical failure"""
    print("\n\n")
    print("=" * 80)
    print("DEMO: ROLLBACK SCENARIO")
    print("=" * 80)
    print()

    workflow = AutomatedWorkflow(batch_size=3)

    demo_dir = Path("demo_rollback_files")
    demo_dir.mkdir(exist_ok=True)

    demo_files = []
    for i in range(3):
        demo_file = demo_dir / f"critical_module_{i}.py"
        demo_file.write_text(f"# Critical Module {i}\nSTATUS = 'stable'\n")
        demo_files.append(str(demo_file))

    print("📝 Scenario: Batch introduces critical bugs\n")

    # Make batch of edits
    print("Making 3 edits that will introduce bugs...\n")
    for i in range(3):
        # Introduce "bugs" by changing file content
        Path(demo_files[i]).write_text(f"# Critical Module {i}\nSTATUS = 'broken'\n")
        workflow.record_and_test(
            demo_files[i], f"Updated critical_module_{i} - CAUTION"
        )

    print("\n⚠️  Tests would show failures here in real scenario")
    print()

    # Demonstrate rollback
    print("--- Performing Rollback ---\n")
    batch_id = len(workflow.tracker.all_batches)

    # Check if backup exists
    backup_dir = workflow.tracker.BACKUP_DIR / f"batch_{batch_id}"
    if backup_dir.exists():
        print(f"✓ Backup found for batch {batch_id}")
        print(f"  Location: {backup_dir}")
        print()

        # Simulate rollback (would be: workflow.rollback(batch_id, confirm=True))
        print(f"To rollback: python batch_test_automation.py rollback {batch_id}")
    else:
        print(f"⚠️  No backup available for batch {batch_id}")

    # Cleanup
    import shutil

    shutil.rmtree(demo_dir)


def demo_status_monitoring():
    """Demonstrate status monitoring"""
    print("\n\n")
    print("=" * 80)
    print("DEMO: STATUS MONITORING")
    print("=" * 80)
    print()

    workflow = AutomatedWorkflow(batch_size=5)

    print("Current workflow status:\n")
    workflow.get_status()

    print("\n\nDetailed batch information:")
    print(f"  Tracking file: {workflow.tracker.TRACKER_FILE}")
    print(f"  Results file: {workflow.tracker.RESULTS_FILE}")
    print(f"  Backup directory: {workflow.tracker.BACKUP_DIR}")

    if workflow.tracker.TRACKER_FILE.exists():
        import json

        with open(workflow.tracker.TRACKER_FILE) as f:
            data = json.load(f)
        print(f"\n  Total batches tracked: {len(data.get('batches', []))}")
        print(f"  Total edits recorded: {data.get('edit_count', 0)}")


def demo_cli_commands():
    """Show available CLI commands"""
    print("\n\n")
    print("=" * 80)
    print("DEMO: CLI COMMAND REFERENCE")
    print("=" * 80)
    print()

    commands = [
        (
            "Record an edit",
            "python batch_test_automation.py record myfile.py --desc 'Fixed bug'",
        ),
        ("Check status", "python batch_test_automation.py status"),
        ("Run tests manually", "python batch_test_automation.py test"),
        ("Analyze batch", "python batch_test_automation.py analyze 1"),
        ("Rollback batch", "python batch_test_automation.py rollback 1"),
        ("Watch directory", "python batch_test_automation.py watch . --recursive"),
        (
            "Custom batch size",
            "python batch_test_automation.py --batch-size 10 record file.py",
        ),
    ]

    for description, command in commands:
        print(f"• {description}:")
        print(f"  $ {command}")
        print()

    print("\nFor comprehensive guide, see: BATCH_TEST_WORKFLOW_GUIDE.md")


def main():
    """Run all demonstrations"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "BATCH TEST WORKFLOW SYSTEM DEMO" + " " * 27 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    print("This demo shows the complete workflow for systematic testing")
    print("after every N code edits with regression isolation and rollback.")
    print()

    demos = [
        ("Basic Workflow", demo_basic_workflow),
        ("Rollback Scenario", demo_rollback_scenario),
        ("Status Monitoring", demo_status_monitoring),
        ("CLI Commands", demo_cli_commands),
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n⚠️  Demo '{name}' encountered error: {e}")
            import traceback

            traceback.print_exc()

    print("\n\n")
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Read the comprehensive guide: BATCH_TEST_WORKFLOW_GUIDE.md")
    print("  2. Run tests: python -m unittest test_batch_workflow_system")
    print("  3. Try it yourself: python batch_test_automation.py record <file>")
    print()


if __name__ == "__main__":
    main()
