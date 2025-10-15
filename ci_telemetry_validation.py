#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CI Contract Enforcement for Telemetry and Auditability
=======================================================

Validates that all orchestrator phases emit complete telemetry and maintain
immutable audit trails per SIN_CARRETA doctrine.

Exit Codes:
- 0: All checks passed
- 1: Telemetry incomplete or audit logs not reproducible
- 2: Contract violations detected

Usage:
    python ci_telemetry_validation.py
    
This script is intended to be run in CI/CD pipelines to enforce auditability.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from orchestrator import create_orchestrator
from infrastructure.telemetry import EventType


class TelemetryValidator:
    """Validates telemetry completeness and audit trail integrity"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_telemetry_completeness(self, orchestrator) -> bool:
        """
        Validate that all phases emitted complete telemetry.
        
        Returns:
            True if complete, False otherwise
        """
        expected_phases = [
            "extract_statements",
            "detect_contradictions",
            "analyze_regulatory_constraints",
            "calculate_coherence_metrics",
            "generate_audit_summary",
            "orchestration_pipeline"
        ]
        
        verification = orchestrator.telemetry.verify_all_phases(expected_phases)
        
        if not verification["all_complete"]:
            self.errors.append(
                f"Telemetry incomplete: {verification['complete_phases']}/{verification['total_phases']} phases complete"
            )
            
            for phase_result in verification["phases"]:
                if not phase_result["complete"]:
                    self.errors.append(
                        f"  Phase '{phase_result['phase_name']}' missing events: {phase_result['missing_events']}"
                    )
            
            return False
        
        return True
    
    def validate_phase_boundaries(self, orchestrator) -> bool:
        """
        Validate that all phases have start and completion events.
        
        Returns:
            True if valid, False otherwise
        """
        phases = [
            "extract_statements",
            "detect_contradictions",
            "analyze_regulatory_constraints",
            "calculate_coherence_metrics",
            "generate_audit_summary"
        ]
        
        for phase in phases:
            events = orchestrator.telemetry.get_events(phase_name=phase)
            
            has_start = any(e.event_type == EventType.PHASE_START for e in events)
            has_completion = any(e.event_type == EventType.PHASE_COMPLETION for e in events)
            
            if not has_start:
                self.errors.append(f"Phase '{phase}' missing PHASE_START event")
            
            if not has_completion:
                self.errors.append(f"Phase '{phase}' missing PHASE_COMPLETION event")
        
        return len(self.errors) == 0
    
    def validate_trace_context_consistency(self, orchestrator) -> bool:
        """
        Validate that trace context is consistent across all events.
        
        Returns:
            True if consistent, False otherwise
        """
        events = orchestrator.telemetry.get_events()
        
        if len(events) == 0:
            self.errors.append("No telemetry events found")
            return False
        
        # All events should have the same trace_id
        trace_ids = set(e.trace_context.trace_id for e in events)
        
        if len(trace_ids) > 1:
            self.errors.append(
                f"Multiple trace IDs detected: {len(trace_ids)} (expected 1)"
            )
            return False
        
        # All events should have the same audit_id
        audit_ids = set(e.trace_context.audit_id for e in events)
        
        if len(audit_ids) > 1:
            self.errors.append(
                f"Multiple audit IDs detected: {len(audit_ids)} (expected 1)"
            )
            return False
        
        return True
    
    def validate_deterministic_hashing(self) -> bool:
        """
        Validate that orchestrator produces deterministic hashes.
        
        Returns:
            True if deterministic, False otherwise
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            text = "Plan de desarrollo municipal para validación CI."
            
            # Run twice with same input
            orch1 = create_orchestrator(log_dir=Path(tmpdir) / "run1")
            orch2 = create_orchestrator(log_dir=Path(tmpdir) / "run2")
            
            result1 = orch1.orchestrate_analysis(text, "PDM_CI_Test", "estratégico")
            result2 = orch2.orchestrate_analysis(text, "PDM_CI_Test", "estratégico")
            
            # Compare phase result hashes
            phase_results1 = orch1._audit_log
            phase_results2 = orch2._audit_log
            
            if len(phase_results1) != len(phase_results2):
                self.errors.append(
                    f"Phase count mismatch: {len(phase_results1)} vs {len(phase_results2)}"
                )
                return False
            
            for r1, r2 in zip(phase_results1, phase_results2):
                if r1.input_hash != r2.input_hash:
                    self.errors.append(
                        f"Non-deterministic input hash for phase '{r1.phase_name}': "
                        f"{r1.input_hash} != {r2.input_hash}"
                    )
                
                if r1.output_hash != r2.output_hash:
                    self.errors.append(
                        f"Non-deterministic output hash for phase '{r1.phase_name}': "
                        f"{r1.output_hash} != {r2.output_hash}"
                    )
        
        return len(self.errors) == 0
    
    def validate_audit_log_immutability(self, orchestrator, log_dir: Path) -> bool:
        """
        Validate that audit logs are immutable (append-only JSONL).
        
        Returns:
            True if immutable, False otherwise
        """
        audit_files = list(log_dir.glob("audit_logs.jsonl"))
        
        if len(audit_files) == 0:
            self.errors.append("No audit log files found")
            return False
        
        # Verify JSONL format (append-only)
        for audit_file in audit_files:
            try:
                with open(audit_file, 'r') as f:
                    lines = f.readlines()
                    
                    if len(lines) == 0:
                        self.warnings.append(f"Empty audit log: {audit_file}")
                        continue
                    
                    # Each line should be valid JSON
                    for i, line in enumerate(lines, 1):
                        try:
                            record = json.loads(line)
                            
                            # Verify required fields
                            required_fields = ["run_id", "orchestrator", "timestamp", "sha256_source"]
                            for field in required_fields:
                                if field not in record:
                                    self.errors.append(
                                        f"Audit log line {i} missing required field '{field}'"
                                    )
                        except json.JSONDecodeError as e:
                            self.errors.append(
                                f"Invalid JSON in audit log line {i}: {e}"
                            )
            except Exception as e:
                self.errors.append(f"Failed to read audit log {audit_file}: {e}")
                return False
        
        return len(self.errors) == 0
    
    def validate_telemetry_persistence(self, log_dir: Path) -> bool:
        """
        Validate that telemetry events are persisted to disk.
        
        Returns:
            True if persisted, False otherwise
        """
        telemetry_files = list(log_dir.glob("telemetry_*.jsonl"))
        
        if len(telemetry_files) == 0:
            self.errors.append("No telemetry files found")
            return False
        
        # Verify JSONL format
        for telemetry_file in telemetry_files:
            try:
                with open(telemetry_file, 'r') as f:
                    lines = f.readlines()
                    
                    if len(lines) == 0:
                        self.warnings.append(f"Empty telemetry file: {telemetry_file}")
                        continue
                    
                    # Each line should be valid JSON
                    for i, line in enumerate(lines, 1):
                        try:
                            event = json.loads(line)
                            
                            # Verify required fields
                            required_fields = ["event_type", "phase_name", "trace_context", "timestamp"]
                            for field in required_fields:
                                if field not in event:
                                    self.errors.append(
                                        f"Telemetry event line {i} missing required field '{field}'"
                                    )
                        except json.JSONDecodeError as e:
                            self.errors.append(
                                f"Invalid JSON in telemetry line {i}: {e}"
                            )
            except Exception as e:
                self.errors.append(f"Failed to read telemetry file {telemetry_file}: {e}")
                return False
        
        return len(self.errors) == 0
    
    def validate_contract_enforcement(self, orchestrator) -> bool:
        """
        Validate that all PhaseResults passed contract validation.
        
        Returns:
            True if all contracts valid, False otherwise
        """
        for phase_result in orchestrator._audit_log:
            try:
                phase_result.validate_contract()
            except Exception as e:
                self.errors.append(
                    f"Contract violation in phase '{phase_result.phase_name}': {e}"
                )
        
        return len(self.errors) == 0
    
    def print_results(self) -> None:
        """Print validation results"""
        print("\n" + "=" * 70)
        print("CI TELEMETRY AND AUDITABILITY VALIDATION")
        print("=" * 70)
        
        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")
        
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  ❌ {error}")
            print("\n" + "=" * 70)
            print("VALIDATION FAILED")
            print("=" * 70)
        else:
            print("\n✅ All validation checks PASSED")
            print("=" * 70)


def main() -> int:
    """
    Run all telemetry and auditability validation checks.
    
    Returns:
        Exit code (0 = success, 1 = validation failed)
    """
    validator = TelemetryValidator()
    
    print("Running CI telemetry and auditability validation...")
    print("This may take a few seconds...\n")
    
    # Create temporary directory for test runs
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        
        # Create orchestrator and run analysis
        orch = create_orchestrator(log_dir=log_dir)
        
        try:
            result = orch.orchestrate_analysis(
                text="Plan de desarrollo municipal para validación completa de auditabilidad.",
                plan_name="PDM_CI_Validation",
                dimension="estratégico"
            )
        except Exception as e:
            validator.errors.append(f"Orchestration failed: {e}")
            validator.print_results()
            return 2
        
        # Run validation checks
        checks = [
            ("Telemetry Completeness", lambda: validator.validate_telemetry_completeness(orch)),
            ("Phase Boundaries", lambda: validator.validate_phase_boundaries(orch)),
            ("Trace Context Consistency", lambda: validator.validate_trace_context_consistency(orch)),
            ("Deterministic Hashing", lambda: validator.validate_deterministic_hashing()),
            ("Audit Log Immutability", lambda: validator.validate_audit_log_immutability(orch, log_dir)),
            ("Telemetry Persistence", lambda: validator.validate_telemetry_persistence(log_dir)),
            ("Contract Enforcement", lambda: validator.validate_contract_enforcement(orch))
        ]
        
        passed_checks = 0
        for check_name, check_func in checks:
            try:
                if check_func():
                    print(f"✅ {check_name}")
                    passed_checks += 1
                else:
                    print(f"❌ {check_name}")
            except Exception as e:
                print(f"❌ {check_name} (exception: {e})")
                validator.errors.append(f"Check '{check_name}' raised exception: {e}")
        
        print(f"\nPassed: {passed_checks}/{len(checks)} checks")
    
    # Print results
    validator.print_results()
    
    # Return exit code
    if len(validator.errors) > 0:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
