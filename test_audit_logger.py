#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Immutable Audit Logger Module
===================================

SIN_CARRETA Compliance Tests:
- Validate append-only semantics
- Verify SHA-256 provenance tracking
- Test JSONL persistence
"""

import json
import tempfile
import unittest
from pathlib import Path

from infrastructure.audit_logger import ImmutableAuditLogger, AuditRecord


class TestImmutableAuditLogger(unittest.TestCase):
    """Test suite for immutable audit logger"""
    
    def setUp(self):
        """Create temporary audit log file"""
        self.temp_dir = tempfile.mkdtemp()
        self.audit_path = Path(self.temp_dir) / "test_audit.jsonl"
        self.logger = ImmutableAuditLogger(self.audit_path)
    
    def test_append_record(self):
        """Test appending audit record"""
        self.logger.append_record(
            run_id="test_run_001",
            orchestrator="TestOrchestrator",
            sha256_source="abc123",
            event="test_event",
            test_data="value"
        )
        
        # Verify in-memory cache
        self.assertEqual(len(self.logger._records), 1)
        record = self.logger._records[0]
        self.assertEqual(record.run_id, "test_run_001")
        self.assertEqual(record.orchestrator, "TestOrchestrator")
        self.assertEqual(record.event, "test_event")
    
    def test_record_persistence(self):
        """Test that records are persisted to disk"""
        self.logger.append_record(
            run_id="test_run_002",
            orchestrator="TestOrchestrator",
            sha256_source="def456",
            event="test_persist"
        )
        
        # Verify file exists and contains record
        self.assertTrue(self.audit_path.exists())
        
        with open(self.audit_path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            
            record_dict = json.loads(lines[0])
            self.assertEqual(record_dict['run_id'], "test_run_002")
            self.assertEqual(record_dict['orchestrator'], "TestOrchestrator")
    
    def test_append_only_semantics(self):
        """Test that records are append-only (no modifications)"""
        self.logger.append_record(
            run_id="test_run_003",
            orchestrator="TestOrchestrator",
            sha256_source="ghi789",
            event="first"
        )
        
        # Append second record
        self.logger.append_record(
            run_id="test_run_004",
            orchestrator="TestOrchestrator",
            sha256_source="jkl012",
            event="second"
        )
        
        # Verify both records exist
        self.assertEqual(len(self.logger._records), 2)
        self.assertEqual(self.logger._records[0].run_id, "test_run_003")
        self.assertEqual(self.logger._records[1].run_id, "test_run_004")
    
    def test_hash_file(self):
        """Test SHA-256 file hashing"""
        # Create temporary file
        temp_file = Path(self.temp_dir) / "test_file.txt"
        temp_file.write_text("Test content for hashing")
        
        # Hash file
        hash_value = ImmutableAuditLogger.hash_file(str(temp_file))
        
        # Verify hash is 64 characters (SHA-256 hex)
        self.assertEqual(len(hash_value), 64)
        
        # Verify deterministic (same content = same hash)
        hash_value2 = ImmutableAuditLogger.hash_file(str(temp_file))
        self.assertEqual(hash_value, hash_value2)
    
    def test_hash_string(self):
        """Test SHA-256 string hashing"""
        hash_value = ImmutableAuditLogger.hash_string("test content")
        
        # Verify hash is 64 characters
        self.assertEqual(len(hash_value), 64)
        
        # Verify deterministic
        hash_value2 = ImmutableAuditLogger.hash_string("test content")
        self.assertEqual(hash_value, hash_value2)
        
        # Verify different content = different hash
        hash_value3 = ImmutableAuditLogger.hash_string("different content")
        self.assertNotEqual(hash_value, hash_value3)
    
    def test_get_recent_records(self):
        """Test retrieving recent records"""
        # Add multiple records
        for i in range(5):
            self.logger.append_record(
                run_id=f"test_run_{i:03d}",
                orchestrator="TestOrchestrator",
                sha256_source=f"hash_{i}",
                event="test"
            )
        
        # Get recent records (default limit=10)
        recent = self.logger.get_recent_records(limit=3)
        
        # Verify returns most recent first
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0]['run_id'], "test_run_004")  # Most recent
        self.assertEqual(recent[1]['run_id'], "test_run_003")
        self.assertEqual(recent[2]['run_id'], "test_run_002")
    
    def test_get_records_by_run_id(self):
        """Test filtering records by run_id"""
        self.logger.append_record(
            run_id="target_run",
            orchestrator="TestOrchestrator",
            sha256_source="hash1",
            event="event1"
        )
        self.logger.append_record(
            run_id="other_run",
            orchestrator="TestOrchestrator",
            sha256_source="hash2",
            event="event2"
        )
        self.logger.append_record(
            run_id="target_run",
            orchestrator="TestOrchestrator",
            sha256_source="hash3",
            event="event3"
        )
        
        # Filter by run_id
        filtered = self.logger.get_records_by_run_id("target_run")
        
        # Verify correct records returned
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(r['run_id'] == "target_run" for r in filtered))
    
    def test_get_statistics(self):
        """Test audit log statistics"""
        # Add records
        self.logger.append_record(
            run_id="run1",
            orchestrator="OrchestratorA",
            sha256_source="hash1",
            event="test"
        )
        self.logger.append_record(
            run_id="run2",
            orchestrator="OrchestratorB",
            sha256_source="hash2",
            event="test"
        )
        self.logger.append_record(
            run_id="run3",
            orchestrator="OrchestratorA",
            sha256_source="hash1",  # Same source
            event="test"
        )
        
        # Get statistics
        stats = self.logger.get_statistics()
        
        # Verify statistics
        self.assertEqual(stats['total_records'], 3)
        self.assertEqual(stats['orchestrators']['OrchestratorA'], 2)
        self.assertEqual(stats['orchestrators']['OrchestratorB'], 1)
        self.assertEqual(stats['unique_sources'], 2)  # hash1 and hash2
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()
