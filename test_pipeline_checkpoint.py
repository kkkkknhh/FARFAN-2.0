#!/usr/bin/env python3
"""
Tests for Pipeline Checkpoint System
"""

import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from pipeline_checkpoint import (
    ENCRYPTION_AVAILABLE,
    CheckpointIndex,
    CheckpointMetadata,
    CheckpointVersion,
    CloudBackupConfig,
    CloudBackupProvider,
    PipelineCheckpoint,
    RetentionPolicy,
    generate_encryption_key,
    load_encryption_key,
)


class TestCheckpointMetadata(unittest.TestCase):
    """Test CheckpointMetadata dataclass"""

    def test_to_dict(self):
        metadata = CheckpointMetadata(
            checkpoint_id="test_001",
            timestamp="2024-01-01T12:00:00",
            version="1.0.0",
            file_path="/tmp/test.ckpt",
            hash_sha256="abc123",
            is_encrypted=True,
            size_bytes=1024,
        )

        d = metadata.to_dict()
        self.assertEqual(d["checkpoint_id"], "test_001")
        self.assertEqual(d["hash_sha256"], "abc123")
        self.assertTrue(d["is_encrypted"])

    def test_from_dict(self):
        d = {
            "checkpoint_id": "test_001",
            "timestamp": "2024-01-01T12:00:00",
            "version": "1.0.0",
            "file_path": "/tmp/test.ckpt",
            "hash_sha256": "abc123",
            "is_encrypted": False,
            "is_incremental": False,
            "base_checkpoint_id": None,
            "size_bytes": 1024,
            "state_keys": ["key1", "key2"],
            "custom_metadata": {},
        }

        metadata = CheckpointMetadata.from_dict(d)
        self.assertEqual(metadata.checkpoint_id, "test_001")
        self.assertEqual(metadata.hash_sha256, "abc123")
        self.assertEqual(metadata.size_bytes, 1024)


class TestCheckpointIndex(unittest.TestCase):
    """Test CheckpointIndex dataclass"""

    def test_to_from_dict(self):
        metadata1 = CheckpointMetadata(
            checkpoint_id="test_001",
            timestamp="2024-01-01T12:00:00",
            version="1.0.0",
            file_path="/tmp/test1.ckpt",
            hash_sha256="abc123",
            size_bytes=1024,
        )

        index = CheckpointIndex(
            checkpoints={"test_001": metadata1}, last_full_checkpoint_id="test_001"
        )

        d = index.to_dict()
        restored = CheckpointIndex.from_dict(d)

        self.assertIn("test_001", restored.checkpoints)
        self.assertEqual(restored.last_full_checkpoint_id, "test_001")
        self.assertEqual(restored.checkpoints["test_001"].hash_sha256, "abc123")


class TestPipelineCheckpointBasic(unittest.TestCase):
    """Test basic checkpoint functionality"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint = PipelineCheckpoint(
            checkpoint_dir=self.temp_dir, enable_incremental=False
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load(self):
        state = {
            "stage": 1,
            "data": [1, 2, 3, 4, 5],
            "metrics": {"accuracy": 0.95, "loss": 0.05},
        }

        checkpoint_id = self.checkpoint.save(state, checkpoint_id="test_basic")
        self.assertEqual(checkpoint_id, "test_basic")

        restored = self.checkpoint.load("test_basic")
        self.assertEqual(restored, state)

    def test_hash_verification(self):
        state = {"data": "test"}
        checkpoint_id = self.checkpoint.save(state)

        # Load with verification
        restored = self.checkpoint.load(checkpoint_id, verify_hash=True)
        self.assertEqual(restored, state)

        # Corrupt the file
        metadata = self.checkpoint.get_checkpoint_info(checkpoint_id)
        checkpoint_path = Path(metadata.file_path)
        data = checkpoint_path.read_bytes()
        checkpoint_path.write_bytes(data + b"corrupted")

        # Should raise error on hash verification
        with self.assertRaises(ValueError) as ctx:
            self.checkpoint.load(checkpoint_id, verify_hash=True)
        self.assertIn("Hash verification failed", str(ctx.exception))

    def test_list_checkpoints(self):
        states = [
            {"stage": 1, "value": 100},
            {"stage": 2, "value": 200},
            {"stage": 3, "value": 300},
        ]

        for i, state in enumerate(states):
            self.checkpoint.save(state, checkpoint_id=f"stage_{i + 1}")
            time.sleep(0.01)  # Ensure different timestamps

        checkpoints = self.checkpoint.list_checkpoints()
        self.assertEqual(len(checkpoints), 3)

        # Should be sorted by timestamp descending
        self.assertEqual(checkpoints[0].checkpoint_id, "stage_3")
        self.assertEqual(checkpoints[-1].checkpoint_id, "stage_1")

    def test_delete_checkpoint(self):
        state = {"data": "test"}
        checkpoint_id = self.checkpoint.save(state)

        self.assertIsNotNone(self.checkpoint.get_checkpoint_info(checkpoint_id))

        self.checkpoint.delete_checkpoint(checkpoint_id)

        self.assertIsNone(self.checkpoint.get_checkpoint_info(checkpoint_id))
        with self.assertRaises(ValueError):
            self.checkpoint.load(checkpoint_id)

    def test_custom_metadata(self):
        state = {"data": "test"}
        custom = {"user": "alice", "experiment": "exp_001"}

        checkpoint_id = self.checkpoint.save(state, custom_metadata=custom)

        metadata = self.checkpoint.get_checkpoint_info(checkpoint_id)
        self.assertEqual(metadata.custom_metadata, custom)

    def test_get_statistics(self):
        for i in range(5):
            state = {"value": i}
            self.checkpoint.save(state, checkpoint_id=f"ckpt_{i}")

        stats = self.checkpoint.get_statistics()

        self.assertEqual(stats["total_checkpoints"], 5)
        self.assertEqual(stats["full_checkpoints"], 5)
        self.assertEqual(stats["incremental_checkpoints"], 0)
        self.assertGreater(stats["total_size_bytes"], 0)


class TestIncrementalCheckpointing(unittest.TestCase):
    """Test incremental checkpointing with delta detection"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint = PipelineCheckpoint(
            checkpoint_dir=self.temp_dir, enable_incremental=True
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_incremental_checkpoint(self):
        # Save full checkpoint
        state1 = {"stage": 1, "data": [1, 2, 3], "metrics": {"accuracy": 0.90}}
        id1 = self.checkpoint.save(state1, checkpoint_id="full_1")

        # Save incremental checkpoint (only change metrics)
        state2 = {"stage": 1, "data": [1, 2, 3], "metrics": {"accuracy": 0.95}}
        id2 = self.checkpoint.save(state2, checkpoint_id="incr_1")

        # Check metadata
        meta1 = self.checkpoint.get_checkpoint_info(id1)
        meta2 = self.checkpoint.get_checkpoint_info(id2)

        self.assertFalse(meta1.is_incremental)
        self.assertTrue(meta2.is_incremental)
        self.assertEqual(meta2.base_checkpoint_id, id1)

        # Incremental should be smaller
        self.assertLess(meta2.size_bytes, meta1.size_bytes)

        # Load incremental checkpoint
        restored = self.checkpoint.load(id2)
        self.assertEqual(restored, state2)

    def test_force_full_checkpoint(self):
        state1 = {"data": [1, 2, 3]}
        id1 = self.checkpoint.save(state1, checkpoint_id="full_1")

        state2 = {"data": [1, 2, 3, 4]}
        id2 = self.checkpoint.save(state2, checkpoint_id="full_2", force_full=True)

        meta2 = self.checkpoint.get_checkpoint_info(id2)
        self.assertFalse(meta2.is_incremental)

    def test_delta_with_deleted_keys(self):
        state1 = {"key1": "value1", "key2": "value2", "key3": "value3"}
        id1 = self.checkpoint.save(state1, checkpoint_id="full_1")

        state2 = {"key1": "value1", "key3": "modified"}
        id2 = self.checkpoint.save(state2, checkpoint_id="incr_1")

        restored = self.checkpoint.load(id2)
        self.assertEqual(restored, state2)
        self.assertNotIn("key2", restored)


class TestRetentionPolicies(unittest.TestCase):
    """Test retention policies for automatic cleanup"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_keep_n_recent(self):
        checkpoint = PipelineCheckpoint(
            checkpoint_dir=self.temp_dir,
            retention_policy=RetentionPolicy.KEEP_N_RECENT,
            retention_count=3,
            enable_incremental=False,
        )

        # Create 5 checkpoints
        for i in range(5):
            state = {"value": i}
            checkpoint.save(state, checkpoint_id=f"ckpt_{i}")
            time.sleep(0.01)

        # Should only keep 3 most recent
        checkpoints = checkpoint.list_checkpoints()
        self.assertEqual(len(checkpoints), 3)

        # Should keep ckpt_2, ckpt_3, ckpt_4
        ids = [c.checkpoint_id for c in checkpoints]
        self.assertIn("ckpt_2", ids)
        self.assertIn("ckpt_3", ids)
        self.assertIn("ckpt_4", ids)
        self.assertNotIn("ckpt_0", ids)
        self.assertNotIn("ckpt_1", ids)

    def test_delete_older_than(self):
        checkpoint = PipelineCheckpoint(
            checkpoint_dir=self.temp_dir,
            retention_policy=RetentionPolicy.DELETE_OLDER_THAN,
            retention_age_days=7,
            enable_incremental=False,
        )

        # Create checkpoint with old timestamp
        state1 = {"value": 1}
        id1 = checkpoint.save(state1, checkpoint_id="old")

        # Manually modify timestamp to be 10 days old
        checkpoint.index.checkpoints[id1].timestamp = (
            datetime.now() - timedelta(days=10)
        ).isoformat()
        checkpoint._save_index()

        # Create new checkpoint
        state2 = {"value": 2}
        id2 = checkpoint.save(state2, checkpoint_id="new")

        # Old checkpoint should be deleted
        checkpoints = checkpoint.list_checkpoints()
        ids = [c.checkpoint_id for c in checkpoints]
        self.assertNotIn("old", ids)
        self.assertIn("new", ids)

    def test_keep_all(self):
        checkpoint = PipelineCheckpoint(
            checkpoint_dir=self.temp_dir,
            retention_policy=RetentionPolicy.KEEP_ALL,
            enable_incremental=False,
        )

        # Create many checkpoints
        for i in range(10):
            state = {"value": i}
            checkpoint.save(state, checkpoint_id=f"ckpt_{i}")

        # Should keep all
        checkpoints = checkpoint.list_checkpoints()
        self.assertEqual(len(checkpoints), 10)


@unittest.skipIf(not ENCRYPTION_AVAILABLE, "cryptography library not available")
class TestEncryption(unittest.TestCase):
    """Test encryption functionality"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_encryption_key_generation(self):
        key_path = os.path.join(self.temp_dir, "test.key")
        key = generate_encryption_key(key_path)

        self.assertTrue(os.path.exists(key_path))
        loaded_key = load_encryption_key(key_path)
        self.assertEqual(key, loaded_key)

    def test_save_load_encrypted(self):
        checkpoint = PipelineCheckpoint(
            checkpoint_dir=self.temp_dir,
            enable_encryption=True,
            enable_incremental=False,
        )

        state = {
            "sensitive_data": "secret_password",
            "plan_info": "confidential_details",
        }

        checkpoint_id = checkpoint.save(state, checkpoint_id="encrypted")

        # Check metadata
        metadata = checkpoint.get_checkpoint_info(checkpoint_id)
        self.assertTrue(metadata.is_encrypted)

        # Load and verify
        restored = checkpoint.load(checkpoint_id)
        self.assertEqual(restored, state)

    def test_encrypted_file_not_readable(self):
        checkpoint = PipelineCheckpoint(
            checkpoint_dir=self.temp_dir,
            enable_encryption=True,
            enable_incremental=False,
        )

        state = {"secret": "value"}
        checkpoint_id = checkpoint.save(state)

        # Read raw file - should be encrypted and not contain plaintext
        metadata = checkpoint.get_checkpoint_info(checkpoint_id)
        raw_data = Path(metadata.file_path).read_bytes()

        self.assertNotIn(b"secret", raw_data)
        self.assertNotIn(b"value", raw_data)

    def test_encryption_key_required_to_load(self):
        key = generate_encryption_key()

        checkpoint1 = PipelineCheckpoint(
            checkpoint_dir=self.temp_dir,
            enable_encryption=True,
            encryption_key=key,
            enable_incremental=False,
        )

        state = {"data": "secret"}
        checkpoint_id = checkpoint1.save(state)

        # Try to load without encryption enabled
        checkpoint2 = PipelineCheckpoint(
            checkpoint_dir=self.temp_dir,
            enable_encryption=False,
            enable_incremental=False,
        )

        with self.assertRaises(RuntimeError) as ctx:
            checkpoint2.load(checkpoint_id)
        self.assertIn("encrypted", str(ctx.exception).lower())


class TestCloudBackupStubs(unittest.TestCase):
    """Test cloud backup stub methods"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cloud_config = CloudBackupConfig(
            enabled=True,
            provider=CloudBackupProvider.AWS_S3,
            bucket_name="test-bucket",
            region="us-west-2",
            auto_sync=False,
        )
        self.checkpoint = PipelineCheckpoint(
            checkpoint_dir=self.temp_dir,
            cloud_backup_config=self.cloud_config,
            enable_incremental=False,
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cloud_config(self):
        self.assertTrue(self.checkpoint.cloud_backup_config.enabled)
        self.assertEqual(
            self.checkpoint.cloud_backup_config.provider, CloudBackupProvider.AWS_S3
        )

    def test_configure_cloud_backup(self):
        new_config = CloudBackupConfig(
            enabled=True,
            provider=CloudBackupProvider.AZURE_BLOB,
            bucket_name="azure-container",
        )
        self.checkpoint.configure_cloud_backup(new_config)

        self.assertEqual(
            self.checkpoint.cloud_backup_config.provider, CloudBackupProvider.AZURE_BLOB
        )

    def test_sync_to_cloud_stub(self):
        state = {"data": "test"}
        checkpoint_id = self.checkpoint.save(state)

        # Should not raise error (it's a stub)
        self.checkpoint.sync_to_cloud([checkpoint_id])

    def test_restore_from_cloud_not_implemented(self):
        with self.assertRaises(NotImplementedError) as ctx:
            self.checkpoint.restore_from_cloud("test_id")
        self.assertIn("stub", str(ctx.exception).lower())

    def test_list_cloud_checkpoints_not_implemented(self):
        with self.assertRaises(NotImplementedError) as ctx:
            self.checkpoint.list_cloud_checkpoints()
        self.assertIn("stub", str(ctx.exception).lower())


class TestFilterAndSort(unittest.TestCase):
    """Test filtering and sorting of checkpoints"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint = PipelineCheckpoint(
            checkpoint_dir=self.temp_dir, enable_incremental=True
        )

        # Create mix of checkpoints
        for i in range(5):
            state = {"value": i, "large_data": "x" * (1000 * i)}
            self.checkpoint.save(state, checkpoint_id=f"ckpt_{i}")
            time.sleep(0.01)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sort_by_size(self):
        checkpoints = self.checkpoint.list_checkpoints(
            sort_by="size_bytes", reverse=True
        )

        # Largest first
        for i in range(len(checkpoints) - 1):
            self.assertGreaterEqual(
                checkpoints[i].size_bytes, checkpoints[i + 1].size_bytes
            )

    def test_filter_function(self):
        # Filter to get only checkpoints larger than 1KB
        checkpoints = self.checkpoint.list_checkpoints(
            filter_fn=lambda c: c.size_bytes > 1024
        )

        for c in checkpoints:
            self.assertGreater(c.size_bytes, 1024)

    def test_filter_incremental(self):
        # Filter to get only full checkpoints
        checkpoints = self.checkpoint.list_checkpoints(
            filter_fn=lambda c: not c.is_incremental
        )

        # Should have at least one full checkpoint
        self.assertGreater(len(checkpoints), 0)
        for c in checkpoints:
            self.assertFalse(c.is_incremental)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint = PipelineCheckpoint(
            checkpoint_dir=self.temp_dir, enable_incremental=False
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_state(self):
        state = {}
        checkpoint_id = self.checkpoint.save(state)
        restored = self.checkpoint.load(checkpoint_id)
        self.assertEqual(restored, {})

    def test_complex_state(self):
        state = {
            "nested": {"deep": {"value": [1, 2, {"key": "value"}]}},
            "list": [1, 2, 3],
            "tuple": (4, 5, 6),
            "set": {7, 8, 9},
        }

        checkpoint_id = self.checkpoint.save(state)
        restored = self.checkpoint.load(checkpoint_id)

        self.assertEqual(restored["nested"]["deep"]["value"], [1, 2, {"key": "value"}])
        self.assertEqual(restored["tuple"], (4, 5, 6))

    def test_load_nonexistent_checkpoint(self):
        with self.assertRaises(ValueError) as ctx:
            self.checkpoint.load("nonexistent")
        self.assertIn("not found", str(ctx.exception).lower())

    def test_delete_nonexistent_checkpoint(self):
        # Should not raise error
        self.checkpoint.delete_checkpoint("nonexistent")

    def test_auto_generate_checkpoint_id(self):
        state = {"data": "test"}
        checkpoint_id = self.checkpoint.save(state)

        self.assertTrue(checkpoint_id.startswith("ckpt_"))
        self.assertIsNotNone(self.checkpoint.get_checkpoint_info(checkpoint_id))

    def test_statistics_empty(self):
        checkpoint = PipelineCheckpoint(checkpoint_dir=self.temp_dir)
        stats = checkpoint.get_statistics()

        self.assertEqual(stats["total_checkpoints"], 0)
        self.assertEqual(stats["total_size_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
