#!/usr/bin/env python3
"""
Pipeline Checkpoint System
Provides checkpointing capabilities with compression, encryption, versioning, and retention policies
"""

import gzip
import hashlib
import json
import logging
import os
import pickle
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Optional dependencies
try:
    from cryptography.fernet import Fernet

    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    Fernet = None

logger = logging.getLogger(__name__)


class CheckpointVersion:
    """Checkpoint format version for backward compatibility"""

    CURRENT = "1.0.0"


class RetentionPolicy(Enum):
    """Retention policy types for checkpoint cleanup"""

    KEEP_N_RECENT = "keep_n_recent"
    DELETE_OLDER_THAN = "delete_older_than"
    KEEP_ALL = "keep_all"


@dataclass
class CheckpointMetadata:
    """Metadata for a single checkpoint"""

    checkpoint_id: str
    timestamp: str
    version: str
    file_path: str
    hash_sha256: str
    is_encrypted: bool = False
    is_incremental: bool = False
    base_checkpoint_id: Optional[str] = None
    size_bytes: int = 0
    state_keys: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "CheckpointMetadata":
        """Create from dictionary"""
        return cls(**data)


@dataclass
class CheckpointIndex:
    """Index tracking all checkpoints"""

    checkpoints: Dict[str, CheckpointMetadata] = field(default_factory=dict)
    last_full_checkpoint_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "checkpoints": {k: v.to_dict() for k, v in self.checkpoints.items()},
            "last_full_checkpoint_id": self.last_full_checkpoint_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CheckpointIndex":
        """Create from dictionary"""
        checkpoints = {
            k: CheckpointMetadata.from_dict(v)
            for k, v in data.get("checkpoints", {}).items()
        }
        return cls(
            checkpoints=checkpoints,
            last_full_checkpoint_id=data.get("last_full_checkpoint_id"),
        )


class CloudBackupProvider(Enum):
    """Supported cloud backup providers (for future implementation)"""

    AWS_S3 = "aws_s3"
    AZURE_BLOB = "azure_blob"
    GCP_STORAGE = "gcp_storage"
    LOCAL_ONLY = "local_only"


@dataclass
class CloudBackupConfig:
    """Configuration for cloud backup (stub for future implementation)"""

    enabled: bool = False
    provider: CloudBackupProvider = CloudBackupProvider.LOCAL_ONLY
    bucket_name: Optional[str] = None
    region: Optional[str] = None
    credentials_path: Optional[str] = None
    auto_sync: bool = False
    sync_interval_seconds: int = 3600


class PipelineCheckpoint:
    """
    Manages pipeline state checkpointing with compression, encryption, versioning, and retention.

    Features:
    - Gzip compression for space efficiency
    - SHA256 hash verification for integrity
    - Versioning for format compatibility
    - Incremental checkpointing with delta detection
    - Optional Fernet encryption for sensitive data
    - Configurable retention policies
    - Checkpoint index for querying and management
    - Stub methods for future cloud backup integration

    Example:
        >>> checkpoint = PipelineCheckpoint(checkpoint_dir="./checkpoints")
        >>> state = {'stage': 1, 'data': [1, 2, 3], 'metrics': {'accuracy': 0.95}}
        >>> checkpoint.save(state, checkpoint_id="stage_1")
        >>> restored = checkpoint.load("stage_1")
        >>> assert restored == state
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        encryption_key: Optional[bytes] = None,
        enable_encryption: bool = False,
        enable_incremental: bool = True,
        retention_policy: RetentionPolicy = RetentionPolicy.KEEP_N_RECENT,
        retention_count: int = 10,
        retention_age_days: int = 30,
        cloud_backup_config: Optional[CloudBackupConfig] = None,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            encryption_key: Fernet encryption key (if None and enable_encryption=True, generates new key)
            enable_encryption: Enable Fernet encryption for checkpoints
            enable_incremental: Enable incremental checkpointing
            retention_policy: Policy for automatic cleanup
            retention_count: Number of recent checkpoints to keep (for KEEP_N_RECENT)
            retention_age_days: Delete checkpoints older than this (for DELETE_OLDER_THAN)
            cloud_backup_config: Configuration for cloud backup (optional)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.checkpoint_dir / "checkpoint_index.json"
        self.enable_incremental = enable_incremental
        self.retention_policy = retention_policy
        self.retention_count = retention_count
        self.retention_age_days = retention_age_days
        self.cloud_backup_config = cloud_backup_config or CloudBackupConfig()

        # Encryption setup
        self.enable_encryption = enable_encryption
        self.cipher = None
        if enable_encryption:
            if not ENCRYPTION_AVAILABLE:
                raise ImportError(
                    "Encryption requires cryptography library. "
                    "Install with: pip install cryptography"
                )
            if encryption_key is None:
                logger.warning("No encryption key provided, generating new key")
                encryption_key = Fernet.generate_key()
                key_path = self.checkpoint_dir / "encryption.key"
                key_path.write_bytes(encryption_key)
                logger.info(f"Encryption key saved to: {key_path}")
            self.cipher = Fernet(encryption_key)

        # Load or create index
        self.index = self._load_index()

        logger.info(f"PipelineCheckpoint initialized at {self.checkpoint_dir}")
        logger.info(
            f"Encryption: {self.enable_encryption}, Incremental: {self.enable_incremental}"
        )
        logger.info(
            f"Retention: {self.retention_policy.value} (count={self.retention_count}, age={self.retention_age_days}d)"
        )

    def _load_index(self) -> CheckpointIndex:
        """Load checkpoint index from disk"""
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    data = json.load(f)
                return CheckpointIndex.from_dict(data)
            except Exception as e:
                logger.error(f"Error loading checkpoint index: {e}, creating new index")
                return CheckpointIndex()
        return CheckpointIndex()

    def _save_index(self):
        """Save checkpoint index to disk"""
        with open(self.index_path, "w") as f:
            json.dump(self.index.to_dict(), f, indent=2)

    def _compute_hash(self, data: bytes) -> str:
        """Compute SHA256 hash of data"""
        return hashlib.sha256(data).hexdigest()

    def _serialize_state(self, state: Dict[str, Any]) -> bytes:
        """Serialize state to bytes using pickle"""
        return pickle.dumps(state)

    def _deserialize_state(self, data: bytes) -> Dict[str, Any]:
        """Deserialize state from bytes using pickle"""
        return pickle.loads(data)

    def _compress(self, data: bytes) -> bytes:
        """Compress data using gzip"""
        return gzip.compress(data)

    def _decompress(self, data: bytes) -> bytes:
        """Decompress gzip data"""
        return gzip.decompress(data)

    def _encrypt(self, data: bytes) -> bytes:
        """Encrypt data using Fernet"""
        if self.cipher is None:
            raise RuntimeError("Encryption not enabled")
        return self.cipher.encrypt(data)

    def _decrypt(self, data: bytes) -> bytes:
        """Decrypt data using Fernet"""
        if self.cipher is None:
            raise RuntimeError("Encryption not enabled")
        return self.cipher.decrypt(data)

    def _detect_delta(
        self, current_state: Dict[str, Any], previous_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect changes between current and previous state.

        Returns a delta containing only changed fields.
        """
        delta = {}

        # Find added or modified keys
        for key, value in current_state.items():
            if key not in previous_state or previous_state[key] != value:
                delta[key] = value

        # Mark deleted keys
        deleted_keys = set(previous_state.keys()) - set(current_state.keys())
        if deleted_keys:
            delta["__deleted_keys__"] = list(deleted_keys)

        return delta

    def _apply_delta(
        self, base_state: Dict[str, Any], delta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply delta to base state to reconstruct full state"""
        result = base_state.copy()

        # Apply changes and additions
        for key, value in delta.items():
            if key != "__deleted_keys__":
                result[key] = value

        # Remove deleted keys
        if "__deleted_keys__" in delta:
            for key in delta["__deleted_keys__"]:
                result.pop(key, None)

        return result

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get file path for checkpoint"""
        return self.checkpoint_dir / f"{checkpoint_id}.ckpt.gz"

    def save(
        self,
        state: Dict[str, Any],
        checkpoint_id: Optional[str] = None,
        force_full: bool = False,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save pipeline state to disk.

        Args:
            state: State dictionary to checkpoint
            checkpoint_id: Unique identifier (auto-generated if None)
            force_full: Force full checkpoint even if incremental is enabled
            custom_metadata: Additional metadata to store with checkpoint

        Returns:
            checkpoint_id of saved checkpoint
        """
        if checkpoint_id is None:
            checkpoint_id = f"ckpt_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info(f"Saving checkpoint: {checkpoint_id}")

        # Determine if incremental
        is_incremental = False
        base_checkpoint_id = None
        state_to_save = state

        if (
            self.enable_incremental
            and not force_full
            and self.index.last_full_checkpoint_id
        ):
            # Try to create incremental checkpoint
            base_checkpoint_id = self.index.last_full_checkpoint_id
            try:
                base_state = self.load(base_checkpoint_id)
                delta = self._detect_delta(state, base_state)

                # Only use incremental if delta is significantly smaller
                delta_size = len(self._serialize_state(delta))
                full_size = len(self._serialize_state(state))
                if delta_size < full_size * 0.7:  # 70% threshold
                    state_to_save = delta
                    is_incremental = True
                    logger.info(
                        f"Creating incremental checkpoint (delta size: {delta_size}, full size: {full_size})"
                    )
                else:
                    logger.info("Delta not beneficial, creating full checkpoint")
            except Exception as e:
                logger.warning(
                    f"Failed to create incremental checkpoint: {e}, creating full checkpoint"
                )

        # Serialize
        serialized = self._serialize_state(state_to_save)

        # Compress
        compressed = self._compress(serialized)

        # Optionally encrypt
        final_data = compressed
        if self.enable_encryption:
            final_data = self._encrypt(compressed)

        # Compute hash of final data
        hash_sha256 = self._compute_hash(final_data)

        # Write to disk
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        checkpoint_path.write_bytes(final_data)

        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now().isoformat(),
            version=CheckpointVersion.CURRENT,
            file_path=str(checkpoint_path),
            hash_sha256=hash_sha256,
            is_encrypted=self.enable_encryption,
            is_incremental=is_incremental,
            base_checkpoint_id=base_checkpoint_id,
            size_bytes=len(final_data),
            state_keys=list(state.keys()),
            custom_metadata=custom_metadata or {},
        )

        # Update index
        self.index.checkpoints[checkpoint_id] = metadata
        if not is_incremental:
            self.index.last_full_checkpoint_id = checkpoint_id
        self._save_index()

        logger.info(
            f"Checkpoint saved: {checkpoint_id} ({metadata.size_bytes} bytes, hash: {hash_sha256[:16]}...)"
        )

        # Apply retention policy
        self._apply_retention_policy()

        # Cloud backup (stub)
        if self.cloud_backup_config.enabled and self.cloud_backup_config.auto_sync:
            self._cloud_backup_checkpoint(checkpoint_id)

        return checkpoint_id

    def load(self, checkpoint_id: str, verify_hash: bool = True) -> Dict[str, Any]:
        """
        Load pipeline state from disk.

        Args:
            checkpoint_id: Checkpoint to load
            verify_hash: Verify SHA256 hash before loading

        Returns:
            Restored state dictionary

        Raises:
            ValueError: If checkpoint not found or hash verification fails
        """
        if checkpoint_id not in self.index.checkpoints:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        metadata = self.index.checkpoints[checkpoint_id]
        checkpoint_path = Path(metadata.file_path)

        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint file not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_id}")

        # Read from disk
        final_data = checkpoint_path.read_bytes()

        # Verify hash
        if verify_hash:
            computed_hash = self._compute_hash(final_data)
            if computed_hash != metadata.hash_sha256:
                raise ValueError(
                    f"Hash verification failed for checkpoint {checkpoint_id}. "
                    f"Expected: {metadata.hash_sha256}, Got: {computed_hash}"
                )
            logger.debug(f"Hash verified: {computed_hash[:16]}...")

        # Optionally decrypt
        compressed = final_data
        if metadata.is_encrypted:
            if self.cipher is None:
                raise RuntimeError(
                    "Checkpoint is encrypted but encryption is not enabled"
                )
            compressed = self._decrypt(final_data)

        # Decompress
        serialized = self._decompress(compressed)

        # Deserialize
        state = self._deserialize_state(serialized)

        # If incremental, reconstruct full state
        if metadata.is_incremental:
            if metadata.base_checkpoint_id is None:
                raise ValueError(
                    f"Incremental checkpoint {checkpoint_id} missing base_checkpoint_id"
                )
            logger.info(
                f"Reconstructing from base checkpoint: {metadata.base_checkpoint_id}"
            )
            base_state = self.load(metadata.base_checkpoint_id, verify_hash=verify_hash)
            state = self._apply_delta(base_state, state)

        logger.info(f"Checkpoint loaded: {checkpoint_id}")
        return state

    def list_checkpoints(
        self,
        sort_by: str = "timestamp",
        reverse: bool = True,
        filter_fn: Optional[Callable[[CheckpointMetadata], bool]] = None,
    ) -> List[CheckpointMetadata]:
        """
        List all checkpoints.

        Args:
            sort_by: Sort key ('timestamp', 'size_bytes', 'checkpoint_id')
            reverse: Reverse sort order
            filter_fn: Optional filter function

        Returns:
            List of checkpoint metadata
        """
        checkpoints = list(self.index.checkpoints.values())

        if filter_fn:
            checkpoints = [c for c in checkpoints if filter_fn(c)]

        if sort_by == "timestamp":
            checkpoints.sort(key=lambda c: c.timestamp, reverse=reverse)
        elif sort_by == "size_bytes":
            checkpoints.sort(key=lambda c: c.size_bytes, reverse=reverse)
        elif sort_by == "checkpoint_id":
            checkpoints.sort(key=lambda c: c.checkpoint_id, reverse=reverse)

        return checkpoints

    def delete_checkpoint(self, checkpoint_id: str):
        """Delete a specific checkpoint"""
        if checkpoint_id not in self.index.checkpoints:
            logger.warning(f"Checkpoint not found: {checkpoint_id}")
            return

        metadata = self.index.checkpoints[checkpoint_id]
        checkpoint_path = Path(metadata.file_path)

        # Delete file
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Deleted checkpoint file: {checkpoint_path}")

        # Remove from index
        del self.index.checkpoints[checkpoint_id]

        # Update last_full_checkpoint_id if needed
        if self.index.last_full_checkpoint_id == checkpoint_id:
            # Find most recent full checkpoint
            full_checkpoints = [
                c for c in self.index.checkpoints.values() if not c.is_incremental
            ]
            if full_checkpoints:
                full_checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
                self.index.last_full_checkpoint_id = full_checkpoints[0].checkpoint_id
            else:
                self.index.last_full_checkpoint_id = None

        self._save_index()
        logger.info(f"Deleted checkpoint: {checkpoint_id}")

    def _apply_retention_policy(self):
        """Apply configured retention policy to clean up old checkpoints"""
        if self.retention_policy == RetentionPolicy.KEEP_ALL:
            return

        checkpoints = self.list_checkpoints(sort_by="timestamp", reverse=True)
        to_delete = []

        if self.retention_policy == RetentionPolicy.KEEP_N_RECENT:
            if len(checkpoints) > self.retention_count:
                to_delete = checkpoints[self.retention_count :]

        elif self.retention_policy == RetentionPolicy.DELETE_OLDER_THAN:
            cutoff = datetime.now() - timedelta(days=self.retention_age_days)
            to_delete = [
                c for c in checkpoints if datetime.fromisoformat(c.timestamp) < cutoff
            ]

        if to_delete:
            logger.info(f"Retention policy: deleting {len(to_delete)} old checkpoints")
            for checkpoint in to_delete:
                self.delete_checkpoint(checkpoint.checkpoint_id)

    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a specific checkpoint"""
        return self.index.checkpoints.get(checkpoint_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about checkpoints"""
        checkpoints = list(self.index.checkpoints.values())

        if not checkpoints:
            return {
                "total_checkpoints": 0,
                "total_size_bytes": 0,
                "full_checkpoints": 0,
                "incremental_checkpoints": 0,
                "encrypted_checkpoints": 0,
            }

        total_size = sum(c.size_bytes for c in checkpoints)
        full_count = sum(1 for c in checkpoints if not c.is_incremental)
        incremental_count = sum(1 for c in checkpoints if c.is_incremental)
        encrypted_count = sum(1 for c in checkpoints if c.is_encrypted)

        oldest = min(checkpoints, key=lambda c: c.timestamp)
        newest = max(checkpoints, key=lambda c: c.timestamp)

        return {
            "total_checkpoints": len(checkpoints),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "full_checkpoints": full_count,
            "incremental_checkpoints": incremental_count,
            "encrypted_checkpoints": encrypted_count,
            "oldest_checkpoint": oldest.checkpoint_id,
            "oldest_timestamp": oldest.timestamp,
            "newest_checkpoint": newest.checkpoint_id,
            "newest_timestamp": newest.timestamp,
        }

    # Cloud backup stub methods (for future implementation)

    def _cloud_backup_checkpoint(self, checkpoint_id: str):
        """
        Upload checkpoint to cloud storage (STUB - not implemented).

        This is a placeholder for future cloud backup integration.
        When implemented, it should:
        1. Authenticate with cloud provider
        2. Upload checkpoint file to configured bucket/container
        3. Update checkpoint metadata with cloud URL
        4. Handle errors and retries
        """
        if not self.cloud_backup_config.enabled:
            return

        logger.debug(f"Cloud backup stub called for checkpoint: {checkpoint_id}")
        logger.debug(f"Provider: {self.cloud_backup_config.provider.value}")
        logger.debug(f"Bucket: {self.cloud_backup_config.bucket_name}")

        # Future implementation would call provider-specific upload methods
        # Example:
        # if self.cloud_backup_config.provider == CloudBackupProvider.AWS_S3:
        #     self._upload_to_s3(checkpoint_id)
        # elif self.cloud_backup_config.provider == CloudBackupProvider.AZURE_BLOB:
        #     self._upload_to_azure(checkpoint_id)
        # elif self.cloud_backup_config.provider == CloudBackupProvider.GCP_STORAGE:
        #     self._upload_to_gcp(checkpoint_id)

    def configure_cloud_backup(self, config: CloudBackupConfig):
        """
        Configure cloud backup settings.

        Args:
            config: CloudBackupConfig with provider details
        """
        self.cloud_backup_config = config
        logger.info(
            f"Cloud backup configured: {config.provider.value} (enabled: {config.enabled})"
        )

    def sync_to_cloud(self, checkpoint_ids: Optional[List[str]] = None):
        """
        Manually sync checkpoints to cloud storage (STUB - not implemented).

        Args:
            checkpoint_ids: List of checkpoint IDs to sync (None = sync all)
        """
        if not self.cloud_backup_config.enabled:
            logger.warning("Cloud backup not enabled")
            return

        if checkpoint_ids is None:
            checkpoint_ids = list(self.index.checkpoints.keys())

        logger.info(f"Syncing {len(checkpoint_ids)} checkpoints to cloud (stub)")

        for checkpoint_id in checkpoint_ids:
            self._cloud_backup_checkpoint(checkpoint_id)

    def restore_from_cloud(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Restore checkpoint from cloud storage (STUB - not implemented).

        Args:
            checkpoint_id: Checkpoint to restore from cloud

        Returns:
            Restored state dictionary
        """
        if not self.cloud_backup_config.enabled:
            raise RuntimeError("Cloud backup not enabled")

        logger.info(f"Restoring checkpoint from cloud (stub): {checkpoint_id}")

        # Future implementation would:
        # 1. Download checkpoint from cloud to local temp directory
        # 2. Verify integrity
        # 3. Load using normal load() method
        # 4. Optionally save to local checkpoint directory

        raise NotImplementedError(
            "Cloud restore not yet implemented. "
            "This is a stub method for future cloud backup integration."
        )

    def list_cloud_checkpoints(self) -> List[str]:
        """
        List checkpoints available in cloud storage (STUB - not implemented).

        Returns:
            List of checkpoint IDs in cloud storage
        """
        if not self.cloud_backup_config.enabled:
            raise RuntimeError("Cloud backup not enabled")

        logger.info("Listing cloud checkpoints (stub)")

        # Future implementation would query cloud storage and return list

        raise NotImplementedError(
            "Cloud listing not yet implemented. "
            "This is a stub method for future cloud backup integration."
        )


def load_encryption_key(key_path: str) -> bytes:
    """
    Load encryption key from file.

    Args:
        key_path: Path to encryption key file

    Returns:
        Encryption key bytes
    """
    return Path(key_path).read_bytes()


def generate_encryption_key(output_path: Optional[str] = None) -> bytes:
    """
    Generate a new Fernet encryption key.

    Args:
        output_path: Optional path to save key

    Returns:
        Generated encryption key
    """
    if not ENCRYPTION_AVAILABLE:
        raise ImportError("Encryption requires cryptography library")

    key = Fernet.generate_key()

    if output_path:
        Path(output_path).write_bytes(key)
        logger.info(f"Encryption key saved to: {output_path}")

    return key
