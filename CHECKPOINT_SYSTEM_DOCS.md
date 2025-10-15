# Pipeline Checkpoint System Documentation

## Overview

The Pipeline Checkpoint System provides robust state persistence for the FARFAN 2.0 orchestrator with compression, encryption, versioning, and automatic retention management.

## Features

### ✅ Core Features
- **Gzip Compression**: Automatic compression for space efficiency
- **SHA256 Verification**: Integrity checking on load
- **Versioning**: Format version tracking for backward compatibility
- **Checkpoint Index**: JSON-based metadata tracking with queryable interface

### 🔄 Incremental Checkpointing
- **Delta Detection**: Automatically detects changes between states
- **Smart Compression**: Only stores modified fields for incremental checkpoints
- **Threshold-based**: Falls back to full checkpoint if delta isn't beneficial (70% threshold)
- **Automatic Reconstruction**: Seamlessly reconstructs full state from base + delta

### 🔒 Security
- **Fernet Encryption**: Optional symmetric encryption for sensitive data
- **Key Management**: Automatic key generation and secure storage
- **Encrypted at Rest**: Checkpoint files are encrypted before writing to disk

### 🗂️ Retention Policies
- **KEEP_N_RECENT**: Maintain only N most recent checkpoints
- **DELETE_OLDER_THAN**: Remove checkpoints older than specified age
- **KEEP_ALL**: Preserve all checkpoints (manual cleanup only)

### ☁️ Cloud Backup (Stub)
- Configuration interface for AWS S3, Azure Blob, and GCP Storage
- Stub methods prepared for future implementation
- Auto-sync capabilities with configurable intervals

## Installation

### Basic Installation
```bash
# No additional dependencies required for basic functionality
pip install -r requirements.txt
```

### With Encryption Support
```bash
pip install cryptography
```

## Quick Start

### Basic Usage
```python
from pipeline_checkpoint import PipelineCheckpoint

# Initialize checkpoint manager
checkpoint = PipelineCheckpoint(checkpoint_dir="./checkpoints")

# Save pipeline state
state = {
    'stage': 'document_extraction',
    'raw_text': 'content...',
    'tables': [...]
}
checkpoint_id = checkpoint.save(state, checkpoint_id="stage_1")

# Load checkpoint
restored_state = checkpoint.load(checkpoint_id)
```

### With Incremental Checkpointing
```python
checkpoint = PipelineCheckpoint(
    checkpoint_dir="./checkpoints",
    enable_incremental=True
)

# Full checkpoint
state = {'stage': 1, 'data': [...]}
checkpoint.save(state, checkpoint_id="full_stage1")

# Incremental checkpoint (only stores changes)
state['stage'] = 2
state['new_field'] = 'value'
checkpoint.save(state, checkpoint_id="incr_stage2")
```

### With Encryption
```python
from pipeline_checkpoint import PipelineCheckpoint, generate_encryption_key

# Generate encryption key
key = generate_encryption_key(output_path="./encryption.key")

# Create encrypted checkpoint manager
checkpoint = PipelineCheckpoint(
    checkpoint_dir="./checkpoints",
    enable_encryption=True,
    encryption_key=key
)

# Sensitive data is now encrypted at rest
sensitive_state = {
    'financial_data': {...},
    'sensitive_indicators': {...}
}
checkpoint.save(sensitive_state, checkpoint_id="sensitive")
```

### With Retention Policies
```python
from pipeline_checkpoint import PipelineCheckpoint, RetentionPolicy

# Keep only 10 most recent checkpoints
checkpoint = PipelineCheckpoint(
    checkpoint_dir="./checkpoints",
    retention_policy=RetentionPolicy.KEEP_N_RECENT,
    retention_count=10
)

# Or delete checkpoints older than 30 days
checkpoint = PipelineCheckpoint(
    checkpoint_dir="./checkpoints",
    retention_policy=RetentionPolicy.DELETE_OLDER_THAN,
    retention_age_days=30
)
```

## API Reference

### PipelineCheckpoint

#### Constructor
```python
PipelineCheckpoint(
    checkpoint_dir: str = "./checkpoints",
    encryption_key: Optional[bytes] = None,
    enable_encryption: bool = False,
    enable_incremental: bool = True,
    retention_policy: RetentionPolicy = RetentionPolicy.KEEP_N_RECENT,
    retention_count: int = 10,
    retention_age_days: int = 30,
    cloud_backup_config: Optional[CloudBackupConfig] = None
)
```

#### Methods

**save()**
```python
checkpoint_id = checkpoint.save(
    state: Dict[str, Any],
    checkpoint_id: Optional[str] = None,
    force_full: bool = False,
    custom_metadata: Optional[Dict[str, Any]] = None
) -> str
```
Saves pipeline state to disk with compression and optional encryption.

**load()**
```python
state = checkpoint.load(
    checkpoint_id: str,
    verify_hash: bool = True
) -> Dict[str, Any]
```
Loads and verifies checkpoint from disk.

**list_checkpoints()**
```python
checkpoints = checkpoint.list_checkpoints(
    sort_by: str = 'timestamp',
    reverse: bool = True,
    filter_fn: Optional[Callable[[CheckpointMetadata], bool]] = None
) -> List[CheckpointMetadata]
```
Lists all checkpoints with optional filtering and sorting.

**delete_checkpoint()**
```python
checkpoint.delete_checkpoint(checkpoint_id: str)
```
Deletes a specific checkpoint from disk and index.

**get_checkpoint_info()**
```python
metadata = checkpoint.get_checkpoint_info(checkpoint_id: str) -> Optional[CheckpointMetadata]
```
Retrieves metadata for a specific checkpoint.

**get_statistics()**
```python
stats = checkpoint.get_statistics() -> Dict[str, Any]
```
Returns statistics about all checkpoints.

#### Cloud Backup Methods (Stub)

**configure_cloud_backup()**
```python
checkpoint.configure_cloud_backup(config: CloudBackupConfig)
```
Configures cloud backup settings.

**sync_to_cloud()**
```python
checkpoint.sync_to_cloud(checkpoint_ids: Optional[List[str]] = None)
```
Manually syncs checkpoints to cloud (stub - not implemented).

**restore_from_cloud()**
```python
state = checkpoint.restore_from_cloud(checkpoint_id: str) -> Dict[str, Any]
```
Restores checkpoint from cloud storage (stub - not implemented).

**list_cloud_checkpoints()**
```python
checkpoint_ids = checkpoint.list_cloud_checkpoints() -> List[str]
```
Lists checkpoints in cloud storage (stub - not implemented).

### CheckpointMetadata

Metadata tracked for each checkpoint:

```python
@dataclass
class CheckpointMetadata:
    checkpoint_id: str
    timestamp: str
    version: str
    file_path: str
    hash_sha256: str
    is_encrypted: bool
    is_incremental: bool
    base_checkpoint_id: Optional[str]
    size_bytes: int
    state_keys: List[str]
    custom_metadata: Dict[str, Any]
```

### RetentionPolicy

```python
class RetentionPolicy(Enum):
    KEEP_N_RECENT = "keep_n_recent"
    DELETE_OLDER_THAN = "delete_older_than"
    KEEP_ALL = "keep_all"
```

### CloudBackupConfig

```python
@dataclass
class CloudBackupConfig:
    enabled: bool = False
    provider: CloudBackupProvider = CloudBackupProvider.LOCAL_ONLY
    bucket_name: Optional[str] = None
    region: Optional[str] = None
    credentials_path: Optional[str] = None
    auto_sync: bool = False
    sync_interval_seconds: int = 3600
```

## Use Cases

### 1. Checkpointing Orchestrator Pipeline

```python
from pipeline_checkpoint import PipelineCheckpoint

checkpoint = PipelineCheckpoint(
    checkpoint_dir="./orchestrator_checkpoints",
    enable_incremental=True,
    retention_policy=RetentionPolicy.KEEP_N_RECENT,
    retention_count=5
)

# Stage 1: Document extraction
state = {'policy_code': 'PDM2024', 'raw_text': '...', 'tables': [...]}
checkpoint.save(state, checkpoint_id="stage_1_extraction")

# Stage 2: Add semantic analysis
state['semantic_analysis'] = {...}
checkpoint.save(state, checkpoint_id="stage_2_semantic")

# Stage 3: Add causal extraction
state['causal_graph'] = {...}
checkpoint.save(state, checkpoint_id="stage_3_causal")

# If failure occurs, recover from last checkpoint
if error_occurred:
    state = checkpoint.load("stage_2_semantic")
    # Continue from stage 3
```

### 2. Sensitive Plan Data with Encryption

```python
from pipeline_checkpoint import PipelineCheckpoint, generate_encryption_key

# Generate and save key
key = generate_encryption_key("./encryption.key")

checkpoint = PipelineCheckpoint(
    checkpoint_dir="./sensitive_checkpoints",
    enable_encryption=True,
    encryption_key=key
)

# Save sensitive financial and demographic data
sensitive_state = {
    'financial_data': {
        'presupuesto_total': 150_000_000_000,
        'allocations': {...}
    },
    'population_data': {
        'poblacion_victimas': 12500,
        'poblacion_desplazada': 3400
    }
}

checkpoint.save(sensitive_state, checkpoint_id="plan_sensitive")
```

### 3. Long-Running Experiments with Retention

```python
checkpoint = PipelineCheckpoint(
    checkpoint_dir="./experiment_checkpoints",
    retention_policy=RetentionPolicy.DELETE_OLDER_THAN,
    retention_age_days=7  # Keep only last week
)

# Save experiment states
for epoch in range(100):
    experiment_state = {
        'epoch': epoch,
        'metrics': {...},
        'model_weights': {...}
    }
    checkpoint.save(
        experiment_state,
        checkpoint_id=f"epoch_{epoch}",
        custom_metadata={'experiment_id': 'exp_001', 'user': 'researcher'}
    )
```

### 4. Querying Checkpoint Metadata

```python
# Find checkpoints with high accuracy
high_accuracy = checkpoint.list_checkpoints(
    filter_fn=lambda c: c.custom_metadata.get('accuracy', 0) > 0.90
)

# Get largest checkpoints
largest = checkpoint.list_checkpoints(
    sort_by='size_bytes',
    reverse=True
)[:5]

# Find checkpoints from specific experiment
exp_checkpoints = checkpoint.list_checkpoints(
    filter_fn=lambda c: c.custom_metadata.get('experiment_id') == 'exp_001'
)

# Get statistics
stats = checkpoint.get_statistics()
print(f"Total: {stats['total_checkpoints']} checkpoints")
print(f"Size: {stats['total_size_mb']} MB")
```

### 5. Cloud Backup Configuration (Future)

```python
from pipeline_checkpoint import CloudBackupConfig, CloudBackupProvider

# Configure AWS S3 backup
cloud_config = CloudBackupConfig(
    enabled=True,
    provider=CloudBackupProvider.AWS_S3,
    bucket_name="farfan-checkpoints",
    region="us-east-1",
    credentials_path="~/.aws/credentials",
    auto_sync=True,
    sync_interval_seconds=3600
)

checkpoint = PipelineCheckpoint(
    checkpoint_dir="./checkpoints",
    cloud_backup_config=cloud_config
)

# Manually sync specific checkpoints
checkpoint.sync_to_cloud(["important_checkpoint_1", "important_checkpoint_2"])

# Note: These are stub methods - actual cloud upload not implemented yet
```

## Integration with FARFAN Orchestrator

```python
from orchestrator import FARFANOrchestrator, PipelineContext
from pipeline_checkpoint import PipelineCheckpoint, RetentionPolicy

# Initialize orchestrator
orchestrator = FARFANOrchestrator(output_dir="./results")

# Initialize checkpoint manager
checkpoint = PipelineCheckpoint(
    checkpoint_dir="./orchestrator_checkpoints",
    enable_incremental=True,
    retention_policy=RetentionPolicy.KEEP_N_RECENT,
    retention_count=10
)

# Process plan with checkpointing
ctx = PipelineContext(
    pdf_path=Path("plan.pdf"),
    policy_code="PDM2024-ANT-MED",
    output_dir=Path("./results")
)

# Save checkpoint after each stage
try:
    # Stage 1-2: Document extraction
    ctx = orchestrator._stage_extract_document(ctx)
    checkpoint.save(asdict(ctx), checkpoint_id=f"{ctx.policy_code}_stage_1")
    
    # Stage 3: Semantic analysis
    ctx = orchestrator._stage_semantic_analysis(ctx)
    checkpoint.save(asdict(ctx), checkpoint_id=f"{ctx.policy_code}_stage_3")
    
    # Stage 4: Causal extraction
    ctx = orchestrator._stage_causal_extraction(ctx)
    checkpoint.save(asdict(ctx), checkpoint_id=f"{ctx.policy_code}_stage_4")
    
    # ... continue for all stages
    
except Exception as e:
    # Recover from last successful checkpoint
    logger.error(f"Error in pipeline: {e}")
    last_checkpoint = checkpoint.list_checkpoints()[0]
    ctx_dict = checkpoint.load(last_checkpoint.checkpoint_id)
    # Resume from recovered state
```

## File Structure

```
checkpoints/
├── checkpoint_index.json          # Index of all checkpoints
├── encryption.key                 # Encryption key (if enabled)
├── ckpt_20240115_103000.ckpt.gz  # Compressed checkpoint files
├── ckpt_20240115_110000.ckpt.gz
└── ckpt_20240115_113000.ckpt.gz
```

### checkpoint_index.json Format
```json
{
  "checkpoints": {
    "stage_1": {
      "checkpoint_id": "stage_1",
      "timestamp": "2024-01-15T10:30:00.123456",
      "version": "1.0.0",
      "file_path": "checkpoints/stage_1.ckpt.gz",
      "hash_sha256": "abc123...",
      "is_encrypted": false,
      "is_incremental": false,
      "base_checkpoint_id": null,
      "size_bytes": 12345,
      "state_keys": ["stage", "data", "metrics"],
      "custom_metadata": {}
    }
  },
  "last_full_checkpoint_id": "stage_1"
}
```

## Performance Characteristics

### Compression Ratios
- Typical compression: 70-85% size reduction for text-heavy states
- Incremental checkpoints: 50-90% smaller than full checkpoints (depending on delta size)

### Speed
- Save operation: ~10-50ms for typical pipeline states (1-10MB uncompressed)
- Load operation: ~5-30ms
- Hash verification: ~1-5ms

### Storage Requirements
Example for FARFAN pipeline (10 stages, 5 retained checkpoints):
- Without incremental: ~50-100 MB
- With incremental: ~15-30 MB
- With encryption: +5-10% overhead

## Best Practices

### 1. Checkpoint Frequency
- Checkpoint after expensive/time-consuming stages
- Checkpoint before stages that modify state significantly
- Don't checkpoint on every minor operation (overhead)

### 2. Retention Policy Selection
- Development: Use `KEEP_N_RECENT` with count=5-10
- Production: Use `DELETE_OLDER_THAN` with age=7-30 days
- Critical data: Use `KEEP_ALL` with manual cleanup

### 3. Encryption
- Enable encryption for:
  - Financial data (presupuesto, allocations)
  - Demographic data (población víctimas, desplazados)
  - Sensitive indicators (seguridad, conflicto)
- Keep encryption key secure and backed up separately

### 4. Incremental Checkpointing
- Enable for pipelines with large states that change incrementally
- Force full checkpoint every N incremental checkpoints (e.g., every 5th)
- Use `force_full=True` for final/milestone checkpoints

### 5. Custom Metadata
- Store experiment IDs, user info, timestamps
- Use for filtering and organizing checkpoints
- Include version numbers for reproducibility

## Troubleshooting

### Hash Verification Failed
```python
# If hash verification fails, checkpoint file may be corrupted
try:
    state = checkpoint.load(checkpoint_id, verify_hash=True)
except ValueError as e:
    logger.error(f"Checkpoint corrupted: {e}")
    # Try loading from previous checkpoint
    checkpoints = checkpoint.list_checkpoints()
    state = checkpoint.load(checkpoints[1].checkpoint_id)
```

### Encryption Key Lost
```python
# If encryption key is lost, checkpoints cannot be decrypted
# Always back up encryption.key file separately
# Consider using key management service (KMS) in production
```

### Disk Space Issues
```python
# Monitor checkpoint storage
stats = checkpoint.get_statistics()
if stats['total_size_mb'] > 1000:  # 1GB threshold
    logger.warning("Checkpoint storage exceeds threshold")
    # Adjust retention policy or manually clean up
    old_checkpoints = checkpoint.list_checkpoints()[-10:]
    for ckpt in old_checkpoints:
        checkpoint.delete_checkpoint(ckpt.checkpoint_id)
```

## Future Enhancements

### Cloud Backup Implementation
When implementing cloud backup, the following should be added:

1. **AWS S3 Integration**
   - Use boto3 for S3 operations
   - Implement `_upload_to_s3()` method
   - Handle authentication with IAM roles

2. **Azure Blob Storage**
   - Use azure-storage-blob SDK
   - Implement `_upload_to_azure()` method
   - Handle authentication with SAS tokens

3. **GCP Storage**
   - Use google-cloud-storage SDK
   - Implement `_upload_to_gcp()` method
   - Handle authentication with service accounts

4. **Sync Features**
   - Background sync thread with configurable interval
   - Retry logic with exponential backoff
   - Conflict resolution for concurrent uploads

## Examples

See `ejemplo_checkpoint_orchestrator.py` for complete working examples:
- Basic save/load
- Incremental checkpointing
- Encryption
- Retention policies
- Querying metadata
- Cloud backup stubs
- Complete workflow integration

## License

Part of FARFAN 2.0 framework. See main LICENSE file.

## Support

For issues or questions:
- Check test cases in `test_pipeline_checkpoint.py`
- Review examples in `ejemplo_checkpoint_orchestrator.py`
- See main FARFAN documentation in README.md
