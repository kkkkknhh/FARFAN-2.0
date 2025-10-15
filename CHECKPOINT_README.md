# Pipeline Checkpoint System

Robust state persistence for FARFAN 2.0 with compression, encryption, and retention management.

## Quick Start

```python
from pipeline_checkpoint import PipelineCheckpoint

# Initialize
checkpoint = PipelineCheckpoint(checkpoint_dir="./checkpoints")

# Save state
state = {'stage': 1, 'data': [...], 'metrics': {...}}
checkpoint_id = checkpoint.save(state, checkpoint_id="stage_1")

# Load state
restored = checkpoint.load(checkpoint_id)
```

## Key Features

✅ **Gzip Compression** - 70-85% size reduction  
✅ **SHA256 Verification** - Integrity checking on load  
✅ **Versioning** - Format compatibility tracking  
✅ **Incremental Checkpoints** - Store only changed fields (50-90% smaller)  
✅ **Fernet Encryption** - Optional symmetric encryption for sensitive data  
✅ **Retention Policies** - Automatic cleanup (keep N recent, delete old)  
✅ **Queryable Index** - Filter and sort by metadata  
✅ **Cloud Backup Stubs** - Ready for AWS S3, Azure Blob, GCP Storage integration  

## Usage Examples

### Incremental Checkpointing
```python
checkpoint = PipelineCheckpoint(
    checkpoint_dir="./checkpoints",
    enable_incremental=True
)

# Full checkpoint
state = {'stage': 1, 'data': [1,2,3]}
checkpoint.save(state, checkpoint_id="full")

# Incremental (only stores changes)
state['stage'] = 2
state['new_field'] = 'value'
checkpoint.save(state, checkpoint_id="incr")  # Automatically detects delta
```

### Encryption for Sensitive Data
```python
from pipeline_checkpoint import generate_encryption_key

key = generate_encryption_key("./encryption.key")

checkpoint = PipelineCheckpoint(
    checkpoint_dir="./checkpoints",
    enable_encryption=True,
    encryption_key=key
)

sensitive = {'financial_data': {...}, 'population_data': {...}}
checkpoint.save(sensitive, checkpoint_id="sensitive")
```

### Retention Policies
```python
from pipeline_checkpoint import RetentionPolicy

# Keep only 10 most recent
checkpoint = PipelineCheckpoint(
    checkpoint_dir="./checkpoints",
    retention_policy=RetentionPolicy.KEEP_N_RECENT,
    retention_count=10
)

# Or delete older than 30 days
checkpoint = PipelineCheckpoint(
    checkpoint_dir="./checkpoints",
    retention_policy=RetentionPolicy.DELETE_OLDER_THAN,
    retention_age_days=30
)
```

### Querying Checkpoints
```python
# List all checkpoints (sorted by timestamp)
checkpoints = checkpoint.list_checkpoints()

# Filter by custom metadata
filtered = checkpoint.list_checkpoints(
    filter_fn=lambda c: c.custom_metadata.get('accuracy', 0) > 0.90
)

# Sort by size
largest = checkpoint.list_checkpoints(sort_by='size_bytes', reverse=True)

# Get statistics
stats = checkpoint.get_statistics()
print(f"Total: {stats['total_checkpoints']}, Size: {stats['total_size_mb']} MB")
```

## Integration with FARFAN Orchestrator

```python
from orchestrator import FARFANOrchestrator
from pipeline_checkpoint import PipelineCheckpoint

orchestrator = FARFANOrchestrator(output_dir="./results")
checkpoint = PipelineCheckpoint(
    checkpoint_dir="./checkpoints",
    enable_incremental=True
)

try:
    # Process with checkpointing after each stage
    ctx = orchestrator._stage_extract_document(ctx)
    checkpoint.save(asdict(ctx), checkpoint_id="stage_1")
    
    ctx = orchestrator._stage_semantic_analysis(ctx)
    checkpoint.save(asdict(ctx), checkpoint_id="stage_2")
    
    # ... continue for all stages
    
except Exception as e:
    # Recover from last checkpoint
    last = checkpoint.list_checkpoints()[0]
    ctx = checkpoint.load(last.checkpoint_id)
    # Resume processing
```

## Files

- **`pipeline_checkpoint.py`** (713 lines) - Main implementation
- **`test_pipeline_checkpoint.py`** (608 lines) - Comprehensive test suite
- **`ejemplo_checkpoint_orchestrator.py`** (428 lines) - Usage examples
- **`CHECKPOINT_SYSTEM_DOCS.md`** (606 lines) - Complete documentation

## Testing

```bash
# Run all tests
python3 -m unittest test_pipeline_checkpoint.py -v

# Output: 33 tests (29 passed, 4 skipped without cryptography)
```

## Requirements

**Core**: Python 3.7+ (no additional dependencies)  
**Encryption**: `pip install cryptography`  

## API Reference

### PipelineCheckpoint
- `save(state, checkpoint_id=None, force_full=False, custom_metadata=None)` → checkpoint_id
- `load(checkpoint_id, verify_hash=True)` → state dict
- `list_checkpoints(sort_by='timestamp', reverse=True, filter_fn=None)` → list
- `delete_checkpoint(checkpoint_id)`
- `get_checkpoint_info(checkpoint_id)` → CheckpointMetadata
- `get_statistics()` → dict

### Cloud Backup (Stub Methods)
- `configure_cloud_backup(config)`
- `sync_to_cloud(checkpoint_ids=None)` - Stub for future AWS/Azure/GCP integration
- `restore_from_cloud(checkpoint_id)` - Stub
- `list_cloud_checkpoints()` - Stub

## Performance

- **Save**: 10-50ms for 1-10MB states
- **Load**: 5-30ms
- **Compression**: 70-85% size reduction
- **Incremental**: 50-90% smaller than full checkpoints

## Cloud Backup Future Implementation

The system includes stub methods ready for cloud provider integration:

```python
from pipeline_checkpoint import CloudBackupConfig, CloudBackupProvider

cloud_config = CloudBackupConfig(
    enabled=True,
    provider=CloudBackupProvider.AWS_S3,
    bucket_name="farfan-checkpoints",
    region="us-east-1",
    auto_sync=True
)

checkpoint = PipelineCheckpoint(
    checkpoint_dir="./checkpoints",
    cloud_backup_config=cloud_config
)

# Stub methods (not yet implemented)
checkpoint.sync_to_cloud(["important_checkpoint"])
```

**To implement**:
1. Add boto3/azure-storage-blob/google-cloud-storage dependencies
2. Implement `_upload_to_s3()`, `_upload_to_azure()`, `_upload_to_gcp()` methods
3. Add background sync thread with retry logic
4. Handle authentication with IAM/SAS/service accounts

## Documentation

See **`CHECKPOINT_SYSTEM_DOCS.md`** for:
- Complete API reference
- Detailed usage examples
- Integration patterns
- Best practices
- Troubleshooting guide
- Future enhancement roadmap

## License

Part of FARFAN 2.0 framework. See main LICENSE file.
