# Resource Pool Manager (F4.3)

## Overview

The Resource Pool Manager provides GPU/CPU resource management with timeout and memory limit enforcement, preventing memory exhaustion and ensuring system stability under load.

## Key Features

- **Worker Pool Management**: Pre-populated pool of workers with configurable size
- **Async Context Manager**: Safe resource acquisition with automatic cleanup
- **Timeout Enforcement**: Automatic worker termination when timeout is exceeded
- **Memory Limit Enforcement**: Monitor and enforce memory limits per worker
- **Task Tracking**: Track active tasks with audit trail
- **Governance Standard Compliance**: Implements resource governance standards

## Architecture

```
ResourcePool
├── ResourceConfig: Configuration for pool settings
├── Worker: Individual computational worker with limits
├── ResourcePool: Manages worker pool and monitoring
└── BayesianInferenceEngine: Integration with Bayesian inference
```

## Usage

### Basic Configuration

```python
from infrastructure import ResourceConfig, ResourcePool

# Create configuration
config = ResourceConfig(
    max_workers=4,
    worker_timeout_secs=300,
    worker_memory_mb=2048,
    devices=["cpu", "cpu", "cuda:0", "cuda:1"]
)

# Initialize pool
pool = ResourcePool(config)
```

### Worker Acquisition

```python
async with pool.acquire_worker("task_id") as worker:
    # Worker is automatically monitored for timeout and memory
    result = await worker.run_mcmc_sampling(link)
    # Worker is automatically returned to pool
```

### Integration with Bayesian Inference

```python
from infrastructure import BayesianInferenceEngine

# Create engine with resource pool
engine = BayesianInferenceEngine(pool)

# Run inference with automatic resource management
result = await engine.infer_mechanism(causal_link)
```

### Monitoring Pool Status

```python
status = pool.get_pool_status()
print(f"Available workers: {status['available_workers']}/{status['total_workers']}")
print(f"Active tasks: {status['active_tasks']}")
print(f"Task IDs: {status['active_task_ids']}")
```

## Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `max_workers` | int | Maximum number of workers in pool | Required |
| `worker_timeout_secs` | int | Timeout in seconds per worker | Required |
| `worker_memory_mb` | int | Memory limit in MB per worker | Required |
| `devices` | List[str] | Device assignments (e.g., ["cpu", "cuda:0"]) | `["cpu"]` |

## Error Handling

### WorkerTimeoutError

Raised when a worker exceeds the configured timeout:

```python
from infrastructure import WorkerTimeoutError

try:
    async with pool.acquire_worker("long_task") as worker:
        await long_running_task(worker)
except WorkerTimeoutError as e:
    logger.error(f"Task timed out: {e}")
    # Handle timeout gracefully
```

### WorkerMemoryError

Raised when a worker exceeds the configured memory limit:

```python
from infrastructure import WorkerMemoryError

try:
    async with pool.acquire_worker("memory_intensive_task") as worker:
        await memory_intensive_task(worker)
except WorkerMemoryError as e:
    logger.error(f"Memory limit exceeded: {e}")
    # Handle memory error gracefully
```

## Governance Standard Compliance

The Resource Pool Manager implements the following governance standards:

1. **Timeout Enforcement**: Workers are automatically terminated if they exceed the configured timeout
2. **Memory Monitoring**: Continuous monitoring of worker memory usage
3. **Resource Limits**: Hard limits on worker resources to prevent exhaustion
4. **Audit Trail**: All worker acquisitions and releases are logged
5. **Graceful Degradation**: System continues operating even if individual workers fail

## Examples

### Example 1: Basic Usage

```python
async def basic_example():
    config = ResourceConfig(
        max_workers=3,
        worker_timeout_secs=60,
        worker_memory_mb=1024
    )
    
    pool = ResourcePool(config)
    
    async with pool.acquire_worker("example_task") as worker:
        # Use worker for computation
        result = await worker.run_mcmc_sampling(link)
```

### Example 2: Concurrent Inference

```python
async def concurrent_inference():
    config = ResourceConfig(
        max_workers=4,
        worker_timeout_secs=120,
        worker_memory_mb=2048
    )
    
    pool = ResourcePool(config)
    engine = BayesianInferenceEngine(pool)
    
    # Process multiple links concurrently
    results = await asyncio.gather(
        *[engine.infer_mechanism(link) for link in causal_links]
    )
```

### Example 3: Pool Monitoring

```python
async def monitor_pool():
    while True:
        await asyncio.sleep(5)
        status = pool.get_pool_status()
        logger.info(
            f"Pool Status - Available: {status['available_workers']}, "
            f"Active: {status['active_tasks']}"
        )
```

## Testing

Run the comprehensive test suite:

```bash
python test_resource_pool.py
```

Run the example demonstrations:

```bash
python example_resource_pool.py
```

## Implementation Details

### Worker Monitoring

Each worker is monitored in a background task that:
- Checks elapsed time every second
- Monitors memory usage
- Enforces timeout limits
- Enforces memory limits
- Logs periodic status updates

### Resource Cleanup

The async context manager ensures:
- Monitoring tasks are cancelled
- Workers are returned to pool
- Active task tracking is updated
- Resources are cleaned up even on exceptions

### Thread Safety

The resource pool uses asyncio primitives for thread safety:
- `asyncio.Queue` for worker pool
- Async context managers for safe acquisition
- Proper cleanup in finally blocks

## Benefits

1. **Prevents Memory Exhaustion**: Hard memory limits prevent runaway processes
2. **Implements Timeouts**: Automatic termination of long-running tasks
3. **Maintains System Stability**: Pool-based resource management prevents overload
4. **Governance Compliance**: Implements standard resource governance policies
5. **Easy Integration**: Simple async context manager interface
6. **Comprehensive Monitoring**: Full visibility into resource usage

## Future Enhancements

Potential improvements for future versions:

- GPU memory monitoring with CUDA integration
- Dynamic pool sizing based on load
- Resource usage metrics and reporting
- Integration with distributed task queues
- Advanced scheduling algorithms
- Resource reservation system

## See Also

- `test_resource_pool.py`: Comprehensive test suite
- `example_resource_pool.py`: Usage examples and demonstrations
- `infrastructure/resource_pool.py`: Implementation source code
