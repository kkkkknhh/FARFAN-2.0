#!/usr/bin/env python3
"""
Memory Benchmark: Streaming vs Batch Processing
================================================
Compares memory consumption of streaming evidence pipeline
against batch processing approach.
"""

import asyncio
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from choreography.event_bus import EventBus, PDMEvent
    from choreography.evidence_stream import (
        EvidenceStream,
        MechanismPrior,
        PosteriorDistribution,
        StreamingBayesianUpdater,
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Required modules not available: {e}")
    print("Install dependencies: pip install -r requirements.txt")
    MODULES_AVAILABLE = False
    sys.exit(1)


def create_test_chunks(count: int) -> List[Dict]:
    """Create test evidence chunks."""
    return [
        {
            "chunk_id": f"chunk_{i}",
            "content": f"educación calidad evidencia prueba mecanismo {i % 50}",
            "embedding": None,
            "metadata": {"source": f"page_{i // 10}"},
            "pdq_context": None,
            "token_count": 15,
            "position": (i * 100, (i + 1) * 100),
        }
        for i in range(count)
    ]


async def benchmark_streaming_approach(chunks: List[Dict]) -> Dict[str, Any]:
    """Benchmark streaming processing with EvidenceStream."""
    print(f"  Running streaming approach ({len(chunks)} chunks)...")
    
    # Start memory tracking
    tracemalloc.start()
    start_time = time.time()
    start_mem = tracemalloc.get_traced_memory()[0]
    
    # Setup
    event_bus = EventBus()
    updater = StreamingBayesianUpdater(event_bus=None)  # Disable events for fair comparison
    stream = EvidenceStream(chunks, batch_size=1)
    prior = MechanismPrior(
        mechanism_name="educación",
        prior_mean=0.5,
        prior_std=0.2,
        confidence=0.5
    )
    
    # Process stream
    posterior = await updater.update_from_stream(stream, prior, run_id="streaming_bench")
    
    # Measure
    end_time = time.time()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "method": "streaming",
        "chunks": len(chunks),
        "start_memory_mb": start_mem / 1024 / 1024,
        "current_memory_mb": current_mem / 1024 / 1024,
        "peak_memory_mb": peak_mem / 1024 / 1024,
        "memory_delta_mb": (peak_mem - start_mem) / 1024 / 1024,
        "elapsed_seconds": end_time - start_time,
        "final_posterior_mean": posterior.posterior_mean,
        "final_posterior_std": posterior.posterior_std,
        "evidence_count": posterior.evidence_count,
    }


async def benchmark_batch_approach(chunks: List[Dict]) -> Dict[str, Any]:
    """Benchmark batch processing (load all into memory)."""
    print(f"  Running batch approach ({len(chunks)} chunks)...")
    
    # Start memory tracking
    tracemalloc.start()
    start_time = time.time()
    start_mem = tracemalloc.get_traced_memory()[0]
    
    # Setup
    updater = StreamingBayesianUpdater(event_bus=None)
    prior = MechanismPrior(
        mechanism_name="educación",
        prior_mean=0.5,
        prior_std=0.2,
        confidence=0.5
    )
    
    # Load ALL chunks into memory at once (batch approach)
    all_chunks_in_memory = list(chunks)  # Creates full copy in memory
    
    # Initialize posterior
    current_posterior = PosteriorDistribution(
        mechanism_name=prior.mechanism_name,
        posterior_mean=prior.prior_mean,
        posterior_std=prior.prior_std,
        evidence_count=0,
    )
    
    evidence_count = 0
    
    # Process all chunks sequentially (but all loaded in memory)
    for chunk in all_chunks_in_memory:
        if await updater._is_relevant(chunk, prior.mechanism_name):
            likelihood = await updater._compute_likelihood(chunk, prior.mechanism_name)
            current_posterior = updater._bayesian_update(current_posterior, likelihood)
            evidence_count += 1
            current_posterior.evidence_count = evidence_count
    
    # Measure
    end_time = time.time()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "method": "batch",
        "chunks": len(chunks),
        "start_memory_mb": start_mem / 1024 / 1024,
        "current_memory_mb": current_mem / 1024 / 1024,
        "peak_memory_mb": peak_mem / 1024 / 1024,
        "memory_delta_mb": (peak_mem - start_mem) / 1024 / 1024,
        "elapsed_seconds": end_time - start_time,
        "final_posterior_mean": current_posterior.posterior_mean,
        "final_posterior_std": current_posterior.posterior_std,
        "evidence_count": evidence_count,
    }


def print_section(title: str, width: int = 80):
    """Print formatted section."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_benchmark_result(result: Dict[str, Any]):
    """Print benchmark results."""
    print(f"\nMethod: {result['method'].upper()}")
    print(f"  Chunks Processed: {result['chunks']}")
    print(f"  Evidence Incorporated: {result['evidence_count']}")
    print(f"  Memory Delta: {result['memory_delta_mb']:.2f} MB")
    print(f"  Peak Memory: {result['peak_memory_mb']:.2f} MB")
    print(f"  Elapsed Time: {result['elapsed_seconds']:.3f}s")
    print(f"  Final Posterior Mean: {result['final_posterior_mean']:.4f}")
    print(f"  Final Posterior Std: {result['final_posterior_std']:.4f}")


async def run_benchmarks():
    """Run comprehensive memory benchmarks."""
    print_section("MEMORY BENCHMARK: STREAMING VS BATCH PROCESSING")
    
    chunk_sizes = [100, 500, 1000]
    
    all_results = []
    
    for chunk_count in chunk_sizes:
        print(f"\n{'─' * 80}")
        print(f"Testing with {chunk_count} chunks")
        print(f"{'─' * 80}")
        
        # Create test data
        chunks = create_test_chunks(chunk_count)
        
        # Run streaming
        streaming_result = await benchmark_streaming_approach(chunks)
        print_benchmark_result(streaming_result)
        
        # Small delay to let GC clean up
        await asyncio.sleep(0.5)
        
        # Run batch
        batch_result = await benchmark_batch_approach(chunks)
        print_benchmark_result(batch_result)
        
        # Compare
        memory_saving = batch_result['memory_delta_mb'] - streaming_result['memory_delta_mb']
        memory_saving_pct = (memory_saving / batch_result['memory_delta_mb'] * 100) if batch_result['memory_delta_mb'] > 0 else 0
        
        time_diff = streaming_result['elapsed_seconds'] - batch_result['elapsed_seconds']
        time_diff_pct = (time_diff / batch_result['elapsed_seconds'] * 100) if batch_result['elapsed_seconds'] > 0 else 0
        
        comparison = {
            "chunk_count": chunk_count,
            "streaming": streaming_result,
            "batch": batch_result,
            "memory_saving_mb": memory_saving,
            "memory_saving_percent": memory_saving_pct,
            "time_overhead_seconds": time_diff,
            "time_overhead_percent": time_diff_pct,
        }
        
        all_results.append(comparison)
        
        print(f"\n  📊 Comparison:")
        print(f"     Memory Saving: {memory_saving:.2f} MB ({memory_saving_pct:+.1f}%)")
        print(f"     Time Overhead: {time_diff:+.3f}s ({time_diff_pct:+.1f}%)")
        
        if memory_saving > 0:
            print(f"     ✓ Streaming uses LESS memory")
        else:
            print(f"     ⚠️  Streaming uses MORE memory")
    
    # Summary
    print_section("SUMMARY")
    
    print("\n📈 Results by Dataset Size:\n")
    print(f"{'Chunks':<10} {'Memory Δ (MB)':<15} {'Time Δ (s)':<15} {'Efficiency':<20}")
    print(f"{'-' * 70}")
    
    for result in all_results:
        chunks = result['chunk_count']
        mem_saving = result['memory_saving_mb']
        time_overhead = result['time_overhead_seconds']
        
        if mem_saving > 0 and time_overhead < 0.5:
            efficiency = "✓ Streaming Better"
        elif mem_saving > 0:
            efficiency = "~ Streaming Saves Mem"
        elif time_overhead < 0:
            efficiency = "~ Streaming Faster"
        else:
            efficiency = "⚠️  Batch Better"
        
        print(f"{chunks:<10} {mem_saving:>+14.2f} {time_overhead:>+14.3f} {efficiency:<20}")
    
    print(f"\n{'─' * 80}")
    
    avg_mem_saving = sum(r['memory_saving_mb'] for r in all_results) / len(all_results)
    avg_time_overhead = sum(r['time_overhead_seconds'] for r in all_results) / len(all_results)
    
    print(f"\nAverage Memory Saving: {avg_mem_saving:+.2f} MB")
    print(f"Average Time Overhead: {avg_time_overhead:+.3f}s")
    
    print("\n🔍 Analysis:")
    
    if avg_mem_saving > 0:
        print(f"  ✓ Streaming approach is more memory-efficient on average")
        print(f"    Saves ~{avg_mem_saving:.1f} MB compared to batch processing")
    else:
        print(f"  ⚠️  Streaming approach uses more memory on average")
        print(f"    Uses ~{abs(avg_mem_saving):.1f} MB more than batch processing")
    
    if avg_time_overhead < 0.1:
        print(f"  ✓ Streaming has minimal time overhead (~{abs(avg_time_overhead):.3f}s)")
    else:
        print(f"  ⚠️  Streaming has noticeable time overhead (~{avg_time_overhead:.3f}s)")
    
    print(f"\n💡 Recommendation:")
    
    if avg_mem_saving > 0.5:
        print(f"  Use STREAMING approach for large documents to save memory")
        print(f"  Memory savings scale with document size")
    elif avg_mem_saving > 0 and avg_time_overhead < 0.2:
        print(f"  Use STREAMING approach for balanced performance")
    else:
        print(f"  Current implementation shows minimal difference between approaches")
        print(f"  Consider profiling with larger datasets (10k+ chunks)")
    
    print_section("BENCHMARK COMPLETE")


if __name__ == "__main__":
    try:
        asyncio.run(run_benchmarks())
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
