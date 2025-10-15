#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Example: Resource Pool with Existing Bayesian Engine
Shows how to integrate the Resource Pool Manager with the existing BayesianSamplingEngine
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from infrastructure import ResourceConfig, ResourcePool


class BayesianEngineWithResourcePool:
    """
    Wrapper for existing BayesianSamplingEngine with resource pool integration.
    
    This demonstrates how to integrate the Resource Pool Manager with the
    existing inference/bayesian_engine.py module.
    """
    
    def __init__(self, resource_pool: ResourcePool):
        """
        Initialize with resource pool.
        
        Args:
            resource_pool: ResourcePool instance for worker management
        """
        self.resource_pool = resource_pool
        
        # Import existing Bayesian engine
        try:
            from inference.bayesian_engine import BayesianSamplingEngine
            self.sampling_engine = BayesianSamplingEngine()
        except ImportError:
            print("Warning: Could not import BayesianSamplingEngine, using mock")
            self.sampling_engine = None
    
    async def infer_mechanism_with_resources(self, link, context):
        """
        Run mechanism inference with resource management.
        
        This wraps the existing BayesianSamplingEngine.sample() method
        with resource pool management for timeout and memory limits.
        
        Args:
            link: CausalLink object
            context: ColombianMunicipalContext object
            
        Returns:
            PosteriorDistribution from Bayesian sampling
        """
        task_id = f"infer_{link.cause_id}_{link.effect_id}"
        
        async with self.resource_pool.acquire_worker(task_id) as worker:
            # Worker is now assigned and monitored
            
            if self.sampling_engine:
                # Call existing Bayesian engine
                # Note: The existing engine's sample() method is synchronous,
                # so we run it in an executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self.sampling_engine.sample,
                    link,
                    context
                )
                return result
            else:
                # Mock result for demonstration
                return {
                    'posterior_mean': 0.75,
                    'posterior_std': 0.1,
                    'device': worker.device,
                    'worker_id': worker.id
                }


async def demo_integration():
    """Demonstrate integration with existing Bayesian engine"""
    print("\n" + "=" * 70)
    print("  Integration: Resource Pool + Existing Bayesian Engine")
    print("=" * 70)
    
    # Configure resource pool
    config = ResourceConfig(
        max_workers=2,
        worker_timeout_secs=300,  # 5 minutes per inference
        worker_memory_mb=4096,    # 4GB per worker
        devices=["cpu", "cpu"]
    )
    
    pool = ResourcePool(config)
    engine = BayesianEngineWithResourcePool(pool)
    
    print(f"\n✓ Integrated engine created with {config.max_workers} workers")
    print(f"  Timeout: {config.worker_timeout_secs}s")
    print(f"  Memory limit: {config.worker_memory_mb}MB")
    
    # Create mock link (would normally come from extraction pipeline)
    from dataclasses import dataclass
    
    @dataclass
    class MockLink:
        cause_id: str
        effect_id: str
        cause_emb: list = None
        effect_emb: list = None
        
        def __post_init__(self):
            if self.cause_emb is None:
                self.cause_emb = [0.1] * 384  # Mock embedding
            if self.effect_emb is None:
                self.effect_emb = [0.2] * 384  # Mock embedding
    
    @dataclass
    class MockContext:
        overall_pdm_embedding: list = None
        municipality_name: str = "Ejemplo"
        
        def __post_init__(self):
            if self.overall_pdm_embedding is None:
                self.overall_pdm_embedding = [0.15] * 384  # Mock embedding
    
    # Create test data
    links = [
        MockLink("programa_1", "resultado_1"),
        MockLink("programa_2", "resultado_2"),
    ]
    context = MockContext()
    
    print("\nRunning inference with resource management:")
    for link in links:
        result = await engine.infer_mechanism_with_resources(link, context)
        print(f"  {link.cause_id} → {link.effect_id}: {result}")
    
    # Check pool status
    status = pool.get_pool_status()
    print(f"\n✓ Pool status after inference: {status}")
    
    print("\n" + "=" * 70)
    print("  Integration Complete!")
    print("=" * 70)


if __name__ == '__main__':
    asyncio.run(demo_integration())
