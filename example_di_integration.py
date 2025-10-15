#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DI Container Integration with Existing FARFAN Modules
======================================================

Demonstrates how to integrate the DI Container with existing framework components:
- dereck_beach (CDAF Framework)
- policy_processor (Industrial Policy Processor)
- inference/bayesian_engine (Bayesian Inference)
- extraction/extraction_pipeline (Document Extraction)
"""

import logging
from pathlib import Path
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from infrastructure import (
    DIContainer,
    DeviceConfig,
    configure_container,
    IExtractor,
    ICausalBuilder,
    IBayesianEngine,
)


# ============================================================================
# Integration Example 1: Configuration-based Setup
# ============================================================================

def example_config_integration():
    """
    Demonstrate integration with CDAFConfigSchema-style configuration.
    
    This shows how the DI container can work with the existing Pydantic
    configuration system used in dereck_beach.
    """
    print("\n=== Integration Example 1: Configuration Setup ===\n")
    
    # Simulate the CDAFConfigSchema structure
    @dataclass
    class MockCDAFConfig:
        use_gpu: bool = False
        nlp_model: str = 'es_core_news_lg'
        cache_embeddings: bool = True
        max_context_length: int = 1000
    
    config = MockCDAFConfig(use_gpu=False)
    
    # Configure container with the config
    container = configure_container(config)
    
    # The container can now be passed to modules that need it
    print(f"Container configured with config: {config}")
    
    # Resolve device config
    device_config = container.resolve(DeviceConfig)
    print(f"Device configuration: {device_config.device} (GPU: {device_config.use_gpu})")
    
    return container


# ============================================================================
# Integration Example 2: Wrapping Existing Components
# ============================================================================

class PolicyProcessorAdapter(IExtractor):
    """
    Adapter for policy_processor.IndustrialPolicyProcessor
    
    This wraps the existing processor to conform to IExtractor interface,
    enabling DI container integration without modifying the original code.
    """
    
    def __init__(self, config=None):
        # In real implementation, import and initialize:
        # from policy_processor import IndustrialPolicyProcessor, ProcessorConfig
        # self.processor = IndustrialPolicyProcessor(ProcessorConfig.from_legacy(**config))
        self.config = config
        print(f"PolicyProcessorAdapter initialized with config: {config}")
    
    def extract(self, document_path: str) -> dict:
        """Extract policy data from document"""
        # In real implementation:
        # return self.processor.process(text)
        return {
            'adapter': 'PolicyProcessorAdapter',
            'document': document_path,
            'mock_result': 'Would call IndustrialPolicyProcessor.process()'
        }


class BayesianEngineAdapter(IBayesianEngine):
    """
    Adapter for inference.bayesian_engine components
    
    Wraps BayesianSamplingEngine, BayesianPriorBuilder, etc.
    """
    
    def __init__(self, config=None):
        # In real implementation:
        # from inference.bayesian_engine import BayesianSamplingEngine, SamplingConfig
        # self.engine = BayesianSamplingEngine(SamplingConfig(**config))
        self.config = config
        print(f"BayesianEngineAdapter initialized")
    
    def infer(self, graph: dict) -> dict:
        """Perform Bayesian inference"""
        # In real implementation:
        # return self.engine.sample(...)
        return {
            'adapter': 'BayesianEngineAdapter',
            'mock_inference': 'Would call BayesianSamplingEngine.sample()',
            'graph_nodes': len(graph.get('nodes', []))
        }


def example_adapter_pattern():
    """Demonstrate adapter pattern for existing components"""
    print("\n=== Integration Example 2: Adapter Pattern ===\n")
    
    container = DIContainer({'env': 'integration'})
    
    # Register adapters
    container.register_transient(IExtractor, PolicyProcessorAdapter)
    container.register_singleton(IBayesianEngine, BayesianEngineAdapter)
    
    # Use them through the container
    extractor = container.resolve(IExtractor)
    engine = container.resolve(IBayesianEngine)
    
    # Execute workflow
    data = extractor.extract('/path/to/pdm.pdf')
    print(f"Extracted: {data}")
    
    inference = engine.infer({'nodes': ['A', 'B', 'C'], 'edges': []})
    print(f"Inference: {inference}")


# ============================================================================
# Integration Example 3: Factory Functions for Complex Setup
# ============================================================================

def create_cdaf_framework(config, output_dir: Path, log_level: str = "INFO"):
    """
    Factory function for CDAFFramework initialization.
    
    This demonstrates how to create factory functions for components
    that require complex initialization.
    """
    # In real implementation:
    # from dereck_beach import CDAFFramework
    # return CDAFFramework(config, output_dir, log_level)
    
    print(f"Creating CDAF Framework:")
    print(f"  - Output dir: {output_dir}")
    print(f"  - Log level: {log_level}")
    
    return {
        'framework': 'CDAF',
        'initialized': True,
        'output_dir': str(output_dir),
        'log_level': log_level
    }


class ICDAFFramework:
    """Interface for CDAF Framework"""
    pass


def example_factory_functions():
    """Demonstrate factory function registration"""
    print("\n=== Integration Example 3: Factory Functions ===\n")
    
    container = DIContainer()
    
    # Register with factory function
    output_dir = Path('./output')
    container.register_singleton(
        ICDAFFramework,  # Using proper interface
        lambda: create_cdaf_framework(
            config={'test': True},
            output_dir=output_dir,
            log_level='DEBUG'
        )
    )
    
    # Resolve
    framework = container.resolve(ICDAFFramework)
    print(f"Framework created: {framework}")


# ============================================================================
# Integration Example 4: Dependency Chain with Real Components
# ============================================================================

class OrchestratorWithDI:
    """
    Example orchestrator using DI for all dependencies.
    
    This shows how a high-level orchestrator can depend on multiple
    components injected via DI container.
    """
    
    def __init__(
        self,
        extractor: IExtractor,
        engine: IBayesianEngine,
        device_config: DeviceConfig
    ):
        self.extractor = extractor
        self.engine = engine
        self.device_config = device_config
        
        print(f"Orchestrator initialized on device: {device_config.device}")
    
    def analyze_document(self, document_path: str) -> dict:
        """Complete analysis workflow"""
        print(f"\nAnalyzing document: {document_path}")
        
        # Step 1: Extract
        extracted_data = self.extractor.extract(document_path)
        print(f"  1. Extracted data")
        
        # Step 2: Build graph (would use ICausalBuilder in real implementation)
        graph = {
            'nodes': extracted_data.get('nodes', []),
            'edges': extracted_data.get('edges', [])
        }
        print(f"  2. Built graph")
        
        # Step 3: Infer
        inference = self.engine.infer(graph)
        print(f"  3. Performed inference")
        
        return {
            'document': document_path,
            'extracted': extracted_data,
            'graph': graph,
            'inference': inference,
            'device': self.device_config.device
        }


def example_dependency_chain():
    """Demonstrate complex dependency chain"""
    print("\n=== Integration Example 4: Dependency Chain ===\n")
    
    config = {'use_gpu': False}
    container = configure_container(config)
    
    # Register components
    container.register_transient(IExtractor, PolicyProcessorAdapter)
    container.register_singleton(IBayesianEngine, BayesianEngineAdapter)
    container.register_transient(OrchestratorWithDI, OrchestratorWithDI)
    
    # Resolve orchestrator - all dependencies injected automatically!
    orchestrator = container.resolve(OrchestratorWithDI)
    
    # Use it
    result = orchestrator.analyze_document('/test/pdm_municipality.pdf')
    print(f"\nAnalysis complete: {result.keys()}")


# ============================================================================
# Integration Example 5: Environment-specific Configuration
# ============================================================================

def configure_for_environment(env: str) -> DIContainer:
    """
    Configure DI container based on environment.
    
    This demonstrates how to have different configurations for:
    - Development (fast, with mocks)
    - Testing (deterministic, with mocks)
    - Production (real components, optimized)
    """
    print(f"\n=== Configuring for environment: {env} ===\n")
    
    if env == 'development':
        # Fast startup, minimal dependencies
        container = DIContainer({'env': 'development'})
        container.register_transient(IExtractor, PolicyProcessorAdapter)
        container.register_singleton(DeviceConfig, lambda: DeviceConfig(device='cpu'))
        print("Development config: Using lightweight components")
    
    elif env == 'testing':
        # Deterministic, with mocks
        container = DIContainer({'env': 'testing'})
        
        # Use simple mocks
        class MockExtractor(IExtractor):
            def extract(self, document_path: str):
                return {'mock': 'data', 'deterministic': True}
        
        container.register_singleton(IExtractor, MockExtractor)
        container.register_singleton(DeviceConfig, lambda: DeviceConfig(device='cpu'))
        print("Testing config: Using mocks for deterministic tests")
    
    elif env == 'production':
        # Real components, optimized
        config = {'use_gpu': True, 'cache_embeddings': True}
        container = configure_container(config)
        container.register_transient(IExtractor, PolicyProcessorAdapter)
        container.register_singleton(IBayesianEngine, BayesianEngineAdapter)
        print("Production config: Using real components with GPU support")
    
    else:
        raise ValueError(f"Unknown environment: {env}")
    
    return container


def example_environment_config():
    """Demonstrate environment-specific configuration"""
    print("\n=== Integration Example 5: Environment Configuration ===\n")
    
    # Test each environment
    for env in ['development', 'testing', 'production']:
        container = configure_for_environment(env)
        
        if container.is_registered(IExtractor):
            extractor = container.resolve(IExtractor)
            result = extractor.extract('/test/doc.pdf')
            print(f"  -> Extracted: {list(result.keys())}\n")


# ============================================================================
# Main: Run All Integration Examples
# ============================================================================

def main():
    """Run all integration examples"""
    print("\n" + "="*70)
    print("DI Container Integration with Existing FARFAN Modules")
    print("="*70)
    
    example_config_integration()
    example_adapter_pattern()
    example_factory_functions()
    example_dependency_chain()
    example_environment_config()
    
    print("\n" + "="*70)
    print("All integration examples completed successfully!")
    print("="*70 + "\n")
    
    print("\nNext Steps for Real Integration:")
    print("1. Uncomment imports in adapter classes")
    print("2. Replace mock implementations with real components")
    print("3. Add container to main application entry point")
    print("4. Update tests to use DI for better isolation")
    print("5. Gradually migrate existing code to use DI")


if __name__ == '__main__':
    main()
