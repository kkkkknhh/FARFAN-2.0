#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DI Container Usage Examples
============================

Demonstrates how to use the Dependency Injection Container for:
- Component registration and resolution
- Graceful degradation patterns
- Testing with mocks
- Configuration management
"""

import logging
from pathlib import Path

from infrastructure import (
    DeviceConfig,
    DIContainer,
    IBayesianEngine,
    ICausalBuilder,
    IExtractor,
    configure_container,
)

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ============================================================================
# Example 1: Basic Container Usage
# ============================================================================


def example_basic_usage():
    """Demonstrate basic container registration and resolution"""
    print("\n=== Example 1: Basic Container Usage ===\n")

    # Create a container
    container = DIContainer()

    # Check device configuration
    container.register_singleton(DeviceConfig, lambda: DeviceConfig(device="cpu"))

    device_config = container.resolve(DeviceConfig)
    print(f"Device: {device_config.device}")
    print(f"Using GPU: {device_config.use_gpu}")

    # Verify singleton behavior
    device_config2 = container.resolve(DeviceConfig)
    print(f"Same instance: {device_config is device_config2}")


# ============================================================================
# Example 2: Factory Configuration with Graceful Degradation
# ============================================================================


def example_graceful_degradation():
    """Demonstrate graceful degradation with NLP models"""
    print("\n=== Example 2: Graceful Degradation ===\n")

    # Use the factory configuration
    config = {"use_gpu": False}
    container = configure_container(config)

    # Check what got configured
    if container.is_registered(DeviceConfig):
        device_config = container.resolve(DeviceConfig)
        print(f"Configured device: {device_config.device}")

    # Note: spaCy.Language would be registered if spaCy is installed
    # This demonstrates the fallback chain:
    # 1. Try es_dep_news_trf (transformer)
    # 2. Fall back to es_core_news_lg (large)
    # 3. Fall back to es_core_news_sm (small)
    # 4. Log error if none available


# ============================================================================
# Example 3: Mock Components for Testing
# ============================================================================


class MockPDFProcessor(IExtractor):
    """Mock PDF processor for testing"""

    def __init__(self, config=None):
        self.config = config
        self.extracted_count = 0

    def extract(self, document_path: str) -> dict:
        self.extracted_count += 1
        return {
            "text": f"Mock extraction from {document_path}",
            "pages": 10,
            "tables": [],
        }


class MockGraphBuilder(ICausalBuilder):
    """Mock graph builder for testing"""

    def __init__(self, extractor: IExtractor):
        self.extractor = extractor

    def build_graph(self, extracted_data: dict) -> dict:
        # Use the injected extractor
        return {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "B"), ("B", "C")],
            "source_data": extracted_data,
        }


def example_testing_with_mocks():
    """Demonstrate using DI container for testing"""
    print("\n=== Example 3: Testing with Mocks ===\n")

    # Create container for testing
    container = DIContainer({"env": "test"})

    # Register mock implementations
    container.register_singleton(IExtractor, MockPDFProcessor)
    container.register_transient(ICausalBuilder, MockGraphBuilder)

    # Resolve components
    extractor = container.resolve(IExtractor)
    builder = container.resolve(ICausalBuilder)

    # Use them
    data = extractor.extract("/test/document.pdf")
    print(f"Extracted: {data['text']}")

    graph = builder.build_graph(data)
    print(f"Built graph with nodes: {graph['nodes']}")
    print(
        f"Builder has access to extractor: {isinstance(builder.extractor, MockPDFProcessor)}"
    )


# ============================================================================
# Example 4: Automatic Dependency Resolution
# ============================================================================


class ServiceA:
    """Service with no dependencies"""

    def __init__(self):
        self.name = "ServiceA"

    def do_work(self):
        return f"{self.name} working"


class ServiceB:
    """Service that depends on ServiceA"""

    def __init__(self, service_a: ServiceA):
        self.service_a = service_a
        self.name = "ServiceB"

    def do_work(self):
        a_result = self.service_a.do_work()
        return f"{self.name} using {a_result}"


class ServiceC:
    """Service that depends on both A and B"""

    def __init__(self, service_a: ServiceA, service_b: ServiceB):
        self.service_a = service_a
        self.service_b = service_b
        self.name = "ServiceC"

    def do_work(self):
        b_result = self.service_b.do_work()
        return f"{self.name} orchestrating: {b_result}"


def example_automatic_dependency_injection():
    """Demonstrate automatic dependency resolution"""
    print("\n=== Example 4: Automatic Dependency Injection ===\n")

    container = DIContainer()

    # Register all services
    container.register_singleton(ServiceA, ServiceA)
    container.register_transient(ServiceB, ServiceB)
    container.register_transient(ServiceC, ServiceC)

    # Resolve ServiceC - dependencies are automatically resolved!
    service_c = container.resolve(ServiceC)

    # Verify the dependency chain
    print(f"ServiceC has ServiceB: {isinstance(service_c.service_b, ServiceB)}")
    print(f"ServiceC has ServiceA: {isinstance(service_c.service_a, ServiceA)}")
    print(
        f"ServiceB has ServiceA: {isinstance(service_c.service_b.service_a, ServiceA)}"
    )

    # Execute
    result = service_c.do_work()
    print(f"Result: {result}")

    # Verify singleton vs transient
    service_c2 = container.resolve(ServiceC)
    print(f"ServiceC is transient (different instance): {service_c is not service_c2}")
    print(
        f"ServiceA is singleton (same instance): {service_c.service_a is service_c2.service_a}"
    )


# ============================================================================
# Example 5: Real-world Integration Pattern
# ============================================================================


def example_real_world_integration():
    """Demonstrate real-world integration pattern"""
    print("\n=== Example 5: Real-world Integration ===\n")

    # Simulate a configuration object (like CDAFConfig)
    class AppConfig:
        def __init__(self):
            self.use_gpu = False
            self.nlp_model = "es_core_news_lg"
            self.batch_size = 32
            self.cache_enabled = True

    config = AppConfig()

    # Configure container with the config
    container = configure_container(config)

    # Now your application can resolve components
    device_config = container.resolve(DeviceConfig)

    print(f"Application initialized with:")
    print(f"  - Device: {device_config.device}")
    print(f"  - GPU: {device_config.use_gpu}")

    # In a real application, you would register your actual components:
    # container.register_transient(IExtractor, PDFProcessor)
    # container.register_singleton(IBayesianEngine, BayesianSamplingEngine)

    # And then use them throughout your application:
    # processor = container.resolve(IExtractor)
    # engine = container.resolve(IBayesianEngine)


# ============================================================================
# Main: Run All Examples
# ============================================================================


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("DI Container Usage Examples")
    print("=" * 70)

    example_basic_usage()
    example_graceful_degradation()
    example_testing_with_mocks()
    example_automatic_dependency_injection()
    example_real_world_integration()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
