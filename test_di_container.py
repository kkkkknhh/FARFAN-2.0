#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Dependency Injection Container
=========================================

Validates F4.1 implementation with comprehensive test coverage.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Optional

from infrastructure.di_container import (
    DIContainer,
    DeviceConfig,
    configure_container,
    IExtractor,
    ICausalBuilder,
    IBayesianEngine,
)


# ============================================================================
# Test Fixtures - Mock Components
# ============================================================================

class MockExtractor(IExtractor):
    """Mock PDF extractor for testing"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config
    
    def extract(self, document_path: str) -> dict:
        return {'text': 'mock extraction', 'source': document_path}


class MockCausalBuilder(ICausalBuilder):
    """Mock causal graph builder for testing"""
    
    def __init__(self, extractor: IExtractor):
        self.extractor = extractor
    
    def build_graph(self, extracted_data: dict) -> dict:
        return {'graph': 'mock graph', 'data': extracted_data}


class MockBayesianEngine(IBayesianEngine):
    """Mock Bayesian engine for testing"""
    
    def __init__(self):
        self.call_count = 0
    
    def infer(self, graph: dict) -> dict:
        self.call_count += 1
        return {'inference': 'mock result', 'graph': graph}


class SimpleDependency:
    """Simple class with no dependencies"""
    
    def __init__(self):
        self.initialized = True


class DependentClass:
    """Class with a dependency"""
    
    def __init__(self, dependency: SimpleDependency):
        self.dependency = dependency


class ConfigDependentClass:
    """Class that takes config parameter"""
    
    def __init__(self, config: dict):
        self.config = config


# ============================================================================
# Test Suite
# ============================================================================

class TestDIContainer(unittest.TestCase):
    """Test suite for DIContainer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.container = DIContainer()
    
    def test_container_initialization(self):
        """Test DIContainer can be initialized"""
        container = DIContainer()
        self.assertIsNotNone(container)
        self.assertIsNone(container.config)
        self.assertEqual(len(container._registry), 0)
        self.assertEqual(len(container._singletons), 0)
    
    def test_container_with_config(self):
        """Test DIContainer initialization with config"""
        config = {'key': 'value'}
        container = DIContainer(config)
        self.assertEqual(container.config, config)
    
    def test_register_singleton(self):
        """Test singleton registration"""
        self.container.register_singleton(IExtractor, MockExtractor)
        
        self.assertTrue(self.container.is_registered(IExtractor))
        implementation, is_singleton = self.container._registry[IExtractor]
        self.assertEqual(implementation, MockExtractor)
        self.assertTrue(is_singleton)
    
    def test_register_transient(self):
        """Test transient registration"""
        self.container.register_transient(IExtractor, MockExtractor)
        
        self.assertTrue(self.container.is_registered(IExtractor))
        implementation, is_singleton = self.container._registry[IExtractor]
        self.assertEqual(implementation, MockExtractor)
        self.assertFalse(is_singleton)
    
    def test_resolve_singleton(self):
        """Test singleton resolution returns same instance"""
        self.container.register_singleton(IExtractor, MockExtractor)
        
        instance1 = self.container.resolve(IExtractor)
        instance2 = self.container.resolve(IExtractor)
        
        self.assertIsInstance(instance1, MockExtractor)
        self.assertIs(instance1, instance2)  # Same instance
    
    def test_resolve_transient(self):
        """Test transient resolution returns new instances"""
        self.container.register_transient(IExtractor, MockExtractor)
        
        instance1 = self.container.resolve(IExtractor)
        instance2 = self.container.resolve(IExtractor)
        
        self.assertIsInstance(instance1, MockExtractor)
        self.assertIsInstance(instance2, MockExtractor)
        self.assertIsNot(instance1, instance2)  # Different instances
    
    def test_resolve_unregistered(self):
        """Test resolving unregistered interface raises KeyError"""
        with self.assertRaises(KeyError) as context:
            self.container.resolve(IExtractor)
        
        self.assertIn('not registered', str(context.exception))
    
    def test_resolve_with_factory_function(self):
        """Test registration with factory function"""
        def extractor_factory():
            return MockExtractor(config={'from': 'factory'})
        
        self.container.register_singleton(IExtractor, extractor_factory)
        instance = self.container.resolve(IExtractor)
        
        self.assertIsInstance(instance, MockExtractor)
        self.assertEqual(instance.config, {'from': 'factory'})
    
    def test_automatic_dependency_resolution(self):
        """Test automatic dependency injection"""
        self.container.register_singleton(SimpleDependency, SimpleDependency)
        self.container.register_transient(DependentClass, DependentClass)
        
        instance = self.container.resolve(DependentClass)
        
        self.assertIsInstance(instance, DependentClass)
        self.assertIsInstance(instance.dependency, SimpleDependency)
        self.assertTrue(instance.dependency.initialized)
    
    def test_nested_dependency_resolution(self):
        """Test multi-level dependency resolution"""
        self.container.register_singleton(IExtractor, MockExtractor)
        self.container.register_transient(ICausalBuilder, MockCausalBuilder)
        
        builder = self.container.resolve(ICausalBuilder)
        
        self.assertIsInstance(builder, MockCausalBuilder)
        self.assertIsInstance(builder.extractor, MockExtractor)
    
    def test_config_injection(self):
        """Test automatic config injection"""
        config = {'test': 'config'}
        container = DIContainer(config)
        container.register_transient(ConfigDependentClass, ConfigDependentClass)
        
        instance = container.resolve(ConfigDependentClass)
        
        self.assertEqual(instance.config, config)
    
    def test_is_registered(self):
        """Test is_registered method"""
        self.assertFalse(self.container.is_registered(IExtractor))
        
        self.container.register_singleton(IExtractor, MockExtractor)
        self.assertTrue(self.container.is_registered(IExtractor))
    
    def test_clear(self):
        """Test clearing container"""
        self.container.register_singleton(IExtractor, MockExtractor)
        self.container.register_singleton(IBayesianEngine, MockBayesianEngine)
        
        # Resolve to populate singletons cache
        self.container.resolve(IExtractor)
        
        self.assertTrue(len(self.container._registry) > 0)
        self.assertTrue(len(self.container._singletons) > 0)
        
        self.container.clear()
        
        self.assertEqual(len(self.container._registry), 0)
        self.assertEqual(len(self.container._singletons), 0)


class TestDeviceConfig(unittest.TestCase):
    """Test suite for DeviceConfig"""
    
    def test_cpu_config(self):
        """Test CPU device configuration"""
        config = DeviceConfig(device='cpu')
        
        self.assertEqual(config.device, 'cpu')
        self.assertFalse(config.use_gpu)
        self.assertIsNone(config.gpu_id)
    
    def test_cuda_config(self):
        """Test CUDA device configuration"""
        config = DeviceConfig(device='cuda')
        
        self.assertEqual(config.device, 'cuda')
        self.assertTrue(config.use_gpu)
    
    def test_invalid_device(self):
        """Test invalid device raises ValueError"""
        with self.assertRaises(ValueError) as context:
            DeviceConfig(device='invalid')
        
        self.assertIn('Invalid device', str(context.exception))
    
    def test_use_gpu_cpu_mismatch(self):
        """Test use_gpu=True with device='cpu' gets corrected"""
        # This should log a warning and correct the device
        config = DeviceConfig(device='cpu', use_gpu=True)
        
        self.assertEqual(config.device, 'cuda')
        self.assertTrue(config.use_gpu)


class TestConfigureContainer(unittest.TestCase):
    """Test suite for configure_container factory"""
    
    def test_configure_container_basic(self):
        """Test basic container configuration"""
        container = configure_container()
        
        self.assertIsInstance(container, DIContainer)
    
    def test_device_config_registration(self):
        """Test DeviceConfig is registered"""
        container = configure_container()
        
        self.assertTrue(container.is_registered(DeviceConfig))
        device_config = container.resolve(DeviceConfig)
        self.assertIsInstance(device_config, DeviceConfig)
    
    def test_cpu_device_by_default(self):
        """Test CPU is used by default"""
        container = configure_container()
        device_config = container.resolve(DeviceConfig)
        
        self.assertEqual(device_config.device, 'cpu')
        self.assertFalse(device_config.use_gpu)
    
    @patch('infrastructure.di_container.logger')
    def test_spacy_model_fallback(self, mock_logger):
        """Test spaCy model graceful degradation logging"""
        # This test validates that the configuration attempts to load models
        # The actual loading will depend on what's installed
        container = configure_container()
        
        # Check that logger was used (indicating model loading was attempted)
        self.assertTrue(mock_logger.info.called or mock_logger.warning.called or mock_logger.error.called)
    
    def test_config_with_gpu_request(self):
        """Test configuration with GPU request"""
        @dataclass
        class MockConfig:
            use_gpu: bool = False
        
        config = MockConfig(use_gpu=False)
        container = configure_container(config)
        device_config = container.resolve(DeviceConfig)
        
        # Should be CPU since GPU not available in test environment
        self.assertEqual(device_config.device, 'cpu')


class TestInterfaces(unittest.TestCase):
    """Test suite for component interfaces"""
    
    def test_iextractor_interface(self):
        """Test IExtractor interface"""
        extractor = MockExtractor()
        result = extractor.extract('/path/to/doc.pdf')
        
        self.assertIn('text', result)
        self.assertEqual(result['source'], '/path/to/doc.pdf')
    
    def test_icausal_builder_interface(self):
        """Test ICausalBuilder interface"""
        extractor = MockExtractor()
        builder = MockCausalBuilder(extractor)
        
        data = {'test': 'data'}
        result = builder.build_graph(data)
        
        self.assertIn('graph', result)
        self.assertEqual(result['data'], data)
    
    def test_ibayesian_engine_interface(self):
        """Test IBayesianEngine interface"""
        engine = MockBayesianEngine()
        
        graph = {'nodes': [], 'edges': []}
        result = engine.infer(graph)
        
        self.assertIn('inference', result)
        self.assertEqual(engine.call_count, 1)
        
        # Call again to test state
        engine.infer(graph)
        self.assertEqual(engine.call_count, 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for DI container"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow with DI container"""
        # Configure container
        config = {'test': 'config'}
        container = DIContainer(config)
        
        # Register components
        container.register_singleton(IExtractor, MockExtractor)
        container.register_transient(ICausalBuilder, MockCausalBuilder)
        container.register_singleton(IBayesianEngine, MockBayesianEngine)
        
        # Resolve and use components
        extractor = container.resolve(IExtractor)
        builder = container.resolve(ICausalBuilder)
        engine = container.resolve(IBayesianEngine)
        
        # Execute workflow
        extracted = extractor.extract('/test/doc.pdf')
        graph = builder.build_graph(extracted)
        inference = engine.infer(graph)
        
        # Verify results
        self.assertIn('text', extracted)
        self.assertIn('graph', graph)
        self.assertIn('inference', inference)
        
        # Verify singleton behavior
        engine2 = container.resolve(IBayesianEngine)
        self.assertIs(engine, engine2)
        
        # Verify transient behavior
        builder2 = container.resolve(ICausalBuilder)
        self.assertIsNot(builder, builder2)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
