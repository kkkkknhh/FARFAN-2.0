#!/usr/bin/env python3
"""
Unit tests for Module Interfaces and Dependency Injection
Tests Protocol classes, DI container, and adapters
"""

import unittest
from typing import Any, Dict, List, Optional
from pathlib import Path
import networkx as nx

from module_interfaces import (
    IPDFProcessor,
    ICausalExtractor,
    IMechanismExtractor,
    IFinancialAuditor,
    IDNPValidator,
    IQuestionAnsweringEngine,
    IReportGenerator,
    CDAFAdapter,
    ModuleDependencies,
    DependencyInjectionContainer
)


# ============================================================================
# MOCK IMPLEMENTATIONS FOR TESTING
# ============================================================================

class MockPDFProcessor:
    """Mock PDF processor for testing"""
    
    def load_document(self, pdf_path: Path) -> bool:
        return True
    
    def extract_text(self) -> str:
        return "Sample text"
    
    def extract_tables(self) -> List[Any]:
        return []
    
    def extract_sections(self) -> Dict[str, str]:
        return {"intro": "Introduction text"}


class MockCausalExtractor:
    """Mock causal extractor for testing"""
    
    def __init__(self):
        self._nodes = {}
        self._causal_chains = []
    
    def extract_causal_hierarchy(self, text: str) -> nx.DiGraph:
        return nx.DiGraph()
    
    @property
    def nodes(self) -> Dict[str, Any]:
        return self._nodes
    
    @property
    def causal_chains(self) -> List[Any]:
        return self._causal_chains


class MockMechanismExtractor:
    """Mock mechanism extractor for testing"""
    
    def extract_entity_activity(self, text: str) -> Optional[Any]:
        return None


class MockFinancialAuditor:
    """Mock financial auditor for testing"""
    
    def __init__(self):
        self._financial_data = {}
    
    def trace_financial_allocation(self, tables: List[Any], 
                                   nodes: Dict[str, Any]) -> Dict[str, float]:
        return {}
    
    @property
    def financial_data(self) -> Dict[str, Any]:
        return self._financial_data


class MockDNPValidator:
    """Mock DNP validator for testing"""
    
    def validar_proyecto_integral(self, sector: str, descripcion: str,
                                  indicadores_propuestos: List[str],
                                  presupuesto: float = 0.0,
                                  es_rural: bool = False,
                                  poblacion_victimas: bool = False) -> Any:
        # Return a mock result
        class MockResult:
            score_total = 75.0
        return MockResult()


class MockQAEngine:
    """Mock question answering engine for testing"""
    
    def answer_all_questions(self, context: Any) -> Dict[str, Dict]:
        return {}


class MockReportGenerator:
    """Mock report generator for testing"""
    
    def generate_micro_report(self, responses: Dict[str, Dict], 
                             policy_code: str) -> Dict:
        return {}
    
    def generate_meso_report(self, responses: Dict[str, Dict],
                            policy_code: str) -> Dict:
        return {}
    
    def generate_macro_report(self, responses: Dict[str, Dict],
                             compliance_score: float,
                             policy_code: str) -> Dict:
        return {}


# ============================================================================
# TEST CASES
# ============================================================================

class TestProtocolInterfaces(unittest.TestCase):
    """Test that mock implementations satisfy Protocol interfaces"""
    
    def test_pdf_processor_protocol(self):
        """Test that MockPDFProcessor satisfies IPDFProcessor"""
        processor: IPDFProcessor = MockPDFProcessor()
        
        self.assertTrue(processor.load_document(Path("test.pdf")))
        self.assertEqual(processor.extract_text(), "Sample text")
        self.assertEqual(processor.extract_tables(), [])
        self.assertIsInstance(processor.extract_sections(), dict)
    
    def test_causal_extractor_protocol(self):
        """Test that MockCausalExtractor satisfies ICausalExtractor"""
        extractor: ICausalExtractor = MockCausalExtractor()
        
        graph = extractor.extract_causal_hierarchy("test text")
        self.assertIsInstance(graph, nx.DiGraph)
        self.assertIsInstance(extractor.nodes, dict)
        self.assertIsInstance(extractor.causal_chains, list)
    
    def test_mechanism_extractor_protocol(self):
        """Test that MockMechanismExtractor satisfies IMechanismExtractor"""
        extractor: IMechanismExtractor = MockMechanismExtractor()
        
        result = extractor.extract_entity_activity("test text")
        self.assertIsNone(result)
    
    def test_financial_auditor_protocol(self):
        """Test that MockFinancialAuditor satisfies IFinancialAuditor"""
        auditor: IFinancialAuditor = MockFinancialAuditor()
        
        result = auditor.trace_financial_allocation([], {})
        self.assertIsInstance(result, dict)
        self.assertIsInstance(auditor.financial_data, dict)
    
    def test_dnp_validator_protocol(self):
        """Test that MockDNPValidator satisfies IDNPValidator"""
        validator: IDNPValidator = MockDNPValidator()
        
        result = validator.validar_proyecto_integral(
            sector="educacion",
            descripcion="Test project",
            indicadores_propuestos=[]
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.score_total, 75.0)
    
    def test_qa_engine_protocol(self):
        """Test that MockQAEngine satisfies IQuestionAnsweringEngine"""
        engine: IQuestionAnsweringEngine = MockQAEngine()
        
        result = engine.answer_all_questions(None)
        self.assertIsInstance(result, dict)
    
    def test_report_generator_protocol(self):
        """Test that MockReportGenerator satisfies IReportGenerator"""
        generator: IReportGenerator = MockReportGenerator()
        
        micro = generator.generate_micro_report({}, "TEST-001")
        meso = generator.generate_meso_report({}, "TEST-001")
        macro = generator.generate_macro_report({}, 75.0, "TEST-001")
        
        self.assertIsInstance(micro, dict)
        self.assertIsInstance(meso, dict)
        self.assertIsInstance(macro, dict)


class TestModuleDependencies(unittest.TestCase):
    """Test ModuleDependencies container"""
    
    def test_empty_dependencies(self):
        """Test empty dependencies container"""
        deps = ModuleDependencies()
        
        self.assertFalse(deps.is_complete())
        self.assertEqual(deps.get_available_modules(), [])
        self.assertGreater(len(deps.get_missing_modules()), 0)
    
    def test_partial_dependencies(self):
        """Test partially filled dependencies"""
        deps = ModuleDependencies(
            pdf_processor=MockPDFProcessor(),
            causal_extractor=MockCausalExtractor()
        )
        
        self.assertFalse(deps.is_complete())
        available = deps.get_available_modules()
        self.assertIn('pdf_processor', available)
        self.assertIn('causal_extractor', available)
        self.assertEqual(len(available), 2)
    
    def test_complete_dependencies(self):
        """Test complete dependencies"""
        deps = ModuleDependencies(
            pdf_processor=MockPDFProcessor(),
            causal_extractor=MockCausalExtractor(),
            qa_engine=MockQAEngine(),
            report_generator=MockReportGenerator()
        )
        
        self.assertTrue(deps.is_complete())
        self.assertEqual(len(deps.get_missing_modules()), 0)
    
    def test_get_available_modules(self):
        """Test getting available modules list"""
        deps = ModuleDependencies(
            pdf_processor=MockPDFProcessor(),
            dnp_validator=MockDNPValidator(),
            qa_engine=MockQAEngine()
        )
        
        available = deps.get_available_modules()
        self.assertIn('pdf_processor', available)
        self.assertIn('dnp_validator', available)
        self.assertIn('qa_engine', available)
        self.assertEqual(len(available), 3)


class TestDependencyInjectionContainer(unittest.TestCase):
    """Test DependencyInjectionContainer"""
    
    def test_register_and_get(self):
        """Test registering and retrieving modules"""
        container = DependencyInjectionContainer()
        processor = MockPDFProcessor()
        
        container.register('pdf_processor', processor)
        retrieved = container.get('pdf_processor')
        
        self.assertIs(retrieved, processor)
    
    def test_get_nonexistent(self):
        """Test getting non-existent module"""
        container = DependencyInjectionContainer()
        
        result = container.get('nonexistent')
        self.assertIsNone(result)
    
    def test_factory_registration(self):
        """Test lazy loading with factory"""
        container = DependencyInjectionContainer()
        
        # Register a factory that creates the module on-demand
        def factory():
            return MockPDFProcessor()
        
        container.register_factory('pdf_processor', factory)
        
        # First get should create the instance
        first = container.get('pdf_processor')
        self.assertIsInstance(first, MockPDFProcessor)
        
        # Second get should return the same instance
        second = container.get('pdf_processor')
        self.assertIs(first, second)
    
    def test_validate_empty(self):
        """Test validation of empty container"""
        container = DependencyInjectionContainer()
        
        is_valid, missing = container.validate()
        self.assertFalse(is_valid)
        self.assertGreater(len(missing), 0)
    
    def test_validate_complete(self):
        """Test validation of complete container"""
        container = DependencyInjectionContainer()
        
        container.register('pdf_processor', MockPDFProcessor())
        container.register('causal_extractor', MockCausalExtractor())
        container.register('qa_engine', MockQAEngine())
        container.register('report_generator', MockReportGenerator())
        
        is_valid, missing = container.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(missing), 0)
    
    def test_get_dependencies(self):
        """Test getting dependency container"""
        container = DependencyInjectionContainer()
        processor = MockPDFProcessor()
        
        container.register('pdf_processor', processor)
        deps = container.get_dependencies()
        
        self.assertIsInstance(deps, ModuleDependencies)
        self.assertIs(deps.pdf_processor, processor)


class TestCDAFAdapter(unittest.TestCase):
    """Test CDAF adapter for legacy compatibility"""
    
    def test_adapter_creation(self):
        """Test creating adapter with mock CDAF"""
        
        class MockCDAF:
            def __init__(self):
                self.pdf_processor = MockPDFProcessor()
                self.causal_extractor = MockCausalExtractor()
                self.mechanism_extractor = MockMechanismExtractor()
                self.financial_auditor = MockFinancialAuditor()
        
        cdaf = MockCDAF()
        adapter = CDAFAdapter(cdaf)
        
        # Test that adapter provides correct interfaces
        pdf = adapter.get_pdf_processor()
        self.assertIsInstance(pdf, MockPDFProcessor)
        
        causal = adapter.get_causal_extractor()
        self.assertIsInstance(causal, MockCausalExtractor)
        
        mechanism = adapter.get_mechanism_extractor()
        self.assertIsInstance(mechanism, MockMechanismExtractor)
        
        financial = adapter.get_financial_auditor()
        self.assertIsInstance(financial, MockFinancialAuditor)


if __name__ == "__main__":
    unittest.main(verbosity=2)
