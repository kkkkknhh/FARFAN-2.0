#!/usr/bin/env python3
"""
Module Interfaces for FARFAN 2.0
Defines formal Protocol classes for all module interfaces
Following Category 2.1 requirement for explicit interface contracts

This module provides:
1. Protocol classes for all major modules
2. Type-safe contracts for module interactions
3. Clear documentation of input/output contracts
"""

from typing import Protocol, Dict, List, Any, Optional, Tuple
from pathlib import Path
import networkx as nx
from dataclasses import dataclass


# ============================================================================
# PROTOCOL CLASSES - Formal Interface Contracts
# ============================================================================

class IPDFProcessor(Protocol):
    """
    Interface for PDF document processing
    
    Input Contract:
        - pdf_path: Path object pointing to valid PDF file
    
    Output Contract:
        - raw_text: String containing extracted text
        - tables: List of table data structures
        - sections: Dict mapping section names to content
    
    Preconditions:
        - PDF file must exist and be readable
        - PDF must not be encrypted
    
    Postconditions:
        - Text extraction preserves document structure
        - Tables maintain row/column relationships
        - Sections are properly identified
    """
    
    def load_document(self, pdf_path: Path) -> bool:
        """
        Load a PDF document
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if successful, False otherwise
        """
        ...
    
    def extract_text(self) -> str:
        """
        Extract raw text from loaded document
        
        Returns:
            Full text content of document
        """
        ...
    
    def extract_tables(self) -> List[Any]:
        """
        Extract table data from document
        
        Returns:
            List of table structures
        """
        ...
    
    def extract_sections(self) -> Dict[str, str]:
        """
        Extract document sections
        
        Returns:
            Dictionary mapping section names to content
        """
        ...


class ICausalExtractor(Protocol):
    """
    Interface for causal hierarchy extraction
    
    Input Contract:
        - text: String containing policy document text
    
    Output Contract:
        - causal_graph: NetworkX DiGraph with causal relationships
        - nodes: Dict of MetaNode objects indexed by ID
        - causal_chains: List of causal link chains
    
    Preconditions:
        - Text must be non-empty
        - Text should be in Spanish
    
    Postconditions:
        - Graph is acyclic (DAG structure)
        - All nodes have proper type classification
        - Edges represent valid causal relationships
    """
    
    def extract_causal_hierarchy(self, text: str) -> nx.DiGraph:
        """
        Extract causal hierarchy from text
        
        Args:
            text: Policy document text
            
        Returns:
            Directed acyclic graph of causal relationships
        """
        ...
    
    @property
    def nodes(self) -> Dict[str, Any]:
        """Access to extracted nodes"""
        ...
    
    @property
    def causal_chains(self) -> List[Any]:
        """Access to causal link chains"""
        ...


class IMechanismExtractor(Protocol):
    """
    Interface for mechanism part extraction
    
    Input Contract:
        - text: String describing a mechanism or intervention
    
    Output Contract:
        - entity_activity: Tuple of (entity, activity, confidence)
    
    Preconditions:
        - Text describes a concrete action or intervention
    
    Postconditions:
        - Entity represents the agent/object
        - Activity represents the action/transformation
        - Confidence in range [0, 1]
    """
    
    def extract_entity_activity(self, text: str) -> Optional[Any]:
        """
        Extract entity-activity pairs from text
        
        Args:
            text: Description of mechanism or intervention
            
        Returns:
            EntityActivity object or None if not found
        """
        ...


class IFinancialAuditor(Protocol):
    """
    Interface for financial audit and traceability
    
    Input Contract:
        - tables: List of financial tables from document
        - nodes: Dict of policy nodes to trace funding for
    
    Output Contract:
        - unit_costs: Dict mapping node IDs to unit costs
        - financial_data: Dict of financial allocations
    
    Preconditions:
        - Tables contain budget/financial information
        - Nodes have financial_allocation attribute
    
    Postconditions:
        - All identified allocations sum to total budget
        - Unit costs are positive numbers
    """
    
    def trace_financial_allocation(self, tables: List[Any], 
                                   nodes: Dict[str, Any]) -> Dict[str, float]:
        """
        Trace financial allocations from tables to nodes
        
        Args:
            tables: List of extracted tables
            nodes: Dictionary of policy nodes
            
        Returns:
            Dictionary of unit costs per node
        """
        ...
    
    @property
    def financial_data(self) -> Dict[str, Any]:
        """Access to financial allocation data"""
        ...


class IDNPValidator(Protocol):
    """
    Interface for DNP standards validation
    
    Input Contract:
        - sector: String identifying policy sector
        - descripcion: Description of project/intervention
        - indicadores_propuestos: List of proposed indicators
        - presupuesto: Budget amount (optional)
    
    Output Contract:
        - resultado: Validation result with score and details
    
    Preconditions:
        - Sector must be valid DNP sector code
        - Description must be non-empty
    
    Postconditions:
        - Score in range [0, 100]
        - Result includes specific compliance issues
    """
    
    def validar_proyecto_integral(self, sector: str, descripcion: str,
                                  indicadores_propuestos: List[str],
                                  presupuesto: float = 0.0,
                                  es_rural: bool = False,
                                  poblacion_victimas: bool = False) -> Any:
        """
        Validate project against DNP standards
        
        Args:
            sector: Policy sector code
            descripcion: Project description
            indicadores_propuestos: Proposed indicators
            presupuesto: Budget amount
            es_rural: Whether project is in rural area
            poblacion_victimas: Whether targets victim population
            
        Returns:
            ValidationResult with score and details
        """
        ...


class IQuestionAnsweringEngine(Protocol):
    """
    Interface for question answering engine
    
    Input Contract:
        - pipeline_context: PipelineContext with all extracted data
    
    Output Contract:
        - question_responses: Dict mapping question IDs to responses
    
    Preconditions:
        - Context contains minimum required data (text, nodes)
    
    Postconditions:
        - All 300 questions have responses
        - Each response has score, evidence, and argumentation
    """
    
    def answer_all_questions(self, context: Any) -> Dict[str, Dict]:
        """
        Answer all policy evaluation questions
        
        Args:
            context: PipelineContext with extraction results
            
        Returns:
            Dictionary mapping question IDs to responses
        """
        ...


class IReportGenerator(Protocol):
    """
    Interface for report generation
    
    Input Contract:
        - question_responses: Dict of answers to 300 questions
        - policy_code: String identifying the policy
    
    Output Contract:
        - micro_report: Individual question-level results
        - meso_report: Cluster/dimension aggregations
        - macro_report: Overall evaluation summary
    
    Preconditions:
        - Responses contain valid scores and evidence
    
    Postconditions:
        - Reports are consistent across all levels
        - Files are written to output directory
    """
    
    def generate_micro_report(self, responses: Dict[str, Dict], 
                             policy_code: str) -> Dict:
        """Generate micro-level report"""
        ...
    
    def generate_meso_report(self, responses: Dict[str, Dict],
                            policy_code: str) -> Dict:
        """Generate meso-level report"""
        ...
    
    def generate_macro_report(self, responses: Dict[str, Dict],
                             compliance_score: float,
                             policy_code: str) -> Dict:
        """Generate macro-level report"""
        ...


# ============================================================================
# ADAPTER PATTERN - For Legacy Module Compatibility
# ============================================================================

class CDAFAdapter:
    """
    Adapter for dereck_beach CDAF Framework
    Provides stable interface even if underlying API changes
    """
    
    def __init__(self, cdaf_framework: Any):
        """
        Initialize adapter with CDAF framework instance
        
        Args:
            cdaf_framework: Instance of CDAFFramework from dereck_beach
        """
        self._cdaf = cdaf_framework
    
    def get_pdf_processor(self) -> IPDFProcessor:
        """Get PDF processor conforming to interface"""
        return self._cdaf.pdf_processor
    
    def get_causal_extractor(self) -> ICausalExtractor:
        """Get causal extractor conforming to interface"""
        return self._cdaf.causal_extractor
    
    def get_mechanism_extractor(self) -> IMechanismExtractor:
        """Get mechanism extractor conforming to interface"""
        return self._cdaf.mechanism_extractor
    
    def get_financial_auditor(self) -> IFinancialAuditor:
        """Get financial auditor conforming to interface"""
        return self._cdaf.financial_auditor


# ============================================================================
# DEPENDENCY INJECTION CONTAINER
# ============================================================================

@dataclass
class ModuleDependencies:
    """
    Container for all module dependencies
    Enables dependency injection for testing and flexibility
    """
    
    # Core processing modules
    pdf_processor: Optional[IPDFProcessor] = None
    causal_extractor: Optional[ICausalExtractor] = None
    mechanism_extractor: Optional[IMechanismExtractor] = None
    financial_auditor: Optional[IFinancialAuditor] = None
    
    # Validation and standards modules
    dnp_validator: Optional[IDNPValidator] = None
    competencias: Optional[Any] = None
    mga_catalog: Optional[Any] = None
    pdet_lineamientos: Optional[Any] = None
    
    # Processing engines
    qa_engine: Optional[IQuestionAnsweringEngine] = None
    report_generator: Optional[IReportGenerator] = None
    
    # Orchestration
    choreographer: Optional[Any] = None
    
    def is_complete(self) -> bool:
        """Check if all required dependencies are available"""
        return all([
            self.pdf_processor is not None,
            self.causal_extractor is not None,
            self.qa_engine is not None,
            self.report_generator is not None
        ])
    
    def get_available_modules(self) -> List[str]:
        """Get list of available module names"""
        available = []
        if self.pdf_processor: available.append('pdf_processor')
        if self.causal_extractor: available.append('causal_extractor')
        if self.mechanism_extractor: available.append('mechanism_extractor')
        if self.financial_auditor: available.append('financial_auditor')
        if self.dnp_validator: available.append('dnp_validator')
        if self.competencias: available.append('competencias')
        if self.mga_catalog: available.append('mga_catalog')
        if self.pdet_lineamientos: available.append('pdet_lineamientos')
        if self.qa_engine: available.append('qa_engine')
        if self.report_generator: available.append('report_generator')
        if self.choreographer: available.append('choreographer')
        return available
    
    def get_missing_modules(self) -> List[str]:
        """Get list of missing required module names"""
        missing = []
        if not self.pdf_processor: missing.append('pdf_processor')
        if not self.causal_extractor: missing.append('causal_extractor')
        if not self.qa_engine: missing.append('qa_engine')
        if not self.report_generator: missing.append('report_generator')
        return missing


class DependencyInjectionContainer:
    """
    Dependency Injection Container for FARFAN modules
    
    Benefits:
    - Testability: Easy to inject mocks for testing
    - Flexibility: Swap implementations without code changes
    - Clarity: Explicit dependencies, no hidden coupling
    """
    
    def __init__(self):
        self._dependencies = ModuleDependencies()
        self._factories = {}
    
    def register_factory(self, name: str, factory_fn):
        """
        Register a factory function for lazy loading
        
        Args:
            name: Module name
            factory_fn: Function that creates the module instance
        """
        self._factories[name] = factory_fn
    
    def register(self, name: str, instance: Any):
        """
        Register a module instance
        
        Args:
            name: Module name (e.g., 'pdf_processor')
            instance: Module instance
        """
        setattr(self._dependencies, name, instance)
    
    def get(self, name: str) -> Optional[Any]:
        """
        Get a module instance, creating it if necessary
        
        Args:
            name: Module name
            
        Returns:
            Module instance or None if not available
        """
        # Check if already instantiated
        instance = getattr(self._dependencies, name, None)
        if instance is not None:
            return instance
        
        # Check if factory is available
        if name in self._factories:
            instance = self._factories[name]()
            setattr(self._dependencies, name, instance)
            return instance
        
        return None
    
    def get_dependencies(self) -> ModuleDependencies:
        """Get the dependency container"""
        return self._dependencies
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate that all required dependencies are available
        
        Returns:
            Tuple of (is_valid, missing_modules)
        """
        missing = self._dependencies.get_missing_modules()
        return len(missing) == 0, missing
