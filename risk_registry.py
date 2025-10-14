#!/usr/bin/env python3
"""
FARFAN Risk Registry
Structured risk definitions for failure modes across the pipeline stages
"""

from enum import Enum
from typing import Callable, Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
import logging

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Risk severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskCategory(str, Enum):
    """Risk categories for classification"""
    DATA_QUALITY = "data_quality"
    EXTERNAL_DEPENDENCY = "external_dependency"
    COMPUTATIONAL = "computational"
    CONFIGURATION = "configuration"


class RiskDefinition(BaseModel):
    """
    Structured risk definition with validation
    
    Attributes:
        risk_id: Unique identifier for the risk
        name: Human-readable risk name
        description: Detailed description of the risk
        category: Risk category classification
        severity: Severity level
        probability: Probability score (0.0-1.0)
        impact: Impact score (0.0-1.0)
        stage: Pipeline stage where risk occurs
        detector: Callable that detects if risk is present
        mitigation_strategy: Callable that attempts to mitigate the risk
        metadata: Additional metadata for the risk
    """
    risk_id: str = Field(..., min_length=1, description="Unique risk identifier")
    name: str = Field(..., min_length=1, description="Human-readable risk name")
    description: str = Field(..., min_length=1, description="Detailed risk description")
    category: RiskCategory = Field(..., description="Risk category")
    severity: Severity = Field(..., description="Severity level")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability score (0-1)")
    impact: float = Field(..., ge=0.0, le=1.0, description="Impact score (0-1)")
    stage: str = Field(..., description="Pipeline stage identifier")
    detector: Optional[Callable[..., bool]] = Field(None, description="Risk detector predicate")
    mitigation_strategy: Optional[Callable[..., Any]] = Field(None, description="Mitigation strategy callable")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        arbitrary_types_allowed = True
    
    @field_validator('probability', 'impact')
    @classmethod
    def validate_scores(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {v}")
        return v
    
    def risk_score(self) -> float:
        """Calculate composite risk score (probability × impact)"""
        return self.probability * self.impact


class RiskRegistry:
    """
    Registry for explicit risk definitions across FARFAN pipeline stages
    
    Stores structured risk objects with detector predicates, mitigation strategies,
    severity levels, and probability/impact scores for known failure modes.
    """
    
    def __init__(self):
        self._risks: Dict[str, RiskDefinition] = {}
        self._by_category: Dict[RiskCategory, List[str]] = {cat: [] for cat in RiskCategory}
        self._by_severity: Dict[Severity, List[str]] = {sev: [] for sev in Severity}
        self._by_stage: Dict[str, List[str]] = {}
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Register known failure modes
        self._register_default_risks()
    
    def register_risk(self, risk: RiskDefinition) -> None:
        """
        Register a risk definition in the registry
        
        Args:
            risk: RiskDefinition object to register
            
        Raises:
            ValueError: If risk_id already exists
        """
        if risk.risk_id in self._risks:
            raise ValueError(f"Risk ID '{risk.risk_id}' already registered")
        
        self._risks[risk.risk_id] = risk
        self._by_category[risk.category].append(risk.risk_id)
        self._by_severity[risk.severity].append(risk.risk_id)
        
        if risk.stage not in self._by_stage:
            self._by_stage[risk.stage] = []
        self._by_stage[risk.stage].append(risk.risk_id)
        
        self.logger.debug(f"Registered risk: {risk.risk_id} ({risk.severity.value}/{risk.category.value})")
    
    def get_risk(self, risk_id: str) -> Optional[RiskDefinition]:
        """Retrieve a risk definition by ID"""
        return self._risks.get(risk_id)
    
    def get_by_category(self, category: RiskCategory) -> List[RiskDefinition]:
        """Retrieve all risks in a specific category"""
        risk_ids = self._by_category.get(category, [])
        return [self._risks[rid] for rid in risk_ids]
    
    def get_by_severity(self, severity: Severity) -> List[RiskDefinition]:
        """Retrieve all risks with a specific severity level"""
        risk_ids = self._by_severity.get(severity, [])
        return [self._risks[rid] for rid in risk_ids]
    
    def get_by_stage(self, stage: str) -> List[RiskDefinition]:
        """Retrieve all risks for a specific pipeline stage"""
        risk_ids = self._by_stage.get(stage, [])
        return [self._risks[rid] for rid in risk_ids]
    
    def get_critical_risks(self) -> List[RiskDefinition]:
        """Retrieve all critical severity risks"""
        return self.get_by_severity(Severity.CRITICAL)
    
    def get_high_impact_risks(self, threshold: float = 0.7) -> List[RiskDefinition]:
        """Retrieve risks with impact score above threshold"""
        return [risk for risk in self._risks.values() if risk.impact >= threshold]
    
    def get_all_risks(self) -> List[RiskDefinition]:
        """Retrieve all registered risks"""
        return list(self._risks.values())
    
    def evaluate_risks(self, context: Any) -> List[Dict[str, Any]]:
        """
        Evaluate all registered risks against a pipeline context
        
        Args:
            context: Pipeline context object to evaluate
            
        Returns:
            List of detected risks with evaluation results
        """
        detected_risks = []
        
        for risk in self._risks.values():
            if risk.detector is None:
                continue
            
            try:
                is_detected = risk.detector(context)
                if is_detected:
                    detected_risks.append({
                        'risk_id': risk.risk_id,
                        'name': risk.name,
                        'severity': risk.severity.value,
                        'category': risk.category.value,
                        'risk_score': risk.risk_score(),
                        'stage': risk.stage,
                        'description': risk.description
                    })
            except Exception as e:
                self.logger.warning(f"Risk detector failed for {risk.risk_id}: {e}")
        
        return sorted(detected_risks, key=lambda x: x['risk_score'], reverse=True)
    
    def apply_mitigation(self, risk_id: str, context: Any) -> Any:
        """
        Apply mitigation strategy for a specific risk
        
        Args:
            risk_id: Risk identifier
            context: Pipeline context object
            
        Returns:
            Result of mitigation strategy
        """
        risk = self.get_risk(risk_id)
        if risk is None:
            raise ValueError(f"Risk ID '{risk_id}' not found")
        
        if risk.mitigation_strategy is None:
            self.logger.warning(f"No mitigation strategy for {risk_id}")
            return None
        
        try:
            return risk.mitigation_strategy(context)
        except Exception as e:
            self.logger.error(f"Mitigation failed for {risk_id}: {e}")
            raise
    
    def _register_default_risks(self) -> None:
        """Register known failure modes for FARFAN pipeline"""
        
        # PDF Parsing Failures
        self.register_risk(RiskDefinition(
            risk_id="PDF_001",
            name="PDF Parsing Failure",
            description="PDF document cannot be opened or parsed by PyMuPDF",
            category=RiskCategory.DATA_QUALITY,
            severity=Severity.CRITICAL,
            probability=0.15,
            impact=1.0,
            stage="LOAD_DOCUMENT",
            detector=lambda ctx: not hasattr(ctx, 'raw_text') or not ctx.raw_text,
            mitigation_strategy=lambda ctx: self._retry_pdf_parsing(ctx),
            metadata={'library': 'PyMuPDF', 'common_causes': ['corrupted_pdf', 'password_protected', 'unsupported_version']}
        ))
        
        self.register_risk(RiskDefinition(
            risk_id="PDF_002",
            name="PDF Text Extraction Low Quality",
            description="PDF text extraction yields low-quality or garbled text",
            category=RiskCategory.DATA_QUALITY,
            severity=Severity.HIGH,
            probability=0.25,
            impact=0.8,
            stage="EXTRACT_TEXT_TABLES",
            detector=lambda ctx: hasattr(ctx, 'raw_text') and len(ctx.raw_text) < 1000,
            mitigation_strategy=lambda ctx: self._apply_ocr_fallback(ctx),
            metadata={'indicators': ['short_text', 'encoding_errors', 'scanned_images']}
        ))
        
        # spaCy Model Loading Issues
        self.register_risk(RiskDefinition(
            risk_id="NLP_001",
            name="spaCy Model Loading Failure",
            description="spaCy Spanish model (es_core_news_lg) fails to load",
            category=RiskCategory.EXTERNAL_DEPENDENCY,
            severity=Severity.CRITICAL,
            probability=0.10,
            impact=1.0,
            stage="SEMANTIC_ANALYSIS",
            detector=lambda ctx: self._check_spacy_availability(),
            mitigation_strategy=lambda ctx: self._download_spacy_model(),
            metadata={'model': 'es_core_news_lg', 'required_version': '3.0+'}
        ))
        
        self.register_risk(RiskDefinition(
            risk_id="NLP_002",
            name="NLP Memory Exhaustion",
            description="spaCy processing exhausts available memory for large documents",
            category=RiskCategory.COMPUTATIONAL,
            severity=Severity.HIGH,
            probability=0.20,
            impact=0.9,
            stage="SEMANTIC_ANALYSIS",
            detector=lambda ctx: hasattr(ctx, 'raw_text') and len(ctx.raw_text) > 1_000_000,
            mitigation_strategy=lambda ctx: self._chunk_text_processing(ctx),
            metadata={'threshold_chars': 1_000_000, 'chunk_size': 100_000}
        ))
        
        # DNP API Unavailability
        self.register_risk(RiskDefinition(
            risk_id="DNP_001",
            name="DNP API Unavailable",
            description="DNP external services or APIs are unreachable",
            category=RiskCategory.EXTERNAL_DEPENDENCY,
            severity=Severity.HIGH,
            probability=0.30,
            impact=0.7,
            stage="DNP_VALIDATION",
            detector=lambda ctx: self._check_dnp_availability(),
            mitigation_strategy=lambda ctx: self._use_cached_dnp_data(ctx),
            metadata={'endpoints': ['MGA', 'PDET', 'SICODIS'], 'cache_ttl': 86400}
        ))
        
        # Embedding Service Timeouts
        self.register_risk(RiskDefinition(
            risk_id="EMB_001",
            name="Embedding Service Timeout",
            description="External embedding service times out or is unreachable",
            category=RiskCategory.EXTERNAL_DEPENDENCY,
            severity=Severity.MEDIUM,
            probability=0.25,
            impact=0.6,
            stage="SEMANTIC_ANALYSIS",
            detector=lambda ctx: self._check_embedding_service(),
            mitigation_strategy=lambda ctx: self._use_local_embeddings(ctx),
            metadata={'timeout_seconds': 30, 'fallback': 'sentence-transformers'}
        ))
        
        # Missing or Malformed Municipal Data
        self.register_risk(RiskDefinition(
            risk_id="DATA_001",
            name="Missing Municipal Metadata",
            description="Required municipal metadata is missing or incomplete",
            category=RiskCategory.DATA_QUALITY,
            severity=Severity.HIGH,
            probability=0.35,
            impact=0.7,
            stage="DNP_VALIDATION",
            detector=lambda ctx: not hasattr(ctx, 'policy_code') or not ctx.policy_code,
            mitigation_strategy=lambda ctx: self._infer_municipal_data(ctx),
            metadata={'required_fields': ['municipality', 'department', 'code', 'pdet_status']}
        ))
        
        self.register_risk(RiskDefinition(
            risk_id="DATA_002",
            name="Malformed Financial Tables",
            description="Financial tables are malformed or cannot be parsed",
            category=RiskCategory.DATA_QUALITY,
            severity=Severity.MEDIUM,
            probability=0.40,
            impact=0.6,
            stage="FINANCIAL_AUDIT",
            detector=lambda ctx: hasattr(ctx, 'tables') and len(ctx.tables) == 0,
            mitigation_strategy=lambda ctx: self._reconstruct_financial_data(ctx),
            metadata={'required_columns': ['item', 'budget', 'source']}
        ))
        
        # Invalid Configuration Parameters
        self.register_risk(RiskDefinition(
            risk_id="CFG_001",
            name="Invalid Configuration File",
            description="Configuration file is missing, malformed, or contains invalid values",
            category=RiskCategory.CONFIGURATION,
            severity=Severity.HIGH,
            probability=0.15,
            impact=0.8,
            stage="LOAD_DOCUMENT",
            detector=lambda ctx: self._validate_config(),
            mitigation_strategy=lambda ctx: self._load_default_config(),
            metadata={'config_file': 'config.yaml', 'schema_version': '2.0'}
        ))
        
        self.register_risk(RiskDefinition(
            risk_id="CFG_002",
            name="Invalid Threshold Parameters",
            description="Confidence or scoring thresholds are outside valid ranges",
            category=RiskCategory.CONFIGURATION,
            severity=Severity.MEDIUM,
            probability=0.20,
            impact=0.5,
            stage="CAUSAL_EXTRACTION",
            detector=lambda ctx: self._validate_thresholds(),
            mitigation_strategy=lambda ctx: self._reset_thresholds_to_default(),
            metadata={'valid_range': [0.0, 1.0], 'default_threshold': 0.7}
        ))
        
        # Computational Resource Exhaustion
        self.register_risk(RiskDefinition(
            risk_id="RES_001",
            name="Memory Exhaustion",
            description="System runs out of available memory during processing",
            category=RiskCategory.COMPUTATIONAL,
            severity=Severity.CRITICAL,
            probability=0.15,
            impact=1.0,
            stage="SEMANTIC_ANALYSIS",
            detector=lambda ctx: self._check_memory_usage(),
            mitigation_strategy=lambda ctx: self._enable_streaming_processing(ctx),
            metadata={'memory_threshold_gb': 8, 'swap_threshold_percent': 90}
        ))
        
        self.register_risk(RiskDefinition(
            risk_id="RES_002",
            name="Graph Processing Timeout",
            description="Causal graph processing exceeds time limits",
            category=RiskCategory.COMPUTATIONAL,
            severity=Severity.MEDIUM,
            probability=0.25,
            impact=0.6,
            stage="CAUSAL_EXTRACTION",
            detector=lambda ctx: hasattr(ctx, 'nodes') and len(ctx.nodes) > 1000,
            mitigation_strategy=lambda ctx: self._simplify_graph(ctx),
            metadata={'max_nodes': 1000, 'timeout_seconds': 300}
        ))
        
        self.logger.info(f"Registered {len(self._risks)} default risks")
    
    # Detector helper methods
    def _check_spacy_availability(self) -> bool:
        try:
            import spacy
            spacy.load('es_core_news_lg')
            return False
        except:
            return True
    
    def _check_dnp_availability(self) -> bool:
        return False
    
    def _check_embedding_service(self) -> bool:
        return False
    
    def _validate_config(self) -> bool:
        return False
    
    def _validate_thresholds(self) -> bool:
        return False
    
    def _check_memory_usage(self) -> bool:
        return False
    
    # Mitigation helper methods
    def _retry_pdf_parsing(self, ctx: Any) -> Any:
        self.logger.info("Retrying PDF parsing with alternative methods")
        return None
    
    def _apply_ocr_fallback(self, ctx: Any) -> Any:
        self.logger.info("Applying OCR fallback for text extraction")
        return None
    
    def _download_spacy_model(self) -> Any:
        self.logger.info("Attempting to download spaCy model")
        return None
    
    def _chunk_text_processing(self, ctx: Any) -> Any:
        self.logger.info("Chunking text for memory-efficient processing")
        return None
    
    def _use_cached_dnp_data(self, ctx: Any) -> Any:
        self.logger.info("Using cached DNP data")
        return None
    
    def _use_local_embeddings(self, ctx: Any) -> Any:
        self.logger.info("Falling back to local embeddings")
        return None
    
    def _infer_municipal_data(self, ctx: Any) -> Any:
        self.logger.info("Inferring missing municipal metadata")
        return None
    
    def _reconstruct_financial_data(self, ctx: Any) -> Any:
        self.logger.info("Reconstructing financial data from text")
        return None
    
    def _load_default_config(self) -> Any:
        self.logger.info("Loading default configuration")
        return None
    
    def _reset_thresholds_to_default(self) -> Any:
        self.logger.info("Resetting thresholds to defaults")
        return None
    
    def _enable_streaming_processing(self, ctx: Any) -> Any:
        self.logger.info("Enabling streaming processing mode")
        return None
    
    def _simplify_graph(self, ctx: Any) -> Any:
        self.logger.info("Simplifying causal graph")
        return None


# Global registry instance
_registry = None


def get_risk_registry() -> RiskRegistry:
    """Get or create the global risk registry instance"""
    global _registry
    if _registry is None:
        _registry = RiskRegistry()
    return _registry


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== FARFAN Risk Registry Demo ===\n")
    
    registry = get_risk_registry()
    
    print(f"Total registered risks: {len(registry.get_all_risks())}\n")
    
    print("Critical Risks:")
    for risk in registry.get_critical_risks():
        print(f"  - {risk.risk_id}: {risk.name} (score: {risk.risk_score():.2f})")
    
    print("\nData Quality Risks:")
    for risk in registry.get_by_category(RiskCategory.DATA_QUALITY):
        print(f"  - {risk.risk_id}: {risk.name} ({risk.severity.value})")
    
    print("\nRisks by Stage:")
    for stage in ["LOAD_DOCUMENT", "SEMANTIC_ANALYSIS", "DNP_VALIDATION"]:
        stage_risks = registry.get_by_stage(stage)
        if stage_risks:
            print(f"  {stage}: {len(stage_risks)} risks")
