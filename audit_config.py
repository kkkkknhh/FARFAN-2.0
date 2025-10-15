#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit Configuration Module
===========================

Implements Audit Point 1.4: External Resource Fallback
Manages fail-closed for core resources and fail-open (degraded gracefully) for external resources.

Core Resources (Fail-Closed):
- CDAFConfigSchema
- TeoriaCambio lexicons
- Critical Pydantic schemas

External Resources (Fail-Open/Degraded):
- ValidadorDNP
- MGA indicators
- PDET lineamientos
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Resource type classification for fallback policy"""
    CORE = "core"  # Fail-closed: missing causes hard failure
    EXTERNAL = "external"  # Fail-open: missing causes degraded mode


class FallbackMode(Enum):
    """Operational modes after resource loading"""
    FULL = "full"  # All resources loaded successfully
    DEGRADED = "degraded"  # Some external resources missing
    FAILED = "failed"  # Core resources missing - cannot proceed


@dataclass
class ResourceStatus:
    """Status of a loaded resource"""
    name: str
    resource_type: ResourceType
    loaded: bool
    error: Optional[str] = None
    degradation_impact: Optional[str] = None


@dataclass
class AuditConfig:
    """
    Audit configuration tracking resource loading status.
    
    Audit Point 1.4: External Resource Fallback
    Tracks which resources loaded successfully and operational mode.
    """
    mode: FallbackMode = FallbackMode.FULL
    resources: List[ResourceStatus] = field(default_factory=list)
    degradation_score: float = 0.0  # 0.0 = full, 1.0 = max degradation
    warnings: List[str] = field(default_factory=list)
    
    def add_resource(self, status: ResourceStatus) -> None:
        """Add resource loading status"""
        self.resources.append(status)
        
        if not status.loaded:
            if status.resource_type == ResourceType.CORE:
                self.mode = FallbackMode.FAILED
                self.warnings.append(
                    f"CRITICAL: Core resource '{status.name}' failed to load: {status.error}"
                )
            else:
                if self.mode == FallbackMode.FULL:
                    self.mode = FallbackMode.DEGRADED
                self.warnings.append(
                    f"WARNING: External resource '{status.name}' unavailable - degraded mode. "
                    f"Impact: {status.degradation_impact}"
                )
    
    def calculate_degradation(self) -> float:
        """
        Calculate overall degradation score.
        
        Audit Point 1.4: Degraded mode should have <10% accuracy loss.
        
        Returns:
            Degradation score (0.0 = no degradation, 1.0 = max degradation)
        """
        if self.mode == FallbackMode.FAILED:
            return 1.0
        
        external_resources = [r for r in self.resources if r.resource_type == ResourceType.EXTERNAL]
        if not external_resources:
            return 0.0
        
        failed_external = sum(1 for r in external_resources if not r.loaded)
        degradation = failed_external / len(external_resources)
        
        self.degradation_score = degradation
        return degradation
    
    def can_proceed(self) -> bool:
        """Check if pipeline can proceed (not in FAILED mode)"""
        return self.mode != FallbackMode.FAILED
    
    def get_summary(self) -> Dict[str, Any]:
        """Get audit summary"""
        return {
            "mode": self.mode.value,
            "degradation_score": self.calculate_degradation(),
            "core_resources_loaded": sum(
                1 for r in self.resources 
                if r.resource_type == ResourceType.CORE and r.loaded
            ),
            "external_resources_loaded": sum(
                1 for r in self.resources 
                if r.resource_type == ResourceType.EXTERNAL and r.loaded
            ),
            "warnings": self.warnings,
            "can_proceed": self.can_proceed()
        }


class ResourceLoader:
    """
    Safe resource loader with fail-closed/fail-open policy.
    
    Audit Point 1.4: External Resource Fallback
    """
    
    def __init__(self):
        self.config = AuditConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_resource(
        self,
        name: str,
        loader_func: Callable[[], Any],
        resource_type: ResourceType,
        degradation_impact: Optional[str] = None
    ) -> Optional[Any]:
        """
        Load resource with appropriate fallback policy.
        
        Audit Point 1.4:
        - Core resources: Fail-Closed (raise exception)
        - External resources: Fail-Open (return None, log warning)
        
        Args:
            name: Resource identifier
            loader_func: Function that loads the resource
            resource_type: CORE or EXTERNAL
            degradation_impact: Description of impact if external resource unavailable
            
        Returns:
            Loaded resource or None for failed external resources
            
        Raises:
            RuntimeError: If core resource fails to load (fail-closed)
        """
        try:
            resource = loader_func()
            status = ResourceStatus(
                name=name,
                resource_type=resource_type,
                loaded=True
            )
            self.config.add_resource(status)
            self.logger.info(f"[Audit 1.4] Resource '{name}' loaded successfully ({resource_type.value})")
            return resource
            
        except Exception as e:
            error_msg = str(e)
            status = ResourceStatus(
                name=name,
                resource_type=resource_type,
                loaded=False,
                error=error_msg,
                degradation_impact=degradation_impact
            )
            self.config.add_resource(status)
            
            if resource_type == ResourceType.CORE:
                # Fail-Closed: Core resource failure is unacceptable
                self.logger.error(
                    f"[Audit 1.4 FAIL-CLOSED] Core resource '{name}' failed: {error_msg}"
                )
                raise RuntimeError(
                    f"Critical failure loading core resource '{name}': {error_msg}. "
                    f"Pipeline cannot proceed (Audit Point 1.4 fail-closed policy)."
                ) from e
            else:
                # Fail-Open: External resource failure causes degradation
                self.logger.warning(
                    f"[Audit 1.4 FAIL-OPEN] External resource '{name}' unavailable: {error_msg}. "
                    f"Operating in degraded mode. Impact: {degradation_impact or 'Unknown'}"
                )
                return None
    
    def get_audit_config(self) -> AuditConfig:
        """Get current audit configuration status"""
        return self.config


# ============================================================================
# Pre-configured Resource Loaders
# ============================================================================

def load_core_schemas() -> Dict[str, Any]:
    """
    Load core Pydantic schemas (fail-closed).
    
    Audit Point 1.4: Core resource - must succeed or pipeline fails.
    """
    from extraction.extraction_pipeline import (
        SemanticChunk,
        ExtractedTable,
        DataQualityMetrics,
        ExtractionResult
    )
    
    return {
        "SemanticChunk": SemanticChunk,
        "ExtractedTable": ExtractedTable,
        "DataQualityMetrics": DataQualityMetrics,
        "ExtractionResult": ExtractionResult
    }


def load_teoria_cambio() -> Any:
    """
    Load TeoriaCambio (fail-closed).
    
    Audit Point 1.4: Core resource - must succeed or pipeline fails.
    """
    from teoria_cambio import TeoriaCambio
    return TeoriaCambio()


def load_dnp_validator(es_municipio_pdet: bool = False) -> Optional[Any]:
    """
    Load ValidadorDNP (fail-open).
    
    Audit Point 1.4: External resource - degraded mode if unavailable.
    """
    try:
        from dnp_integration import ValidadorDNP
        return ValidadorDNP(es_municipio_pdet=es_municipio_pdet)
    except ImportError as e:
        # Let ResourceLoader handle this gracefully
        raise ImportError(f"ValidadorDNP module unavailable: {e}")


def load_mga_indicators() -> Optional[Any]:
    """
    Load MGA indicators catalog (fail-open).
    
    Audit Point 1.4: External resource - degraded mode if unavailable.
    """
    try:
        from mga_indicadores import CATALOGO_MGA
        return CATALOGO_MGA
    except ImportError as e:
        raise ImportError(f"MGA indicators unavailable: {e}")


def load_pdet_lineamientos() -> Optional[Any]:
    """
    Load PDET lineamientos (fail-open).
    
    Audit Point 1.4: External resource - degraded mode if unavailable.
    """
    try:
        from pdet_lineamientos import LINEAMIENTOS_PDET
        return LINEAMIENTOS_PDET
    except ImportError as e:
        raise ImportError(f"PDET lineamientos unavailable: {e}")


# ============================================================================
# Main Audit Configuration Factory
# ============================================================================

def create_audit_configuration(es_municipio_pdet: bool = False) -> tuple[AuditConfig, Dict[str, Any]]:
    """
    Create audit configuration with all resources loaded.
    
    Audit Point 1.4: External Resource Fallback
    Implements fail-closed for core, fail-open for external.
    
    Args:
        es_municipio_pdet: Whether municipality is PDET (affects resource loading)
        
    Returns:
        Tuple of (AuditConfig, loaded_resources)
        
    Raises:
        RuntimeError: If any core resource fails to load
    """
    loader = ResourceLoader()
    resources = {}
    
    # Core resources (fail-closed)
    resources["schemas"] = loader.load_resource(
        name="CorePydanticSchemas",
        loader_func=load_core_schemas,
        resource_type=ResourceType.CORE
    )
    
    resources["teoria_cambio"] = loader.load_resource(
        name="TeoriaCambio",
        loader_func=load_teoria_cambio,
        resource_type=ResourceType.CORE
    )
    
    # External resources (fail-open)
    resources["dnp_validator"] = loader.load_resource(
        name="ValidadorDNP",
        loader_func=lambda: load_dnp_validator(es_municipio_pdet),
        resource_type=ResourceType.EXTERNAL,
        degradation_impact="DNP compliance validation disabled - manual review required"
    )
    
    resources["mga_indicators"] = loader.load_resource(
        name="MGAIndicators",
        loader_func=load_mga_indicators,
        resource_type=ResourceType.EXTERNAL,
        degradation_impact="MGA indicator validation disabled - score penalty ~5%"
    )
    
    if es_municipio_pdet:
        resources["pdet_lineamientos"] = loader.load_resource(
            name="PDETLineamientos",
            loader_func=load_pdet_lineamientos,
            resource_type=ResourceType.EXTERNAL,
            degradation_impact="PDET compliance validation disabled - score penalty ~10%"
        )
    
    audit_config = loader.get_audit_config()
    
    # Log final status
    summary = audit_config.get_summary()
    logger.info(f"[Audit 1.4] Resource loading complete: {summary}")
    
    if not audit_config.can_proceed():
        raise RuntimeError(
            "Pipeline cannot proceed - core resources failed to load. "
            f"Check audit warnings: {audit_config.warnings}"
        )
    
    if audit_config.mode == FallbackMode.DEGRADED:
        logger.warning(
            f"[Audit 1.4] Operating in DEGRADED mode. "
            f"Degradation score: {audit_config.degradation_score:.2%}"
        )
    
    return audit_config, resources


if __name__ == "__main__":
    """Test audit configuration"""
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Audit Point 1.4: External Resource Fallback")
    print("=" * 70)
    
    try:
        config, resources = create_audit_configuration(es_municipio_pdet=False)
        print("\nAudit Configuration Summary:")
        print(f"  Mode: {config.mode.value}")
        print(f"  Degradation Score: {config.degradation_score:.2%}")
        print(f"  Can Proceed: {config.can_proceed()}")
        print(f"\nLoaded Resources:")
        for name, resource in resources.items():
            status = "✓" if resource is not None else "✗"
            print(f"  {status} {name}")
        
        if config.warnings:
            print(f"\nWarnings ({len(config.warnings)}):")
            for warning in config.warnings:
                print(f"  - {warning}")
    
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        print("Pipeline halted (fail-closed policy)")
