#!/usr/bin/env python3
"""
Test script para Risk Mitigation Layer
Valida funcionalidad básica del módulo
"""

from risk_mitigation_layer import (
    RiskSeverity, RiskCategory, Risk, MitigationResult,
    RiskRegistry, RiskMitigationLayer,
    CriticalRiskUnmitigatedException, HighRiskUnmitigatedException,
    create_default_risk_registry
)
from dataclasses import dataclass


@dataclass
class MockContext:
    """Mock del PipelineContext para testing"""
    raw_text: str = ""
    sections: dict = None
    causal_chains: list = None
    nodes: dict = None
    financial_allocations: dict = None
    
    def __post_init__(self):
        if self.sections is None:
            self.sections = {}
        if self.causal_chains is None:
            self.causal_chains = []
        if self.nodes is None:
            self.nodes = {}
        if self.financial_allocations is None:
            self.financial_allocations = {}


def test_risk_registry():
    """Test RiskRegistry básico"""
    print("\n[TEST] RiskRegistry...")
    
    registry = RiskRegistry()
    
    # Registrar un riesgo de prueba
    test_risk = Risk(
        category=RiskCategory.EMPTY_DOCUMENT,
        severity=RiskSeverity.CRITICAL,
        probability=0.8,
        impact=1.0,
        detector_predicate=lambda ctx: len(ctx.raw_text) < 100,
        mitigation_strategy=lambda ctx: "fallback",
        description="Test risk"
    )
    
    registry.register_risk("TEST_STAGE", test_risk)
    
    # Verificar recuperación
    risks = registry.get_risks_for_stage("TEST_STAGE")
    assert len(risks) == 1, "Should have 1 risk registered"
    assert risks[0].category == RiskCategory.EMPTY_DOCUMENT
    
    print("  ✓ RiskRegistry works correctly")


def test_default_registry():
    """Test del registry predefinido"""
    print("\n[TEST] Default RiskRegistry...")
    
    registry = create_default_risk_registry()
    stats = registry.get_statistics()
    
    print(f"  Total risks: {stats['total_risks']}")
    print(f"  Stages covered: {stats['stages_covered']}")
    print(f"  By severity: {stats['by_severity']}")
    
    assert stats['total_risks'] > 0, "Should have predefined risks"
    print("  ✓ Default registry created successfully")


def test_risk_detection():
    """Test detección de riesgos"""
    print("\n[TEST] Risk Detection...")
    
    registry = create_default_risk_registry()
    mitigation_layer = RiskMitigationLayer(registry)
    
    # Contexto con documento vacío (debería detectar EMPTY_DOCUMENT)
    empty_ctx = MockContext(raw_text="short")
    detected = mitigation_layer.assess_stage_risks("STAGE_1_2", empty_ctx)
    
    assert len(detected) > 0, "Should detect empty document risk"
    print(f"  ✓ Detected {len(detected)} risks in empty document")
    
    # Contexto normal (no debería detectar riesgo EMPTY_DOCUMENT)
    normal_ctx = MockContext(raw_text="x" * 200)
    detected_normal = mitigation_layer.assess_stage_risks("STAGE_1_2", normal_ctx)
    
    # Puede detectar MISSING_SECTIONS pero no EMPTY_DOCUMENT
    empty_doc_detected = any(
        r.category == RiskCategory.EMPTY_DOCUMENT 
        for r in detected_normal
    )
    assert not empty_doc_detected, "Should not detect empty doc in normal context"
    print("  ✓ Risk detection working correctly")


def test_mitigation_execution():
    """Test ejecución de mitigación"""
    print("\n[TEST] Mitigation Execution...")
    
    # Crear riesgo que se puede mitigar
    mitigable_risk = Risk(
        category=RiskCategory.MISSING_SECTIONS,
        severity=RiskSeverity.MEDIUM,
        probability=0.5,
        impact=0.6,
        detector_predicate=lambda ctx: len(ctx.sections) < 3,
        mitigation_strategy=lambda ctx: setattr(ctx, 'sections', {'a': 1, 'b': 2, 'c': 3}) or "mitigated",
        description="Test mitigable risk"
    )
    
    registry = RiskRegistry()
    registry.register_risk("TEST_STAGE", mitigable_risk)
    
    mitigation_layer = RiskMitigationLayer(registry)
    
    ctx = MockContext(sections={'a': 1})  # Solo 1 sección
    result = mitigation_layer.execute_mitigation(mitigable_risk, ctx)
    
    assert result.attempts > 0, "Should have attempted mitigation"
    print(f"  ✓ Mitigation attempted {result.attempts} times")
    print(f"  ✓ Success: {result.success}")


def test_severity_escalation():
    """Test escalación por severidad"""
    print("\n[TEST] Severity-based Escalation...")
    
    # Test que CRITICAL debería tener 1 intento
    mitigation_layer = RiskMitigationLayer(RiskRegistry())
    
    critical_attempts = mitigation_layer._get_max_attempts(RiskSeverity.CRITICAL)
    high_attempts = mitigation_layer._get_max_attempts(RiskSeverity.HIGH)
    medium_attempts = mitigation_layer._get_max_attempts(RiskSeverity.MEDIUM)
    low_attempts = mitigation_layer._get_max_attempts(RiskSeverity.LOW)
    
    assert critical_attempts == 1, "CRITICAL should have 1 attempt"
    assert high_attempts == 2, "HIGH should have 2 attempts (1 retry)"
    assert medium_attempts == 3, "MEDIUM should have 3 attempts (2 retries)"
    assert low_attempts == 1, "LOW should have 1 attempt"
    
    print(f"  ✓ CRITICAL: {critical_attempts} attempt(s)")
    print(f"  ✓ HIGH: {high_attempts} attempt(s)")
    print(f"  ✓ MEDIUM: {medium_attempts} attempt(s)")
    print(f"  ✓ LOW: {low_attempts} attempt(s)")


def test_wrap_stage_execution():
    """Test wrapper de ejecución de etapa"""
    print("\n[TEST] Stage Execution Wrapper...")
    
    registry = RiskRegistry()
    mitigation_layer = RiskMitigationLayer(registry)
    
    # Mock stage function
    def mock_stage(ctx):
        ctx.stage_executed = True
        return ctx
    
    ctx = MockContext(raw_text="x" * 200)  # Normal context
    
    result = mitigation_layer.wrap_stage_execution(
        "TEST_STAGE",
        mock_stage,
        ctx
    )
    
    assert result.stage_executed, "Stage should have been executed"
    print("  ✓ Stage wrapper executed successfully")


def test_mitigation_report():
    """Test generación de reporte"""
    print("\n[TEST] Mitigation Report...")
    
    registry = create_default_risk_registry()
    mitigation_layer = RiskMitigationLayer(registry)
    
    # Ejecutar algunas detecciones
    ctx = MockContext(raw_text="short")
    detected = mitigation_layer.assess_stage_risks("STAGE_1_2", ctx)
    
    # Intentar mitigar
    for risk in detected[:2]:  # Solo los primeros 2
        mitigation_layer.execute_mitigation(risk, ctx)
    
    report = mitigation_layer.get_mitigation_report()
    
    print(f"  Total mitigations: {report['total_mitigations']}")
    print(f"  Successful: {report['successful']}")
    print(f"  Failed: {report['failed']}")
    
    assert report['total_mitigations'] > 0, "Should have mitigation records"
    print("  ✓ Mitigation report generated successfully")


def main():
    print("="*80)
    print("RISK MITIGATION LAYER - TEST SUITE")
    print("="*80)
    
    try:
        test_risk_registry()
        test_default_registry()
        test_risk_detection()
        test_mitigation_execution()
        test_severity_escalation()
        test_wrap_stage_execution()
        test_mitigation_report()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
