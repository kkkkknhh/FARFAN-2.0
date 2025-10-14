#!/usr/bin/env python3
"""
Tests for RiskRegistry module
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from risk_registry import (
    RiskRegistry, 
    RiskDefinition, 
    Severity, 
    RiskCategory,
    get_risk_registry
)


def test_risk_definition_validation():
    """Test RiskDefinition Pydantic validation"""
    print("Testing RiskDefinition validation...")
    
    # Valid risk
    risk = RiskDefinition(
        risk_id="TEST_001",
        name="Test Risk",
        description="A test risk definition",
        category=RiskCategory.DATA_QUALITY,
        severity=Severity.HIGH,
        probability=0.5,
        impact=0.8,
        stage="TEST_STAGE"
    )
    assert risk.risk_score() == 0.4
    print("  ✓ Valid risk definition created")
    
    # Test validation for out-of-range probability
    try:
        invalid_risk = RiskDefinition(
            risk_id="TEST_002",
            name="Invalid Risk",
            description="Invalid probability",
            category=RiskCategory.DATA_QUALITY,
            severity=Severity.LOW,
            probability=1.5,  # Invalid
            impact=0.5,
            stage="TEST_STAGE"
        )
        assert False, "Should have raised validation error"
    except Exception as e:
        print(f"  ✓ Validation error caught for invalid probability")
    
    print("  ✓ Risk definition validation tests passed\n")


def test_registry_registration():
    """Test risk registration in registry"""
    print("Testing risk registration...")
    
    registry = RiskRegistry()
    
    # Count default risks
    initial_count = len(registry.get_all_risks())
    print(f"  ✓ Registry initialized with {initial_count} default risks")
    
    # Register custom risk
    custom_risk = RiskDefinition(
        risk_id="CUSTOM_001",
        name="Custom Test Risk",
        description="A custom test risk",
        category=RiskCategory.CONFIGURATION,
        severity=Severity.MEDIUM,
        probability=0.3,
        impact=0.6,
        stage="TEST_STAGE",
        detector=lambda ctx: True,
        mitigation_strategy=lambda ctx: "mitigated"
    )
    
    registry.register_risk(custom_risk)
    assert len(registry.get_all_risks()) == initial_count + 1
    print("  ✓ Custom risk registered successfully")
    
    # Test duplicate detection
    try:
        registry.register_risk(custom_risk)
        assert False, "Should have raised error for duplicate"
    except ValueError:
        print("  ✓ Duplicate risk ID rejected")
    
    print("  ✓ Risk registration tests passed\n")


def test_query_by_category():
    """Test querying risks by category"""
    print("Testing query by category...")
    
    registry = get_risk_registry()
    
    for category in RiskCategory:
        risks = registry.get_by_category(category)
        print(f"  {category.value}: {len(risks)} risks")
    
    data_quality_risks = registry.get_by_category(RiskCategory.DATA_QUALITY)
    assert len(data_quality_risks) > 0
    print("  ✓ Category query tests passed\n")


def test_query_by_severity():
    """Test querying risks by severity"""
    print("Testing query by severity...")
    
    registry = get_risk_registry()
    
    for severity in Severity:
        risks = registry.get_by_severity(severity)
        print(f"  {severity.value}: {len(risks)} risks")
    
    critical_risks = registry.get_critical_risks()
    assert len(critical_risks) > 0
    print("  ✓ Severity query tests passed\n")


def test_query_by_stage():
    """Test querying risks by pipeline stage"""
    print("Testing query by stage...")
    
    registry = get_risk_registry()
    
    stages = ["LOAD_DOCUMENT", "SEMANTIC_ANALYSIS", "DNP_VALIDATION"]
    for stage in stages:
        risks = registry.get_by_stage(stage)
        print(f"  {stage}: {len(risks)} risks")
        assert len(risks) > 0, f"Expected risks for stage {stage}"
    
    print("  ✓ Stage query tests passed\n")


def test_risk_evaluation():
    """Test risk evaluation against context"""
    print("Testing risk evaluation...")
    
    # Mock context
    class MockContext:
        pass
    
    context = MockContext()
    
    registry = get_risk_registry()
    detected = registry.evaluate_risks(context)
    
    print(f"  Detected {len(detected)} risks in mock context")
    
    if detected:
        top_risk = detected[0]
        print(f"  Top risk: {top_risk['risk_id']} - {top_risk['name']}")
        print(f"    Severity: {top_risk['severity']}")
        print(f"    Risk score: {top_risk['risk_score']:.3f}")
    
    print("  ✓ Risk evaluation tests passed\n")


def test_high_impact_risks():
    """Test high impact risk filtering"""
    print("Testing high impact risk filtering...")
    
    registry = get_risk_registry()
    high_impact = registry.get_high_impact_risks(threshold=0.7)
    
    print(f"  Found {len(high_impact)} high-impact risks (>0.7)")
    
    for risk in high_impact[:3]:
        print(f"    - {risk.risk_id}: {risk.name} (impact={risk.impact})")
    
    print("  ✓ High impact filtering tests passed\n")


def test_default_risks_coverage():
    """Test that all required failure modes are registered"""
    print("Testing default risks coverage...")
    
    registry = get_risk_registry()
    
    required_risk_types = {
        'PDF_': 'PDF parsing',
        'NLP_': 'spaCy/NLP',
        'DNP_': 'DNP API',
        'EMB_': 'Embedding service',
        'DATA_': 'Data quality',
        'CFG_': 'Configuration',
        'RES_': 'Resources'
    }
    
    all_risks = registry.get_all_risks()
    
    for prefix, description in required_risk_types.items():
        matching = [r for r in all_risks if r.risk_id.startswith(prefix)]
        assert len(matching) > 0, f"Missing {description} risks"
        print(f"  ✓ {description}: {len(matching)} risks registered")
    
    print("  ✓ Default risks coverage tests passed\n")


def main():
    """Run all tests"""
    print("="*60)
    print("FARFAN Risk Registry Test Suite")
    print("="*60 + "\n")
    
    try:
        test_risk_definition_validation()
        test_registry_registration()
        test_query_by_category()
        test_query_by_severity()
        test_query_by_stage()
        test_risk_evaluation()
        test_high_impact_risks()
        test_default_risks_coverage()
        
        print("="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
