#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration of IoR Causal Axiomatic-Bayesian Integration
Shows how the three audit points work in practice

This script demonstrates:
- Audit Point 2.1: Structural Veto (D6-Q2) - Caps posterior for impermissible links
- Audit Point 2.2: Mechanism Necessity Hoop Test - Validates Entity, Activity, Budget
- Audit Point 2.3: Policy Alignment Dual Constraint - Risk multiplier for low alignment
"""

import logging
from typing import Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_audit_point_21():
    """Demonstrate Audit Point 2.1: Structural Veto (D6-Q2)"""
    print("\n" + "=" * 80)
    print("AUDIT POINT 2.1: Structural Veto (D6-Q2)")
    print("=" * 80)
    print("\nPurpose: Ensure structural rules constrain Bayesian inferences")
    print("Per Goertz & Mahoney 2012 on set-theoretic constraints\n")
    
    # Scenario 1: Valid link (producto → resultado)
    print("Scenario 1: Valid Link (producto → resultado)")
    print("-" * 40)
    source_type = 'producto'
    target_type = 'resultado'
    posterior_semantic = 0.85  # High semantic similarity
    
    violation = check_structural_violation(source_type, target_type)
    if violation:
        posterior_final = min(posterior_semantic, 0.6)
        print(f"❌ VIOLATION DETECTED: {violation}")
        print(f"   Posterior capped from {posterior_semantic:.2f} to {posterior_final:.2f}")
    else:
        posterior_final = posterior_semantic
        print(f"✓ Valid structural link")
        print(f"   Posterior: {posterior_final:.2f} (unchanged)")
    
    # Scenario 2: Invalid link (producto → impacto)
    print("\nScenario 2: Invalid Link (producto → impacto)")
    print("-" * 40)
    source_type = 'producto'
    target_type = 'impacto'
    posterior_semantic = 0.92  # Very high semantic similarity
    
    violation = check_structural_violation(source_type, target_type)
    if violation:
        posterior_final = min(posterior_semantic, 0.6)
        print(f"❌ VIOLATION DETECTED: {violation}")
        print(f"   Posterior capped from {posterior_semantic:.2f} to {posterior_final:.2f}")
        print(f"   Despite high semantic evidence, structural rules prevail")
    else:
        posterior_final = posterior_semantic
        print(f"✓ Valid structural link")
        print(f"   Posterior: {posterior_final:.2f}")
    
    # Scenario 3: Reverse causation (impacto → producto)
    print("\nScenario 3: Reverse Causation (impacto → producto)")
    print("-" * 40)
    source_type = 'impacto'
    target_type = 'producto'
    posterior_semantic = 0.78
    
    violation = check_structural_violation(source_type, target_type)
    if violation:
        posterior_final = min(posterior_semantic, 0.6)
        print(f"❌ VIOLATION DETECTED: {violation}")
        print(f"   Posterior capped from {posterior_semantic:.2f} to {posterior_final:.2f}")
    else:
        posterior_final = posterior_semantic
        print(f"✓ Valid structural link")
        print(f"   Posterior: {posterior_final:.2f}")


def demonstrate_audit_point_22():
    """Demonstrate Audit Point 2.2: Mechanism Necessity Hoop Test"""
    print("\n" + "=" * 80)
    print("AUDIT POINT 2.2: Mechanism Necessity Hoop Test")
    print("=" * 80)
    print("\nPurpose: Validate documented Entity, Activity, Budget")
    print("Per Beach 2017 Hoop Tests & Falleti-Lynch 2009 mechanism depth\n")
    
    # Scenario 1: Complete documentation
    print("Scenario 1: Complete Documentation")
    print("-" * 40)
    observations = {
        'entity_activity': {'entity': 'Secretaría de Planeación'},
        'entities': ['Secretaría de Planeación'],
        'verbs': ['implementar', 'ejecutar', 'coordinar', 'supervisar'],
        'budget': 75000000
    }
    
    result = evaluate_necessity_hoop_test(observations)
    print(f"Entity: ✓ {observations['entity_activity']['entity']}")
    print(f"Activity: ✓ {len(observations['verbs'])} documented verbs")
    print(f"Budget: ✓ COP ${observations['budget']:,}")
    print(f"\n→ Hoop Test: {'PASSED' if result['is_necessary'] else 'FAILED'}")
    print(f"   Necessity Score: {result['score']:.2f}")
    
    # Scenario 2: Missing entity
    print("\nScenario 2: Missing Entity")
    print("-" * 40)
    observations = {
        'entity_activity': None,
        'entities': [],
        'verbs': ['implementar', 'ejecutar'],
        'budget': 50000000
    }
    
    result = evaluate_necessity_hoop_test(observations)
    print(f"Entity: ✗ Not documented")
    print(f"Activity: ✓ {len(observations['verbs'])} documented verbs")
    print(f"Budget: ✓ COP ${observations['budget']:,}")
    print(f"\n→ Hoop Test: {'PASSED' if result['is_necessary'] else 'FAILED'}")
    print(f"   Missing: {', '.join(result['missing_components'])}")
    print(f"   Necessity Score: {result['score']:.2f}")
    
    # Scenario 3: Missing multiple components
    print("\nScenario 3: Multiple Missing Components")
    print("-" * 40)
    observations = {
        'entity_activity': None,
        'entities': [],
        'verbs': [],
        'budget': None
    }
    
    result = evaluate_necessity_hoop_test(observations)
    print(f"Entity: ✗ Not documented")
    print(f"Activity: ✗ Not documented")
    print(f"Budget: ✗ Not documented")
    print(f"\n→ Hoop Test: {'PASSED' if result['is_necessary'] else 'FAILED'}")
    print(f"   Missing: {', '.join(result['missing_components'])}")
    print(f"   Necessity Score: {result['score']:.2f}")
    print(f"   Status: CRITICAL - Mechanism cannot be validated")


def demonstrate_audit_point_23():
    """Demonstrate Audit Point 2.3: Policy Alignment Dual Constraint"""
    print("\n" + "=" * 80)
    print("AUDIT POINT 2.3: Policy Alignment Dual Constraint")
    print("=" * 80)
    print("\nPurpose: Integrate macro-micro causality via alignment scores")
    print("Per Lieberman 2015 & UN 2020 ODS benchmarks\n")
    
    # Scenario 1: High alignment, low base risk
    print("Scenario 1: High Alignment (0.75), Low Base Risk (0.08)")
    print("-" * 40)
    base_risk = 0.08
    pdet_alignment = 0.75
    
    result = calculate_systemic_risk_with_alignment(base_risk, pdet_alignment)
    print(f"Base Risk Score: {base_risk:.3f}")
    print(f"PDET Alignment: {pdet_alignment:.2f}")
    print(f"Alignment Penalty: {'Applied' if result['alignment_penalty_applied'] else 'Not Applied'}")
    print(f"Final Risk Score: {result['risk_score']:.3f}")
    print(f"Quality Rating: {result['d5_q4_quality'].upper()}")
    
    # Scenario 2: Low alignment, low base risk
    print("\nScenario 2: Low Alignment (0.50), Low Base Risk (0.08)")
    print("-" * 40)
    base_risk = 0.08
    pdet_alignment = 0.50
    
    result = calculate_systemic_risk_with_alignment(base_risk, pdet_alignment)
    print(f"Base Risk Score: {base_risk:.3f}")
    print(f"PDET Alignment: {pdet_alignment:.2f}")
    print(f"Alignment Penalty: {'Applied (1.2×)' if result['alignment_penalty_applied'] else 'Not Applied'}")
    print(f"Final Risk Score: {result['risk_score']:.3f} (escalated from {base_risk:.3f})")
    print(f"Quality Rating: {result['d5_q4_quality'].upper()}")
    if result['alignment_causing_failure']:
        print(f"⚠️  WARNING: Low alignment degraded quality from EXCELENTE to {result['d5_q4_quality'].upper()}")
    
    # Scenario 3: Alignment penalty pushes over threshold
    print("\nScenario 3: Alignment Penalty Causes Quality Downgrade")
    print("-" * 40)
    base_risk = 0.09
    pdet_alignment = 0.55
    
    result = calculate_systemic_risk_with_alignment(base_risk, pdet_alignment)
    print(f"Base Risk Score: {base_risk:.3f}")
    print(f"PDET Alignment: {pdet_alignment:.2f}")
    print(f"Alignment Penalty: {'Applied (1.2×)' if result['alignment_penalty_applied'] else 'Not Applied'}")
    print(f"Final Risk Score: {result['risk_score']:.3f}")
    print(f"Quality Rating: {result['d5_q4_quality'].upper()}")
    print(f"\nRisk Thresholds:")
    print(f"  - Excelente: < {result['risk_thresholds']['excellent']:.2f}")
    print(f"  - Bueno: < {result['risk_thresholds']['good']:.2f}")
    print(f"  - Aceptable: < {result['risk_thresholds']['acceptable']:.2f}")
    if result['alignment_causing_failure']:
        print(f"\n⚠️  CRITICAL: Alignment penalty caused quality failure (D5-Q4)")
        print(f"   Without penalty: Risk={base_risk:.3f} → EXCELENTE")
        print(f"   With penalty: Risk={result['risk_score']:.3f} → {result['d5_q4_quality'].upper()}")


# Helper functions (extracted from implementation)

def check_structural_violation(source_type: str, target_type: str) -> Optional[str]:
    """Check if causal link violates structural hierarchy"""
    hierarchy_levels = {
        'programa': 1,
        'producto': 2,
        'resultado': 3,
        'impacto': 4
    }
    
    source_level = hierarchy_levels.get(source_type, 0)
    target_level = hierarchy_levels.get(target_type, 0)
    
    if target_level < source_level:
        return f"reverse_causation:{source_type}→{target_type}"
    
    if target_level - source_level > 2:
        return f"level_skip:{source_type}→{target_type} (skips {target_level - source_level - 1} levels)"
    
    if source_type == 'producto' and target_type == 'impacto':
        return f"missing_intermediate:producto→impacto requires resultado"
    
    return None


def evaluate_necessity_hoop_test(observations: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate necessity hoop test"""
    missing_components = []
    
    # Check Entity
    entity_activity = observations.get('entity_activity')
    if not entity_activity or not entity_activity.get('entity'):
        missing_components.append('entity')
    
    # Check Activity
    verbs = observations.get('verbs', [])
    if not verbs or len(verbs) < 1:
        missing_components.append('activity')
    
    # Check Budget
    budget = observations.get('budget')
    if budget is None or budget <= 0:
        missing_components.append('budget')
    
    is_necessary = len(missing_components) == 0
    max_components = 3
    present_components = max_components - len([c for c in missing_components if c in ['entity', 'activity', 'budget']])
    necessity_score = present_components / max_components
    
    return {
        'score': necessity_score,
        'is_necessary': is_necessary,
        'missing_components': missing_components,
        'hoop_test_passed': is_necessary
    }


def calculate_systemic_risk_with_alignment(base_risk: float, pdet_alignment: Optional[float]) -> Dict[str, Any]:
    """Calculate systemic risk with alignment constraint"""
    alignment_threshold = 0.60
    alignment_multiplier = 1.2
    
    risk_score = base_risk
    original_risk = base_risk
    alignment_penalty_applied = False
    
    if pdet_alignment is not None and pdet_alignment <= alignment_threshold:
        risk_score = risk_score * alignment_multiplier
        alignment_penalty_applied = True
    
    # Quality assessment
    d5_q4_quality = 'insuficiente'
    risk_threshold_excellent = 0.10
    risk_threshold_good = 0.20
    risk_threshold_acceptable = 0.35
    
    if risk_score < risk_threshold_excellent:
        d5_q4_quality = 'excelente'
    elif risk_score < risk_threshold_good:
        d5_q4_quality = 'bueno'
    elif risk_score < risk_threshold_acceptable:
        d5_q4_quality = 'aceptable'
    
    alignment_causing_failure = (
        alignment_penalty_applied and
        original_risk < risk_threshold_excellent and
        risk_score >= risk_threshold_excellent
    )
    
    return {
        'risk_score': risk_score,
        'alignment_penalty_applied': alignment_penalty_applied,
        'd5_q4_quality': d5_q4_quality,
        'alignment_causing_failure': alignment_causing_failure,
        'risk_thresholds': {
            'excellent': risk_threshold_excellent,
            'good': risk_threshold_good,
            'acceptable': risk_threshold_acceptable
        }
    }


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("IoR CAUSAL AXIOMATIC-BAYESIAN INTEGRATION - DEMONSTRATION")
    print("Part 2: Phase II/III Wiring")
    print("=" * 80)
    
    demonstrate_audit_point_21()
    demonstrate_audit_point_22()
    demonstrate_audit_point_23()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Structural rules cap Bayesian posteriors at 0.6 for invalid links (D6-Q2)")
    print("2. Hoop tests require Entity, Activity, Budget documentation (Beach 2017)")
    print("3. Low alignment scores (≤0.60) escalate systemic risk by 1.2× (D5-Q4)")
    print("\nAll three audit points ensure structural rules constrain Bayesian inferences,")
    print("preventing logical jumps and over-inference per SOTA axiomatic-Bayesian fusion.\n")
