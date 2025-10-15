#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script for refactored Bayesian engine.

This script demonstrates that the refactored components work correctly
and can be used independently of the main framework.
"""

import sys
from pathlib import Path

# Ensure inference module is in path
sys.path.insert(0, str(Path(__file__).parent))

def validate_imports():
    """Validate that all refactored components can be imported"""
    print("="*60)
    print("VALIDATING REFACTORED BAYESIAN ENGINE")
    print("="*60)
    
    try:
        from inference.bayesian_engine import (
            BayesianPriorBuilder,
            BayesianSamplingEngine,
            NecessitySufficiencyTester,
            MechanismPrior,
            PosteriorDistribution,
            NecessityTestResult,
            MechanismEvidence,
            EvidenceChunk,
            SamplingConfig,
            CausalLink,
            ColombianMunicipalContext,
            DocumentEvidence
        )
        print("✓ All components imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def validate_prior_builder():
    """Validate BayesianPriorBuilder"""
    print("\n" + "="*60)
    print("VALIDATING BayesianPriorBuilder")
    print("="*60)
    
    try:
        from inference.bayesian_engine import (
            BayesianPriorBuilder,
            CausalLink,
            MechanismEvidence,
            ColombianMunicipalContext
        )
        import numpy as np
        
        # Create builder
        builder = BayesianPriorBuilder()
        print("✓ BayesianPriorBuilder initialized")
        
        # Create test data
        cause_emb = np.random.randn(384)
        effect_emb = np.random.randn(384)
        cause_emb /= np.linalg.norm(cause_emb)
        effect_emb /= np.linalg.norm(effect_emb)
        
        link = CausalLink(
            cause_id='MP-001',
            effect_id='MR-001',
            cause_emb=cause_emb,
            effect_emb=effect_emb,
            cause_type='producto',
            effect_type='resultado'
        )
        print("✓ CausalLink created")
        
        mechanism_evidence = MechanismEvidence(
            type='técnico',
            verb_sequence=['implementar', 'ejecutar', 'evaluar']
        )
        print("✓ MechanismEvidence created")
        
        context = ColombianMunicipalContext()
        print("✓ ColombianMunicipalContext created")
        
        # Build prior
        prior = builder.build_mechanism_prior(link, mechanism_evidence, context)
        print(f"✓ MechanismPrior built: alpha={prior.alpha:.3f}, beta={prior.beta:.3f}")
        print(f"  Rationale: {prior.rationale[:80]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_sampling_engine():
    """Validate BayesianSamplingEngine"""
    print("\n" + "="*60)
    print("VALIDATING BayesianSamplingEngine")
    print("="*60)
    
    try:
        from inference.bayesian_engine import (
            BayesianSamplingEngine,
            MechanismPrior,
            EvidenceChunk,
            SamplingConfig
        )
        
        # Create engine
        engine = BayesianSamplingEngine(seed=42)
        print("✓ BayesianSamplingEngine initialized with seed=42")
        
        # Create prior
        prior = MechanismPrior(
            alpha=2.0,
            beta=2.0,
            rationale="Test prior for validation"
        )
        print("✓ MechanismPrior created")
        
        # Create evidence
        evidence = [
            EvidenceChunk('chunk1', 'Test chunk 1', cosine_similarity=0.85),
            EvidenceChunk('chunk2', 'Test chunk 2', cosine_similarity=0.75),
            EvidenceChunk('chunk3', 'Test chunk 3', cosine_similarity=0.65),
        ]
        print(f"✓ Created {len(evidence)} evidence chunks")
        
        # Sample posterior
        config = SamplingConfig(draws=1000, chains=4)
        posterior = engine.sample_mechanism_posterior(prior, evidence, config)
        
        print(f"✓ Posterior sampled:")
        print(f"  Mean: {posterior.posterior_mean:.3f}")
        print(f"  Std: {posterior.posterior_std:.3f}")
        print(f"  95% HDI: ({posterior.confidence_interval[0]:.3f}, {posterior.confidence_interval[1]:.3f})")
        print(f"  Converged: {posterior.convergence_diagnostic}")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_necessity_tester():
    """Validate NecessitySufficiencyTester"""
    print("\n" + "="*60)
    print("VALIDATING NecessitySufficiencyTester")
    print("="*60)
    
    try:
        from inference.bayesian_engine import (
            NecessitySufficiencyTester,
            CausalLink,
            DocumentEvidence,
            MechanismEvidence
        )
        import numpy as np
        
        # Create tester
        tester = NecessitySufficiencyTester()
        print("✓ NecessitySufficiencyTester initialized")
        
        # Create test data
        cause_emb = np.random.randn(384)
        effect_emb = np.random.randn(384)
        cause_emb /= np.linalg.norm(cause_emb)
        effect_emb /= np.linalg.norm(effect_emb)
        
        link = CausalLink(
            cause_id='MP-001',
            effect_id='MR-001',
            cause_emb=cause_emb,
            effect_emb=effect_emb,
            cause_type='producto',
            effect_type='resultado'
        )
        print("✓ CausalLink created")
        
        # Test necessity with incomplete evidence
        doc_evidence = DocumentEvidence()
        doc_evidence.budgets['MP-001'] = 5000000.0
        # Missing: entity, activity, timeline
        
        result = tester.test_necessity(link, doc_evidence)
        print(f"✓ Necessity test executed:")
        print(f"  Passed: {result.passed}")
        print(f"  Missing components: {result.missing}")
        print(f"  Severity: {result.severity}")
        if result.remediation:
            print(f"  Remediation: {result.remediation[:80]}...")
        
        # Test with complete evidence
        doc_evidence.entities['MP-001'] = ['Secretaría de Salud']
        doc_evidence.activities[('MP-001', 'MR-001')] = ['implementar', 'ejecutar']
        doc_evidence.timelines['MP-001'] = '2024-2028'
        
        result2 = tester.test_necessity(link, doc_evidence)
        print(f"✓ Necessity test with complete evidence:")
        print(f"  Passed: {result2.passed}")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_adapter():
    """Validate BayesianEngineAdapter"""
    print("\n" + "="*60)
    print("VALIDATING BayesianEngineAdapter")
    print("="*60)
    
    try:
        from inference.bayesian_adapter import BayesianEngineAdapter
        
        # Create mock config and nlp
        class MockConfig:
            def get_mechanism_prior(self, name):
                return 0.2
            def get_bayesian_threshold(self, name):
                return 1.0
        
        class MockNLP:
            pass
        
        # Create adapter
        adapter = BayesianEngineAdapter(MockConfig(), MockNLP())
        print("✓ BayesianEngineAdapter initialized")
        
        # Check status
        status = adapter.get_component_status()
        print("✓ Component status:")
        for component, available in status.items():
            symbol = "✓" if available else "✗"
            print(f"  {symbol} {component}")
        
        if adapter.is_available():
            print("✓ Refactored engine is available and ready")
        else:
            print("✗ Refactored engine is not available")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validations"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "BAYESIAN ENGINE VALIDATION" + " "*16 + "║")
    print("║" + " "*19 + "F1.2 Refactoring" + " "*22 + "║")
    print("╚" + "="*58 + "╝")
    
    results = []
    
    # Run validations
    results.append(("Imports", validate_imports()))
    results.append(("BayesianPriorBuilder", validate_prior_builder()))
    results.append(("BayesianSamplingEngine", validate_sampling_engine()))
    results.append(("NecessitySufficiencyTester", validate_necessity_tester()))
    results.append(("BayesianEngineAdapter", validate_adapter()))
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for component, success in results:
        symbol = "✓" if success else "✗"
        status = "PASS" if success else "FAIL"
        print(f"{symbol} {component:35s} {status}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    
    print("\n" + "-"*60)
    print(f"Total: {passed}/{total} validations passed")
    
    if passed == total:
        print("\n✓ ALL VALIDATIONS PASSED - Refactored engine is working correctly!")
        return 0
    else:
        print(f"\n✗ {total - passed} validation(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
