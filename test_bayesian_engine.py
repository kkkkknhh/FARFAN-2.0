#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for refactored Bayesian Engine
Tests BayesianPriorBuilder, BayesianSamplingEngine, and NecessitySufficiencyTester
"""

import unittest
import numpy as np
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

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


class TestBayesianPriorBuilder(unittest.TestCase):
    """Test suite for BayesianPriorBuilder"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.builder = BayesianPriorBuilder()
        
        # Create sample embeddings
        self.cause_emb = np.random.randn(384)
        self.effect_emb = np.random.randn(384)
        self.context_emb = np.random.randn(384)
        
        # Normalize embeddings
        self.cause_emb /= np.linalg.norm(self.cause_emb)
        self.effect_emb /= np.linalg.norm(self.effect_emb)
        self.context_emb /= np.linalg.norm(self.context_emb)
    
    def test_builder_initialization(self):
        """Test that builder initializes correctly"""
        self.assertIsNotNone(self.builder)
        self.assertIsInstance(self.builder.type_transitions, dict)
        self.assertIsInstance(self.builder.mechanism_type_verbs, dict)
    
    def test_build_mechanism_prior(self):
        """Test building a mechanism prior"""
        link = CausalLink(
            cause_id='MP-001',
            effect_id='MR-001',
            cause_emb=self.cause_emb,
            effect_emb=self.effect_emb,
            cause_type='producto',
            effect_type='resultado'
        )
        
        mechanism_evidence = MechanismEvidence(
            type='técnico',
            verb_sequence=['implementar', 'ejecutar', 'evaluar']
        )
        
        context = ColombianMunicipalContext(
            overall_pdm_embedding=self.context_emb
        )
        
        prior = self.builder.build_mechanism_prior(link, mechanism_evidence, context)
        
        self.assertIsInstance(prior, MechanismPrior)
        self.assertGreater(prior.alpha, 0)
        self.assertGreater(prior.beta, 0)
        self.assertIsInstance(prior.rationale, str)
    
    def test_semantic_distance_calculation(self):
        """Test semantic distance calculation"""
        # Create two similar embeddings
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.9, 0.1, 0.0])
        
        # Normalize
        emb1 /= np.linalg.norm(emb1)
        emb2 /= np.linalg.norm(emb2)
        
        distance = self.builder._calculate_semantic_distance(emb1, emb2)
        
        self.assertGreaterEqual(distance, 0.0)
        self.assertLessEqual(distance, 1.0)
    
    def test_type_transition_prior(self):
        """Test type transition prior calculation"""
        # Test known transition
        prior = self.builder._get_type_transition_prior('producto', 'resultado')
        self.assertEqual(prior, 0.8)
        
        # Test unknown transition (should return default)
        prior = self.builder._get_type_transition_prior('unknown', 'unknown')
        self.assertEqual(prior, 0.5)
    
    def test_mechanism_type_coherence(self):
        """Test mechanism type coherence validation"""
        # Technical verbs should score high for technical type
        verb_sequence = ['implementar', 'diseñar', 'ejecutar']
        coherence = self.builder._validate_mechanism_type_coherence(
            verb_sequence,
            'producto',
            'resultado'
        )
        
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)
    
    def test_beta_params_computation(self):
        """Test Beta parameter computation"""
        alpha, beta = self.builder._compute_beta_params(
            base_strength=0.7,
            type_coherence=0.8,
            semantic_distance=0.2,
            type_transition=0.75,
            historical_priors=[]
        )
        
        self.assertGreater(alpha, 0)
        self.assertGreater(beta, 0)
        # Alpha should be larger than beta for high strength
        self.assertGreater(alpha, beta)


class TestBayesianSamplingEngine(unittest.TestCase):
    """Test suite for BayesianSamplingEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = BayesianSamplingEngine(seed=42)
    
    def test_engine_initialization(self):
        """Test that engine initializes correctly"""
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.rng)
    
    def test_sample_mechanism_posterior_with_evidence(self):
        """Test sampling posterior with evidence"""
        prior = MechanismPrior(
            alpha=2.0,
            beta=2.0,
            rationale="Test prior"
        )
        
        evidence = [
            EvidenceChunk(
                chunk_id='chunk1',
                text='Test chunk 1',
                cosine_similarity=0.85
            ),
            EvidenceChunk(
                chunk_id='chunk2',
                text='Test chunk 2',
                cosine_similarity=0.75
            ),
            EvidenceChunk(
                chunk_id='chunk3',
                text='Test chunk 3',
                cosine_similarity=0.65
            )
        ]
        
        config = SamplingConfig(draws=1000, chains=4)
        
        posterior = self.engine.sample_mechanism_posterior(prior, evidence, config)
        
        self.assertIsInstance(posterior, PosteriorDistribution)
        self.assertGreaterEqual(posterior.posterior_mean, 0.0)
        self.assertLessEqual(posterior.posterior_mean, 1.0)
        self.assertGreaterEqual(posterior.posterior_std, 0.0)
        self.assertIsNotNone(posterior.samples)
    
    def test_sample_mechanism_posterior_no_evidence(self):
        """Test sampling posterior without evidence (prior as posterior)"""
        prior = MechanismPrior(
            alpha=2.0,
            beta=2.0,
            rationale="Test prior"
        )
        
        config = SamplingConfig(draws=1000)
        
        posterior = self.engine.sample_mechanism_posterior(prior, [], config)
        
        self.assertIsInstance(posterior, PosteriorDistribution)
        # Should be close to prior mean (0.5 for alpha=beta=2)
        self.assertAlmostEqual(posterior.posterior_mean, 0.5, delta=0.1)
    
    def test_similarity_to_probability(self):
        """Test similarity to probability conversion"""
        # High similarity should give high probability
        prob_high = self.engine._similarity_to_probability(0.9, tau=1.0)
        self.assertGreater(prob_high, 0.7)
        
        # Low similarity should give low probability
        prob_low = self.engine._similarity_to_probability(0.1, tau=1.0)
        self.assertLess(prob_low, 0.3)
        
        # Mid similarity should be around 0.5
        prob_mid = self.engine._similarity_to_probability(0.5, tau=1.0)
        self.assertAlmostEqual(prob_mid, 0.5, delta=0.1)
    
    def test_hdi_extraction(self):
        """Test HDI extraction"""
        samples = np.random.beta(5, 2, size=1000)
        hdi = self.engine._extract_hdi(samples, credible_mass=0.95)
        
        self.assertIsInstance(hdi, tuple)
        self.assertEqual(len(hdi), 2)
        self.assertLess(hdi[0], hdi[1])
    
    def test_reproducibility(self):
        """Test that sampling is reproducible with same seed"""
        prior = MechanismPrior(alpha=3.0, beta=2.0, rationale="Test")
        evidence = [
            EvidenceChunk('c1', 'text', 0.8),
            EvidenceChunk('c2', 'text', 0.7)
        ]
        config = SamplingConfig(draws=100)
        
        engine1 = BayesianSamplingEngine(seed=123)
        engine2 = BayesianSamplingEngine(seed=123)
        
        posterior1 = engine1.sample_mechanism_posterior(prior, evidence, config)
        posterior2 = engine2.sample_mechanism_posterior(prior, evidence, config)
        
        # Means should be very close
        self.assertAlmostEqual(
            posterior1.posterior_mean,
            posterior2.posterior_mean,
            delta=0.01
        )


class TestNecessitySufficiencyTester(unittest.TestCase):
    """Test suite for NecessitySufficiencyTester"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tester = NecessitySufficiencyTester()
        
        # Create sample embeddings
        self.cause_emb = np.random.randn(384)
        self.effect_emb = np.random.randn(384)
        
        # Normalize
        self.cause_emb /= np.linalg.norm(self.cause_emb)
        self.effect_emb /= np.linalg.norm(self.effect_emb)
    
    def test_tester_initialization(self):
        """Test that tester initializes correctly"""
        self.assertIsNotNone(self.tester)
    
    def test_necessity_test_pass(self):
        """Test necessity test when all components present"""
        link = CausalLink(
            cause_id='MP-001',
            effect_id='MR-001',
            cause_emb=self.cause_emb,
            effect_emb=self.effect_emb,
            cause_type='producto',
            effect_type='resultado'
        )
        
        doc_evidence = DocumentEvidence()
        doc_evidence.entities['MP-001'] = ['Secretaría de Salud']
        doc_evidence.activities[('MP-001', 'MR-001')] = ['implementar', 'ejecutar']
        doc_evidence.budgets['MP-001'] = 5000000.0
        doc_evidence.timelines['MP-001'] = '2024-2028'
        
        result = self.tester.test_necessity(link, doc_evidence)
        
        self.assertIsInstance(result, NecessityTestResult)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.missing), 0)
    
    def test_necessity_test_fail(self):
        """Test necessity test when components missing"""
        link = CausalLink(
            cause_id='MP-001',
            effect_id='MR-001',
            cause_emb=self.cause_emb,
            effect_emb=self.effect_emb,
            cause_type='producto',
            effect_type='resultado'
        )
        
        doc_evidence = DocumentEvidence()
        # Only budget present, missing entity, activity, timeline
        doc_evidence.budgets['MP-001'] = 5000000.0
        
        result = self.tester.test_necessity(link, doc_evidence)
        
        self.assertIsInstance(result, NecessityTestResult)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.missing), 0)
        self.assertIn('entity', result.missing)
        self.assertIn('activity', result.missing)
        self.assertEqual(result.severity, 'critical')
    
    def test_sufficiency_test_pass(self):
        """Test sufficiency test when components adequate"""
        link = CausalLink(
            cause_id='MP-001',
            effect_id='MR-001',
            cause_emb=self.cause_emb,
            effect_emb=self.effect_emb,
            cause_type='producto',
            effect_type='resultado'
        )
        
        doc_evidence = DocumentEvidence()
        doc_evidence.entities['MP-001'] = ['Secretaría de Salud']
        
        mechanism_evidence = MechanismEvidence(
            type='técnico',
            verb_sequence=['planificar', 'implementar', 'ejecutar', 'evaluar'],
            entity='Secretaría de Salud',
            budget=10000000.0
        )
        
        result = self.tester.test_sufficiency(link, doc_evidence, mechanism_evidence)
        
        self.assertIsInstance(result, NecessityTestResult)
        self.assertTrue(result.passed)
    
    def test_sufficiency_test_fail_inadequate_budget(self):
        """Test sufficiency test with inadequate budget"""
        link = CausalLink(
            cause_id='MP-001',
            effect_id='MR-001',
            cause_emb=self.cause_emb,
            effect_emb=self.effect_emb,
            cause_type='producto',
            effect_type='resultado'
        )
        
        doc_evidence = DocumentEvidence()
        
        mechanism_evidence = MechanismEvidence(
            type='técnico',
            verb_sequence=['planificar', 'implementar', 'ejecutar'],
            entity='Secretaría de Salud',
            budget=500000.0  # Below threshold
        )
        
        result = self.tester.test_sufficiency(link, doc_evidence, mechanism_evidence)
        
        self.assertIsInstance(result, NecessityTestResult)
        self.assertFalse(result.passed)
        self.assertIn('adequate_budget', result.missing)
    
    def test_necessity_result_to_dict(self):
        """Test conversion of result to dictionary"""
        result = NecessityTestResult(
            passed=False,
            missing=['entity', 'budget'],
            severity='critical',
            remediation='Test remediation'
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['passed'], False)
        self.assertEqual(len(result_dict['missing']), 2)
        self.assertEqual(result_dict['severity'], 'critical')


class TestDataStructures(unittest.TestCase):
    """Test suite for data structures"""
    
    def test_mechanism_prior_validation(self):
        """Test MechanismPrior validation"""
        # Valid prior
        prior = MechanismPrior(alpha=2.0, beta=3.0, rationale="Test")
        self.assertEqual(prior.alpha, 2.0)
        self.assertEqual(prior.beta, 3.0)
        
        # Invalid prior (negative alpha)
        with self.assertRaises(ValueError):
            MechanismPrior(alpha=-1.0, beta=2.0, rationale="Test")
        
        # Invalid prior (zero beta)
        with self.assertRaises(ValueError):
            MechanismPrior(alpha=2.0, beta=0.0, rationale="Test")
    
    def test_posterior_distribution_hdi(self):
        """Test PosteriorDistribution HDI calculation"""
        samples = np.random.beta(5, 2, size=1000)
        posterior = PosteriorDistribution(
            posterior_mean=np.mean(samples),
            posterior_std=np.std(samples),
            samples=samples
        )
        
        hdi = posterior.get_hdi(credible_mass=0.95)
        
        self.assertIsInstance(hdi, tuple)
        self.assertEqual(len(hdi), 2)
        self.assertLess(hdi[0], hdi[1])
    
    def test_document_evidence_methods(self):
        """Test DocumentEvidence helper methods"""
        doc_ev = DocumentEvidence()
        
        # Test has_entity
        self.assertFalse(doc_ev.has_entity('MP-001'))
        doc_ev.entities['MP-001'] = ['Entity1']
        self.assertTrue(doc_ev.has_entity('MP-001'))
        
        # Test has_activity_sequence
        self.assertFalse(doc_ev.has_activity_sequence('MP-001', 'MR-001'))
        doc_ev.activities[('MP-001', 'MR-001')] = ['verb1', 'verb2']
        self.assertTrue(doc_ev.has_activity_sequence('MP-001', 'MR-001'))
        
        # Test has_budget_trace
        self.assertFalse(doc_ev.has_budget_trace('MP-001'))
        doc_ev.budgets['MP-001'] = 1000000.0
        self.assertTrue(doc_ev.has_budget_trace('MP-001'))


if __name__ == '__main__':
    unittest.main()
