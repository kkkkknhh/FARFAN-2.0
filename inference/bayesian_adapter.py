#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapter/Bridge for integrating refactored Bayesian engine with existing code.

This module provides backwards compatibility while using the new refactored
BayesianPriorBuilder, BayesianSamplingEngine, and NecessitySufficiencyTester.
"""

from typing import Any, Dict, List, Optional
import logging
import numpy as np

try:
    from inference.bayesian_engine import (
        BayesianPriorBuilder,
        BayesianSamplingEngine,
        NecessitySufficiencyTester,
        MechanismPrior,
        MechanismEvidence,
        DocumentEvidence,
        CausalLink,
        ColombianMunicipalContext
    )
    REFACTORED_ENGINE_AVAILABLE = True
except ImportError:
    REFACTORED_ENGINE_AVAILABLE = False


class BayesianEngineAdapter:
    """
    Adapter to use refactored Bayesian engine components in existing code.
    
    This allows gradual migration from the monolithic BayesianMechanismInference
    to the separated BayesianPriorBuilder, BayesianSamplingEngine, and
    NecessitySufficiencyTester classes.
    """
    
    def __init__(self, config, nlp_model):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.nlp = nlp_model
        
        if REFACTORED_ENGINE_AVAILABLE:
            self.logger.info("Using refactored Bayesian engine components")
            self.prior_builder = BayesianPriorBuilder()
            self.sampling_engine = BayesianSamplingEngine(seed=42)
            self.necessity_tester = NecessitySufficiencyTester()
        else:
            self.logger.warning("Refactored engine not available, will use legacy methods")
            self.prior_builder = None
            self.sampling_engine = None
            self.necessity_tester = None
    
    def is_available(self) -> bool:
        """Check if refactored engine is available"""
        return REFACTORED_ENGINE_AVAILABLE
    
    def test_necessity_from_observations(
        self,
        node_id: str,
        observations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test necessity using refactored NecessitySufficiencyTester.
        
        Converts old observations format to new DocumentEvidence format.
        """
        if not self.necessity_tester:
            # Fallback to legacy format
            return self._legacy_necessity_test(observations)
        
        # Create DocumentEvidence from observations
        doc_evidence = DocumentEvidence()
        
        if observations.get('entity_activity'):
            entity = observations['entity_activity'].get('entity')
            if entity:
                doc_evidence.entities[node_id] = [entity]
        
        if observations.get('verbs'):
            # Assume verbs represent activities
            # In real implementation, would need cause-effect pairs
            pass
        
        if observations.get('budget'):
            doc_evidence.budgets[node_id] = observations['budget']
        
        # Create a simple CausalLink (would need more context in real use)
        # For now, just testing necessity of node itself
        link = CausalLink(
            cause_id=node_id,
            effect_id=f"{node_id}_effect",
            cause_emb=np.zeros(384),  # Placeholder
            effect_emb=np.zeros(384),  # Placeholder
            cause_type='producto',
            effect_type='resultado'
        )
        
        result = self.necessity_tester.test_necessity(link, doc_evidence)
        
        # Convert to legacy format
        return {
            'score': 1.0 if result.passed else 0.5,
            'is_necessary': result.passed,
            'alternatives_likely': not result.passed,
            'missing_components': result.missing,
            'remediation': result.remediation
        }
    
    def _legacy_necessity_test(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy necessity test format"""
        # Simple heuristic-based test
        entities = observations.get('entities', [])
        unique_entity = len(set(entities)) == 1 if entities else False
        
        verbs = observations.get('verbs', [])
        specific_verbs = len([v for v in verbs if v in [
            'implementar', 'ejecutar', 'realizar', 'desarrollar'
        ]]) > 0
        
        necessity_score = (
            (0.5 if unique_entity else 0.3) +
            (0.5 if specific_verbs else 0.3)
        )
        
        return {
            'score': necessity_score,
            'is_necessary': necessity_score >= 0.7,
            'alternatives_likely': necessity_score < 0.5
        }
    
    def build_prior_from_node(
        self,
        node,
        observations: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[MechanismPrior]:
        """
        Build a MechanismPrior from node and observations.
        
        This is a helper for gradual migration.
        """
        if not self.prior_builder:
            return None
        
        # Extract mechanism evidence from observations
        mechanism_evidence = MechanismEvidence(
            type='administrativo',  # Default, would infer in real use
            verb_sequence=observations.get('verbs', []),
            entity=observations.get('entity_activity', {}).get('entity'),
            budget=observations.get('budget')
        )
        
        # Create causal link (simplified - would need more context)
        link = CausalLink(
            cause_id=node.id,
            effect_id=f"{node.id}_effect",
            cause_emb=np.zeros(384),  # Would use real embeddings
            effect_emb=np.zeros(384),
            cause_type=node.type,
            effect_type='resultado'
        )
        
        # Create context
        pdm_context = ColombianMunicipalContext()
        
        # Build prior
        prior = self.prior_builder.build_mechanism_prior(
            link,
            mechanism_evidence,
            pdm_context
        )
        
        return prior
    
    def get_component_status(self) -> Dict[str, bool]:
        """Get status of refactored components"""
        return {
            'refactored_engine_available': REFACTORED_ENGINE_AVAILABLE,
            'prior_builder_ready': self.prior_builder is not None,
            'sampling_engine_ready': self.sampling_engine is not None,
            'necessity_tester_ready': self.necessity_tester is not None
        }
