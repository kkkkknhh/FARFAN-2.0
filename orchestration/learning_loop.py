#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Learning Loop
Implements Front D.1: Prior Learning desde failures históricos
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class MechanismPrior:
    """Prior distribution for mechanism type"""
    mechanism_type: str
    alpha: float
    beta: float = 2.0
    last_updated: str = ""
    update_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mechanism_type': self.mechanism_type,
            'alpha': self.alpha,
            'beta': self.beta,
            'last_updated': self.last_updated,
            'update_count': self.update_count
        }


class PriorHistoryStore:
    """
    Store and manage historical priors for mechanism types.
    Implements immutable snapshots for audit trail.
    """
    
    def __init__(self, store_path: Optional[Path] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.store_path = store_path or Path("prior_history.json")
        self.priors: Dict[str, MechanismPrior] = {}
        self.snapshots: List[Dict[str, Any]] = []
        
        # Load existing priors if available
        self._load_priors()
    
    def _load_priors(self) -> None:
        """Load priors from persistent storage"""
        if self.store_path.exists():
            try:
                with open(self.store_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if 'priors' in data:
                    for mech_type, prior_data in data['priors'].items():
                        self.priors[mech_type] = MechanismPrior(
                            mechanism_type=mech_type,
                            alpha=prior_data.get('alpha', 2.0),
                            beta=prior_data.get('beta', 2.0),
                            last_updated=prior_data.get('last_updated', ''),
                            update_count=prior_data.get('update_count', 0)
                        )
                
                if 'snapshots' in data:
                    self.snapshots = data['snapshots']
                
                self.logger.info(f"Loaded {len(self.priors)} priors from {self.store_path}")
            except Exception as e:
                self.logger.warning(f"Could not load priors from {self.store_path}: {e}")
    
    def get_mechanism_prior(self, mechanism_type: str) -> MechanismPrior:
        """Get prior for mechanism type, creating default if not exists"""
        if mechanism_type not in self.priors:
            # Default prior (weakly informative)
            self.priors[mechanism_type] = MechanismPrior(
                mechanism_type=mechanism_type,
                alpha=2.0,
                beta=2.0,
                last_updated=pd.Timestamp.now().isoformat(),
                update_count=0
            )
        return self.priors[mechanism_type]
    
    def update_mechanism_prior(
        self,
        mechanism_type: str,
        new_alpha: float,
        reason: str,
        timestamp: Optional[pd.Timestamp] = None
    ) -> None:
        """Update prior for mechanism type"""
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        prior = self.get_mechanism_prior(mechanism_type)
        old_alpha = prior.alpha
        
        prior.alpha = new_alpha
        prior.last_updated = timestamp.isoformat()
        prior.update_count += 1
        
        self.logger.info(
            f"Updated prior for {mechanism_type}: "
            f"alpha {old_alpha:.3f} -> {new_alpha:.3f} ({reason})"
        )
    
    def save_snapshot(self) -> None:
        """Save current state as immutable snapshot"""
        snapshot = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'priors': {k: v.to_dict() for k, v in self.priors.items()}
        }
        self.snapshots.append(snapshot)
        
        # Persist to disk
        try:
            data = {
                'priors': {k: v.to_dict() for k, v in self.priors.items()},
                'snapshots': self.snapshots
            }
            
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.store_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved prior snapshot to {self.store_path}")
        except Exception as e:
            self.logger.error(f"Failed to save prior snapshot: {e}")
    
    def get_history(self, mechanism_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get historical snapshots for mechanism type"""
        if mechanism_type is None:
            return self.snapshots
        
        # Filter snapshots for specific mechanism type
        filtered = []
        for snapshot in self.snapshots:
            if mechanism_type in snapshot.get('priors', {}):
                filtered.append({
                    'timestamp': snapshot['timestamp'],
                    'prior': snapshot['priors'][mechanism_type]
                })
        return filtered


@dataclass
class Feedback:
    """Feedback extracted from analysis results"""
    failed_mechanism_types: List[str] = field(default_factory=list)
    passed_mechanism_types: List[str] = field(default_factory=list)
    necessity_test_failures: Dict[str, List[str]] = field(default_factory=dict)
    overall_quality: float = 0.0


class FeedbackExtractor:
    """
    Extract learning feedback from analysis results.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_from_result(self, analysis_result: Any) -> Feedback:
        """
        Extract feedback from AnalysisResult.
        
        Args:
            analysis_result: AnalysisResult object with mechanism_results
        
        Returns:
            Feedback object with extracted learning signals
        """
        feedback = Feedback()
        
        # Extract from mechanism results
        if hasattr(analysis_result, 'mechanism_results'):
            for mech in analysis_result.mechanism_results:
                mech_type = getattr(mech, 'type', 'unknown')
                necessity_test = getattr(mech, 'necessity_test', {})
                
                if isinstance(necessity_test, dict):
                    passed = necessity_test.get('passed', True)
                    missing = necessity_test.get('missing', [])
                else:
                    passed = getattr(necessity_test, 'passed', True)
                    missing = getattr(necessity_test, 'missing', [])
                
                if not passed:
                    feedback.failed_mechanism_types.append(mech_type)
                    feedback.necessity_test_failures[mech_type] = missing
                else:
                    feedback.passed_mechanism_types.append(mech_type)
        
        # Extract overall quality
        if hasattr(analysis_result, 'quality_score'):
            quality_score = analysis_result.quality_score
            if hasattr(quality_score, 'overall_score'):
                feedback.overall_quality = quality_score.overall_score
            elif isinstance(quality_score, (int, float)):
                feedback.overall_quality = float(quality_score)
        
        self.logger.debug(
            f"Extracted feedback: {len(feedback.failed_mechanism_types)} failed mechanisms, "
            f"{len(feedback.passed_mechanism_types)} passed mechanisms"
        )
        
        return feedback


class AdaptiveLearningLoop:
    """
    Implementa Front D.1: Prior Learning desde failures históricos.
    """
    
    def __init__(self, config: Any):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract config values
        if hasattr(config, 'self_reflection'):
            self_reflection = config.self_reflection
            prior_history_path = getattr(self_reflection, 'prior_history_path', None)
            self.enabled = getattr(self_reflection, 'enable_prior_learning', False)
            self.prior_decay_factor = getattr(config, 'prior_decay_factor', 0.9)
        else:
            prior_history_path = None
            self.enabled = False
            self.prior_decay_factor = 0.9
        
        # Initialize components
        if prior_history_path:
            self.prior_store = PriorHistoryStore(Path(prior_history_path))
        else:
            self.prior_store = PriorHistoryStore()
        
        self.feedback_extractor = FeedbackExtractor()
        
        self.logger.info(
            f"AdaptiveLearningLoop initialized (enabled={self.enabled}, "
            f"decay_factor={self.prior_decay_factor})"
        )
    
    def extract_and_update_priors(
        self,
        analysis_result: Any
    ) -> None:
        """
        Extrae feedback de failures y actualiza priors para futuro.
        
        Args:
            analysis_result: AnalysisResult object from PDMOrchestrator
        """
        if not self.enabled:
            self.logger.debug("Prior learning disabled, skipping update")
            return
        
        # Extract feedback
        feedback = self.feedback_extractor.extract_from_result(analysis_result)
        
        # Identificar mechanism types que fallaron necessity tests
        failed_mechanisms = feedback.failed_mechanism_types
        
        for mech_type in set(failed_mechanisms):  # Use set to avoid duplicates
            # Decay del prior para este tipo (Front D.1)
            current_prior = self.prior_store.get_mechanism_prior(mech_type)
            updated_alpha = current_prior.alpha * self.prior_decay_factor
            
            # Get failure reason
            missing_items = feedback.necessity_test_failures.get(mech_type, [])
            reason = f"Necessity test failed: {missing_items}" if missing_items else "Necessity test failed"
            
            self.prior_store.update_mechanism_prior(
                mechanism_type=mech_type,
                new_alpha=updated_alpha,
                reason=reason,
                timestamp=pd.Timestamp.now()
            )
        
        # Optionally boost priors for passed mechanisms
        passed_mechanisms = feedback.passed_mechanism_types
        for mech_type in set(passed_mechanisms):
            current_prior = self.prior_store.get_mechanism_prior(mech_type)
            # Small boost for passing (less aggressive than decay)
            boost_factor = 1.0 + (1.0 - self.prior_decay_factor) / 2.0
            updated_alpha = min(current_prior.alpha * boost_factor, 10.0)  # Cap at 10.0
            
            if updated_alpha != current_prior.alpha:
                self.prior_store.update_mechanism_prior(
                    mechanism_type=mech_type,
                    new_alpha=updated_alpha,
                    reason="Necessity test passed",
                    timestamp=pd.Timestamp.now()
                )
        
        # Persistir histórico inmutable
        self.prior_store.save_snapshot()
        
        self.logger.info(
            f"Updated priors: {len(failed_mechanisms)} decayed, "
            f"{len(passed_mechanisms)} boosted"
        )
    
    def get_current_prior(self, mechanism_type: str) -> float:
        """Get current prior alpha for mechanism type"""
        prior = self.prior_store.get_mechanism_prior(mechanism_type)
        return prior.alpha
    
    def get_prior_history(self, mechanism_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get historical prior snapshots"""
        return self.prior_store.get_history(mechanism_type)
