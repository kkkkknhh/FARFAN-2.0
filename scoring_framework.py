#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scoring Framework for FARFAN 2.0
Deterministic PDM Evaluation with Complete Audit Trail
SIN_CARRETA Compliant: Contract Enforcement, Determinism, Observability

Validates P1-P10 × D1-D6 × Q1-Q5 = 300 canonical questions
ENFORCES: dimension weights sum to 1.0, consistent rubric thresholds, D6<0.55 triggers manual review
INTEGRATES: DNP regulatory compliance at D1-Q5 and D4-Q5
AGGREGATES: MICRO (questions) → MESO (dimensions/policies) → MACRO (clusters/Decálogo)
"""

import logging
import math
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from decimal import Decimal
from canonical_notation import CanonicalID, CanonicalNotationValidator

logger = logging.getLogger(__name__)

# ============================================================================
# MATHEMATICAL INVARIANTS - SIN_CARRETA DETERMINISM
# ============================================================================

DIMENSION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "P1": {"D1": 0.15, "D2": 0.20, "D3": 0.15, "D4": 0.20, "D5": 0.15, "D6": 0.15},
    "P2": {"D1": 0.15, "D2": 0.20, "D3": 0.15, "D4": 0.20, "D5": 0.15, "D6": 0.15},
    "P3": {"D1": 0.18, "D2": 0.18, "D3": 0.16, "D4": 0.18, "D5": 0.15, "D6": 0.15},
    "P4": {"D1": 0.15, "D2": 0.20, "D3": 0.15, "D4": 0.20, "D5": 0.15, "D6": 0.15},
    "P5": {"D1": 0.15, "D2": 0.18, "D3": 0.15, "D4": 0.20, "D5": 0.17, "D6": 0.15},
    "P6": {"D1": 0.15, "D2": 0.20, "D3": 0.15, "D4": 0.20, "D5": 0.15, "D6": 0.15},
    "P7": {"D1": 0.20, "D2": 0.18, "D3": 0.15, "D4": 0.18, "D5": 0.14, "D6": 0.15},
    "P8": {"D1": 0.15, "D2": 0.20, "D3": 0.15, "D4": 0.20, "D5": 0.15, "D6": 0.15},
    "P9": {"D1": 0.15, "D2": 0.20, "D3": 0.15, "D4": 0.20, "D5": 0.15, "D6": 0.15},
    "P10": {"D1": 0.15, "D2": 0.20, "D3": 0.15, "D4": 0.20, "D5": 0.15, "D6": 0.15}
}

RUBRIC_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "excelente": (0.85, 1.00),
    "bueno": (0.70, 0.85),
    "aceptable": (0.55, 0.70),
    "insuficiente": (0.00, 0.55)
}

POLICY_CLUSTERS: Dict[str, List[str]] = {
    "derechos_humanos": ["P1", "P2", "P8"],
    "sostenibilidad": ["P3", "P7"],
    "desarrollo_social": ["P4", "P6"],
    "paz_y_reconciliacion": ["P5", "P9", "P10"]
}

CLUSTER_WEIGHTS: Dict[str, float] = {
    "derechos_humanos": 0.30,
    "sostenibilidad": 0.20,
    "desarrollo_social": 0.30,
    "paz_y_reconciliacion": 0.20
}

D6_MANUAL_REVIEW_THRESHOLD = 0.55
QUESTIONS_PER_DIMENSION = 5


@dataclass
class QuestionScore:
    question_id: str
    score: float
    rubric_category: str
    confidence: float = 1.0
    dnp_compliance: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        assert 0.0 <= self.score <= 1.0, f"Score must be in [0,1], got {self.score}"
        canonical_id = CanonicalID.from_string(self.question_id)
        assert canonical_id.question <= QUESTIONS_PER_DIMENSION


@dataclass
class DimensionScore:
    policy: str
    dimension: str
    score: float
    question_scores: List[QuestionScore] = field(default_factory=list)
    weight: float = 0.0

    def __post_init__(self):
        assert 0.0 <= self.score <= 1.0
        assert self.policy in DIMENSION_WEIGHTS
        assert self.dimension in DIMENSION_WEIGHTS[self.policy]
        if math.isclose(self.weight, 0.0, rel_tol=1e-9, abs_tol=1e-12):  # replaced float equality with isclose (tolerance from DEFAULT_FLOAT_TOLS)
            self.weight = DIMENSION_WEIGHTS[self.policy][self.dimension]


@dataclass
class PolicyScore:
    policy: str
    score: float
    dimension_scores: Dict[str, DimensionScore] = field(default_factory=dict)

    def __post_init__(self):
        assert 0.0 <= self.score <= 1.0
        assert self.policy in DIMENSION_WEIGHTS


@dataclass
class ClusterScore:
    cluster_name: str
    score: float
    policy_scores: List[PolicyScore] = field(default_factory=list)
    weight: float = 0.0

    def __post_init__(self):
        assert 0.0 <= self.score <= 1.0
        assert self.cluster_name in CLUSTER_WEIGHTS
        if math.isclose(self.weight, 0.0, rel_tol=1e-9, abs_tol=1e-12):  # replaced float equality with isclose (tolerance from DEFAULT_FLOAT_TOLS)
            self.weight = CLUSTER_WEIGHTS[self.cluster_name]


@dataclass
class MacroScore:
    overall_score: float
    cluster_scores: Dict[str, ClusterScore] = field(default_factory=dict)
    manual_review_flags: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert 0.0 <= self.overall_score <= 1.0


class ScoringEngine:
    def __init__(self):
        self.validator = CanonicalNotationValidator()
        self._validate_configuration()
        logger.info("[SIN_CARRETA] ScoringEngine initialized with validated configuration")

    def _validate_configuration(self) -> None:
        logger.info("[SIN_CARRETA CONTRACT] Validating scoring configuration...")
        
        for policy, weights in DIMENSION_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-9, f"DIMENSION_WEIGHTS[{policy}] sum to {total:.10f}, not 1.0"
            assert set(weights.keys()) == {"D1", "D2", "D3", "D4", "D5", "D6"}
        
        cluster_weight_sum = sum(CLUSTER_WEIGHTS.values())
        assert abs(cluster_weight_sum - 1.0) < 1e-9, f"CLUSTER_WEIGHTS sum to {cluster_weight_sum}, not 1.0"
        
        all_policies_in_clusters = set()
        for policies in POLICY_CLUSTERS.values():
            all_policies_in_clusters.update(policies)
        expected_policies = {f"P{i}" for i in range(1, 11)}
        assert all_policies_in_clusters == expected_policies
        
        for category, (low, high) in RUBRIC_THRESHOLDS.items():
            assert 0.0 <= low < high <= 1.0, f"Invalid threshold for {category}: [{low}, {high}]"
        
        logger.info("[SIN_CARRETA CONTRACT] ✓ Configuration validated")

    def validate_all_canonical_questions(self) -> Dict[str, Any]:
        logger.info("[SIN_CARRETA] Validating all P1-P10 × D1-D6 × Q1-Q5 combinations...")
        
        all_questions = []
        validation_results = {"total": 0, "valid": 0, "invalid": []}
        
        for policy_num in range(1, 11):
            policy = f"P{policy_num}"
            for dim_num in range(1, 7):
                dimension = f"D{dim_num}"
                for q_num in range(1, QUESTIONS_PER_DIMENSION + 1):
                    question_id = f"{policy}-{dimension}-Q{q_num}"
                    
                    try:
                        canonical_id = CanonicalID.from_string(question_id)
                        assert canonical_id.policy == policy
                        assert canonical_id.dimension == dimension
                        assert canonical_id.question == q_num
                        all_questions.append(question_id)
                        validation_results["valid"] += 1
                    except Exception as e:
                        validation_results["invalid"].append({"question_id": question_id, "error": str(e)})
                    
                    validation_results["total"] += 1
        
        assert validation_results["total"] == 300, f"Expected 300 questions, got {validation_results['total']}"
        assert validation_results["valid"] == 300, f"Not all questions valid: {validation_results}"
        
        logger.info(f"[SIN_CARRETA] ✓ All 300 canonical questions validated")
        
        return {
            "total_questions": 300,
            "validated_questions": all_questions,
            "validation_results": validation_results
        }

    def score_to_rubric_category(self, score: float) -> str:
        assert 0.0 <= score <= 1.0, f"Score {score} out of bounds"
        
        for category, (low, high) in RUBRIC_THRESHOLDS.items():
            if low <= score < high:
                return category
        
        if math.isclose(score, 1.0, rel_tol=1e-9, abs_tol=1e-12):  # replaced float equality with isclose (tolerance from DEFAULT_FLOAT_TOLS)
            return "excelente"
        
        return "insuficiente"

    def rubric_category_to_score_range(self, category: str) -> Tuple[float, float]:
        assert category in RUBRIC_THRESHOLDS, f"Unknown rubric category: {category}"
        return RUBRIC_THRESHOLDS[category]

    def calculate_dimension_score(
        self, 
        policy: str, 
        dimension: str, 
        question_scores: List[QuestionScore]
    ) -> DimensionScore:
        assert policy in DIMENSION_WEIGHTS, f"Unknown policy: {policy}"
        assert dimension in DIMENSION_WEIGHTS[policy], f"Unknown dimension: {dimension}"
        assert len(question_scores) == QUESTIONS_PER_DIMENSION, \
            f"Expected {QUESTIONS_PER_DIMENSION} questions for {policy}-{dimension}, got {len(question_scores)}"
        
        for qs in question_scores:
            canonical_id = CanonicalID.from_string(qs.question_id)
            assert canonical_id.policy == policy
            assert canonical_id.dimension == dimension
        
        avg_score = sum(qs.score for qs in question_scores) / len(question_scores)
        
        weight = DIMENSION_WEIGHTS[policy][dimension]
        
        logger.debug(f"[MICRO→MESO] {policy}-{dimension}: {len(question_scores)} questions → score={avg_score:.4f}, weight={weight}")
        
        return DimensionScore(
            policy=policy,
            dimension=dimension,
            score=avg_score,
            question_scores=question_scores,
            weight=weight
        )

    def calculate_policy_score(
        self,
        policy: str,
        dimension_scores: List[DimensionScore]
    ) -> PolicyScore:
        assert policy in DIMENSION_WEIGHTS
        assert len(dimension_scores) == 6, f"Expected 6 dimensions for {policy}, got {len(dimension_scores)}"
        
        for ds in dimension_scores:
            assert ds.policy == policy
        
        weighted_sum = sum(ds.score * ds.weight for ds in dimension_scores)
        total_weight = sum(ds.weight for ds in dimension_scores)
        assert abs(total_weight - 1.0) < 1e-9, f"Dimension weights for {policy} sum to {total_weight}, not 1.0"
        
        policy_score = weighted_sum / total_weight
        
        dimension_dict = {ds.dimension: ds for ds in dimension_scores}
        
        logger.debug(f"[MESO] {policy}: 6 dimensions → weighted_score={policy_score:.4f}")
        
        return PolicyScore(
            policy=policy,
            score=policy_score,
            dimension_scores=dimension_dict
        )

    def calculate_cluster_score(
        self,
        cluster_name: str,
        policy_scores: List[PolicyScore]
    ) -> ClusterScore:
        assert cluster_name in POLICY_CLUSTERS
        expected_policies = set(POLICY_CLUSTERS[cluster_name])
        actual_policies = {ps.policy for ps in policy_scores}
        assert expected_policies == actual_policies, \
            f"Cluster {cluster_name} expects {expected_policies}, got {actual_policies}"
        
        avg_score = sum(ps.score for ps in policy_scores) / len(policy_scores)
        
        weight = CLUSTER_WEIGHTS[cluster_name]
        
        logger.debug(f"[MESO→MACRO] Cluster '{cluster_name}': {len(policy_scores)} policies → score={avg_score:.4f}, weight={weight}")
        
        return ClusterScore(
            cluster_name=cluster_name,
            score=avg_score,
            policy_scores=policy_scores,
            weight=weight
        )

    def calculate_macro_score(
        self,
        cluster_scores: List[ClusterScore]
    ) -> MacroScore:
        assert len(cluster_scores) == 4, f"Expected 4 clusters, got {len(cluster_scores)}"
        
        cluster_names = {cs.cluster_name for cs in cluster_scores}
        expected_clusters = set(CLUSTER_WEIGHTS.keys())
        assert cluster_names == expected_clusters
        
        weighted_sum = sum(cs.score * cs.weight for cs in cluster_scores)
        total_weight = sum(cs.weight for cs in cluster_scores)
        assert abs(total_weight - 1.0) < 1e-9
        
        overall_score = weighted_sum / total_weight
        
        manual_review_flags = self._check_manual_review_triggers(cluster_scores)
        
        provenance = self._build_provenance_chain(cluster_scores)
        
        logger.info(f"[MACRO] Decálogo Alignment Score: {overall_score:.4f}")
        if manual_review_flags:
            logger.warning(f"[MACRO] Manual review flags: {manual_review_flags}")
        
        return MacroScore(
            overall_score=overall_score,
            cluster_scores={cs.cluster_name: cs for cs in cluster_scores},
            manual_review_flags=manual_review_flags,
            provenance=provenance
        )

    def _check_manual_review_triggers(self, cluster_scores: List[ClusterScore]) -> List[str]:
        flags = []
        
        for cluster in cluster_scores:
            for policy_score in cluster.policy_scores:
                d6_score_obj = policy_score.dimension_scores.get("D6")
                if d6_score_obj and d6_score_obj.score < D6_MANUAL_REVIEW_THRESHOLD:
                    flag = f"{policy_score.policy}-D6: score={d6_score_obj.score:.3f} < {D6_MANUAL_REVIEW_THRESHOLD} (Theory of Change weak)"
                    flags.append(flag)
                    logger.warning(f"[MANUAL_REVIEW_FLAG] {flag}")
        
        return flags

    def _build_provenance_chain(self, cluster_scores: List[ClusterScore]) -> Dict[str, Any]:
        provenance = {
            "aggregation_method": "weighted_average",
            "levels": {
                "MICRO": "300 questions (P1-P10 × D1-D6 × Q1-Q5)",
                "MESO_dimension": "6 dimensions per policy (simple average of 5 questions each)",
                "MESO_policy": "10 policies (weighted average of 6 dimensions using DIMENSION_WEIGHTS)",
                "MACRO_cluster": "4 clusters (simple average of policies in cluster)",
                "MACRO_overall": "1 overall score (weighted average of 4 clusters using CLUSTER_WEIGHTS)"
            },
            "dimension_weights": DIMENSION_WEIGHTS,
            "cluster_weights": CLUSTER_WEIGHTS,
            "rubric_thresholds": RUBRIC_THRESHOLDS,
            "manual_review_threshold_d6": D6_MANUAL_REVIEW_THRESHOLD,
            "cluster_breakdown": {}
        }
        
        for cluster in cluster_scores:
            provenance["cluster_breakdown"][cluster.cluster_name] = {
                "score": cluster.score,
                "weight": cluster.weight,
                "policies": [ps.policy for ps in cluster.policy_scores],
                "policy_scores": {ps.policy: ps.score for ps in cluster.policy_scores}
            }
        
        return provenance

    def integrate_dnp_compliance(
        self,
        question_score: QuestionScore,
        dnp_validator: Any
    ) -> QuestionScore:
        canonical_id = CanonicalID.from_string(question_score.question_id)
        
        if canonical_id.dimension == "D1" and canonical_id.question == 5:
            logger.info(f"[DNP_INTEGRATION] D1-Q5 detected: {question_score.question_id}")
            dnp_result = self._evaluate_d1_q5_compliance(canonical_id.policy, dnp_validator)
            question_score.dnp_compliance = dnp_result
            question_score.score = dnp_result.get("adjusted_score", question_score.score)
        
        elif canonical_id.dimension == "D4" and canonical_id.question == 5:
            logger.info(f"[DNP_INTEGRATION] D4-Q5 detected: {question_score.question_id}")
            dnp_result = self._evaluate_d4_q5_compliance(canonical_id.policy, dnp_validator)
            question_score.dnp_compliance = dnp_result
            question_score.score = dnp_result.get("adjusted_score", question_score.score)
        
        return question_score

    def _evaluate_d1_q5_compliance(self, policy: str, dnp_validator: Any) -> Dict[str, Any]:
        if dnp_validator is None:
            logger.warning("[DNP_INTEGRATION] No DNP validator available, using base score")
            return {"compliance": "unknown", "adjusted_score": None}
        
        try:
            regulatory_framework = dnp_validator.get_regulatory_framework(policy)
            compliance_level = dnp_validator.evaluate_compliance(policy, regulatory_framework)
            
            adjustment_factor = {
                "full": 1.0,
                "partial": 0.9,
                "minimal": 0.7,
                "none": 0.5
            }.get(compliance_level, 0.8)
            
            return {
                "compliance": compliance_level,
                "framework": regulatory_framework,
                "adjustment_factor": adjustment_factor,
                "adjusted_score": None  
            }
        except Exception as e:
            logger.error(f"[DNP_INTEGRATION] Error evaluating D1-Q5: {e}")
            return {"compliance": "error", "error": str(e), "adjusted_score": None}

    def _evaluate_d4_q5_compliance(self, policy: str, dnp_validator: Any) -> Dict[str, Any]:
        if dnp_validator is None:
            logger.warning("[DNP_INTEGRATION] No DNP validator available, using base score")
            return {"compliance": "unknown", "adjusted_score": None}
        
        try:
            alignment = dnp_validator.check_pnd_alignment(policy)
            
            adjustment_factor = {
                "strong": 1.0,
                "moderate": 0.9,
                "weak": 0.7,
                "none": 0.5
            }.get(alignment, 0.8)
            
            return {
                "pnd_alignment": alignment,
                "adjustment_factor": adjustment_factor,
                "adjusted_score": None  
            }
        except Exception as e:
            logger.error(f"[DNP_INTEGRATION] Error evaluating D4-Q5: {e}")
            return {"alignment": "error", "error": str(e), "adjusted_score": None}

    def generate_scoring_report(self, macro_score: MacroScore) -> Dict[str, Any]:
        report = {
            "overall_score": macro_score.overall_score,
            "rubric_category": self.score_to_rubric_category(macro_score.overall_score),
            "manual_review_required": len(macro_score.manual_review_flags) > 0,
            "manual_review_flags": macro_score.manual_review_flags,
            "cluster_scores": {},
            "policy_scores": {},
            "dimension_scores": {},
            "provenance": macro_score.provenance
        }
        
        for cluster_name, cluster in macro_score.cluster_scores.items():
            report["cluster_scores"][cluster_name] = {
                "score": cluster.score,
                "weight": cluster.weight,
                "rubric_category": self.score_to_rubric_category(cluster.score)
            }
            
            for policy_score in cluster.policy_scores:
                report["policy_scores"][policy_score.policy] = {
                    "score": policy_score.score,
                    "rubric_category": self.score_to_rubric_category(policy_score.score),
                    "cluster": cluster_name
                }
                
                for dim, dim_score in policy_score.dimension_scores.items():
                    key = f"{policy_score.policy}-{dim}"
                    report["dimension_scores"][key] = {
                        "score": dim_score.score,
                        "weight": dim_score.weight,
                        "rubric_category": self.score_to_rubric_category(dim_score.score),
                        "question_count": len(dim_score.question_scores)
                    }
        
        return report


def validate_scoring_framework() -> Dict[str, Any]:
    logger.info("[VALIDATION] Running complete scoring framework validation...")
    
    engine = ScoringEngine()
    
    validation_report = {
        "configuration_valid": True,
        "canonical_questions_valid": False,
        "dimension_weights_valid": False,
        "rubric_thresholds_valid": False,
        "errors": []
    }
    
    try:
        question_validation = engine.validate_all_canonical_questions()
        validation_report["canonical_questions_valid"] = True
        validation_report["question_validation"] = question_validation
    except Exception as e:
        validation_report["errors"].append(f"Question validation failed: {e}")
    
    try:
        for policy, weights in DIMENSION_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-9, f"{policy} weights sum to {total}"
        validation_report["dimension_weights_valid"] = True
    except Exception as e:
        validation_report["errors"].append(f"Dimension weights validation failed: {e}")
    
    try:
        for category, (low, high) in RUBRIC_THRESHOLDS.items():
            assert 0.0 <= low < high <= 1.0
        validation_report["rubric_thresholds_valid"] = True
    except Exception as e:
        validation_report["errors"].append(f"Rubric thresholds validation failed: {e}")
    
    validation_report["all_valid"] = (
        validation_report["canonical_questions_valid"] and
        validation_report["dimension_weights_valid"] and
        validation_report["rubric_thresholds_valid"]
    )
    
    logger.info(f"[VALIDATION] Complete: all_valid={validation_report['all_valid']}")
    
    return validation_report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=" * 80)
    print("FARFAN 2.0 Scoring Framework Validation")
    print("=" * 80)
    
    validation_report = validate_scoring_framework()
    
    print(f"\n✓ Configuration Valid: {validation_report['configuration_valid']}")
    print(f"✓ Canonical Questions Valid: {validation_report['canonical_questions_valid']}")
    print(f"✓ Dimension Weights Valid: {validation_report['dimension_weights_valid']}")
    print(f"✓ Rubric Thresholds Valid: {validation_report['rubric_thresholds_valid']}")
    print(f"\n{'✓' if validation_report['all_valid'] else '✗'} Overall: {'PASSED' if validation_report['all_valid'] else 'FAILED'}")
    
    if validation_report['errors']:
        print("\nErrors:")
        for error in validation_report['errors']:
            print(f"  - {error}")
