#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scoring System Audit Module for FARFAN 2.0
==========================================

Comprehensive audit function that traces scoring from 300-question matrix
through all aggregation levels (MICRO → MESO → MACRO).

Author: AI Systems Architect
Version: 1.0.0
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("scoring_audit")

EXPECTED_POLICIES = 10
EXPECTED_DIMENSIONS = 6
EXPECTED_QUESTIONS_PER_DIM = 5
EXPECTED_TOTAL_QUESTIONS = 300
THRESHOLD_EXCELENTE = 0.70
THRESHOLD_BUENO = 0.55
THRESHOLD_ACEPTABLE = 0.40
THRESHOLD_D6_CRITICAL = 0.55
WEIGHT_TOLERANCE = 0.001


class QualityBand(Enum):
    EXCELENTE = "excelente"
    BUENO = "bueno"
    ACEPTABLE = "aceptable"
    INSUFICIENTE = "insuficiente"


class ClusterMeso(Enum):
    C1_SEGURIDAD_PAZ = "C1"
    C2_DERECHOS_SOCIALES = "C2"
    C3_TERRITORIO_AMBIENTE = "C3"
    C4_POBLACIONES_ESPECIALES = "C4"


CLUSTER_TO_POLICIES = {
    ClusterMeso.C1_SEGURIDAD_PAZ: ["P1", "P2", "P8"],
    ClusterMeso.C2_DERECHOS_SOCIALES: ["P4", "P5", "P6"],
    ClusterMeso.C3_TERRITORIO_AMBIENTE: ["P3", "P7"],
    ClusterMeso.C4_POBLACIONES_ESPECIALES: ["P9", "P10"],
}

POLICY_TO_CLUSTER = {
    p: cluster for cluster, policies in CLUSTER_TO_POLICIES.items() for p in policies
}


@dataclass
class AuditIssue:
    category: str
    severity: str
    description: str
    location: str
    expected: Any
    actual: Any
    recommendation: str


@dataclass
class ScoringAuditReport:
    timestamp: str
    total_questions_expected: int = EXPECTED_TOTAL_QUESTIONS
    total_questions_found: int = 0
    matrix_valid: bool = False
    policies_found: Set[str] = field(default_factory=set)
    dimensions_found: Set[str] = field(default_factory=set)
    micro_scores_valid: bool = True
    micro_issues: List[AuditIssue] = field(default_factory=list)
    meso_aggregation_valid: bool = True
    meso_weight_issues: List[AuditIssue] = field(default_factory=list)
    meso_convergence_gaps: List[AuditIssue] = field(default_factory=list)
    macro_alignment_valid: bool = True
    macro_issues: List[AuditIssue] = field(default_factory=list)
    rubric_mappings_valid: bool = True
    rubric_issues: List[AuditIssue] = field(default_factory=list)
    d6_scores_below_threshold: List[Dict[str, Any]] = field(default_factory=list)
    dnp_integration_valid: bool = True
    dnp_issues: List[AuditIssue] = field(default_factory=list)
    total_issues: int = 0
    critical_issues: int = 0
    overall_valid: bool = False


class ScoringSystemAuditor:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("audit_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report = ScoringAuditReport(timestamp=datetime.now().isoformat())

    def audit_complete_system(
        self,
        question_responses: Dict[str, Any],
        dimension_weights: Optional[Dict[str, Dict[str, float]]] = None,
        meso_report: Optional[Dict[str, Any]] = None,
        macro_report: Optional[Dict[str, Any]] = None,
        dnp_results: Optional[Any] = None,
    ) -> ScoringAuditReport:
        logger.info("Starting comprehensive scoring system audit...")
        self.audit_matrix_structure(question_responses)
        self.audit_micro_scores(question_responses)
        if meso_report:
            self.audit_meso_aggregation(
                question_responses, meso_report, dimension_weights
            )
        if macro_report:
            self.audit_macro_alignment(question_responses, macro_report)
        self.audit_rubric_thresholds(question_responses)
        self.audit_d6_theory_of_change(question_responses)
        if dnp_results:
            self.audit_dnp_integration(dnp_results, macro_report)
        self.finalize_report()
        return self.report

    def audit_matrix_structure(self, question_responses: Dict[str, Any]) -> None:
        logger.info("Auditing matrix structure...")
        self.report.total_questions_found = len(question_responses)
        for question_id in question_responses.keys():
            parts = question_id.split("-")
            if len(parts) >= 2:
                self.report.policies_found.add(parts[0])
                self.report.dimensions_found.add(parts[1])
        if self.report.total_questions_found != EXPECTED_TOTAL_QUESTIONS:
            self.report.micro_issues.append(
                AuditIssue(
                    category="matrix_structure",
                    severity="CRITICAL",
                    description="Total question count mismatch",
                    location="MICRO level",
                    expected=EXPECTED_TOTAL_QUESTIONS,
                    actual=self.report.total_questions_found,
                    recommendation=f"Ensure all {EXPECTED_TOTAL_QUESTIONS} questions present",
                )
            )
            self.report.matrix_valid = False
        expected_policies = {f"P{i}" for i in range(1, 11)}
        missing = expected_policies - self.report.policies_found
        if missing:
            self.report.micro_issues.append(
                AuditIssue(
                    category="matrix_structure",
                    severity="CRITICAL",
                    description="Missing policies",
                    location="MICRO level",
                    expected=sorted(expected_policies),
                    actual=sorted(self.report.policies_found),
                    recommendation=f"Add missing: {sorted(missing)}",
                )
            )
            self.report.matrix_valid = False
        expected_dimensions = {f"D{i}" for i in range(1, 7)}
        missing_dim = expected_dimensions - self.report.dimensions_found
        if missing_dim:
            self.report.micro_issues.append(
                AuditIssue(
                    category="matrix_structure",
                    severity="CRITICAL",
                    description="Missing dimensions",
                    location="MICRO level",
                    expected=sorted(expected_dimensions),
                    actual=sorted(self.report.dimensions_found),
                    recommendation=f"Add missing: {sorted(missing_dim)}",
                )
            )
            self.report.matrix_valid = False
        if not self.report.matrix_valid:
            self.report.matrix_valid = (
                len(
                    [
                        i
                        for i in self.report.micro_issues
                        if i.category == "matrix_structure"
                    ]
                )
                == 0
            )
        else:
            self.report.matrix_valid = True

    def audit_micro_scores(self, question_responses: Dict[str, Any]) -> None:
        logger.info("Auditing MICRO scores...")
        for question_id, response in question_responses.items():
            if not hasattr(response, "nota_cuantitativa"):
                self.report.micro_issues.append(
                    AuditIssue(
                        category="micro_scoring",
                        severity="HIGH",
                        description="Missing score",
                        location=question_id,
                        expected="nota_cuantitativa",
                        actual="missing",
                        recommendation=f"Add score to {question_id}",
                    )
                )
                self.report.micro_scores_valid = False
                continue
            score = response.nota_cuantitativa
            if not (0.0 <= score <= 1.0):
                self.report.micro_issues.append(
                    AuditIssue(
                        category="micro_scoring",
                        severity="HIGH",
                        description="Score out of range",
                        location=question_id,
                        expected="[0.0, 1.0]",
                        actual=score,
                        recommendation=f"Normalize {question_id} to [0, 1]",
                    )
                )
                self.report.micro_scores_valid = False

    def audit_meso_aggregation(
        self,
        question_responses: Dict[str, Any],
        meso_report: Dict[str, Any],
        dimension_weights: Optional[Dict[str, Dict[str, float]]],
    ) -> None:
        logger.info("Auditing MESO aggregation...")
        if "clusters" not in meso_report:
            self.report.meso_convergence_gaps.append(
                AuditIssue(
                    category="meso_structure",
                    severity="CRITICAL",
                    description="Missing clusters",
                    location="MESO level",
                    expected="clusters key",
                    actual="missing",
                    recommendation="Add 4 clusters to MESO report",
                )
            )
            self.report.meso_aggregation_valid = False
            return
        expected_clusters = {c.value for c in ClusterMeso}
        found = set(meso_report["clusters"].keys())
        missing = expected_clusters - found
        if missing:
            self.report.meso_convergence_gaps.append(
                AuditIssue(
                    category="meso_structure",
                    severity="HIGH",
                    description="Missing clusters",
                    location="MESO level",
                    expected=sorted(expected_clusters),
                    actual=sorted(found),
                    recommendation=f"Add: {sorted(missing)}",
                )
            )
            self.report.meso_aggregation_valid = False
        if dimension_weights:
            for policy_id, weights in dimension_weights.items():
                total = sum(weights.values())
                if abs(total - 1.0) > WEIGHT_TOLERANCE:
                    self.report.meso_weight_issues.append(
                        AuditIssue(
                            category="dimension_weights",
                            severity="HIGH",
                            description="Weights don't sum to 1.0",
                            location=policy_id,
                            expected=1.0,
                            actual=total,
                            recommendation=f"Adjust weights for {policy_id} (current: {total:.4f})",
                        )
                    )
                    self.report.meso_aggregation_valid = False

    def audit_macro_alignment(
        self, question_responses: Dict[str, Any], macro_report: Dict[str, Any]
    ) -> None:
        logger.info("Auditing MACRO alignment...")
        if "evaluacion_global" not in macro_report:
            self.report.macro_issues.append(
                AuditIssue(
                    category="macro_structure",
                    severity="CRITICAL",
                    description="Missing global evaluation",
                    location="MACRO level",
                    expected="evaluacion_global",
                    actual="missing",
                    recommendation="Add evaluacion_global to MACRO",
                )
            )
            self.report.macro_alignment_valid = False
            return
        evaluacion = macro_report["evaluacion_global"]
        if "score_global" in evaluacion:
            micro_scores = [
                r.nota_cuantitativa
                for r in question_responses.values()
                if hasattr(r, "nota_cuantitativa")
            ]
            if micro_scores:
                expected = sum(micro_scores) / len(micro_scores)
                actual = evaluacion["score_global"]
                if abs(expected - actual) > 0.01:
                    self.report.macro_issues.append(
                        AuditIssue(
                            category="macro_convergence",
                            severity="HIGH",
                            description="MACRO score mismatch",
                            location="evaluacion_global",
                            expected=f"{expected:.4f}",
                            actual=f"{actual:.4f}",
                            recommendation="Recalculate MACRO from all MICRO scores",
                        )
                    )
                    self.report.macro_alignment_valid = False

    def audit_rubric_thresholds(self, question_responses: Dict[str, Any]) -> None:
        logger.info("Auditing rubric thresholds...")
        band_counts = {band: 0 for band in QualityBand}
        for question_id, response in question_responses.items():
            if not hasattr(response, "nota_cuantitativa"):
                continue
            score = response.nota_cuantitativa
            if score >= THRESHOLD_EXCELENTE:
                band = QualityBand.EXCELENTE
            elif score >= THRESHOLD_BUENO:
                band = QualityBand.BUENO
            elif score >= THRESHOLD_ACEPTABLE:
                band = QualityBand.ACEPTABLE
            else:
                band = QualityBand.INSUFICIENTE
            band_counts[band] += 1
        logger.info(
            f"  EXCELENTE (≥{THRESHOLD_EXCELENTE}): {band_counts[QualityBand.EXCELENTE]}"
        )
        logger.info(
            f"  BUENO ({THRESHOLD_BUENO}-{THRESHOLD_EXCELENTE}): {band_counts[QualityBand.BUENO]}"
        )
        logger.info(
            f"  ACEPTABLE ({THRESHOLD_ACEPTABLE}-{THRESHOLD_BUENO}): {band_counts[QualityBand.ACEPTABLE]}"
        )
        logger.info(
            f"  INSUFICIENTE (<{THRESHOLD_ACEPTABLE}): {band_counts[QualityBand.INSUFICIENTE]}"
        )
        if band_counts[QualityBand.INSUFICIENTE] > EXPECTED_TOTAL_QUESTIONS * 0.3:
            self.report.rubric_issues.append(
                AuditIssue(
                    category="rubric_quality",
                    severity="HIGH",
                    description="High proportion of insufficient scores",
                    location="MICRO level",
                    expected="<30% insufficient",
                    actual=f"{band_counts[QualityBand.INSUFICIENTE]}/{EXPECTED_TOTAL_QUESTIONS}",
                    recommendation="Review low-scoring questions",
                )
            )
        self.report.rubric_mappings_valid = len(self.report.rubric_issues) == 0

    def audit_d6_theory_of_change(self, question_responses: Dict[str, Any]) -> None:
        logger.info("Auditing D6 Theory of Change...")
        d6_scores = []
        for question_id, response in question_responses.items():
            if "-D6-" in question_id and hasattr(response, "nota_cuantitativa"):
                score = response.nota_cuantitativa
                d6_scores.append((question_id, score))
                if score < THRESHOLD_D6_CRITICAL:
                    self.report.d6_scores_below_threshold.append(
                        {
                            "question_id": question_id,
                            "score": score,
                            "threshold": THRESHOLD_D6_CRITICAL,
                            "gap": THRESHOLD_D6_CRITICAL - score,
                        }
                    )
        if self.report.d6_scores_below_threshold:
            logger.warning(
                f"⚠ {len(self.report.d6_scores_below_threshold)} D6 scores below {THRESHOLD_D6_CRITICAL}"
            )
            self.report.rubric_issues.append(
                AuditIssue(
                    category="d6_theory_of_change",
                    severity="CRITICAL",
                    description=f"D6 scores below threshold ({THRESHOLD_D6_CRITICAL})",
                    location="D6",
                    expected=f"≥{THRESHOLD_D6_CRITICAL}",
                    actual=f"{len(self.report.d6_scores_below_threshold)} below",
                    recommendation="Strengthen Theory of Change framework",
                )
            )
        if d6_scores:
            avg = sum(s for _, s in d6_scores) / len(d6_scores)
            logger.info(f"  D6 average: {avg:.3f}")

    def audit_dnp_integration(
        self, dnp_results: Any, macro_report: Optional[Dict[str, Any]]
    ) -> None:
        logger.info("Auditing DNP integration...")
        required = [
            "cumple_competencias",
            "cumple_mga",
            "nivel_cumplimiento",
            "score_total",
        ]
        for attr in required:
            if not hasattr(dnp_results, attr):
                self.report.dnp_issues.append(
                    AuditIssue(
                        category="dnp_integration",
                        severity="HIGH",
                        description=f"Missing DNP attribute",
                        location="DNP validation",
                        expected=attr,
                        actual="missing",
                        recommendation=f"Add {attr} to DNP validator",
                    )
                )
                self.report.dnp_integration_valid = False
        if macro_report and "evaluacion_global" in macro_report:
            evaluacion = macro_report["evaluacion_global"]
            if "score_dnp_compliance" not in evaluacion:
                self.report.dnp_issues.append(
                    AuditIssue(
                        category="dnp_integration",
                        severity="HIGH",
                        description="DNP not in MACRO",
                        location="evaluacion_global",
                        expected="score_dnp_compliance",
                        actual="missing",
                        recommendation="Add DNP score to MACRO",
                    )
                )
                self.report.dnp_integration_valid = False

    def finalize_report(self) -> None:
        all_issues = (
            self.report.micro_issues
            + self.report.meso_weight_issues
            + self.report.meso_convergence_gaps
            + self.report.macro_issues
            + self.report.rubric_issues
            + self.report.dnp_issues
        )
        self.report.total_issues = len(all_issues)
        self.report.critical_issues = sum(
            1 for i in all_issues if i.severity == "CRITICAL"
        )
        self.report.overall_valid = (
            self.report.matrix_valid
            and self.report.micro_scores_valid
            and self.report.meso_aggregation_valid
            and self.report.macro_alignment_valid
            and self.report.rubric_mappings_valid
            and self.report.dnp_integration_valid
            and self.report.critical_issues == 0
        )
        logger.info(f"\n{'=' * 60}")
        logger.info(f"AUDIT SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"Status: {'✓ VALID' if self.report.overall_valid else '✗ ISSUES'}")
        logger.info(
            f"Total: {self.report.total_issues} ({self.report.critical_issues} critical)"
        )
        logger.info(f"Matrix: {'✓' if self.report.matrix_valid else '✗'}")
        logger.info(f"MICRO: {'✓' if self.report.micro_scores_valid else '✗'}")
        logger.info(f"MESO: {'✓' if self.report.meso_aggregation_valid else '✗'}")
        logger.info(f"MACRO: {'✓' if self.report.macro_alignment_valid else '✗'}")
        logger.info(f"Rubric: {'✓' if self.report.rubric_mappings_valid else '✗'}")
        logger.info(f"DNP: {'✓' if self.report.dnp_integration_valid else '✗'}")
        logger.info(
            f"D6 Critical: {len(self.report.d6_scores_below_threshold)} below threshold"
        )
        logger.info(f"{'=' * 60}\n")

    def export_report(self, filename: Optional[str] = None) -> Path:
        if filename is None:
            filename = f"scoring_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = self.output_dir / filename
        report_dict = {
            "timestamp": self.report.timestamp,
            "overall_valid": self.report.overall_valid,
            "total_issues": self.report.total_issues,
            "critical_issues": self.report.critical_issues,
            "matrix": {
                "valid": self.report.matrix_valid,
                "expected": self.report.total_questions_expected,
                "found": self.report.total_questions_found,
                "policies": sorted(self.report.policies_found),
                "dimensions": sorted(self.report.dimensions_found),
            },
            "micro": {
                "valid": self.report.micro_scores_valid,
                "issues": [self._issue_to_dict(i) for i in self.report.micro_issues],
            },
            "meso": {
                "valid": self.report.meso_aggregation_valid,
                "weight_issues": [
                    self._issue_to_dict(i) for i in self.report.meso_weight_issues
                ],
                "convergence_gaps": [
                    self._issue_to_dict(i) for i in self.report.meso_convergence_gaps
                ],
            },
            "macro": {
                "valid": self.report.macro_alignment_valid,
                "issues": [self._issue_to_dict(i) for i in self.report.macro_issues],
            },
            "rubric": {
                "valid": self.report.rubric_mappings_valid,
                "issues": [self._issue_to_dict(i) for i in self.report.rubric_issues],
            },
            "d6_theory_of_change": {
                "scores_below_threshold": self.report.d6_scores_below_threshold,
                "threshold": THRESHOLD_D6_CRITICAL,
            },
            "dnp": {
                "valid": self.report.dnp_integration_valid,
                "issues": [self._issue_to_dict(i) for i in self.report.dnp_issues],
            },
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Audit report saved: {output_path}")
        return output_path

    def _issue_to_dict(self, issue: AuditIssue) -> Dict[str, Any]:
        return {
            "category": issue.category,
            "severity": issue.severity,
            "description": issue.description,
            "location": issue.location,
            "expected": str(issue.expected),
            "actual": str(issue.actual),
            "recommendation": issue.recommendation,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Scoring Audit Module - FARFAN 2.0")
    print(
        f"Expected matrix: {EXPECTED_POLICIES}×{EXPECTED_DIMENSIONS}×{EXPECTED_QUESTIONS_PER_DIM} = {EXPECTED_TOTAL_QUESTIONS}"
    )
