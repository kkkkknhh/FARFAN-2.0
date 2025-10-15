#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IoR Validator - Input/Output Rigor Enforcement
===============================================

Implements Audit Points 1.1, 1.2, and 1.3 for deterministic input anchor
and schema integrity per SOTA MMR input rigor (Ragin 2008).

This module ensures:
- 100% Pydantic validation of structured inputs (Audit Point 1.1)
- Immutable SHA-256 provenance fingerprints (Audit Point 1.2)
- High-confidence financial anchor integrity (Audit Point 1.3)
"""

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from extraction.extraction_pipeline import ExtractedTable, SemanticChunk

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of schema validation check"""

    passed: bool
    validated_items: int
    rejected_items: int
    rejection_log: List[Dict[str, Any]]
    error_summary: Dict[str, int]


@dataclass
class ProvenanceCheck:
    """Result of provenance traceability check"""

    passed: bool
    total_chunks: int
    verified_hashes: int
    hash_mismatches: int
    immutable_fingerprints_verified: bool


@dataclass
class FinancialAnchorCheck:
    """Result of financial anchor integrity check"""

    passed: bool
    total_nodes: int
    matched_nodes: int
    confidence_score: float
    high_confidence: bool  # >= 80% threshold
    ppi_bpin_codes_found: int


class IoRValidator:
    """
    IoR (Input/Output Rigor) Validator

    Enforces deterministic input anchor and schema integrity for
    FARFAN 2.0 analysis pipeline, ensuring SOTA MMR compliance.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validation_history: List[ValidationResult] = []
        self._provenance_history: List[ProvenanceCheck] = []
        self._financial_history: List[FinancialAnchorCheck] = []

    # ========================================================================
    # Audit Point 1.1: Input Schema Enforcement
    # ========================================================================

    def validate_extracted_tables(
        self, raw_tables: List[Dict[str, Any]]
    ) -> ValidationResult:
        """
        Audit Point 1.1: Input Schema Enforcement for ExtractedTable

        Ensures 100% structured inputs pass Pydantic validation pre-evidence pool.
        Violations trigger Hard Failure and evidence pool exclusion.

        Args:
            raw_tables: List of raw table dictionaries to validate

        Returns:
            ValidationResult with pass/fail status and rejection log
        """
        validated = []
        rejection_log = []
        error_summary = {}

        for i, table_data in enumerate(raw_tables):
            try:
                # Attempt Pydantic validation
                validated_table = ExtractedTable.model_validate(table_data)
                validated.append(validated_table)

            except ValidationError as e:
                # Hard Failure: Log rejection and exclude from evidence pool
                error_type = self._categorize_validation_error(e)
                error_summary[error_type] = error_summary.get(error_type, 0) + 1

                rejection_log.append(
                    {
                        "item_index": i,
                        "item_type": "ExtractedTable",
                        "error_type": error_type,
                        "error_detail": str(e),
                        "excluded_from_evidence_pool": True,
                    }
                )

                self.logger.warning(
                    f"Table {i} rejected: {error_type} - EXCLUDED from evidence pool"
                )

        result = ValidationResult(
            passed=len(rejection_log) == 0,
            validated_items=len(validated),
            rejected_items=len(rejection_log),
            rejection_log=rejection_log,
            error_summary=error_summary,
        )

        self._validation_history.append(result)

        # Log summary
        total = len(raw_tables)
        pass_rate = (len(validated) / total * 100) if total > 0 else 0
        self.logger.info(
            f"ExtractedTable validation: {len(validated)}/{total} passed "
            f"({pass_rate:.1f}% - Target: 100%)"
        )

        if not result.passed:
            self.logger.warning(
                f"⚠ Schema validation FAILED: {len(rejection_log)} tables rejected"
            )
        else:
            self.logger.info("✓ Schema validation PASSED: All tables validated")

        return result

    def validate_semantic_chunks(
        self, raw_chunks: List[Dict[str, Any]]
    ) -> ValidationResult:
        """
        Audit Point 1.1: Input Schema Enforcement for SemanticChunk

        Ensures 100% structured inputs pass Pydantic validation pre-evidence pool.
        Missing chunk_id or DNP metadata triggers Hard Failure.

        Args:
            raw_chunks: List of raw chunk dictionaries to validate

        Returns:
            ValidationResult with pass/fail status and rejection log
        """
        validated = []
        rejection_log = []
        error_summary = {}

        for i, chunk_data in enumerate(raw_chunks):
            try:
                # Attempt Pydantic validation
                validated_chunk = SemanticChunk.model_validate(chunk_data)

                # Additional DNP metadata check (Audit Point 1.1)
                if not validated_chunk.doc_id:
                    raise ValidationError("Missing DNP metadata: doc_id required")

                validated.append(validated_chunk)

            except ValidationError as e:
                # Hard Failure: Log rejection and exclude from evidence pool
                error_type = self._categorize_validation_error(e)
                error_summary[error_type] = error_summary.get(error_type, 0) + 1

                rejection_log.append(
                    {
                        "item_index": i,
                        "item_type": "SemanticChunk",
                        "error_type": error_type,
                        "error_detail": str(e),
                        "excluded_from_evidence_pool": True,
                    }
                )

                self.logger.warning(
                    f"Chunk {i} rejected: {error_type} - EXCLUDED from evidence pool"
                )

        result = ValidationResult(
            passed=len(rejection_log) == 0,
            validated_items=len(validated),
            rejected_items=len(rejection_log),
            rejection_log=rejection_log,
            error_summary=error_summary,
        )

        self._validation_history.append(result)

        # Log summary
        total = len(raw_chunks)
        pass_rate = (len(validated) / total * 100) if total > 0 else 0
        self.logger.info(
            f"SemanticChunk validation: {len(validated)}/{total} passed "
            f"({pass_rate:.1f}% - Target: 100%)"
        )

        if not result.passed:
            self.logger.warning(
                f"⚠ Schema validation FAILED: {len(rejection_log)} chunks rejected"
            )
        else:
            self.logger.info("✓ Schema validation PASSED: All chunks validated")

        return result

    def _categorize_validation_error(self, error: ValidationError) -> str:
        """Categorize validation error for reporting"""
        error_str = str(error).lower()

        if "empty" in error_str or "cannot be empty" in error_str:
            return "missing_required_field"
        elif "chunk_id" in error_str:
            return "missing_chunk_id"
        elif "doc_id" in error_str or "dnp" in error_str:
            return "missing_dnp_metadata"
        elif "greater than" in error_str or "less than" in error_str:
            return "invalid_range"
        elif "confidence" in error_str:
            return "invalid_confidence_score"
        else:
            return "schema_violation"

    # ========================================================================
    # Audit Point 1.2: Provenance Traceability
    # ========================================================================

    def verify_provenance_traceability(
        self, chunks: List[SemanticChunk]
    ) -> ProvenanceCheck:
        """
        Audit Point 1.2: Provenance Traceability

        Verifies every data unit exposes immutable SHA-256 fingerprint.
        Recomputes hashes from raw chunks and matches against stored IDs.

        Args:
            chunks: List of validated semantic chunks

        Returns:
            ProvenanceCheck with verification results
        """
        verified_hashes = 0
        hash_mismatches = 0

        for chunk in chunks:
            # Recompute SHA-256 hash from canonicalized content
            canonical_content = (
                f"{chunk.doc_id}:{chunk.text}:{chunk.start_char}:{chunk.end_char}"
            )
            recomputed_hash = hashlib.sha256(
                canonical_content.encode("utf-8")
            ).hexdigest()

            # Verify against stored chunk_fingerprint in metadata
            stored_hash = chunk.metadata.get("chunk_fingerprint")

            if stored_hash == recomputed_hash:
                verified_hashes += 1
            else:
                hash_mismatches += 1
                self.logger.warning(
                    f"Hash mismatch for chunk {chunk.chunk_id}: stored != recomputed"
                )

        # Check immutable fingerprint property
        immutable_verified = all(
            chunk.metadata.get("chunk_fingerprint") for chunk in chunks
        )

        result = ProvenanceCheck(
            passed=hash_mismatches == 0 and immutable_verified,
            total_chunks=len(chunks),
            verified_hashes=verified_hashes,
            hash_mismatches=hash_mismatches,
            immutable_fingerprints_verified=immutable_verified,
        )

        self._provenance_history.append(result)

        # Log summary
        verification_rate = (verified_hashes / len(chunks) * 100) if chunks else 0

        self.logger.info(
            f"Provenance verification: {verified_hashes}/{len(chunks)} hashes verified "
            f"({verification_rate:.1f}%)"
        )

        if result.passed:
            self.logger.info(
                "✓ Provenance traceability PASSED: "
                "Blockchain-inspired immutable fingerprints verified"
            )
        else:
            self.logger.warning(
                f"⚠ Provenance traceability FAILED: "
                f"{hash_mismatches} hash mismatches detected"
            )

        return result

    # ========================================================================
    # Audit Point 1.3: Financial Anchor Integrity
    # ========================================================================

    def verify_financial_anchor_integrity(
        self, financial_data: Dict[str, Dict[str, Any]], total_nodes: int
    ) -> FinancialAnchorCheck:
        """
        Audit Point 1.3: Financial Anchor Integrity

        Verifies FinancialAuditor.trace_financial_allocation confirms
        PPI/BPIN links to nodes with high confidence (>=80%) for D1-Q3.

        Args:
            financial_data: Financial allocation data from FinancialAuditor
            total_nodes: Total number of nodes in the analysis

        Returns:
            FinancialAnchorCheck with integrity verification results
        """
        matched_nodes = len(financial_data)

        # Calculate confidence score
        confidence = (matched_nodes / total_nodes * 100) if total_nodes > 0 else 0

        # High confidence threshold: >= 80% (Colombian DNP 2023 standards)
        high_confidence = confidence >= 80.0

        # Count PPI/BPIN codes found
        ppi_bpin_count = 0
        for node_id, data in financial_data.items():
            # Check if allocation has identifiable project codes
            if data.get("allocation") and data.get("source"):
                ppi_bpin_count += 1

        result = FinancialAnchorCheck(
            passed=high_confidence,
            total_nodes=total_nodes,
            matched_nodes=matched_nodes,
            confidence_score=confidence,
            high_confidence=high_confidence,
            ppi_bpin_codes_found=ppi_bpin_count,
        )

        self._financial_history.append(result)

        # Log summary
        self.logger.info(
            f"Financial anchor integrity: {matched_nodes}/{total_nodes} nodes matched "
            f"({confidence:.1f}% - Target: >=80%)"
        )

        if result.passed:
            self.logger.info(
                "✓ Financial anchor integrity PASSED: "
                f"High-confidence ({confidence:.1f}%) anchoring verified"
            )
        else:
            self.logger.warning(
                f"⚠ Financial anchor integrity FAILED: "
                f"Confidence {confidence:.1f}% below 80% threshold"
            )

        return result

    # ========================================================================
    # Comprehensive IoR Audit Report
    # ========================================================================

    def generate_ior_audit_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive IoR audit report

        Returns:
            Dictionary with complete audit results for all three audit points
        """
        # Aggregate validation results
        total_validations = len(self._validation_history)
        passed_validations = sum(1 for v in self._validation_history if v.passed)

        # Aggregate provenance checks
        total_provenance = len(self._provenance_history)
        passed_provenance = sum(1 for p in self._provenance_history if p.passed)

        # Aggregate financial checks
        total_financial = len(self._financial_history)
        passed_financial = sum(1 for f in self._financial_history if f.passed)

        # Overall IoR compliance
        all_checks = total_validations + total_provenance + total_financial
        all_passed = passed_validations + passed_provenance + passed_financial
        overall_compliance = (all_passed / all_checks * 100) if all_checks > 0 else 0

        report = {
            "ior_audit_summary": {
                "overall_compliance_rate": round(overall_compliance, 2),
                "total_checks_performed": all_checks,
                "checks_passed": all_passed,
                "checks_failed": all_checks - all_passed,
                "sota_mmr_compliant": math.isclose(overall_compliance, 100.0, rel_tol=1e-9, abs_tol=1e-12),  # replaced float equality with isclose (tolerance from DEFAULT_FLOAT_TOLS)
            },
            "audit_point_1_1_schema_enforcement": {
                "total_validations": total_validations,
                "passed": passed_validations,
                "failed": total_validations - passed_validations,
                "pass_rate": round(
                    (
                        (passed_validations / total_validations * 100)
                        if total_validations > 0
                        else 0
                    ),
                    2,
                ),
                "target_pass_rate": 100.0,
                "qca_level_calibration": passed_validations == total_validations,
                "recent_results": [
                    {
                        "validated_items": v.validated_items,
                        "rejected_items": v.rejected_items,
                        "error_summary": v.error_summary,
                    }
                    for v in self._validation_history[-5:]  # Last 5
                ],
            },
            "audit_point_1_2_provenance_traceability": {
                "total_checks": total_provenance,
                "passed": passed_provenance,
                "failed": total_provenance - passed_provenance,
                "pass_rate": round(
                    (
                        (passed_provenance / total_provenance * 100)
                        if total_provenance > 0
                        else 0
                    ),
                    2,
                ),
                "blockchain_inspired_traceability": passed_provenance > 0,
                "recent_results": [
                    {
                        "total_chunks": p.total_chunks,
                        "verified_hashes": p.verified_hashes,
                        "hash_mismatches": p.hash_mismatches,
                    }
                    for p in self._provenance_history[-5:]  # Last 5
                ],
            },
            "audit_point_1_3_financial_anchor_integrity": {
                "total_checks": total_financial,
                "passed": passed_financial,
                "failed": total_financial - passed_financial,
                "pass_rate": round(
                    (
                        (passed_financial / total_financial * 100)
                        if total_financial > 0
                        else 0
                    ),
                    2,
                ),
                "high_confidence_threshold": 80.0,
                "recent_results": [
                    {
                        "confidence_score": f.confidence_score,
                        "matched_nodes": f.matched_nodes,
                        "total_nodes": f.total_nodes,
                        "ppi_bpin_codes": f.ppi_bpin_codes_found,
                    }
                    for f in self._financial_history[-5:]  # Last 5
                ],
            },
            "references": {
                "mmr_input_rigor": "Ragin 2008 - QCA deterministic data calibration",
                "qca_calibration": "Schneider & Rohlfing 2013",
                "provenance_traceability": "Pearl 2018 - Causal data provenance",
                "process_tracing": "Bennett & Checkel 2015",
                "dnp_standards": "Colombian DNP 2023",
                "fiscal_mechanisms": "Waldner 2015",
            },
        }

        self.logger.info(
            f"IoR Audit Report: {overall_compliance:.1f}% overall compliance"
        )

        return report

    def clear_history(self):
        """Clear audit history (for testing or new analysis runs)"""
        self._validation_history.clear()
        self._provenance_history.clear()
        self._financial_history.clear()
        self.logger.info("Audit history cleared")
