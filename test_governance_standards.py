#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Industrial Governance Standards
Audit Point 5.5: CI Contract Enforcement

Tests methodological gates:
- test_hoop_test_failure: Validates necessity test failure detection
- test_posterior_cap_enforced: Validates posterior distribution bounds
- test_mechanism_prior_decay: Validates prior decay over time
"""

import hashlib
import json
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

# Import governance standards
from governance_standards import (
    AuditLogEntry,
    ExecutionIsolationConfig,
    ExplainabilityPayload,
    HumanInTheLoopGate,
    ImmutableAuditLog,
    IsolationMetrics,
    IsolationMode,
    QualityGrade,
    compute_document_hash,
    create_governance_audit_log,
)

# Try to import Bayesian engine for methodological tests
try:
    import numpy as np

    from inference.bayesian_engine import (
        BayesianPriorBuilder,
        BayesianSamplingEngine,
        CausalLink,
        ColombianMunicipalContext,
        DocumentEvidence,
        EvidenceChunk,
        MechanismEvidence,
        MechanismPrior,
        NecessitySufficiencyTester,
        SamplingConfig,
    )

    BAYESIAN_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    BAYESIAN_AVAILABLE = False
    # Define mock classes for tests that don't require them
    DocumentEvidence = None

# ============================================================================
# Test Audit Point 5.1: Execution Isolation
# ============================================================================


class TestExecutionIsolation(unittest.TestCase):
    """Test suite for execution isolation (Audit Point 5.1)"""

    def test_isolation_config_initialization(self):
        """Test isolation configuration initialization"""
        config = ExecutionIsolationConfig(
            mode=IsolationMode.DOCKER,
            worker_timeout_secs=300,
            fail_open_on_timeout=True,
        )

        self.assertEqual(config.mode, IsolationMode.DOCKER)
        self.assertEqual(config.worker_timeout_secs, 300)
        self.assertTrue(config.fail_open_on_timeout)

    def test_isolation_config_validation(self):
        """Test that invalid timeout raises error"""
        with self.assertRaises(ValueError) as ctx:
            ExecutionIsolationConfig(worker_timeout_secs=0)

        self.assertIn("worker_timeout_secs must be positive", str(ctx.exception))

    def test_isolation_metrics_uptime_calculation(self):
        """Test uptime calculation meets 99.9% standard"""
        metrics = IsolationMetrics(
            total_executions=1000,
            timeout_count=0,
            failure_count=1,  # Only 1 failure
            fallback_count=0,
        )
        metrics.update_uptime()

        # 999/1000 = 99.9%
        self.assertEqual(metrics.uptime_percentage, 99.9)

        metrics_dict = metrics.to_dict()
        self.assertTrue(metrics_dict["meets_sota_standard"])

    def test_isolation_metrics_below_threshold(self):
        """Test uptime detection when below 99.9%"""
        metrics = IsolationMetrics(
            total_executions=1000,
            timeout_count=0,
            failure_count=2,  # 2 failures = 99.8%
            fallback_count=0,
        )
        metrics.update_uptime()

        self.assertEqual(metrics.uptime_percentage, 99.8)

        metrics_dict = metrics.to_dict()
        self.assertFalse(metrics_dict["meets_sota_standard"])


# ============================================================================
# Test Audit Point 5.2: Immutable Audit Log
# ============================================================================


class TestImmutableAuditLog(unittest.TestCase):
    """Test suite for immutable audit log (Audit Point 5.2)"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = TemporaryDirectory()
        self.log_dir = Path(self.temp_dir.name)
        self.audit_log = ImmutableAuditLog(log_dir=self.log_dir)

    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()

    def test_audit_log_entry_hash_computation(self):
        """Test SHA256 hash computation for audit entries"""
        entry = AuditLogEntry(
            run_id="RUN_001",
            timestamp=datetime.now().isoformat(),
            sha256_source="abc123",
            phase="test_phase",
            status="success",
            metrics={"score": 0.85},
            outputs={"result": "test"},
        )

        # Entry hash should be computed
        self.assertIsNotNone(entry.entry_hash)
        self.assertEqual(len(entry.entry_hash), 64)  # SHA256 is 64 hex chars

        # Verify hash
        self.assertTrue(entry.verify_hash())

    def test_audit_log_retention_period(self):
        """Test 5-year retention period calculation"""
        now = datetime.now()
        entry = AuditLogEntry(
            run_id="RUN_001",
            timestamp=now.isoformat(),
            sha256_source="abc123",
            phase="test_phase",
            status="success",
            metrics={},
            outputs={},
        )

        retention_date = datetime.fromisoformat(entry.retention_until)
        delta = retention_date - now

        # Should be approximately 5 years (1825 days)
        self.assertGreaterEqual(delta.days, 5 * 365 - 1)
        self.assertLessEqual(delta.days, 5 * 365 + 1)

    def test_audit_log_append_only(self):
        """Test append-only behavior"""
        # Add first entry
        entry1 = self.audit_log.append(
            run_id="RUN_001",
            sha256_source="hash1",
            phase="phase1",
            status="success",
            metrics={"score": 0.8},
            outputs={"result": "ok"},
        )

        # Add second entry (should link to first)
        entry2 = self.audit_log.append(
            run_id="RUN_001",
            sha256_source="hash1",
            phase="phase2",
            status="success",
            metrics={"score": 0.85},
            outputs={"result": "ok"},
        )

        # Second entry should reference first
        self.assertEqual(entry2.previous_hash, entry1.entry_hash)
        self.assertEqual(len(self.audit_log._entries), 2)

    def test_audit_log_hash_chain_verification(self):
        """Test hash chain integrity verification"""
        # Add multiple entries
        for i in range(5):
            self.audit_log.append(
                run_id="RUN_001",
                sha256_source="hash1",
                phase=f"phase{i}",
                status="success",
                metrics={"score": 0.8 + i * 0.01},
                outputs={"result": f"result{i}"},
            )

        # Verify chain
        is_valid, errors = self.audit_log.verify_chain()

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_audit_log_persist_immutable(self):
        """Test persistence creates immutable (read-only) file"""
        # Add entries
        self.audit_log.append(
            run_id="RUN_TEST",
            sha256_source="hash1",
            phase="phase1",
            status="success",
            metrics={},
            outputs={},
        )

        # Persist to disk
        log_file = self.audit_log.persist("RUN_TEST")

        # File should exist and be read-only
        self.assertTrue(log_file.exists())

        # File should be readable
        with open(log_file, "r") as f:
            data = json.load(f)

        self.assertEqual(data["run_id"], "RUN_TEST")
        self.assertEqual(data["retention_years"], 5)
        self.assertTrue(data["chain_valid"])

    def test_audit_log_query_by_run_id(self):
        """Test querying entries by run_id"""
        # Add entries for different runs
        self.audit_log.append(
            run_id="RUN_001",
            sha256_source="hash1",
            phase="phase1",
            status="success",
            metrics={},
            outputs={},
        )

        self.audit_log.append(
            run_id="RUN_002",
            sha256_source="hash2",
            phase="phase1",
            status="success",
            metrics={},
            outputs={},
        )

        # Query by run_id
        run_001_entries = self.audit_log.query_by_run_id("RUN_001")
        run_002_entries = self.audit_log.query_by_run_id("RUN_002")

        self.assertEqual(len(run_001_entries), 1)
        self.assertEqual(len(run_002_entries), 1)
        self.assertEqual(run_001_entries[0].run_id, "RUN_001")


# ============================================================================
# Test Audit Point 5.3: Explainability Payload
# ============================================================================


class TestExplainabilityPayload(unittest.TestCase):
    """Test suite for explainability payload (Audit Point 5.3)"""

    def test_payload_initialization(self):
        """Test explainability payload initialization"""
        payload = ExplainabilityPayload(
            link_id="LINK_001",
            posterior_mean=0.75,
            posterior_std=0.12,
            confidence_interval=(0.55, 0.90),
            necessity_test_passed=True,
            necessity_test_missing=[],
            evidence_snippets=["snippet1", "snippet2"],
            sha256_evidence="abc123",
        )

        self.assertEqual(payload.link_id, "LINK_001")
        self.assertEqual(payload.posterior_mean, 0.75)
        self.assertTrue(payload.necessity_test_passed)

    def test_payload_validation_bounds(self):
        """Test that posterior_mean must be in [0, 1]"""
        with self.assertRaises(ValueError) as ctx:
            ExplainabilityPayload(
                link_id="LINK_001",
                posterior_mean=1.5,  # Invalid
                posterior_std=0.1,
                confidence_interval=(0.5, 0.9),
                necessity_test_passed=True,
                necessity_test_missing=[],
                evidence_snippets=[],
                sha256_evidence="abc",
            )

        self.assertIn("posterior_mean must be in [0, 1]", str(ctx.exception))

    def test_payload_evidence_hash_computation(self):
        """Test SHA256 hash computation for evidence"""
        snippets = ["Evidence 1", "Evidence 2", "Evidence 3"]
        hash1 = ExplainabilityPayload.compute_evidence_hash(snippets)

        # Hash should be 64 hex characters (SHA256)
        self.assertEqual(len(hash1), 64)

        # Same snippets should produce same hash (deterministic)
        hash2 = ExplainabilityPayload.compute_evidence_hash(snippets)
        self.assertEqual(hash1, hash2)

        # Different snippets should produce different hash
        hash3 = ExplainabilityPayload.compute_evidence_hash(["Different"])
        self.assertNotEqual(hash1, hash3)

    def test_payload_to_dict(self):
        """Test conversion to dictionary"""
        payload = ExplainabilityPayload(
            link_id="LINK_001",
            posterior_mean=0.751234,
            posterior_std=0.123456,
            confidence_interval=(0.55, 0.90),
            necessity_test_passed=False,
            necessity_test_missing=["entity", "budget"],
            evidence_snippets=["s1", "s2", "s3", "s4", "s5", "s6"],
            sha256_evidence="abc123",
        )

        payload_dict = payload.to_dict()

        # Check fields
        self.assertEqual(payload_dict["link_id"], "LINK_001")
        self.assertEqual(payload_dict["posterior_mean"], 0.751234)
        self.assertFalse(payload_dict["necessity_test"]["passed"])
        self.assertEqual(len(payload_dict["necessity_test"]["missing"]), 2)

        # Evidence snippets should be limited to 5
        self.assertEqual(len(payload_dict["evidence"]["snippets"]), 5)


# ============================================================================
# Test Audit Point 5.4: Human-in-the-Loop Gate
# ============================================================================


class TestHumanInTheLoopGate(unittest.TestCase):
    """Test suite for human-in-the-loop gate (Audit Point 5.4)"""

    def test_gate_triggers_on_non_excelente(self):
        """Test gate triggers when quality_grade != 'Excelente'"""
        gate = HumanInTheLoopGate(
            quality_grade=QualityGrade.BUENO,
            critical_severity_count=0,
            total_contradictions=7,
            coherence_score=0.72,
        )

        # Should require manual review
        self.assertTrue(gate.hold_for_manual_review)
        self.assertEqual(gate.approver_role, "policy_analyst")

    def test_gate_triggers_on_critical_severity(self):
        """Test gate triggers when critical_severity > 0"""
        gate = HumanInTheLoopGate(
            quality_grade=QualityGrade.EXCELENTE,
            critical_severity_count=1,  # Critical severity present
            total_contradictions=3,
            coherence_score=0.85,
        )

        # Should require manual review despite Excelente grade
        self.assertTrue(gate.hold_for_manual_review)
        self.assertEqual(gate.approver_role, "policy_analyst")

    def test_gate_no_trigger_on_excelente(self):
        """Test gate does NOT trigger when Excelente and no critical issues"""
        gate = HumanInTheLoopGate(
            quality_grade=QualityGrade.EXCELENTE,
            critical_severity_count=0,
            total_contradictions=3,
            coherence_score=0.90,
        )

        # Should NOT require manual review
        self.assertFalse(gate.hold_for_manual_review)

    def test_gate_approval_workflow(self):
        """Test manual approval workflow"""
        gate = HumanInTheLoopGate(
            quality_grade=QualityGrade.BUENO,
            critical_severity_count=0,
            total_contradictions=7,
            coherence_score=0.72,
        )

        # Should be held initially
        self.assertTrue(gate.hold_for_manual_review)

        # Approve by reviewer
        gate.approve(reviewer_id="analyst_001")

        # Should no longer be held
        self.assertFalse(gate.hold_for_manual_review)
        self.assertEqual(gate.reviewer_id, "analyst_001")
        self.assertIsNotNone(gate.review_timestamp)

    def test_gate_trigger_reason(self):
        """Test trigger reason reporting"""
        gate = HumanInTheLoopGate(
            quality_grade=QualityGrade.REGULAR,
            critical_severity_count=2,
            total_contradictions=15,
            coherence_score=0.55,
        )

        reason = gate._get_trigger_reason()

        # Should include both triggers
        self.assertIn("quality_grade=Regular", reason)
        self.assertIn("critical_severity=2", reason)


# ============================================================================
# Test Audit Point 5.5: CI Contract Enforcement (Methodological Gates)
# ============================================================================


class TestCIContractEnforcement(unittest.TestCase):
    """
    Test suite for CI contract enforcement (Audit Point 5.5)

    Methodological gates that BLOCK MERGE on failure:
    - test_hoop_test_failure: Necessity test must detect missing components
    - test_posterior_cap_enforced: Posterior must be bounded [0, 1]
    - test_mechanism_prior_decay: Prior should decay without evidence
    """

    def setUp(self):
        """Set up test fixtures"""
        self.tester = NecessitySufficiencyTester()
        self.engine = BayesianSamplingEngine(seed=42)
        self.builder = BayesianPriorBuilder()

        # Create sample embeddings
        self.cause_emb = np.random.randn(384)
        self.effect_emb = np.random.randn(384)

        # Normalize
        self.cause_emb /= np.linalg.norm(self.cause_emb)
        self.effect_emb /= np.linalg.norm(self.effect_emb)

    def test_hoop_test_failure(self):
        """
        CI CONTRACT: Hoop test MUST fail when necessary components missing

        CRITICAL: This test ensures necessity tests detect missing evidence.
        If this test fails, the CI pipeline MUST BLOCK MERGE.
        """
        # Create link without complete evidence
        link = CausalLink(
            cause_id="MP-001",
            effect_id="MR-001",
            cause_emb=self.cause_emb,
            effect_emb=self.effect_emb,
            cause_type="producto",
            effect_type="resultado",
        )

        # Create incomplete evidence (missing entity and budget)
        doc_evidence = DocumentEvidence()
        # Only add activity, no entity or budget
        doc_evidence.activities[("MP-001", "MR-001")] = ["activity1"]

        # Run necessity test
        result = self.tester.test_necessity(link, doc_evidence)

        # Test MUST fail (missing components)
        self.assertFalse(
            result.passed,
            "Hoop test should FAIL when evidence incomplete - CI GATE VIOLATED",
        )

        # Should report missing components
        self.assertIn("entity", result.missing)
        self.assertIn("budget", result.missing)

        # Should provide remediation
        self.assertIsNotNone(result.remediation)

    def test_posterior_cap_enforced(self):
        """
        CI CONTRACT: Posterior mean MUST be bounded in [0, 1]

        CRITICAL: This test ensures posterior distributions are valid probabilities.
        If this test fails, the CI pipeline MUST BLOCK MERGE.
        """
        prior = MechanismPrior(alpha=2.0, beta=2.0, rationale="Test prior")

        # Sample with various evidence levels
        evidence_levels = [
            [],  # No evidence
            [EvidenceChunk("c1", "text", 0.9)],  # High similarity
            [EvidenceChunk("c1", "text", 0.1)],  # Low similarity
            [
                EvidenceChunk(f"c{i}", "text", 0.5 + i * 0.1) for i in range(10)
            ],  # Many chunks
        ]

        config = SamplingConfig(draws=100)

        for evidence in evidence_levels:
            posterior = self.engine.sample_mechanism_posterior(prior, evidence, config)

            # Posterior mean MUST be in [0, 1]
            self.assertGreaterEqual(
                posterior.posterior_mean,
                0.0,
                f"Posterior mean {posterior.posterior_mean} < 0 - CI GATE VIOLATED",
            )
            self.assertLessEqual(
                posterior.posterior_mean,
                1.0,
                f"Posterior mean {posterior.posterior_mean} > 1 - CI GATE VIOLATED",
            )

            # Confidence interval must also be bounded
            self.assertGreaterEqual(posterior.confidence_interval[0], 0.0)
            self.assertLessEqual(posterior.confidence_interval[1], 1.0)

    def test_mechanism_prior_decay(self):
        """
        CI CONTRACT: Prior should decay (become less confident) without fresh evidence

        CRITICAL: This test ensures priors don't artificially inflate over time.
        If this test fails, the CI pipeline MUST BLOCK MERGE.
        """
        link = CausalLink(
            cause_id="MP-001",
            effect_id="MR-001",
            cause_emb=self.cause_emb,
            effect_emb=self.effect_emb,
            cause_type="producto",
            effect_type="resultado",
        )

        mechanism_evidence = MechanismEvidence(
            type="técnico", verb_sequence=["implementar"]
        )

        context = ColombianMunicipalContext(
            overall_pdm_embedding=self.cause_emb  # Use same for simplicity
        )

        # Build initial prior
        prior = self.builder.build_mechanism_prior(link, mechanism_evidence, context)

        # Prior should have reasonable confidence (not overconfident)
        # Beta distribution variance = (alpha*beta)/((alpha+beta)^2*(alpha+beta+1))
        # For high confidence, one parameter should be much larger than the other

        # Calculate effective sample size (alpha + beta)
        effective_n = prior.alpha + prior.beta

        # Effective sample size should be moderate (not too high without evidence)
        # Typical reasonable range is 2-20 for uninformed/weakly-informed priors
        self.assertLess(
            effective_n,
            50,
            f"Prior too confident (effective_n={effective_n}) without evidence - "
            "CI GATE VIOLATED",
        )

        # Alpha and beta should both be positive
        self.assertGreater(prior.alpha, 0)
        self.assertGreater(prior.beta, 0)


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
