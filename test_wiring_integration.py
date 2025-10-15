#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Tests for FARFAN 2.0 Wiring and Scoring
Tests that verify the complete pipeline integration, data flow and scoring.
"""

import unittest
import sys
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

from audit_wiring_and_scoring import WiringAuditor
from module_choreographer import ModuleChoreographer, create_canonical_flow


class TestCanonicalFlow(unittest.TestCase):
    """Test the canonical flow definition and integrity"""
    
    def test_canonical_flow_exists(self):
        """Test that canonical flow is defined and non-empty"""
        flow = create_canonical_flow()
        self.assertIsNotNone(flow)
        self.assertGreater(len(flow), 0)
        self.assertIsInstance(flow, list)
    
    def test_canonical_flow_has_all_stages(self):
        """Test that canonical flow covers all 9 stages"""
        flow = create_canonical_flow()
        stages = {step[0] for step in flow}
        
        # Should have stages 1-9
        expected_stages = {
            "STAGE_1_2", "STAGE_3", "STAGE_4", "STAGE_5",
            "STAGE_6", "STAGE_7", "STAGE_8", "STAGE_9"
        }
        
        for stage in expected_stages:
            self.assertIn(stage, stages, f"Missing stage: {stage}")
    
    def test_canonical_flow_step_format(self):
        """Test that each flow step has correct format"""
        flow = create_canonical_flow()
        
        for step in flow:
            # Each step should be (stage, module, function, inputs)
            self.assertEqual(len(step), 4, f"Invalid step format: {step}")
            
            stage, module, function, inputs = step
            self.assertIsInstance(stage, str)
            self.assertIsInstance(module, str)
            self.assertIsInstance(function, str)
            self.assertIsInstance(inputs, list)
    
    def test_canonical_flow_sequential_stages(self):
        """Test that stages are executed in order"""
        flow = create_canonical_flow()
        
        # Stage order mapping
        stage_order = {
            "STAGE_1_2": 1,
            "STAGE_3": 3,
            "STAGE_4": 4,
            "STAGE_5": 5,
            "STAGE_6": 6,
            "STAGE_7": 7,
            "STAGE_8": 8,
            "STAGE_9": 9
        }
        
        prev_order = 0
        for step in flow:
            stage = step[0]
            current_order = stage_order.get(stage, 0)
            
            # Current stage should be >= previous
            self.assertGreaterEqual(
                current_order, prev_order,
                f"Stage order violation: {stage} comes after stage with order {prev_order}"
            )
            prev_order = current_order


class TestModuleChoreographer(unittest.TestCase):
    """Test the ModuleChoreographer functionality"""
    
    def setUp(self):
        self.choreographer = ModuleChoreographer()
    
    def test_choreographer_initialization(self):
        """Test that choreographer initializes correctly"""
        self.assertIsNotNone(self.choreographer.execution_history)
        self.assertIsNotNone(self.choreographer.accumulators)
        self.assertIsNotNone(self.choreographer.module_registry)
        
        self.assertEqual(len(self.choreographer.execution_history), 0)
        self.assertEqual(len(self.choreographer.accumulators), 0)
        self.assertEqual(len(self.choreographer.module_registry), 0)
    
    def test_register_module(self):
        """Test module registration"""
        mock_module = Mock()
        self.choreographer.register_module("test_module", mock_module)
        
        self.assertIn("test_module", self.choreographer.module_registry)
        self.assertEqual(self.choreographer.module_registry["test_module"], mock_module)
    
    def test_accumulate_for_question(self):
        """Test question response accumulation"""
        question_id = "P1-D1-Q1"
        module_name = "test_module"
        contribution = {"score": 0.8, "evidence": ["test evidence"]}
        
        self.choreographer.accumulate_for_question(
            question_id, module_name, contribution
        )
        
        self.assertIn(question_id, self.choreographer.accumulators)
        accumulator = self.choreographer.accumulators[question_id]
        self.assertIn(module_name, accumulator.module_contributions)
    
    def test_synthesize_responses(self):
        """Test response synthesis"""
        # Add some contributions
        q1 = "P1-D1-Q1"
        self.choreographer.accumulate_for_question(
            q1, "module1", {"score": 0.8}
        )
        self.choreographer.accumulate_for_question(
            q1, "module2", {"score": 0.6}
        )
        
        synthesized = self.choreographer.synthesize_responses()
        
        self.assertIn(q1, synthesized)
        self.assertIn("synthesized_score", synthesized[q1])
        # Average of 0.8 and 0.6 is 0.7
        self.assertAlmostEqual(synthesized[q1]["synthesized_score"], 0.7, places=2)
    
    def test_verify_all_modules_used(self):
        """Test module usage verification"""
        # Register modules
        self.choreographer.register_module("module1", Mock())
        self.choreographer.register_module("module2", Mock())
        
        # Add execution for module1 only
        from module_choreographer import ModuleExecution
        exec1 = ModuleExecution(
            module_name="module1",
            stage="STAGE_1",
            inputs={},
            outputs={},
            execution_time=0.1,
            success=True
        )
        self.choreographer.execution_history.append(exec1)
        
        # Check verification
        all_used, unused = self.choreographer.verify_all_modules_used(
            ["module1", "module2"]
        )
        
        self.assertFalse(all_used)
        self.assertIn("module2", unused)
    
    def test_get_data_transfer_log(self):
        """Test data transfer log generation"""
        from module_choreographer import ModuleExecution
        
        # Add multiple executions
        exec1 = ModuleExecution(
            module_name="module1",
            stage="STAGE_1",
            inputs={},
            outputs={"data": "test"},
            execution_time=0.1,
            success=True
        )
        exec2 = ModuleExecution(
            module_name="module2",
            stage="STAGE_2",
            inputs={"data": "test"},
            outputs={"result": "processed"},
            execution_time=0.2,
            success=True
        )
        
        self.choreographer.execution_history.extend([exec1, exec2])
        
        transfers = self.choreographer.get_data_transfer_log()
        
        self.assertGreater(len(transfers), 0)
        self.assertEqual(transfers[0]["from_module"], "module1")
        self.assertEqual(transfers[0]["to_module"], "module2")


class TestWiringAuditor(unittest.TestCase):
    """Test the WiringAuditor functionality"""
    
    def setUp(self):
        repo_path = Path(__file__).parent
        self.auditor = WiringAuditor(repo_path)
    
    def test_auditor_initialization(self):
        """Test that auditor initializes correctly"""
        self.assertIsNotNone(self.auditor.modules_info)
        self.assertIsNotNone(self.auditor.data_flows)
        self.assertIsNotNone(self.auditor.scoring_components)
        self.assertIsNotNone(self.auditor.canonical_flow)
    
    def test_analyze_modules(self):
        """Test module analysis"""
        self.auditor._analyze_modules()
        
        # Should have analyzed several modules
        self.assertGreater(len(self.auditor.modules_info), 0)
        
        # Check that key modules are present
        key_modules = ["orchestrator", "module_choreographer", "question_answering_engine"]
        for module in key_modules:
            self.assertIn(module, self.auditor.modules_info)
    
    def test_extract_canonical_flow(self):
        """Test canonical flow extraction"""
        self.auditor._extract_canonical_flow()
        
        self.assertGreater(len(self.auditor.canonical_flow), 0)
        # Should have 18 steps as shown in the flow definition
        self.assertEqual(len(self.auditor.canonical_flow), 18)
    
    def test_analyze_scoring_components(self):
        """Test scoring component analysis"""
        self.auditor._analyze_scoring()
        
        self.assertGreater(len(self.auditor.scoring_components), 0)
        
        # Check that scoring covers required aspects
        all_aspects = set()
        for comp in self.auditor.scoring_components:
            all_aspects.update(comp.contributes_to)
        
        required_aspects = {"quantitative", "qualitative", "justification"}
        self.assertTrue(
            required_aspects.issubset(all_aspects),
            f"Missing scoring aspects: {required_aspects - all_aspects}"
        )
    
    def test_analyze_data_flow(self):
        """Test data flow analysis"""
        self.auditor._extract_canonical_flow()
        self.auditor._analyze_data_flow()
        
        self.assertGreater(len(self.auditor.data_flows), 0)
    
    def test_full_audit_generates_report(self):
        """Test that full audit generates a complete report"""
        report = self.auditor.analyze_repository()
        
        # Check report structure
        self.assertIn("metadata", report)
        self.assertIn("flow_summary", report)
        self.assertIn("coverage_evidence", report)
        self.assertIn("scoring_audit", report)
        self.assertIn("issues_detected", report)
        
        # Check metadata
        self.assertIn("total_modules_analyzed", report["metadata"])
        self.assertIn("canonical_flow_steps", report["metadata"])
        
        # Check coverage evidence
        self.assertIn("all_functions_used", report["coverage_evidence"])
        self.assertIn("coverage_percentage", report["coverage_evidence"])
        
        # Check scoring audit
        self.assertIn("verified", report["scoring_audit"])
        self.assertIn("total_components", report["scoring_audit"])


class TestScoringIntegrity(unittest.TestCase):
    """Test scoring system integrity"""
    
    def test_scoring_components_have_weights(self):
        """Test that all scoring components have weights"""
        auditor = WiringAuditor(Path(__file__).parent)
        auditor._analyze_scoring()
        
        for comp in auditor.scoring_components:
            self.assertIsNotNone(comp.weight)
            self.assertGreater(comp.weight, 0)
            self.assertLessEqual(comp.weight, 1.0)
    
    def test_scoring_covers_all_aspects(self):
        """Test that scoring covers quantitative, qualitative, and justification"""
        auditor = WiringAuditor(Path(__file__).parent)
        auditor._analyze_scoring()
        
        all_aspects = set()
        for comp in auditor.scoring_components:
            all_aspects.update(comp.contributes_to)
        
        required_aspects = {"quantitative", "qualitative", "justification"}
        self.assertEqual(
            required_aspects,
            all_aspects & required_aspects,
            "Not all required scoring aspects are covered"
        )
    
    def test_scoring_modules_exist(self):
        """Test that all scoring modules exist"""
        auditor = WiringAuditor(Path(__file__).parent)
        auditor._analyze_modules()
        auditor._analyze_scoring()
        
        scoring_modules = {comp.module for comp in auditor.scoring_components}
        analyzed_modules = set(auditor.modules_info.keys())
        
        # All scoring modules should have been analyzed
        for module in scoring_modules:
            self.assertIn(
                module, analyzed_modules,
                f"Scoring module {module} not found in analyzed modules"
            )


class TestDataFlowIntegrity(unittest.TestCase):
    """Test data flow integrity"""
    
    def test_pipeline_context_transfers(self):
        """Test that PipelineContext properly transfers data between stages"""
        auditor = WiringAuditor(Path(__file__).parent)
        auditor._analyze_data_flow()
        
        # Should have data flows
        self.assertGreater(len(auditor.data_flows), 0)
    
    def test_stage_data_accumulation(self):
        """Test that data accumulates through stages"""
        # Data should be added at each stage, not replaced
        auditor = WiringAuditor(Path(__file__).parent)
        auditor._analyze_data_flow()
        
        # Get unique data descriptions
        data_types = {flow.data_description for flow in auditor.data_flows}
        
        # Should have multiple distinct data types
        self.assertGreater(len(data_types), 5)


class TestFunctionCoverage(unittest.TestCase):
    """Test function usage coverage"""
    
    def test_public_functions_have_coverage_info(self):
        """Test that we can determine function coverage"""
        auditor = WiringAuditor(Path(__file__).parent)
        auditor._analyze_modules()
        auditor._extract_canonical_flow()
        
        function_usage = auditor._verify_function_usage()
        
        self.assertIn("total_public_functions", function_usage)
        self.assertIn("used_in_canonical_flow", function_usage)
        self.assertIn("coverage_percentage", function_usage)
        
        # Should have some functions
        self.assertGreater(function_usage["total_public_functions"], 0)
    
    def test_core_functions_are_used(self):
        """Test that core orchestrator functions are in the flow"""
        flow = create_canonical_flow()
        
        # Core functions that must be present
        core_functions = [
            "PDFProcessor.load_document",
            "PDFProcessor.extract_text",
            "CausalExtractor.extract_causal_hierarchy",
            "QuestionAnsweringEngine.answer_all_questions",
            "ReportGenerator.generate_micro_report"
        ]
        
        flow_functions = {step[2] for step in flow}
        
        for func in core_functions:
            self.assertIn(
                func, flow_functions,
                f"Core function {func} not found in canonical flow"
            )


class TestReportGeneration(unittest.TestCase):
    """Test audit report generation"""
    
    def test_generate_json_report(self):
        """Test JSON report generation"""
        auditor = WiringAuditor(Path(__file__).parent)
        report = auditor.analyze_repository()
        
        # Should be JSON-serializable
        try:
            json_str = json.dumps(report, indent=2)
            self.assertGreater(len(json_str), 0)
        except Exception as e:
            self.fail(f"Report is not JSON serializable: {e}")
    
    def test_generate_markdown_report(self):
        """Test Markdown report generation"""
        auditor = WiringAuditor(Path(__file__).parent)
        report = auditor.analyze_repository()
        markdown = auditor.generate_markdown_report(report)
        
        self.assertGreater(len(markdown), 0)
        self.assertIn("# Auditoría", markdown)
        self.assertIn("## 1. Resumen Ejecutivo", markdown)
        self.assertIn("## 2. Flujo de Datos", markdown)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
