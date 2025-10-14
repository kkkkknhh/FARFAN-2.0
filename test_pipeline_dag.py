#!/usr/bin/env python3
"""
Unit tests for DAG-based Pipeline Configuration
Tests pipeline stages, DAG validation, and execution order
"""

import unittest
import tempfile
from pathlib import Path
import yaml

from pipeline_dag import (
    PipelineStage,
    PipelineDAG,
    create_default_pipeline,
    export_default_pipeline_yaml,
    PipelineExecutor
)


class TestPipelineStage(unittest.TestCase):
    """Test PipelineStage dataclass"""
    
    def test_create_stage(self):
        """Test creating a pipeline stage"""
        stage = PipelineStage(
            id='test_stage',
            module='test_module',
            function='test_function',
            inputs=['input1', 'input2'],
            outputs=['output1'],
            depends_on=['previous_stage']
        )
        
        self.assertEqual(stage.id, 'test_stage')
        self.assertEqual(stage.module, 'test_module')
        self.assertEqual(stage.function, 'test_function')
        self.assertEqual(len(stage.inputs), 2)
        self.assertEqual(len(stage.outputs), 1)
        self.assertEqual(len(stage.depends_on), 1)
        self.assertFalse(stage.optional)
    
    def test_optional_stage(self):
        """Test creating optional stage"""
        stage = PipelineStage(
            id='optional_stage',
            module='test_module',
            function='test_function',
            optional=True
        )
        
        self.assertTrue(stage.optional)
    
    def test_parallel_group(self):
        """Test stage with parallel group"""
        stage = PipelineStage(
            id='parallel_stage',
            module='test_module',
            function='test_function',
            parallel_group='group1'
        )
        
        self.assertEqual(stage.parallel_group, 'group1')


class TestPipelineDAG(unittest.TestCase):
    """Test PipelineDAG class"""
    
    def test_empty_dag(self):
        """Test creating empty DAG"""
        dag = PipelineDAG()
        
        self.assertEqual(len(dag.stages), 0)
        self.assertTrue(dag.validate())
    
    def test_add_single_stage(self):
        """Test adding a single stage"""
        dag = PipelineDAG()
        stage = PipelineStage(
            id='stage1',
            module='module1',
            function='func1'
        )
        
        dag.add_stage(stage)
        
        self.assertEqual(len(dag.stages), 1)
        self.assertIn('stage1', dag.stages)
    
    def test_add_duplicate_stage(self):
        """Test that adding duplicate stage raises error"""
        dag = PipelineDAG()
        stage = PipelineStage(id='stage1', module='module1', function='func1')
        
        dag.add_stage(stage)
        
        with self.assertRaises(ValueError):
            dag.add_stage(stage)
    
    def test_linear_pipeline(self):
        """Test linear pipeline (stage1 -> stage2 -> stage3)"""
        dag = PipelineDAG()
        
        dag.add_stage(PipelineStage(id='stage1', module='m1', function='f1'))
        dag.add_stage(PipelineStage(id='stage2', module='m2', function='f2', 
                                   depends_on=['stage1']))
        dag.add_stage(PipelineStage(id='stage3', module='m3', function='f3',
                                   depends_on=['stage2']))
        
        self.assertTrue(dag.validate())
        order = dag.get_execution_order()
        self.assertEqual(order, ['stage1', 'stage2', 'stage3'])
    
    def test_parallel_branches(self):
        """Test pipeline with parallel branches"""
        dag = PipelineDAG()
        
        # stage1 -> stage2, stage3 (parallel) -> stage4
        dag.add_stage(PipelineStage(id='stage1', module='m1', function='f1'))
        dag.add_stage(PipelineStage(id='stage2', module='m2', function='f2',
                                   depends_on=['stage1']))
        dag.add_stage(PipelineStage(id='stage3', module='m3', function='f3',
                                   depends_on=['stage1']))
        dag.add_stage(PipelineStage(id='stage4', module='m4', function='f4',
                                   depends_on=['stage2', 'stage3']))
        
        self.assertTrue(dag.validate())
        order = dag.get_execution_order()
        
        # stage1 must be first, stage4 must be last
        self.assertEqual(order[0], 'stage1')
        self.assertEqual(order[-1], 'stage4')
        # stage2 and stage3 can be in any order
        self.assertIn('stage2', order)
        self.assertIn('stage3', order)
    
    def test_detect_cycle(self):
        """Test that cycles are detected"""
        dag = PipelineDAG()
        
        dag.add_stage(PipelineStage(id='stage1', module='m1', function='f1',
                                   depends_on=['stage3']))
        dag.add_stage(PipelineStage(id='stage2', module='m2', function='f2',
                                   depends_on=['stage1']))
        dag.add_stage(PipelineStage(id='stage3', module='m3', function='f3',
                                   depends_on=['stage2']))
        
        with self.assertRaises(ValueError) as ctx:
            dag.validate()
        
        self.assertIn("cycle", str(ctx.exception).lower())
    
    def test_get_dependencies(self):
        """Test getting stage dependencies"""
        dag = PipelineDAG()
        
        dag.add_stage(PipelineStage(id='stage1', module='m1', function='f1'))
        dag.add_stage(PipelineStage(id='stage2', module='m2', function='f2',
                                   depends_on=['stage1']))
        
        deps = dag.get_dependencies('stage2')
        self.assertEqual(deps, ['stage1'])
        
        deps = dag.get_dependencies('stage1')
        self.assertEqual(deps, [])
    
    def test_get_dependents(self):
        """Test getting stages that depend on a stage"""
        dag = PipelineDAG()
        
        dag.add_stage(PipelineStage(id='stage1', module='m1', function='f1'))
        dag.add_stage(PipelineStage(id='stage2', module='m2', function='f2',
                                   depends_on=['stage1']))
        dag.add_stage(PipelineStage(id='stage3', module='m3', function='f3',
                                   depends_on=['stage1']))
        
        dependents = dag.get_dependents('stage1')
        self.assertEqual(set(dependents), {'stage2', 'stage3'})
        
        dependents = dag.get_dependents('stage2')
        self.assertEqual(dependents, [])
    
    def test_parallel_groups(self):
        """Test identifying parallel execution groups"""
        dag = PipelineDAG()
        
        dag.add_stage(PipelineStage(id='stage1', module='m1', function='f1'))
        dag.add_stage(PipelineStage(id='stage2', module='m2', function='f2',
                                   parallel_group='group1'))
        dag.add_stage(PipelineStage(id='stage3', module='m3', function='f3',
                                   parallel_group='group1'))
        dag.add_stage(PipelineStage(id='stage4', module='m4', function='f4',
                                   parallel_group='group2'))
        
        groups = dag.get_parallel_groups()
        
        self.assertIn('group1', groups)
        self.assertIn('group2', groups)
        self.assertEqual(set(groups['group1']), {'stage2', 'stage3'})
        self.assertEqual(set(groups['group2']), {'stage4'})
    
    def test_generate_mermaid(self):
        """Test generating Mermaid diagram"""
        dag = PipelineDAG()
        
        dag.add_stage(PipelineStage(id='stage1', module='m1', function='f1'))
        dag.add_stage(PipelineStage(id='stage2', module='m2', function='f2',
                                   depends_on=['stage1']))
        
        mermaid = dag.generate_mermaid()
        
        self.assertIn('```mermaid', mermaid)
        self.assertIn('graph TD', mermaid)
        self.assertIn('stage1', mermaid)
        self.assertIn('stage2', mermaid)
        self.assertIn('-->', mermaid)
    
    def test_to_dict(self):
        """Test exporting DAG to dictionary"""
        dag = PipelineDAG()
        
        dag.add_stage(PipelineStage(
            id='stage1',
            module='m1',
            function='f1',
            inputs=['input1'],
            outputs=['output1']
        ))
        
        config = dag.to_dict()
        
        self.assertIn('stages', config)
        self.assertEqual(len(config['stages']), 1)
        self.assertEqual(config['stages'][0]['id'], 'stage1')
        self.assertEqual(config['stages'][0]['module'], 'm1')
    
    def test_from_dict(self):
        """Test creating DAG from dictionary"""
        config = {
            'stages': [
                {
                    'id': 'stage1',
                    'module': 'm1',
                    'function': 'f1',
                    'inputs': [],
                    'outputs': ['output1'],
                    'depends_on': []
                },
                {
                    'id': 'stage2',
                    'module': 'm2',
                    'function': 'f2',
                    'inputs': ['output1'],
                    'outputs': ['output2'],
                    'depends_on': ['stage1']
                }
            ]
        }
        
        dag = PipelineDAG.from_dict(config)
        
        self.assertEqual(len(dag.stages), 2)
        self.assertIn('stage1', dag.stages)
        self.assertIn('stage2', dag.stages)
        
        order = dag.get_execution_order()
        self.assertEqual(order, ['stage1', 'stage2'])
    
    def test_yaml_roundtrip(self):
        """Test exporting to and loading from YAML"""
        dag1 = PipelineDAG()
        dag1.add_stage(PipelineStage(id='stage1', module='m1', function='f1'))
        dag1.add_stage(PipelineStage(id='stage2', module='m2', function='f2',
                                    depends_on=['stage1']))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = dag1.to_dict()
            yaml.dump(config, f)
            yaml_path = Path(f.name)
        
        try:
            dag2 = PipelineDAG.from_yaml(yaml_path)
            
            self.assertEqual(len(dag2.stages), len(dag1.stages))
            self.assertEqual(dag2.get_execution_order(), dag1.get_execution_order())
        finally:
            yaml_path.unlink()


class TestDefaultPipeline(unittest.TestCase):
    """Test the default FARFAN pipeline configuration"""
    
    def test_create_default_pipeline(self):
        """Test creating default pipeline"""
        dag = create_default_pipeline()
        
        self.assertGreater(len(dag.stages), 0)
        self.assertTrue(dag.validate())
    
    def test_default_pipeline_has_all_stages(self):
        """Test that default pipeline includes all expected stages"""
        dag = create_default_pipeline()
        
        # Check for key stages
        expected_stages = [
            'load_document',
            'extract_text',
            'causal_extraction',
            'question_answering',
            'generate_micro_report'
        ]
        
        for stage_id in expected_stages:
            self.assertIn(stage_id, dag.stages, 
                         f"Expected stage {stage_id} not found")
    
    def test_default_pipeline_execution_order(self):
        """Test that default pipeline has valid execution order"""
        dag = create_default_pipeline()
        
        order = dag.get_execution_order()
        
        # Load document must be first
        self.assertEqual(order[0], 'load_document')
        
        # Question answering must come before report generation
        qa_index = order.index('question_answering')
        micro_index = order.index('generate_micro_report')
        self.assertLess(qa_index, micro_index)
    
    def test_export_default_pipeline_yaml(self):
        """Test exporting default pipeline to YAML"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = Path(f.name)
        
        try:
            export_default_pipeline_yaml(yaml_path)
            
            self.assertTrue(yaml_path.exists())
            
            # Load and validate
            dag = PipelineDAG.from_yaml(yaml_path)
            self.assertTrue(dag.validate())
        finally:
            if yaml_path.exists():
                yaml_path.unlink()


class TestPipelineExecutor(unittest.TestCase):
    """Test PipelineExecutor"""
    
    def test_executor_creation(self):
        """Test creating pipeline executor"""
        dag = PipelineDAG()
        dag.add_stage(PipelineStage(id='stage1', module='m1', function='f1'))
        
        # Mock dependencies
        class MockDeps:
            def get(self, name):
                return None
        
        executor = PipelineExecutor(dag, MockDeps())
        
        self.assertIsNotNone(executor)
        self.assertEqual(executor.dag, dag)
    
    def test_simple_execution(self):
        """Test executing a simple pipeline"""
        dag = PipelineDAG()
        
        # Create a simple stage that just returns a value
        dag.add_stage(PipelineStage(
            id='stage1',
            module='test_module',
            function='test_func',
            inputs=[],
            outputs=['result1']
        ))
        
        # Mock dependencies and module
        class MockModule:
            def test_func(self):
                return "test_result"
        
        class MockDeps:
            def get(self, name):
                if name == 'test_module':
                    return MockModule()
                return None
        
        executor = PipelineExecutor(dag, MockDeps())
        result = executor.execute({})
        
        self.assertIn('result1', result)
        self.assertEqual(result['result1'], "test_result")


if __name__ == "__main__":
    unittest.main(verbosity=2)
