#!/usr/bin/env python3
"""
DAG-based Pipeline Configuration for FARFAN 2.0
Following Category 2.2 requirement for declarative orchestration

This module provides:
1. DAG-based pipeline definition
2. Topological execution order
3. Configurable stage dependencies
4. Hot-swappable module implementations
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Callable
from enum import Enum
import networkx as nx
from pathlib import Path
import yaml

logger = logging.getLogger("pipeline_dag")


# ============================================================================
# PIPELINE STAGE DEFINITIONS
# ============================================================================

@dataclass
class PipelineStage:
    """
    Defines a single stage in the processing pipeline
    
    Input Contract:
        - id: Unique stage identifier
        - module: Module name to execute
        - function: Function name within module
        - inputs: List of required input keys
        - outputs: List of output keys produced
        
    Output Contract:
        - Stage execution produces outputs as specified
        
    Preconditions:
        - All input keys must be available in context
        - Module and function must be registered
        
    Postconditions:
        - All output keys are added to context
        - Stage execution is recorded in history
    """
    
    id: str
    module: str
    function: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    optional: bool = False
    parallel_group: Optional[str] = None
    
    def __hash__(self):
        return hash(self.id)


class PipelineDAG:
    """
    Directed Acyclic Graph for pipeline execution
    
    Manages:
    - Stage dependencies
    - Topological execution order
    - Parallel execution groups
    - Data flow validation
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.stages: Dict[str, PipelineStage] = {}
        logger.info("PipelineDAG initialized")
    
    def add_stage(self, stage: PipelineStage):
        """
        Add a stage to the pipeline DAG
        
        Args:
            stage: PipelineStage to add
        """
        if stage.id in self.stages:
            raise ValueError(f"Stage {stage.id} already exists")
        
        self.stages[stage.id] = stage
        self.graph.add_node(stage.id, stage=stage)
        
        # Add dependency edges
        for dep in stage.depends_on:
            if dep not in self.stages:
                logger.warning(f"Stage {stage.id} depends on undefined stage {dep}")
            self.graph.add_edge(dep, stage.id)
        
        logger.debug(f"Added stage {stage.id}")
    
    def validate(self) -> bool:
        """
        Validate the pipeline DAG
        
        Returns:
            True if DAG is valid
            
        Raises:
            ValueError if cycles detected or dependencies missing
        """
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            raise ValueError(f"Pipeline contains cycles: {cycles}")
        
        # Check that all dependencies exist
        for stage_id, stage in self.stages.items():
            for dep in stage.depends_on:
                if dep not in self.stages:
                    raise ValueError(f"Stage {stage_id} depends on undefined stage {dep}")
        
        logger.info("Pipeline DAG validation passed")
        return True
    
    def get_execution_order(self) -> List[str]:
        """
        Get topological execution order
        
        Returns:
            List of stage IDs in execution order
        """
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError as e:
            logger.error(f"Cannot determine execution order: {e}")
            raise
    
    def get_parallel_groups(self) -> Dict[str, List[str]]:
        """
        Identify stages that can execute in parallel
        
        Returns:
            Dict mapping group ID to list of stage IDs
        """
        parallel_groups = {}
        
        for stage_id, stage in self.stages.items():
            if stage.parallel_group:
                if stage.parallel_group not in parallel_groups:
                    parallel_groups[stage.parallel_group] = []
                parallel_groups[stage.parallel_group].append(stage_id)
        
        return parallel_groups
    
    def get_dependencies(self, stage_id: str) -> List[str]:
        """
        Get direct dependencies of a stage
        
        Args:
            stage_id: Stage identifier
            
        Returns:
            List of stage IDs that must complete before this stage
        """
        if stage_id not in self.stages:
            raise ValueError(f"Unknown stage: {stage_id}")
        
        return self.stages[stage_id].depends_on
    
    def get_dependents(self, stage_id: str) -> List[str]:
        """
        Get stages that depend on this stage
        
        Args:
            stage_id: Stage identifier
            
        Returns:
            List of stage IDs that depend on this stage
        """
        if stage_id not in self.graph:
            raise ValueError(f"Unknown stage: {stage_id}")
        
        return list(self.graph.successors(stage_id))
    
    def generate_mermaid(self) -> str:
        """
        Generate Mermaid diagram of the pipeline
        
        Returns:
            Mermaid markdown diagram
        """
        lines = ["```mermaid", "graph TD"]
        
        # Add nodes
        for stage_id, stage in self.stages.items():
            label = f"{stage_id}<br/>{stage.module}.{stage.function}"
            lines.append(f'    {stage_id}["{label}"]')
        
        # Add edges
        for source, target in self.graph.edges():
            lines.append(f"    {source} --> {target}")
        
        # Mark parallel groups
        parallel_groups = self.get_parallel_groups()
        for group_id, stage_ids in parallel_groups.items():
            lines.append(f"    subgraph {group_id}")
            for stage_id in stage_ids:
                lines.append(f"        {stage_id}")
            lines.append("    end")
        
        lines.append("```")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export pipeline to dictionary format
        
        Returns:
            Dictionary representation of pipeline
        """
        return {
            'stages': [
                {
                    'id': stage.id,
                    'module': stage.module,
                    'function': stage.function,
                    'inputs': stage.inputs,
                    'outputs': stage.outputs,
                    'depends_on': stage.depends_on,
                    'optional': stage.optional,
                    'parallel_group': stage.parallel_group
                }
                for stage in self.stages.values()
            ]
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'PipelineDAG':
        """
        Create pipeline from dictionary configuration
        
        Args:
            config: Dictionary with 'stages' list
            
        Returns:
            PipelineDAG instance
        """
        dag = cls()
        
        for stage_config in config.get('stages', []):
            stage = PipelineStage(
                id=stage_config['id'],
                module=stage_config['module'],
                function=stage_config['function'],
                inputs=stage_config.get('inputs', []),
                outputs=stage_config.get('outputs', []),
                depends_on=stage_config.get('depends_on', []),
                optional=stage_config.get('optional', False),
                parallel_group=stage_config.get('parallel_group')
            )
            dag.add_stage(stage)
        
        dag.validate()
        return dag
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'PipelineDAG':
        """
        Load pipeline from YAML configuration file
        
        Args:
            yaml_path: Path to YAML config file
            
        Returns:
            PipelineDAG instance
        """
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        
        return cls.from_dict(config)


# ============================================================================
# DEFAULT FARFAN PIPELINE CONFIGURATION
# ============================================================================

def create_default_pipeline() -> PipelineDAG:
    """
    Create the default FARFAN 2.0 pipeline
    
    Returns:
        PipelineDAG with standard FARFAN stages
    """
    dag = PipelineDAG()
    
    # Stage 1: Load Document
    dag.add_stage(PipelineStage(
        id='load_document',
        module='pdf_processor',
        function='load_document',
        inputs=['pdf_path'],
        outputs=['document_loaded'],
        depends_on=[]
    ))
    
    # Stage 2a: Extract Text
    dag.add_stage(PipelineStage(
        id='extract_text',
        module='pdf_processor',
        function='extract_text',
        inputs=['document_loaded'],
        outputs=['raw_text'],
        depends_on=['load_document']
    ))
    
    # Stage 2b: Extract Tables
    dag.add_stage(PipelineStage(
        id='extract_tables',
        module='pdf_processor',
        function='extract_tables',
        inputs=['document_loaded'],
        outputs=['tables'],
        depends_on=['load_document'],
        parallel_group='extraction'
    ))
    
    # Stage 2c: Extract Sections
    dag.add_stage(PipelineStage(
        id='extract_sections',
        module='pdf_processor',
        function='extract_sections',
        inputs=['document_loaded'],
        outputs=['sections'],
        depends_on=['load_document'],
        parallel_group='extraction'
    ))
    
    # Stage 3: Semantic Analysis (optional)
    dag.add_stage(PipelineStage(
        id='semantic_analysis',
        module='policy_analyzer',
        function='analyze_document',
        inputs=['raw_text'],
        outputs=['semantic_chunks', 'dimension_scores'],
        depends_on=['extract_text'],
        optional=True
    ))
    
    # Stage 4: Causal Extraction
    dag.add_stage(PipelineStage(
        id='causal_extraction',
        module='causal_extractor',
        function='extract_causal_hierarchy',
        inputs=['raw_text'],
        outputs=['causal_graph', 'nodes', 'causal_chains'],
        depends_on=['extract_text']
    ))
    
    # Stage 5: Mechanism Inference
    dag.add_stage(PipelineStage(
        id='mechanism_inference',
        module='mechanism_extractor',
        function='extract_mechanisms',
        inputs=['nodes'],
        outputs=['mechanism_parts'],
        depends_on=['causal_extraction'],
        parallel_group='analysis'
    ))
    
    # Stage 6: Financial Audit
    dag.add_stage(PipelineStage(
        id='financial_audit',
        module='financial_auditor',
        function='trace_financial_allocation',
        inputs=['tables', 'nodes'],
        outputs=['financial_allocations', 'budget_traceability'],
        depends_on=['extract_tables', 'causal_extraction'],
        parallel_group='analysis'
    ))
    
    # Stage 7: DNP Validation
    dag.add_stage(PipelineStage(
        id='dnp_validation',
        module='dnp_validator',
        function='validate_nodes',
        inputs=['nodes'],
        outputs=['dnp_validation_results', 'compliance_score'],
        depends_on=['causal_extraction']
    ))
    
    # Stage 8: Question Answering
    dag.add_stage(PipelineStage(
        id='question_answering',
        module='qa_engine',
        function='answer_all_questions',
        inputs=['raw_text', 'causal_graph', 'nodes', 'mechanism_parts', 
                'financial_allocations', 'dnp_validation_results'],
        outputs=['question_responses'],
        depends_on=['mechanism_inference', 'financial_audit', 'dnp_validation']
    ))
    
    # Stage 9a: Micro Report
    dag.add_stage(PipelineStage(
        id='generate_micro_report',
        module='report_generator',
        function='generate_micro_report',
        inputs=['question_responses', 'policy_code'],
        outputs=['micro_report'],
        depends_on=['question_answering'],
        parallel_group='reporting'
    ))
    
    # Stage 9b: Meso Report
    dag.add_stage(PipelineStage(
        id='generate_meso_report',
        module='report_generator',
        function='generate_meso_report',
        inputs=['question_responses', 'policy_code'],
        outputs=['meso_report'],
        depends_on=['question_answering'],
        parallel_group='reporting'
    ))
    
    # Stage 9c: Macro Report
    dag.add_stage(PipelineStage(
        id='generate_macro_report',
        module='report_generator',
        function='generate_macro_report',
        inputs=['question_responses', 'compliance_score', 'policy_code'],
        outputs=['macro_report'],
        depends_on=['question_answering'],
        parallel_group='reporting'
    ))
    
    dag.validate()
    return dag


def export_default_pipeline_yaml(output_path: Path):
    """
    Export default pipeline to YAML configuration file
    
    Args:
        output_path: Path to write YAML file
    """
    dag = create_default_pipeline()
    config = dag.to_dict()
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Pipeline configuration exported to {output_path}")


# ============================================================================
# PIPELINE EXECUTOR
# ============================================================================

class PipelineExecutor:
    """
    Executes pipeline stages in topological order
    
    Responsibilities:
    - Execute stages in correct order
    - Manage data flow between stages
    - Handle optional stages gracefully
    - Support parallel execution groups
    """
    
    def __init__(self, dag: PipelineDAG, dependencies: Any, 
                 choreographer: Any = None):
        """
        Initialize executor
        
        Args:
            dag: Pipeline DAG to execute
            dependencies: Dependency injection container
            choreographer: Optional module choreographer for tracking
        """
        self.dag = dag
        self.dependencies = dependencies
        self.choreographer = choreographer
        self.context: Dict[str, Any] = {}
    
    def execute(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline
        
        Args:
            initial_context: Initial context values (e.g., pdf_path, policy_code)
            
        Returns:
            Final context with all outputs
        """
        self.context = initial_context.copy()
        execution_order = self.dag.get_execution_order()
        
        logger.info(f"Executing pipeline with {len(execution_order)} stages")
        
        for stage_id in execution_order:
            stage = self.dag.stages[stage_id]
            
            # Check if inputs are available
            missing_inputs = [inp for inp in stage.inputs 
                            if inp not in self.context]
            
            if missing_inputs:
                if stage.optional:
                    logger.warning(f"Skipping optional stage {stage_id}: "
                                 f"missing inputs {missing_inputs}")
                    continue
                else:
                    raise ValueError(f"Stage {stage_id} missing required inputs: "
                                   f"{missing_inputs}")
            
            # Execute stage
            try:
                outputs = self._execute_stage(stage)
                self.context.update(outputs)
                logger.info(f"✓ Stage {stage_id} completed")
            except Exception as e:
                logger.error(f"✗ Stage {stage_id} failed: {e}")
                if not stage.optional:
                    raise
        
        return self.context
    
    def _execute_stage(self, stage: PipelineStage) -> Dict[str, Any]:
        """
        Execute a single stage
        
        Args:
            stage: Stage to execute
            
        Returns:
            Dictionary of outputs
        """
        # Get module instance
        module = self.dependencies.get(stage.module)
        if module is None:
            raise ValueError(f"Module {stage.module} not available")
        
        # Prepare inputs
        inputs = {inp: self.context[inp] for inp in stage.inputs if inp in self.context}
        
        # Execute through choreographer if available
        if self.choreographer:
            outputs = self.choreographer.execute_module_stage(
                stage_name=stage.id,
                module_name=stage.module,
                function_name=stage.function,
                inputs=inputs
            )
        else:
            # Direct execution
            func = getattr(module, stage.function)
            result = func(**inputs)
            outputs = {'result': result}
        
        # Map outputs
        output_dict = {}
        if len(stage.outputs) == 1:
            output_dict[stage.outputs[0]] = outputs.get('result')
        else:
            # Assume result is a dict or tuple with named outputs
            for i, output_key in enumerate(stage.outputs):
                if isinstance(outputs.get('result'), dict):
                    output_dict[output_key] = outputs['result'].get(output_key)
                elif isinstance(outputs.get('result'), (list, tuple)):
                    output_dict[output_key] = outputs['result'][i]
        
        return output_dict
