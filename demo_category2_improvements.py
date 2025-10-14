#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Category 2 Improvements - Module Wiring
Demonstrates the new interfaces, DI container, and DAG-based pipeline
"""

from pathlib import Path
import json

# Import new components
from module_interfaces import (
    DependencyInjectionContainer,
    ModuleDependencies,
    IPDFProcessor,
    ICausalExtractor
)
from pipeline_dag import (
    PipelineDAG,
    PipelineStage,
    create_default_pipeline,
    export_default_pipeline_yaml
)
from module_choreographer import ModuleChoreographer


def demo_1_protocol_interfaces():
    """Demo 1: Using Protocol interfaces for type safety"""
    print("\n" + "="*80)
    print("DEMO 1: Protocol Interfaces")
    print("="*80)
    
    # Define a mock implementation that satisfies IPDFProcessor
    class MockPDFProcessor:
        def load_document(self, pdf_path: Path) -> bool:
            print(f"  Loading document: {pdf_path}")
            return True
        
        def extract_text(self) -> str:
            return "Sample extracted text from PDF"
        
        def extract_tables(self):
            return [{"table": "data"}]
        
        def extract_sections(self):
            return {"intro": "Introduction", "body": "Main content"}
    
    # Use with type annotation for safety
    processor: IPDFProcessor = MockPDFProcessor()
    
    success = processor.load_document(Path("sample.pdf"))
    text = processor.extract_text()
    
    print(f"  ✓ Document loaded: {success}")
    print(f"  ✓ Text extracted: {len(text)} characters")
    print(f"  ✓ Type-safe interface enforced at development time")


def demo_2_dependency_injection():
    """Demo 2: Dependency Injection Container"""
    print("\n" + "="*80)
    print("DEMO 2: Dependency Injection Container")
    print("="*80)
    
    # Create DI container
    container = DependencyInjectionContainer()
    
    # Register modules
    class MockModule:
        def __init__(self, name):
            self.name = name
            print(f"  Creating module: {name}")
        
        def process(self):
            return f"Processed by {self.name}"
    
    # Direct registration
    container.register('module_a', MockModule('Module A'))
    
    # Factory registration (lazy loading)
    container.register_factory('module_b', lambda: MockModule('Module B'))
    
    print("\n  Retrieving modules:")
    mod_a = container.get('module_a')
    print(f"  ✓ Retrieved: {mod_a.name}")
    
    mod_b = container.get('module_b')  # Created on first access
    print(f"  ✓ Retrieved (lazy): {mod_b.name}")
    
    # Validate dependencies
    is_valid, missing = container.validate()
    print(f"\n  Validation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if missing:
        print(f"  Missing modules: {missing}")


def demo_3_pipeline_dag():
    """Demo 3: DAG-based Pipeline Configuration"""
    print("\n" + "="*80)
    print("DEMO 3: DAG-based Pipeline")
    print("="*80)
    
    # Create a simple pipeline
    dag = PipelineDAG()
    
    # Add stages
    dag.add_stage(PipelineStage(
        id='load_data',
        module='data_loader',
        function='load',
        inputs=['file_path'],
        outputs=['raw_data']
    ))
    
    dag.add_stage(PipelineStage(
        id='process_data',
        module='processor',
        function='process',
        inputs=['raw_data'],
        outputs=['processed_data'],
        depends_on=['load_data']
    ))
    
    dag.add_stage(PipelineStage(
        id='generate_report',
        module='reporter',
        function='generate',
        inputs=['processed_data'],
        outputs=['report'],
        depends_on=['process_data']
    ))
    
    # Validate DAG
    try:
        dag.validate()
        print("  ✓ Pipeline validated successfully")
    except ValueError as e:
        print(f"  ✗ Validation error: {e}")
    
    # Get execution order
    order = dag.get_execution_order()
    print(f"\n  Execution order:")
    for i, stage_id in enumerate(order, 1):
        stage = dag.stages[stage_id]
        print(f"    {i}. {stage_id} ({stage.module}.{stage.function})")
    
    # Generate Mermaid diagram
    print("\n  Mermaid diagram:")
    mermaid = dag.generate_mermaid()
    for line in mermaid.split('\n')[:10]:  # Show first 10 lines
        print(f"    {line}")


def demo_4_choreographer_tracing():
    """Demo 4: Module Choreographer for Execution Tracing"""
    print("\n" + "="*80)
    print("DEMO 4: Module Choreographer - Execution Tracing")
    print("="*80)
    
    # Create choreographer
    choreographer = ModuleChoreographer()
    
    # Register mock modules
    class MockModule:
        def __init__(self, name):
            self.name = name
        
        def process(self, data):
            return f"{self.name} processed: {data}"
    
    choreographer.register_module('module_a', MockModule('Module A'))
    choreographer.register_module('module_b', MockModule('Module B'))
    
    # Execute stages
    print("\n  Executing stages:")
    
    result1 = choreographer.execute_module_stage(
        stage_name='STAGE_1',
        module_name='module_a',
        function_name='process',
        inputs={'data': 'input_data'}
    )
    print(f"  ✓ Stage 1 completed")
    
    result2 = choreographer.execute_module_stage(
        stage_name='STAGE_2',
        module_name='module_b',
        function_name='process',
        inputs={'data': 'intermediate_data'}
    )
    print(f"  ✓ Stage 2 completed")
    
    # Get execution trace
    trace = choreographer.export_execution_trace()
    print(f"\n  Execution Summary:")
    print(f"    Total executions: {trace['total_executions']}")
    print(f"    Successful: {trace['successful_executions']}")
    print(f"    Total time: {trace['total_time']:.3f}s")
    
    # Module usage report
    usage = choreographer.get_module_usage_report()
    print(f"\n  Module Usage:")
    for module_name, stats in usage.items():
        print(f"    {module_name}: {stats['executions']} executions, "
              f"{stats['total_time']:.3f}s")


def demo_5_default_pipeline():
    """Demo 5: Default FARFAN Pipeline"""
    print("\n" + "="*80)
    print("DEMO 5: Default FARFAN Pipeline")
    print("="*80)
    
    # Create default pipeline
    dag = create_default_pipeline()
    
    print(f"  Pipeline created with {len(dag.stages)} stages")
    
    # Show execution order
    order = dag.get_execution_order()
    print(f"\n  First 5 stages:")
    for i, stage_id in enumerate(order[:5], 1):
        stage = dag.stages[stage_id]
        print(f"    {i}. {stage_id}")
        print(f"       Module: {stage.module}")
        print(f"       Depends on: {stage.depends_on if stage.depends_on else 'None'}")
    
    # Show parallel groups
    parallel = dag.get_parallel_groups()
    print(f"\n  Parallel execution groups: {len(parallel)}")
    for group_id, stages in parallel.items():
        print(f"    {group_id}: {stages}")
    
    # Export to YAML
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_path = Path(f.name)
    
    export_default_pipeline_yaml(yaml_path)
    print(f"\n  ✓ Pipeline exported to: {yaml_path}")
    
    # Show sample YAML content
    with open(yaml_path) as f:
        lines = f.readlines()[:15]  # First 15 lines
    
    print("\n  Sample YAML content:")
    for line in lines:
        print(f"    {line.rstrip()}")
    
    # Clean up
    yaml_path.unlink()


def demo_6_execution_visualization():
    """Demo 6: Execution Visualization with Mermaid"""
    print("\n" + "="*80)
    print("DEMO 6: Execution Visualization")
    print("="*80)
    
    choreographer = ModuleChoreographer()
    
    # Simulate some executions
    class DummyModule:
        def step1(self):
            return "result1"
        def step2(self):
            return "result2"
        def step3(self):
            return "result3"
    
    choreographer.register_module('processor', DummyModule())
    
    # Execute multiple stages
    for i in range(1, 4):
        choreographer.execute_module_stage(
            stage_name=f'STAGE_{i}',
            module_name='processor',
            function_name=f'step{i}',
            inputs={}
        )
    
    # Generate ASCII flow diagram
    print("\n  ASCII Flow Diagram:")
    flow = choreographer.generate_flow_diagram()
    for line in flow.split('\n')[:20]:  # Show first 20 lines
        print(f"    {line}")
    
    # Generate Mermaid diagram
    print("\n  Mermaid Diagram:")
    mermaid = choreographer.generate_mermaid_diagram()
    for line in mermaid.split('\n')[:15]:  # Show first 15 lines
        print(f"    {line}")


def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("FARFAN 2.0 - Category 2 Improvements Demo")
    print("Module Wiring, Interfaces, and DAG-based Pipeline")
    print("="*80)
    
    try:
        demo_1_protocol_interfaces()
        demo_2_dependency_injection()
        demo_3_pipeline_dag()
        demo_4_choreographer_tracing()
        demo_5_default_pipeline()
        demo_6_execution_visualization()
        
        print("\n" + "="*80)
        print("✅ All demos completed successfully!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
