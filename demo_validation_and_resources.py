#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration test demonstrating data validation and resource management
This test shows how the validators and resource managers work together in the pipeline
"""

from pipeline_validators import (
    CausalExtractionData,
    DNPValidationData,
    QuestionAnsweringData,
    validate_stage_transition
)
from resource_management import (
    managed_stage_execution,
    MemoryMonitor,
    memory_profiling_decorator
)
from pydantic import ValidationError


def demonstrate_pipeline_validation():
    """Demonstrate how pipeline validation works"""
    print("=" * 80)
    print("DEMONSTRATION: Pipeline Data Validation")
    print("=" * 80)
    
    # Initialize memory monitor
    monitor = MemoryMonitor(log_interval_mb=100.0)
    print(f"\n📊 Memory monitoring started: {monitor.initial_memory:.2f} MB\n")
    
    # Stage 4: Causal Extraction - CRITICAL INVARIANT
    print("--- Stage 4: Causal Extraction ---")
    with managed_stage_execution("STAGE 4"):
        # Simulate successful extraction
        nodes = {
            "node1": {"type": "outcome", "text": "Improve education quality"},
            "node2": {"type": "output", "text": "Build new schools"},
            "node3": {"type": "input", "text": "Allocate budget"}
        }
        
        try:
            stage_data = CausalExtractionData(
                causal_graph=None,
                nodes=nodes,
                causal_chains=[]
            )
            validate_stage_transition("4", stage_data)
            print(f"✅ Stage 4 validation PASSED: {len(nodes)} nodes extracted")
        except ValidationError as e:
            print(f"❌ Stage 4 validation FAILED: {e}")
    
    monitor.check("After Stage 4")
    
    # Stage 4: Show what happens with empty nodes
    print("\n--- Stage 4: Testing Empty Nodes (Should Fail) ---")
    try:
        bad_data = CausalExtractionData(
            causal_graph=None,
            nodes={},  # Empty - violates invariant
            causal_chains=[]
        )
        print("❌ UNEXPECTED: Empty nodes were accepted (this should not happen)")
    except ValidationError as e:
        print("✅ EXPECTED: Empty nodes correctly rejected")
        print(f"   Error: {str(e).split('validation error')[0].strip()}")
    
    # Stage 7: DNP Validation - COMPLIANCE SCORE RANGE
    print("\n--- Stage 7: DNP Validation ---")
    with managed_stage_execution("STAGE 7"):
        # Test normal score
        validation_results = [
            {"node_id": "node1", "resultado": type('obj', (), {'score_total': 85.0})()},
            {"node_id": "node2", "resultado": type('obj', (), {'score_total': 92.0})()},
            {"node_id": "node3", "resultado": type('obj', (), {'score_total': 78.0})()}
        ]
        avg_score = sum(r['resultado'].score_total for r in validation_results) / len(validation_results)
        
        stage_data = DNPValidationData(
            dnp_validation_results=validation_results,
            compliance_score=avg_score
        )
        validate_stage_transition("7", stage_data)
        print(f"✅ Stage 7 validation PASSED: compliance_score = {stage_data.compliance_score:.1f}/100")
    
    monitor.check("After Stage 7")
    
    # Stage 7: Test out-of-range score (should be clamped)
    print("\n--- Stage 7: Testing Out-of-Range Score (Should Clamp) ---")
    out_of_range = DNPValidationData(
        dnp_validation_results=[],
        compliance_score=150.0  # Too high
    )
    print(f"✅ Input: 150.0 → Output: {out_of_range.compliance_score} (clamped)")
    
    # Stage 8: Question Answering - EXACT 300 QUESTIONS
    print("\n--- Stage 8: Question Answering ---")
    with managed_stage_execution("STAGE 8"):
        # Generate exactly 300 questions
        questions = {f"P{i//30+1}-D{(i%30)//5+1}-Q{i%5+1}": {"answer": "test"} for i in range(300)}
        
        try:
            stage_data = QuestionAnsweringData(question_responses=questions)
            validate_stage_transition("8", stage_data)
            print(f"✅ Stage 8 validation PASSED: {len(questions)} questions answered")
        except ValidationError as e:
            print(f"❌ Stage 8 validation FAILED: {e}")
    
    monitor.check("After Stage 8")
    
    # Stage 8: Test wrong number of questions
    print("\n--- Stage 8: Testing Wrong Question Count (Should Fail) ---")
    try:
        bad_qa = QuestionAnsweringData(
            question_responses={f"Q{i}": {} for i in range(299)}  # Only 299
        )
        print("❌ UNEXPECTED: Wrong question count was accepted")
    except ValidationError as e:
        print("✅ EXPECTED: Wrong question count correctly rejected")
        print(f"   Expected: 300, Got: 299")
    
    # Final memory report
    print("\n" + "=" * 80)
    print("Memory Report")
    print("=" * 80)
    report = monitor.report()
    print(f"Initial:     {report['initial_mb']:.2f} MB")
    print(f"Final:       {report['final_mb']:.2f} MB")
    print(f"Peak:        {report['peak_mb']:.2f} MB")
    print(f"Total Delta: {report['total_delta_mb']:+.2f} MB")
    print(f"Peak Delta:  {report['peak_delta_mb']:+.2f} MB")
    
    print("\n✅ All demonstrations completed successfully!")
    print("=" * 80)


@memory_profiling_decorator
def simulate_heavy_operation():
    """Simulate a heavy operation to show memory profiling"""
    print("\n--- Simulating Heavy Operation ---")
    # Allocate some memory
    data = [{"id": i, "value": [0] * 1000} for i in range(1000)]
    print(f"Processed {len(data)} items")
    return len(data)


if __name__ == "__main__":
    demonstrate_pipeline_validation()
    
    # Demonstrate memory profiling
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Memory Profiling")
    print("=" * 80)
    result = simulate_heavy_operation()
    print(f"✅ Heavy operation completed: {result} items processed")
    print("=" * 80)
