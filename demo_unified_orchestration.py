#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Unified Orchestrator Execution
====================================
Demonstrates complete pipeline with mocked components.
"""

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, Mock

import networkx as nx

# Mock pandas if not available
try:
    import pandas as pd
except ImportError:
    class MockPandas:
        class Timestamp:
            @staticmethod
            def now():
                from datetime import datetime
                return type('obj', (object,), {
                    'isoformat': lambda: datetime.now().isoformat()
                })()
    pd = MockPandas()
    sys.modules['pandas'] = pd


from orchestration.unified_orchestrator import UnifiedOrchestrator, UnifiedResult
from orchestration.pdm_orchestrator import (
    ExtractionResult, MechanismResult, ValidationResult, QualityScore
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MockSelfReflection:
    """Mock self-reflection config"""
    enable_prior_learning: bool = True
    prior_history_path: str = "/tmp/demo_prior_history.json"
    feedback_weight: float = 0.1
    min_documents_for_learning: int = 1


@dataclass
class MockConfig:
    """Mock configuration"""
    prior_decay_factor: float = 0.9
    queue_size: int = 10
    max_inflight_jobs: int = 3
    worker_timeout_secs: int = 300
    min_quality_threshold: float = 0.5
    
    def __post_init__(self):
        self.self_reflection = MockSelfReflection()


# ============================================================================
# MOCK COMPONENTS
# ============================================================================

class MockExtractionPipeline:
    """Mock extraction pipeline"""
    
    async def extract_complete(self, pdf_path: str):
        """Mock extraction"""
        print(f"  [Extraction] Processing {pdf_path}")
        return ExtractionResult(
            semantic_chunks=[
                {
                    'text': 'Implementar programa de infraestructura vial',
                    'id': 'chunk_1',
                    'dimension': 'ESTRATEGICO'
                },
                {
                    'text': 'Mejorar cobertura en servicios de salud',
                    'id': 'chunk_2',
                    'dimension': 'DIAGNOSTICO'
                },
                {
                    'text': 'Fortalecer capacidad institucional municipal',
                    'id': 'chunk_3',
                    'dimension': 'PROGRAMATICO'
                }
            ],
            tables=[
                {
                    'title': 'Presupuesto Plurianual',
                    'headers': ['Año', 'Inversión (COP)'],
                    'rows': [[2024, 1000000], [2025, 1200000]]
                }
            ],
            extraction_quality={'score': 0.85}
        )


class MockCausalBuilder:
    """Mock causal graph builder"""
    
    async def build_graph(self, chunks, tables):
        """Build mock causal graph"""
        print(f"  [Graph] Building DAG from {len(chunks)} chunks")
        
        graph = nx.DiGraph()
        
        # Add nodes with types
        graph.add_node('infraestructura_vial', type='producto')
        graph.add_node('conectividad_mejorada', type='resultado')
        graph.add_node('desarrollo_economico', type='impacto')
        
        graph.add_node('servicios_salud', type='producto')
        graph.add_node('salud_poblacional', type='resultado')
        graph.add_node('calidad_vida', type='impacto')
        
        # Add edges with mechanisms
        graph.add_edge('infraestructura_vial', 'conectividad_mejorada', 
                      weight=0.8, mechanism='tecnico')
        graph.add_edge('conectividad_mejorada', 'desarrollo_economico', 
                      weight=0.7, mechanism='mixto')
        graph.add_edge('servicios_salud', 'salud_poblacional', 
                      weight=0.85, mechanism='administrativo')
        graph.add_edge('salud_poblacional', 'calidad_vida', 
                      weight=0.75, mechanism='politico')
        
        print(f"  [Graph] Created graph: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges")
        
        return graph


class MockBayesianEngine:
    """Mock Bayesian inference engine"""
    
    async def infer_all_mechanisms(self, graph, chunks):
        """Mock mechanism inference"""
        print(f"  [Bayesian] Inferring mechanisms for {graph.number_of_edges()} edges")
        
        mechanisms = []
        
        for source, target, data in graph.edges(data=True):
            mech_type = data.get('mechanism', 'mixto')
            
            # Simulate necessity test
            if mech_type == 'administrativo':
                # Fail necessity test for demo
                necessity_test = {
                    'passed': False,
                    'missing': ['timeline', 'budget_allocation']
                }
                posterior_mean = 0.45
            else:
                # Pass necessity test
                necessity_test = {
                    'passed': True,
                    'missing': []
                }
                posterior_mean = 0.75 + (hash(source) % 20) / 100.0
            
            mechanisms.append(MechanismResult(
                type=mech_type,
                necessity_test=necessity_test,
                posterior_mean=posterior_mean
            ))
        
        print(f"  [Bayesian] Completed: {len(mechanisms)} mechanisms inferred")
        print(f"  [Bayesian] Failed necessity tests: "
              f"{sum(1 for m in mechanisms if not m.necessity_test['passed'])}")
        
        return mechanisms


class MockValidator:
    """Mock axiomatic validator"""
    
    def validate_complete(self, graph, chunks, tables):
        """Mock validation"""
        print(f"  [Validation] Validating graph structure and semantics")
        
        from validators.axiomatic_validator import AxiomaticValidationResult
        
        result = AxiomaticValidationResult()
        result.is_valid = True
        result.structural_valid = True
        result.contradiction_density = 0.02
        result.regulatory_score = 78.5
        result.total_nodes = graph.number_of_nodes()
        result.total_edges = graph.number_of_edges()
        
        print(f"  [Validation] Complete: valid={result.is_valid}, "
              f"contradiction_density={result.contradiction_density:.3f}")
        
        return result


class MockScorer:
    """Mock scoring system"""
    
    def calculate_all_levels(self, graph, mechanism_results, validation_result, contradictions):
        """Mock scoring calculation"""
        print(f"  [Scoring] Calculating MICRO→MESO→MACRO scores")
        
        # Simplified scoring
        micro_scores = {
            f'P{i}-D{j}-Q{k}': 0.65 + (i + j + k) / 100.0
            for i in range(1, 4)  # Reduced for demo
            for j in range(1, 4)
            for k in range(1, 3)
        }
        
        meso_scores = {
            'C1': 0.72,
            'C2': 0.68,
            'C3': 0.75,
            'C4': 0.70
        }
        
        macro_score = sum(meso_scores.values()) / len(meso_scores)
        
        print(f"  [Scoring] MICRO: {len(micro_scores)} questions")
        print(f"  [Scoring] MESO: {meso_scores}")
        print(f"  [Scoring] MACRO: {macro_score:.3f}")
        
        return {
            'micro': micro_scores,
            'meso': meso_scores,
            'macro': macro_score
        }


class MockReportGenerator:
    """Mock report generator"""
    
    async def generate(self, result: UnifiedResult, pdf_path: str, run_id: str):
        """Mock report generation"""
        print(f"  [Report] Generating final report")
        
        report_path = Path(f"/tmp/report_{run_id}.json")
        
        import json
        report_data = {
            'run_id': run_id,
            'success': result.success,
            'macro_score': result.macro_score,
            'mechanism_count': len(result.mechanism_results),
            'stage_count': len(result.stage_metrics)
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"  [Report] Saved to {report_path}")
        
        return report_path


# ============================================================================
# DEMO EXECUTION
# ============================================================================

async def main():
    """Run demo orchestration"""
    
    print("="*70)
    print("UNIFIED ORCHESTRATOR DEMO")
    print("="*70)
    
    # Create configuration
    config = MockConfig()
    
    # Create orchestrator
    print("\n[1/3] Initializing UnifiedOrchestrator...")
    orchestrator = UnifiedOrchestrator(config)
    
    # Inject mock components
    print("[2/3] Injecting components...")
    orchestrator.inject_components(
        extraction_pipeline=MockExtractionPipeline(),
        causal_builder=MockCausalBuilder(),
        bayesian_engine=MockBayesianEngine(),
        validator=MockValidator(),
        scorer=MockScorer(),
        report_generator=MockReportGenerator()
    )
    
    # Execute pipeline
    print("[3/3] Executing 9-stage pipeline...\n")
    print("-"*70)
    
    result = await orchestrator.execute_pipeline("/tmp/demo_pdm.pdf")
    
    print("-"*70)
    print(f"\n{'='*70}")
    print("EXECUTION RESULTS")
    print("="*70)
    
    print(f"\nStatus: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
    print(f"Run ID: {result.run_id}")
    print(f"Duration: {result.total_duration:.2f}s")
    
    print(f"\nExtraction:")
    print(f"  - Semantic chunks: {len(result.semantic_chunks)}")
    print(f"  - Tables: {len(result.tables)}")
    
    print(f"\nGraph:")
    if result.causal_graph:
        print(f"  - Nodes: {result.causal_graph.number_of_nodes()}")
        print(f"  - Edges: {result.causal_graph.number_of_edges()}")
    
    print(f"\nBayesian Inference:")
    print(f"  - Mechanisms: {len(result.mechanism_results)}")
    failed = sum(1 for m in result.mechanism_results 
                 if not m.necessity_test.get('passed', True))
    print(f"  - Failed necessity tests: {failed}")
    
    print(f"\nValidation:")
    if result.validation_result:
        print(f"  - Valid: {result.validation_result.is_valid}")
        print(f"  - Contradiction density: {result.validation_result.contradiction_density:.3f}")
    
    print(f"\nScoring:")
    print(f"  - MICRO scores: {len(result.micro_scores)}")
    print(f"  - MESO scores: {len(result.meso_scores)}")
    print(f"  - MACRO score: {result.macro_score:.3f}")
    
    print(f"\nLearning Loop:")
    print(f"  - Penalty factors: {result.penalty_factors}")
    
    print(f"\nStage Metrics:")
    for metric in result.stage_metrics:
        print(f"  - {metric.stage.name}: {metric.duration_seconds:.3f}s "
              f"({metric.items_processed} items)")
    
    # Bottleneck analysis
    print(f"\nBottleneck Analysis:")
    bottlenecks = orchestrator.metrics.get_bottlenecks(top_n=3)
    for i, (stage, duration) in enumerate(bottlenecks, 1):
        print(f"  {i}. {stage}: {duration:.3f}s")
    
    print(f"\n{'='*70}")
    print("✅ Demo completed successfully!")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
