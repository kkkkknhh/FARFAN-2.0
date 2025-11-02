"""
Orchestrator Package
====================

Provides pipeline orchestration and dependency injection for FARFAN 2.0.

Main Components:
- pipeline.py: Pure orchestration logic
- factory.py: Dependency injection container

Usage:
    from orchestrator.factory import create_production_dependencies
    from orchestrator.pipeline import create_pipeline
    
    deps = create_production_dependencies()
    pipeline = create_pipeline(log_port=deps["log_port"])
    
    result = pipeline.orchestrate(pipeline_input)
"""

from orchestrator.pipeline import create_pipeline, PolicyAnalysisPipeline
from orchestrator.factory import (
    create_production_dependencies,
    create_test_dependencies,
)

__all__ = [
    "create_pipeline",
    "PolicyAnalysisPipeline",
    "create_production_dependencies",
    "create_test_dependencies",
]
