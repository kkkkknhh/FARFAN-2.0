# FARFAN 2.0 Examples

This directory contains runnable demonstrations of FARFAN 2.0 functionality.

## Available Examples

### Basic Analysis
**File:** `basic_analysis.py`  
**Usage:** `python -m examples.basic_analysis`

Demonstrates the complete FARFAN 2.0 pipeline using the orchestrator. Shows how to:
- Create production dependencies
- Initialize the pipeline
- Run a complete policy analysis
- Access results through contracts

### Semantic Chunking
**File:** `demo_semantic_chunking.py`  
**Usage:** `python -m examples.demo_semantic_chunking`

Demonstrates semantic analysis of Colombian Municipal Development Plans using BGE-M3 embeddings and Bayesian evidence accumulation.

### Embedding Policy
**File:** `demo_embedding_policy.py`  
**Usage:** `python -m examples.demo_embedding_policy`

Demonstrates advanced embedding and P-D-Q canonical notation analysis for Colombian Municipal Development Plans.

### Contradiction Detection
**File:** `demo_contradiction_detection.py`  
**Usage:** `python -m examples.demo_contradiction_detection`

Demonstrates state-of-the-art contradiction detection in Colombian Municipal Development Plans using transformer models and Bayesian inference.

### Theory of Change
**File:** `demo_teoria_cambio.py`  
**Usage:** `python -m examples.demo_teoria_cambio industrial-check`  
**Usage:** `python -m examples.demo_teoria_cambio stochastic-validation "PDM_EJEMPLO_2024"`

Demonstrates causal validation and acyclicity testing for Colombian Municipal Development Plans using Bayesian DAG inference.

### Financial Viability
**File:** `demo_financiero_viabilidad.py`  
**Usage:** `python -m examples.demo_financiero_viabilidad <path_to_pdf>`

Demonstrates financial viability analysis for Colombian Municipal Development Plans using PDF table extraction and validation.

## Architecture

All examples use production adapters from `infrastructure/` for realistic demonstrations:
- `LocalFileAdapter` for file I/O
- `RequestsHttpAdapter` for HTTP requests
- `OsEnvAdapter` for environment variables
- `SystemClockAdapter` for time operations
- `StandardLogAdapter` for logging

## Development

These examples were created as part of the Phase 1 refactoring to:
1. Remove `__main__` blocks from core modules
2. Demonstrate proper usage with dependency injection
3. Show how to compose the pipeline for different use cases

All examples follow the Ports & Adapters (Hexagonal Architecture) pattern.
