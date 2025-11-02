# FARFAN 2.0 - Public API Surface Map

**Version**: 1.3.0  
**Last Updated**: 2024-10-31  
**Status**: Active Development - Industrialization Phase

## Overview

This document defines the public API surface of FARFAN 2.0 after the industrialization refactoring. It serves as the single source of truth for:

- What constitutes the public API
- Contract versions and compatibility
- Breaking vs non-breaking changes
- Deprecation policy

## Core Principles

1. **Contracts are Law**: All public APIs accept and return TypedDict contracts
2. **No Implicit I/O**: Core modules are pure; I/O only through ports
3. **Versioned Interfaces**: Every contract has a `schema_version` field
4. **Backward Compatibility**: Changes bump version and provide converters
5. **Fail Fast**: Runtime validation at boundaries using Pydantic

## Public API Modules

### Core Contracts (`core_contracts.py`)

**Status**: Stable ✓  
**Version**: 1.3  
**Purpose**: Single source of truth for all data contracts

#### Contract Types

| Contract Group | Input Contract | Output Contract | Schema Version |
|---------------|---------------|-----------------|----------------|
| Semantic Analysis | `SemanticAnalyzerInput` | `SemanticAnalyzerOutput` | `sem-1.3` |
| Contradiction Detection | `ContradictionDetectionInput` | `ContradictionDetectionOutput` | `cd-1.3` |
| Embedding | `EmbeddingInput` | `EmbeddingOutput` | `emb-1.3` |
| Statement Extraction | `StatementExtractionInput` | `StatementExtractionOutput` | `stmt-1.3` |
| Coherence Metrics | `CoherenceMetricsInput` | `CoherenceMetricsOutput` | `coh-1.3` |
| Regulatory Analysis | `RegulatoryAnalysisInput` | `RegulatoryAnalysisOutput` | `reg-1.3` |
| Audit Summary | `AuditSummaryInput` | `AuditSummaryOutput` | `audit-1.3` |
| Pipeline | `PipelineInput` | `PipelineOutput` | `pipe-1.3` |

**Guarantees**:
- All contracts are TypedDict (structural typing)
- All include `schema_version: str` field
- All have documented invariants in docstrings
- All nested structures use typed contracts

**Example Usage**:
```python
from core_contracts import SemanticAnalyzerInput, CURRENT_VERSIONS

input_data: SemanticAnalyzerInput = {
    'text': 'Document text...',
    'segments': [],
    'ontology_params': {},
    'schema_version': CURRENT_VERSIONS['semantic']
}
```

### Runtime Validators (`contracts_runtime.py`)

**Status**: Stable ✓  
**Version**: 1.3  
**Purpose**: Pydantic models for runtime validation at boundaries

**Guarantees**:
- Mirror every TypedDict in `core_contracts.py`
- Enforce all invariants (bounds, patterns, types)
- Strict mode: reject unknown fields (`extra="forbid"`)
- Clear validation error messages

**Example Usage**:
```python
from contracts_runtime import SemanticAnalyzerInputModel

# Validate at boundary
try:
    validated = SemanticAnalyzerInputModel(**user_input)
except ValidationError as e:
    # Handle validation failure
    print(e.errors())
```

### Ports (`ports.py`)

**Status**: Stable ✓  
**Version**: 1.0  
**Purpose**: Abstract protocols for all I/O operations

#### Available Ports

| Port | Purpose | Key Methods |
|------|---------|-------------|
| `FilePort` | File I/O | `read_text`, `write_text`, `read_json`, `write_json`, `exists` |
| `HttpPort` | HTTP operations | `get`, `post` |
| `EnvPort` | Environment variables | `get`, `get_required`, `get_int`, `get_bool` |
| `ClockPort` | Time operations | `now`, `now_iso`, `timestamp` |
| `LogPort` | Structured logging | `debug`, `info`, `warning`, `error`, `critical` |
| `CachePort` | Key-value caching | `get`, `set`, `delete`, `clear` |
| `ModelPort` | ML model operations | `load`, `embed_batch` |

**Guarantees**:
- All are `Protocol` types (structural subtyping)
- No implementation details (pure interfaces)
- Easy to mock for testing

**Example Usage**:
```python
from ports import FilePort

def analyze_document(text: str, file_port: FilePort) -> dict:
    # Function only depends on port interface, not implementation
    metadata = file_port.read_json('metadata.json')
    # ... process ...
    file_port.write_json('results.json', results)
    return results
```

### Infrastructure Adapters (`infrastructure/`)

**Status**: Stable ✓  
**Version**: 1.0  
**Purpose**: Concrete implementations of ports

#### Production Adapters

| Adapter | Implements | Use Case |
|---------|------------|----------|
| `LocalFileAdapter` | `FilePort` | Real filesystem I/O |
| `RequestsHttpAdapter` | `HttpPort` | HTTP with retry logic |
| `OsEnvAdapter` | `EnvPort` | Real environment variables |
| `SystemClockAdapter` | `ClockPort` | Real system time |

#### Testing Adapters

| Adapter | Implements | Use Case |
|---------|------------|----------|
| `InMemoryFileAdapter` | `FilePort` | Fast, isolated unit tests |
| `MockHttpAdapter` | `HttpPort` | Deterministic HTTP responses |
| `DictEnvAdapter` | `EnvPort` | Controlled environment |
| `FixedClockAdapter` | `ClockPort` | Deterministic time for tests |

**Example Usage**:
```python
from infrastructure.filesystem import LocalFileAdapter, InMemoryFileAdapter

# Production
file_port = LocalFileAdapter(base_path='/data')

# Testing
file_port = InMemoryFileAdapter()
file_port.write_text('/test.txt', 'test data')
assert file_port.read_text('/test.txt') == 'test data'
```

## Tools

### Boundary Scanner (`tools/scan_boundaries.py`)

**Status**: Active ✓  
**Version**: 1.0  
**Purpose**: Enforce architectural boundaries in CI

**Features**:
- Detect I/O operations (files, network, subprocess)
- Detect `__main__` blocks outside allowed directories
- Generate SARIF reports for GitHub Code Scanning
- Configurable violation types and allowed paths

**Usage**:
```bash
python tools/scan_boundaries.py \
  --root . \
  --fail-on=io,subprocess,requests,main \
  --allow-path examples \
  --allow-path cli \
  --sarif out/boundaries.sarif \
  --json out/violations.json
```

## Versioning Policy

### Schema Versions

All contracts follow semantic versioning in their `schema_version` field:

- **Pattern**: `{type}-{major}.{minor}`
- **Example**: `sem-1.3` (semantic analyzer, version 1.3)

### Version Bump Rules

| Change Type | Bump | Example |
|-------------|------|---------|
| Add optional field | Minor | `sem-1.2` → `sem-1.3` |
| Add new enum value | Minor | Add `"warning"` to severity |
| Change field type | **Major** | `confidence: float` → `confidence: int` |
| Remove field | **Major** | Remove deprecated field |
| Rename field | **Major** | `plan_id` → `plan_name` |
| Tighten bounds | **Major** | `score: float` → `score: float [0,1]` |

### Compatibility Contract

- **Minor bumps**: Must be backward compatible
- **Major bumps**: Require migration path
- **Converters**: Provided in `compat/` for major bumps
- **Deprecation**: Minimum 2 minor versions notice

Example:
```
v1.2: Add deprecation warning for old_field
v1.3: Still accept old_field, copy to new_field
v2.0: Remove old_field (converter available)
```

## Breaking Change Protocol

When making a breaking change:

1. **Bump Schema Version**: Increment major version
2. **Add Converter**: Create `compat/convert_{old}_{new}.py`
3. **Update CHANGELOG**: Add entry to interface changes table
4. **Update Tests**: Add round-trip conversion test
5. **Document Migration**: Add migration guide

## Non-Breaking Change Guidelines

Safe to make without version bump:

- ✅ Add new optional field with default
- ✅ Add new enum value (if consumers handle unknown)
- ✅ Relax bounds (wider acceptance)
- ✅ Add new method to Protocol (if not enforced)
- ✅ Improve docstrings
- ✅ Add examples

Requires version bump:

- ❌ Remove field
- ❌ Rename field
- ❌ Change field type
- ❌ Tighten bounds
- ❌ Change required → optional
- ❌ Change method signature

## Stability Guarantees

### Stable (✓)

These modules are considered stable and follow strict versioning:

- `core_contracts.py`
- `contracts_runtime.py`
- `ports.py`
- `infrastructure/filesystem.py`
- `infrastructure/http.py`
- `infrastructure/environment.py`

### Beta (β)

These modules may change without major version bumps:

- (None currently)

### Experimental (⚠)

These modules have no stability guarantees:

- (None currently)

## Deprecation Policy

When deprecating a field or contract:

1. **Mark Deprecated**: Add `@deprecated` decorator or comment
2. **Add Warning**: Emit `DeprecationWarning` on use
3. **Document Timeline**: Specify removal version
4. **Provide Alternative**: Show migration path in warning

Minimum timeline: **2 minor versions** before removal.

Example:
```python
# Deprecated in v1.3, will be removed in v2.0
# Use 'plan_name' instead
plan_id: str  # @deprecated
```

## CI Enforcement

All changes are validated by CI:

- **Boundary Scanner**: Blocks I/O in core modules
- **Contract Tests**: Validate schema versions
- **Schema Diff**: Detects contract changes
- **Version Check**: Enforces version bump if contracts change

## Migration Guides

### From v1.2 to v1.3

(No breaking changes - backward compatible)

### From v1.1 to v1.2

(No breaking changes - backward compatible)

## Contact & Questions

For questions about the public API:

1. Check this document first
2. Review contract docstrings in `core_contracts.py`
3. Check test examples in `tests/`
4. Open an issue for clarification

---

**Last Schema Change**: v1.3 (2024-10-31)  
**Next Review**: When adding new contract types
