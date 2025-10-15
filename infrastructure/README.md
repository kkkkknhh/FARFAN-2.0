# F4.2: Circuit Breaker for External Services - Implementation Guide

## Overview

This implementation provides a robust Circuit Breaker pattern for protecting the FARFAN 2.0 pipeline from cascading failures when external services (like DNP validation APIs) become unavailable.

## 🎯 Problem Addressed

**Issue**: External dependencies (ValidadorDNP, APIs) can fail and block the entire analytical pipeline, causing:
- Pipeline stalls waiting for unavailable services
- Cascading failures across dependent components
- Loss of throughput during external service outages
- No graceful degradation mechanism

**Solution**: Circuit Breaker pattern with fail-open policy implementation.

## 📦 Components

### 1. `infrastructure/circuit_breaker.py`

Core circuit breaker implementation following the canonical pattern:

```python
from infrastructure import CircuitBreaker, CircuitOpenError, CircuitState

# Create circuit breaker
breaker = CircuitBreaker(
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=60,      # Try recovery after 60 seconds
    expected_exception=Exception
)

# Use with async functions
async def call_external_api():
    result = await breaker.call(external_service_function, arg1, arg2)
    return result
```

**Features**:
- ✅ Three states: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)
- ✅ Configurable failure thresholds and recovery timeouts
- ✅ Automatic state transitions based on success/failure patterns
- ✅ Comprehensive metrics for observability
- ✅ Support for both async and sync functions

### 2. `infrastructure/resilient_dnp_validator.py`

DNP Validator wrapper with circuit breaker integration and fail-open policy:

```python
from infrastructure import ResilientDNPValidator, PDMData, DNPAPIClient

# Create resilient validator
validator = ResilientDNPValidator(
    dnp_api_client=your_dnp_client,
    failure_threshold=3,        # More aggressive for critical service
    recovery_timeout=120,       # 2 minutes before retry
    fail_open_penalty=0.05      # 5% penalty when skipping
)

# Validate with automatic failure handling
data = PDMData(
    sector="salud",
    descripcion="Programa de salud",
    indicadores_propuestos=["IND001", "IND002"]
)

result = await validator.validate(data)

if result.status == 'skipped':
    logger.warning(f"Validation skipped: {result.reason}")
    # Pipeline continues with minor penalty
elif result.status == 'passed':
    logger.info(f"Validation passed: score={result.score}")
```

**Features**:
- ✅ Fail-open policy: continue with penalty if service unavailable
- ✅ Circuit breaker integration prevents cascading failures
- ✅ Configurable penalty for skipped validations (default: 5%)
- ✅ Full metrics and observability
- ✅ Backwards compatible with existing ValidadorDNP interface

### 3. `infrastructure/__init__.py`

Module exports for easy importing:

```python
from infrastructure import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    ResilientDNPValidator,
    ValidationResult,
    PDMData
)
```

## 🔄 State Transitions

The circuit breaker follows this state machine:

```
CLOSED (Normal)
    │
    ├─ Success → Stay CLOSED
    │
    └─ Failures ≥ threshold
        ↓
    OPEN (Rejecting)
        │
        └─ After recovery_timeout
            ↓
        HALF_OPEN (Testing)
            │
            ├─ Success → CLOSED
            │
            └─ Failure → OPEN
```

## 📊 Fail-Open vs Fail-Closed Policy

### Fail-Open Policy (Implemented)

**When circuit opens**: Continue processing with degraded validation
- Status: `skipped`
- Score: `1.0 - penalty` (e.g., 0.95 with 5% penalty)
- Pipeline: **CONTINUES**
- Use case: Non-critical validations, external enrichment services

**Benefits**:
- ✅ Maintains pipeline throughput
- ✅ Prevents cascading failures
- ✅ Graceful degradation
- ✅ Minor penalty acknowledges incomplete validation

### Fail-Closed Policy (Alternative)

**When circuit opens**: Stop processing and raise error
- Status: `failed`
- Pipeline: **BLOCKS**
- Use case: Critical validations required for compliance

**To implement fail-closed**:
```python
result = await validator.validate(data)
if result.status == 'skipped':
    raise ValidationError("Critical validation unavailable")
```

## 🧪 Testing

Comprehensive test suite included in `test_circuit_breaker.py`:

```bash
python test_circuit_breaker.py
```

**Test Coverage**:
- ✅ Circuit breaker initialization and validation
- ✅ CLOSED state behavior (normal operation)
- ✅ OPEN state behavior (failure handling)
- ✅ HALF_OPEN state recovery
- ✅ Metrics collection
- ✅ Resilient validator success cases
- ✅ Fail-open policy implementation
- ✅ Recovery after failures

**All tests passing**: ✅

## 📈 Metrics and Observability

Monitor circuit breaker health:

```python
# Get circuit metrics
metrics = validator.get_circuit_metrics()

print(f"State: {metrics['state']}")
print(f"Total calls: {metrics['total_calls']}")
print(f"Successful: {metrics['successful_calls']}")
print(f"Failed: {metrics['failed_calls']}")
print(f"Rejected: {metrics['rejected_calls']}")
print(f"Transitions: {metrics['state_transitions']}")
```

**Example output**:
```
State: closed
Total calls: 150
Successful: 145
Failed: 5
Rejected: 0
Transitions: 2
```

## 🔧 Configuration Guidelines

### For DNP Validation
```python
ResilientDNPValidator(
    dnp_api_client=client,
    failure_threshold=3,      # Open after 3 failures
    recovery_timeout=120,     # Try recovery after 2 minutes
    fail_open_penalty=0.05    # 5% penalty (conservative)
)
```

### For External APIs (Less Critical)
```python
CircuitBreaker(
    failure_threshold=5,      # More tolerant
    recovery_timeout=60,      # Faster recovery attempts
    expected_exception=ConnectionError  # Specific exceptions
)
```

### For Critical Services
```python
CircuitBreaker(
    failure_threshold=10,     # Very tolerant
    recovery_timeout=300,     # 5 minutes before retry
    expected_exception=Exception
)
```

## 🔌 Integration Example

### Integrating with Existing ValidadorDNP

```python
from dnp_integration import ValidadorDNP
from infrastructure import ResilientDNPValidator, DNPAPIClient, PDMData

# Create adapter for existing ValidadorDNP
class DNPValidatorAdapter(DNPAPIClient):
    def __init__(self, validador: ValidadorDNP):
        self.validador = validador
    
    async def validate_compliance(self, data: PDMData) -> dict:
        # Adapt synchronous ValidadorDNP to async interface
        result = self.validador.validar_proyecto_integral(
            sector=data.sector,
            descripcion=data.descripcion,
            indicadores_propuestos=data.indicadores_propuestos,
            presupuesto=data.presupuesto,
            es_rural=data.es_rural,
            poblacion_victimas=data.poblacion_victimas
        )
        
        return {
            'cumple': result.cumple_competencias and result.cumple_mga,
            'score_total': result.score_total,
            'nivel_cumplimiento': result.nivel_cumplimiento.value,
            'detalles': result.to_dict()
        }

# Use in CDAF pipeline
validador_dnp = ValidadorDNP(es_municipio_pdet=False)
adapter = DNPValidatorAdapter(validador_dnp)
resilient_validator = ResilientDNPValidator(adapter)

# Validate with circuit breaker protection
result = await resilient_validator.validate(pdm_data)
```

### Using in CDAFFramework

```python
class CDAFFramework:
    def __init__(self, config_path: Path, output_dir: Path):
        # ... existing initialization ...
        
        # Initialize resilient DNP validator
        if DNP_AVAILABLE:
            validador_dnp = ValidadorDNP(es_municipio_pdet=False)
            adapter = DNPValidatorAdapter(validador_dnp)
            self.dnp_validator = ResilientDNPValidator(
                adapter,
                failure_threshold=3,
                recovery_timeout=120
            )
            self.logger.info("Resilient DNP Validator initialized")
    
    async def validate_with_dnp(self, pdm_data):
        """Validate with circuit breaker protection"""
        result = await self.dnp_validator.validate(pdm_data)
        
        if result.status == 'skipped':
            self.logger.warning(
                f"DNP validation skipped - applying {result.score_penalty} penalty"
            )
        
        return result
```

## ✅ Benefits Delivered

1. **Cascading Failure Prevention**: Circuit opens before failures propagate
2. **Throughput Maintenance**: Pipeline continues with degraded mode
3. **Governance Compliance**: Implements fail-open/fail-closed policies
4. **Observability**: Comprehensive metrics for monitoring
5. **Flexibility**: Configurable thresholds and timeouts
6. **Testing**: Complete test coverage for confidence
7. **Documentation**: Clear usage examples and integration guides

## 📁 Files Created

- `infrastructure/circuit_breaker.py` (352 lines)
- `infrastructure/resilient_dnp_validator.py` (286 lines)
- `infrastructure/__init__.py` (45 lines)
- `test_circuit_breaker.py` (354 lines)
- `infrastructure/README.md` (this file)

**Total**: ~1,037 lines of production-quality code with tests

## 🚀 Quick Start

```python
# 1. Import components
from infrastructure import (
    ResilientDNPValidator,
    PDMData,
    DNPAPIClient
)

# 2. Create your API client (implement DNPAPIClient interface)
class MyDNPClient(DNPAPIClient):
    async def validate_compliance(self, data: PDMData) -> dict:
        # Your API call here
        pass

# 3. Create resilient validator
validator = ResilientDNPValidator(MyDNPClient())

# 4. Use it
data = PDMData(sector="salud", descripcion="...", indicadores_propuestos=[...])
result = await validator.validate(data)

# 5. Handle result
if result.status == 'passed':
    print(f"✓ Validation passed: {result.score}")
elif result.status == 'skipped':
    print(f"⚠ Validation skipped: {result.reason} (penalty: {result.score_penalty})")
else:
    print(f"✗ Validation failed: {result.reason}")
```

## 📖 Further Reading

- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html) - Martin Fowler
- [Release It!](https://pragprog.com/titles/mnee2/release-it-second-edition/) - Michael Nygard
- [Resilience Patterns](https://docs.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker) - Microsoft Azure

## 🤝 Support

For questions or issues:
1. Check this README
2. Review test cases in `test_circuit_breaker.py`
3. Examine code documentation in source files
4. Check metrics output for debugging

---

**Version**: 1.0.0  
**Author**: AI Systems Architect  
**Date**: 2025-10-15
