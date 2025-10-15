# F4.2: Circuit Breaker - Quick Start Guide

## 🚀 5-Minute Quick Start

### 1. Basic Circuit Breaker

```python
import asyncio
from infrastructure import CircuitBreaker

# Create circuit breaker
breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=60
)

# Protect an async function
async def call_external_api():
    result = await breaker.call(your_api_function, arg1, arg2)
    return result

# Run it
result = asyncio.run(call_external_api())
```

### 2. Resilient DNP Validator

```python
from infrastructure import ResilientDNPValidator, PDMData, DNPAPIClient

# Implement your DNP client
class MyDNPClient(DNPAPIClient):
    async def validate_compliance(self, data: PDMData) -> dict:
        # Your validation logic here
        return {'cumple': True, 'score_total': 85.0}

# Create resilient validator
validator = ResilientDNPValidator(MyDNPClient())

# Use it
data = PDMData(
    sector="salud",
    descripcion="Programa de salud",
    indicadores_propuestos=["IND001"]
)
result = await validator.validate(data)

# Handle result
if result.status == 'passed':
    print(f"✓ Score: {result.score}")
elif result.status == 'skipped':
    print(f"⚠ Skipped with {result.score_penalty} penalty")
```

### 3. Run Tests

```bash
python test_circuit_breaker.py
```

### 4. Run Examples

```bash
python example_circuit_breaker.py
```

## 📊 Key Metrics

Monitor circuit health:

```python
metrics = validator.get_circuit_metrics()
print(f"State: {metrics['state']}")
print(f"Success rate: {metrics['successful_calls']}/{metrics['total_calls']}")
```

## ⚙️ Configuration Presets

### Conservative (DNP Validation)
```python
ResilientDNPValidator(
    client,
    failure_threshold=3,      # Open after 3 failures
    recovery_timeout=120,     # Wait 2 minutes
    fail_open_penalty=0.05    # 5% penalty
)
```

### Aggressive (Non-critical APIs)
```python
CircuitBreaker(
    failure_threshold=1,      # Open immediately
    recovery_timeout=30       # Quick recovery
)
```

### Tolerant (Flaky services)
```python
CircuitBreaker(
    failure_threshold=10,     # Very tolerant
    recovery_timeout=300      # Wait 5 minutes
)
```

## 📖 Full Documentation

See [`infrastructure/README.md`](infrastructure/README.md) for complete documentation.

## 🧪 Testing

- 9 comprehensive test scenarios
- All tests passing ✅
- Coverage: Circuit states, recovery, metrics, fail-open policy

## 🎯 Benefits

✅ Prevents cascading failures  
✅ Maintains pipeline throughput  
✅ Configurable fail-open/fail-closed policies  
✅ Complete observability  
✅ Production-ready with tests  

## 🔗 Related Files

- `infrastructure/circuit_breaker.py` - Core implementation
- `infrastructure/resilient_dnp_validator.py` - DNP integration
- `test_circuit_breaker.py` - Test suite
- `example_circuit_breaker.py` - Usage examples
- `infrastructure/README.md` - Full documentation
