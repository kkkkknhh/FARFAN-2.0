# DI Container Quick Reference

## Import

```python
from infrastructure import (
    DIContainer,
    DeviceConfig,
    configure_container,
    IExtractor,
    ICausalBuilder,
    IBayesianEngine,
)
```

## Basic Usage

### Create Container

```python
# Factory (recommended) - includes graceful degradation
container = configure_container(config)

# Manual
container = DIContainer(config)
```

### Register Components

```python
# Singleton (shared instance)
container.register_singleton(IExtractor, PDFProcessor)

# Transient (new instance each time)
container.register_transient(ICausalBuilder, GraphBuilder)

# Factory function
container.register_singleton(
    spacy.Language,
    lambda: spacy.load("es_core_news_lg")
)
```

### Resolve Dependencies

```python
# Manual resolution
extractor = container.resolve(IExtractor)

# Automatic injection (via constructor)
class MyService:
    def __init__(self, extractor: IExtractor):
        self.extractor = extractor

container.register_transient(MyService, MyService)
service = container.resolve(MyService)  # extractor auto-injected!
```

## Interfaces

### Define Your Own

```python
from abc import ABC, abstractmethod

class IMyService(ABC):
    @abstractmethod
    def do_work(self) -> dict:
        pass

class MyServiceImpl(IMyService):
    def do_work(self) -> dict:
        return {'status': 'done'}

container.register_singleton(IMyService, MyServiceImpl)
```

### Built-in Interfaces

```python
# Document extraction
class IExtractor(ABC):
    @abstractmethod
    def extract(self, document_path: str) -> Dict[str, Any]:
        pass

# Causal graph building
class ICausalBuilder(ABC):
    @abstractmethod
    def build_graph(self, extracted_data: Dict[str, Any]) -> Any:
        pass

# Bayesian inference
class IBayesianEngine(ABC):
    @abstractmethod
    def infer(self, graph: Any) -> Dict[str, Any]:
        pass
```

## Testing

### Mock Components

```python
class MockExtractor(IExtractor):
    def extract(self, document_path: str):
        return {'text': 'mock data'}

# In tests
test_container = DIContainer()
test_container.register_singleton(IExtractor, MockExtractor)

# Your code uses the mock
component = test_container.resolve(MyComponent)
assert isinstance(component.extractor, MockExtractor)
```

## Device Management

```python
# Resolves to CPU or CUDA based on config and availability
device_config = container.resolve(DeviceConfig)

print(device_config.device)     # 'cpu' or 'cuda'
print(device_config.use_gpu)    # True/False
print(device_config.gpu_id)     # GPU device ID (if multiple)
```

## Graceful Degradation

The `configure_container()` factory implements:

**NLP Models** (Front A.1):
- Try `es_dep_news_trf` (transformer)
- → `es_core_news_lg` (large)
- → `es_core_news_sm` (small)
- → Log error

**Device** (Front A.2):
- Check config for GPU
- → Check CUDA availability
- → Fall back to CPU

## Common Patterns

### Adapter Pattern

```python
# Wrap existing components to conform to interface
class MyComponentAdapter(IExtractor):
    def __init__(self, config=None):
        from existing_module import ExistingComponent
        self.component = ExistingComponent(config)
    
    def extract(self, document_path: str):
        # Adapt the interface
        result = self.component.process(document_path)
        return self._adapt_result(result)
```

### Factory Functions

```python
def create_complex_component(config, option_a, option_b):
    # Complex initialization logic
    component = ComplexComponent()
    component.configure(config)
    component.set_options(option_a, option_b)
    return component

container.register_singleton(
    IComplexComponent,
    lambda: create_complex_component(config, True, False)
)
```

### Environment-specific Setup

```python
def configure_for_environment(env: str) -> DIContainer:
    if env == 'testing':
        container = DIContainer({'env': 'testing'})
        container.register_singleton(IExtractor, MockExtractor)
    elif env == 'production':
        container = configure_container(production_config)
        container.register_transient(IExtractor, RealExtractor)
    return container
```

## API Reference

| Method | Description |
|--------|-------------|
| `DIContainer(config)` | Initialize container |
| `register_singleton(interface, impl)` | Register singleton |
| `register_transient(interface, impl)` | Register transient |
| `resolve(interface)` | Resolve dependency |
| `is_registered(interface)` | Check registration |
| `clear()` | Clear all registrations |

## Troubleshooting

### Interface not registered

```python
# Error: KeyError: Interface IExtractor is not registered

# Fix: Register it first
container.register_singleton(IExtractor, MyExtractor)
```

### Circular dependencies

```python
# Bad: A → B → A
class A:
    def __init__(self, b: B): pass
class B:
    def __init__(self, a: A): pass

# Good: Extract shared dependency
class Shared:
    def __init__(self): pass
class A:
    def __init__(self, shared: Shared): pass
class B:
    def __init__(self, shared: Shared): pass
```

### Missing constructor parameter

```python
# Error: Can't resolve SomeType

# Fix 1: Register the missing dependency
container.register_singleton(SomeType, SomeImpl)

# Fix 2: Use a factory function
container.register_singleton(
    MyComponent,
    lambda: MyComponent(some_param=manual_value)
)
```

## Examples

See:
- `example_di_container.py` - Basic usage examples
- `example_di_integration.py` - Integration with existing modules
- `test_di_container.py` - Complete test suite

## Running Examples

```bash
# Basic examples
python example_di_container.py

# Integration examples
python example_di_integration.py

# Tests
python test_di_container.py
```

## Benefits

✅ **Loose Coupling** - Components don't depend on concrete implementations  
✅ **Easy Testing** - Inject mocks for unit tests  
✅ **Graceful Degradation** - Automatic fallbacks when dependencies unavailable  
✅ **Centralized Config** - Single source of truth for component setup  
✅ **Auto Injection** - No manual wiring of dependencies  

## See Also

- `DI_CONTAINER_README.md` - Full documentation
- `infrastructure/di_container.py` - Implementation
- `.github/copilot-instructions.md` - Orchestration principles
