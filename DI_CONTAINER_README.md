# Dependency Injection Container (F4.1)

## Overview

The Dependency Injection (DI) Container provides centralized dependency management for the FARFAN 2.0 framework. It addresses **Front 4: WIRING & STABILITY** by enabling:

- **Loose coupling** between components
- **Easy testing** with mock implementations
- **Graceful degradation** when dependencies are unavailable
- **Centralized configuration** management
- **Automatic dependency resolution**

## Key Features

### 1. Singleton and Transient Lifetimes

- **Singleton**: Single instance shared across all resolutions (e.g., database connections, NLP models)
- **Transient**: New instance created for each resolution (e.g., request handlers, processors)

### 2. Automatic Dependency Injection

The container uses reflection to automatically resolve constructor dependencies:

```python
class ServiceB:
    def __init__(self, service_a: ServiceA):
        self.service_a = service_a

container.register_singleton(ServiceA, ServiceA)
container.register_transient(ServiceB, ServiceB)

# ServiceA is automatically injected!
service_b = container.resolve(ServiceB)
```

### 3. Graceful Degradation

The `configure_container()` factory implements fallback chains:

**NLP Models** (Front A.1):
1. Try `es_dep_news_trf` (transformer, best quality)
2. Fall back to `es_core_news_lg` (large)
3. Fall back to `es_core_news_sm` (small)
4. Log error if none available

**Device Management** (Front A.2):
1. Check config for GPU request
2. Verify CUDA availability
3. Fall back to CPU if unavailable

## Architecture

```
infrastructure/
├── __init__.py          # Public API exports
├── di_container.py      # Core implementation
```

### Component Interfaces

Three core interfaces define the contract for framework components:

```python
class IExtractor(ABC):
    """PDF/document extraction"""
    @abstractmethod
    def extract(self, document_path: str) -> Dict[str, Any]:
        pass

class ICausalBuilder(ABC):
    """Causal graph construction"""
    @abstractmethod
    def build_graph(self, extracted_data: Dict[str, Any]) -> Any:
        pass

class IBayesianEngine(ABC):
    """Bayesian inference"""
    @abstractmethod
    def infer(self, graph: Any) -> Dict[str, Any]:
        pass
```

## Usage

### Basic Setup

```python
from infrastructure import configure_container, DIContainer

# Use factory for production (includes graceful degradation)
container = configure_container(config)

# Or create manually for testing
container = DIContainer(config)
```

### Registering Components

```python
# Singleton (shared instance)
container.register_singleton(IExtractor, PDFProcessor)

# Transient (new instance each time)
container.register_transient(ICausalBuilder, CausalGraphBuilder)

# Factory function
container.register_singleton(
    spacy.Language,
    lambda: spacy.load("es_core_news_lg")
)
```

### Resolving Dependencies

```python
# Manual resolution
extractor = container.resolve(IExtractor)
data = extractor.extract('/path/to/doc.pdf')

# Automatic dependency injection
class Orchestrator:
    def __init__(self, extractor: IExtractor, builder: ICausalBuilder):
        self.extractor = extractor
        self.builder = builder

container.register_transient(Orchestrator, Orchestrator)
orchestrator = container.resolve(Orchestrator)  # Dependencies auto-injected!
```

## Testing

The DI container makes unit testing trivial with mock implementations:

```python
# Create mock
class MockExtractor(IExtractor):
    def extract(self, document_path: str):
        return {'text': 'mock data'}

# Configure test container
test_container = DIContainer()
test_container.register_singleton(IExtractor, MockExtractor)

# Test your code with mocks
component = test_container.resolve(MyComponent)
assert isinstance(component.extractor, MockExtractor)
```

### Running Tests

```bash
# Run DI container tests
python test_di_container.py

# Expected output: 26 tests passed
```

## Integration with Existing Code

### Example: Integrating with dereck_beach

```python
from infrastructure import configure_container
from dereck_beach import CDAFFramework

# Create config
config = CDAFConfigSchema(...)

# Configure DI container
container = configure_container(config)

# Register CDAF components
container.register_singleton(CDAFFramework, lambda: CDAFFramework(config, output_dir, log_level))

# Resolve and use
framework = container.resolve(CDAFFramework)
```

### Example: Integrating with Bayesian Engine

```python
from infrastructure import DIContainer, IBayesianEngine
from inference.bayesian_engine import BayesianSamplingEngine

container = DIContainer()
container.register_singleton(IBayesianEngine, BayesianSamplingEngine)

engine = container.resolve(IBayesianEngine)
results = engine.infer(graph)
```

## Configuration

The container respects configuration objects for component initialization:

```python
class AppConfig:
    use_gpu: bool = False
    nlp_model: str = 'es_core_news_lg'

config = AppConfig()
container = configure_container(config)

# DeviceConfig is automatically configured based on config.use_gpu
device_config = container.resolve(DeviceConfig)
```

## Troubleshooting

### Common Issues

**1. KeyError: Interface not registered**
```python
# Problem
extractor = container.resolve(IExtractor)
# KeyError: Interface IExtractor is not registered

# Solution: Register the interface first
container.register_singleton(IExtractor, PDFProcessor)
```

**2. Missing dependencies**
```python
# Problem: Constructor parameter can't be resolved

class ServiceA:
    def __init__(self, unknown_dep: UnknownType):
        pass

# Solution: Register all dependencies
container.register_singleton(UnknownType, ConcreteImplementation)
```

**3. Circular dependencies**

The container does not currently support circular dependencies. Refactor your code to avoid them:

```python
# Bad: A depends on B, B depends on A
class A:
    def __init__(self, b: B): pass

class B:
    def __init__(self, a: A): pass

# Good: Extract common functionality to C
class C:
    def __init__(self): pass

class A:
    def __init__(self, c: C): pass

class B:
    def __init__(self, c: C): pass
```

## API Reference

### DIContainer

#### Methods

- `__init__(config=None)` - Initialize container with optional config
- `register_singleton(interface, implementation)` - Register singleton
- `register_transient(interface, implementation)` - Register transient
- `resolve(interface)` - Resolve dependency
- `is_registered(interface)` - Check if interface is registered
- `clear()` - Clear all registrations

### DeviceConfig

Configuration for compute device selection.

#### Attributes

- `device: str` - Device name ('cpu', 'cuda', 'mps')
- `use_gpu: bool` - Whether GPU is enabled
- `gpu_id: Optional[int]` - GPU device ID if multiple GPUs

### configure_container(config=None)

Factory function that creates and configures a DIContainer with:
- NLP model graceful degradation
- Device management
- Logging configuration

## Benefits

### Solves Front A.1: NLP Model Fallback

Before:
```python
# Hardcoded, brittle
nlp = spacy.load("es_dep_news_trf")  # Crashes if not available
```

After:
```python
# Graceful degradation
container = configure_container()
nlp = container.resolve(spacy.Language)  # Falls back automatically
```

### Solves Front A.2: Device Management

Before:
```python
# Scattered GPU checks
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
```

After:
```python
# Centralized device management
container = configure_container(config)
device_config = container.resolve(DeviceConfig)
model.to(device_config.device)
```

### Enables Massive Unit Testing

Before:
```python
# Hard to test - real dependencies
class Processor:
    def __init__(self):
        self.nlp = spacy.load("es_core_news_lg")  # Slow!
        self.db = Database()  # External dependency!
```

After:
```python
# Easy to test - inject mocks
class Processor:
    def __init__(self, nlp: spacy.Language, db: IDatabase):
        self.nlp = nlp
        self.db = db

# In tests
container.register_singleton(spacy.Language, lambda: MockNLP())
container.register_singleton(IDatabase, lambda: MockDB())
```

## Examples

See `example_di_container.py` for comprehensive usage examples:

```bash
python example_di_container.py
```

Examples include:
1. Basic container usage
2. Graceful degradation patterns
3. Testing with mocks
4. Automatic dependency injection
5. Real-world integration

## Future Enhancements

Potential improvements for future iterations:

1. **Named registrations** - Register multiple implementations with names
2. **Scoped lifetimes** - Request-scoped instances for web apps
3. **Decorator support** - `@inject` decorator for cleaner code
4. **Configuration validation** - Validate all dependencies are registered
5. **Circular dependency detection** - Detect and report circular dependencies
6. **Async support** - Async factory functions and resolution

## Contributing

When adding new components:

1. Define an interface (e.g., `IMyComponent`)
2. Implement the interface
3. Register in `configure_container()` or your setup code
4. Add tests for the component
5. Update this documentation

## See Also

- `test_di_container.py` - Comprehensive test suite
- `example_di_container.py` - Usage examples
- `HARMONIC_FRONT_4_IMPLEMENTATION.md` - Overall Front 4 strategy
- `.github/copilot-instructions.md` - Orchestration principles
