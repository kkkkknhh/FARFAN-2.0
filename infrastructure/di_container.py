#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dependency Injection Container for FARFAN 2.0
==============================================

F4.1: Dependency Injection Container
Provides centralized dependency management for:
- Testing and mocking
- Graceful degradation
- Configuration-based component selection

Resolves Front A.1 (NLP model fallback) and Front A.2 (device management).
"""

import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ============================================================================
# Component Interfaces
# ============================================================================


class IExtractor(ABC):
    """Interface for document extraction components"""

    @abstractmethod
    def extract(self, document_path: str) -> Dict[str, Any]:
        """Extract structured data from document"""
        pass


class ICausalBuilder(ABC):
    """Interface for causal graph building components"""

    @abstractmethod
    def build_graph(self, extracted_data: Dict[str, Any]) -> Any:
        """Build causal graph from extracted data"""
        pass


class IBayesianEngine(ABC):
    """Interface for Bayesian inference components"""

    @abstractmethod
    def infer(self, graph: Any) -> Dict[str, Any]:
        """Perform Bayesian inference on causal graph"""
        pass


# ============================================================================
# Configuration Data Classes
# ============================================================================


@dataclass
class DeviceConfig:
    """Device configuration for computation (CPU/GPU)"""

    device: str = "cpu"
    use_gpu: bool = False
    gpu_id: Optional[int] = None

    def __post_init__(self):
        """Validate device configuration"""
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError(
                f"Invalid device: {self.device}. Must be 'cpu', 'cuda', or 'mps'"
            )

        if self.device == "cuda":
            self.use_gpu = True

        if self.use_gpu and self.device == "cpu":
            logger.warning("use_gpu=True but device='cpu'. Setting device='cuda'")
            self.device = "cuda"


# ============================================================================
# Dependency Injection Container
# ============================================================================


class DIContainer:
    """
    Dependency Injection container for centralized dependency management.

    Features:
    - Singleton and transient instance management
    - Automatic dependency resolution via reflection
    - Lazy initialization for performance
    - Graceful degradation support

    Example:
        >>> container = DIContainer(config)
        >>> container.register_singleton(IExtractor, PDFProcessor)
        >>> extractor = container.resolve(IExtractor)
    """

    def __init__(self, config: Any = None):
        """
        Initialize DI container.

        Args:
            config: Configuration object (e.g., CDAFConfig, dict, or None)
        """
        self.config = config
        self._registry: Dict[Type, tuple[Type | Callable, bool]] = {}
        self._singletons: Dict[Type, Any] = {}

        logger.info("DIContainer initialized")

    def register_singleton(
        self, interface: Type[T], implementation: Type[T] | Callable[[], T]
    ) -> None:
        """
        Register a singleton implementation.

        Singleton instances are created once and reused for all resolutions.

        Args:
            interface: Interface or abstract class type
            implementation: Concrete implementation class or factory function
        """
        self._registry[interface] = (implementation, True)
        logger.debug(f"Registered singleton: {interface.__name__} -> {implementation}")

    def register_transient(
        self, interface: Type[T], implementation: Type[T] | Callable[[], T]
    ) -> None:
        """
        Register a transient implementation.

        Transient instances are created new for each resolution.

        Args:
            interface: Interface or abstract class type
            implementation: Concrete implementation class or factory function
        """
        self._registry[interface] = (implementation, False)
        logger.debug(f"Registered transient: {interface.__name__} -> {implementation}")

    def resolve(self, interface: Type[T]) -> T:
        """
        Resolve a dependency with lazy initialization.

        For singletons, returns the cached instance if it exists.
        For transients, always creates a new instance.

        Args:
            interface: Interface type to resolve

        Returns:
            Instance of the registered implementation

        Raises:
            KeyError: If interface is not registered
        """
        # Check if singleton is already instantiated
        if interface in self._singletons:
            logger.debug(f"Returning cached singleton: {interface.__name__}")
            return self._singletons[interface]

        # Get registration
        if interface not in self._registry:
            raise KeyError(
                f"Interface {interface.__name__} is not registered. "
                f"Available interfaces: {list(self._registry.keys())}"
            )

        implementation, is_singleton = self._registry[interface]

        # Instantiate
        instance = self._instantiate_with_deps(implementation)

        # Cache if singleton
        if is_singleton:
            self._singletons[interface] = instance
            logger.debug(f"Cached new singleton: {interface.__name__}")

        return instance

    def _get_constructor_params(self, cls: Type) -> Dict[str, inspect.Parameter]:
        """
        Inspect and collect constructor parameters.

        Args:
            cls: Class to inspect

        Returns:
            Dictionary of parameter names to Parameter objects

        Raises:
            ValueError: If constructor signature cannot be inspected
        """
        try:
            sig = inspect.signature(cls.__init__)
            return {
                name: param for name, param in sig.parameters.items() if name != "self"
            }
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Cannot inspect constructor for {cls}") from e

    def _resolve_parameter(
        self, param_name: str, param: inspect.Parameter
    ) -> tuple[bool, Any]:
        """
        Resolve a single constructor parameter.

        Args:
            param_name: Name of the parameter
            param: Parameter object from signature inspection

        Returns:
            Tuple of (should_include, resolved_value)
        """
        # Check if parameter type is registered
        if (
            param.annotation != inspect.Parameter.empty
            and param.annotation in self._registry
        ):
            logger.debug(
                f"Resolving dependency: {param_name} -> {param.annotation.__name__}"
            )
            return (True, self.resolve(param.annotation))

        # If parameter has default, skip it
        if param.default != inspect.Parameter.empty:
            return (False, None)

        # If it's the config parameter, inject our config
        if param_name == "config" and self.config is not None:
            return (True, self.config)

        return (False, None)

    def _instantiate_with_deps(self, cls: Type | Callable) -> Any:
        """
        Instantiate a class with automatic dependency resolution.

        Uses reflection to inspect constructor parameters and automatically
        resolves registered dependencies.

        Args:
            cls: Class or factory function to instantiate

        Returns:
            Instance of the class
        """
        # If it's a callable (factory function), just call it
        if callable(cls) and not inspect.isclass(cls):
            logger.debug(f"Calling factory function: {cls}")
            return cls()

        # Get constructor parameters
        try:
            params = self._get_constructor_params(cls)
        except ValueError:
            # No __init__ or can't inspect, try to instantiate directly
            logger.debug(f"No inspectable __init__, instantiating directly: {cls}")
            return cls()

        # Resolve dependencies
        kwargs = {}
        for param_name, param in params.items():
            should_include, value = self._resolve_parameter(param_name, param)
            if should_include:
                kwargs[param_name] = value

        logger.debug(
            f"Instantiating {cls.__name__} with dependencies: {list(kwargs.keys())}"
        )
        return cls(**kwargs)

    def is_registered(self, interface: Type) -> bool:
        """
        Check if an interface is registered.

        Args:
            interface: Interface type to check

        Returns:
            True if registered, False otherwise
        """
        return interface in self._registry

    def clear(self) -> None:
        """Clear all registrations and cached singletons."""
        self._registry.clear()
        self._singletons.clear()
        logger.info("DIContainer cleared")


# ============================================================================
# Configuration Factory
# ============================================================================


def configure_container(config: Any = None) -> DIContainer:
    """
    Configure DI container with default component registrations.

    Implements graceful degradation for:
    - NLP models (transformer -> large -> small)
    - Device selection (GPU -> CPU)
    - Component availability

    Args:
        config: Configuration object with settings like:
            - use_gpu: bool
            - nlp_model: str

    Returns:
        Configured DIContainer instance

    Example:
        >>> from infrastructure import configure_container
        >>> container = configure_container(config)
        >>> nlp = container.resolve(spacy.Language)
    """
    container = DIContainer(config)

    # ========================================================================
    # Front A.1: NLP Model with Graceful Degradation
    # ========================================================================

    try:
        import spacy

        # Try transformer model first (best quality)
        try:
            nlp = spacy.load("es_dep_news_trf")
            logger.info("Loaded transformer model: es_dep_news_trf")
            container.register_singleton(spacy.Language, lambda: nlp)

        except (ImportError, OSError):
            # Fall back to large model
            try:
                nlp = spacy.load("es_core_news_lg")
                logger.warning(
                    "Transformer model unavailable, using core model: es_core_news_lg"
                )
                container.register_singleton(spacy.Language, lambda: nlp)

            except (ImportError, OSError):
                # Fall back to small model
                try:
                    nlp = spacy.load("es_core_news_sm")
                    logger.warning(
                        "Large model unavailable, using small model: es_core_news_sm"
                    )
                    container.register_singleton(spacy.Language, lambda: nlp)

                except (ImportError, OSError):
                    logger.error(
                        "No spaCy Spanish model available. Run: python -m spacy download es_core_news_lg"
                    )

    except ImportError:
        logger.error("spaCy not installed. Install with: pip install spacy")

    # ========================================================================
    # Front A.2: Device Management (GPU/CPU)
    # ========================================================================

    device = "cpu"
    use_gpu = False

    # Check if config specifies GPU usage
    if config is not None:
        if hasattr(config, "use_gpu"):
            use_gpu = config.use_gpu
        elif isinstance(config, dict):
            use_gpu = config.get("use_gpu", False)

    # Detect GPU availability
    if use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("GPU requested but CUDA not available, using CPU")
                device = "cpu"
        except ImportError:
            logger.warning("PyTorch not installed, GPU support unavailable")
            device = "cpu"

    device_config = DeviceConfig(device=device, use_gpu=(device == "cuda"))
    container.register_singleton(DeviceConfig, lambda: device_config)

    # ========================================================================
    # Component Interfaces with Explicit Registrations
    # ========================================================================

    logger.info("DIContainer configured with graceful degradation")
    return container
