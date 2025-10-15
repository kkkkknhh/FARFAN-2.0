#!/usr/bin/env python3
"""
Circuit Breaker - Patrón de resiliencia para el pipeline FARFAN

Implementa el patrón Circuit Breaker para manejar fallos transitorios:
- CLOSED: Operación normal, permite ejecución
- OPEN: Demasiados fallos, bloquea ejecuciones
- HALF_OPEN: Prueba si el servicio se recuperó

Previene cascadas de fallos y permite recuperación automática.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, List, Optional

logger = logging.getLogger("circuit_breaker")

# Module-level constants
CIRCUIT_BREAKER_OPEN = "Circuit breaker is OPEN"


class CircuitState(Enum):
    """Estados del circuit breaker"""

    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Failing, blocking calls
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuración del circuit breaker"""

    failure_threshold: int = 3  # Fallos antes de abrir
    success_threshold: int = 2  # Éxitos para cerrar desde half-open
    timeout: float = 60.0  # Segundos antes de intentar half-open
    expected_exceptions: tuple = (Exception,)


@dataclass
class CircuitTransition:
    """Registro de transición de estado"""

    from_state: CircuitState
    to_state: CircuitState
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


class CircuitBreakerError(Exception):
    """Excepción cuando el circuit breaker está abierto"""

    pass


class CircuitBreaker:
    """
    Circuit Breaker para proteger contra fallos transitorios

    Uso:
        breaker = CircuitBreaker(name="pdf_extraction")
        result = breaker.call(extract_pdf, pdf_path)
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.opened_at: Optional[datetime] = None

        self.transitions: List[CircuitTransition] = []
        self.total_calls = 0
        self.total_successes = 0
        self.total_failures = 0

        logger.info(
            f"CircuitBreaker '{name}' inicializado: {self.config.failure_threshold} fallos, {self.config.timeout}s timeout"
        )

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Ejecuta función protegida por circuit breaker

        Args:
            func: Función a ejecutar
            *args, **kwargs: Argumentos para la función

        Returns:
            Resultado de la función

        Raises:
            CircuitBreakerError: Si el circuit está abierto
        """
        self.total_calls += 1

        # Check current state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition(
                    CircuitState.HALF_OPEN, "Timeout expired, testing recovery"
                )
            else:
                raise CircuitBreakerError(
                    f"CircuitBreaker '{self.name}' {CIRCUIT_BREAKER_OPEN}. "
                    f"Opened {(datetime.now() - self.opened_at).seconds}s ago. "
                    f"Will retry in {self.config.timeout - (datetime.now() - self.opened_at).total_seconds():.1f}s"
                )

        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exceptions as e:
            self._on_failure(e)
            raise

    def _on_success(self):
        """Maneja ejecución exitosa"""
        self.total_successes += 1

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.info(
                f"CircuitBreaker '{self.name}': Success in HALF_OPEN ({self.success_count}/{self.config.success_threshold})"
            )

            if self.success_count >= self.config.success_threshold:
                self._transition(
                    CircuitState.CLOSED, "Recovered after success threshold"
                )
                self.failure_count = 0
                self.success_count = 0

        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def _on_failure(self, exception: Exception):
        """Maneja fallo de ejecución"""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        logger.warning(
            f"CircuitBreaker '{self.name}': Failure ({self.failure_count}/{self.config.failure_threshold}) - {exception}"
        )

        if self.state == CircuitState.HALF_OPEN:
            self._transition(CircuitState.OPEN, f"Failed in HALF_OPEN: {exception}")
            self.opened_at = datetime.now()
            self.success_count = 0

        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._transition(
                    CircuitState.OPEN,
                    f"Failure threshold reached: {self.failure_count} failures",
                )
                self.opened_at = datetime.now()

    def _should_attempt_reset(self) -> bool:
        """Verifica si debe intentar reset (OPEN -> HALF_OPEN)"""
        if not self.opened_at:
            return False

        elapsed = (datetime.now() - self.opened_at).total_seconds()
        return elapsed >= self.config.timeout

    def _transition(self, new_state: CircuitState, reason: str):
        """Registra transición de estado"""
        old_state = self.state
        self.state = new_state

        transition = CircuitTransition(
            from_state=old_state, to_state=new_state, reason=reason
        )
        self.transitions.append(transition)

        logger.warning(
            f"CircuitBreaker '{self.name}': {old_state.value} -> {new_state.value} ({reason})"
        )

    def reset(self):
        """Resetea manualmente el circuit breaker"""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.opened_at = None
        logger.info(
            f"CircuitBreaker '{self.name}' manually reset from {old_state.value}"
        )

    def get_stats(self) -> dict:
        """Obtiene estadísticas del circuit breaker"""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.total_calls,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "failure_count": self.failure_count,
            "success_rate": (
                self.total_successes / self.total_calls if self.total_calls > 0 else 0.0
            ),
            "transitions": len(self.transitions),
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
        }

    def get_transitions(self) -> List[CircuitTransition]:
        """Obtiene historial de transiciones"""
        return self.transitions.copy()


class CircuitBreakerRegistry:
    """
    Registro centralizado de circuit breakers por etapa
    """

    def __init__(self):
        self.breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Obtiene o crea circuit breaker"""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]

    def get_all_stats(self) -> dict:
        """Obtiene estadísticas de todos los breakers"""
        return {name: breaker.get_stats() for name, breaker in self.breakers.items()}

    def get_all_transitions(self) -> List[CircuitTransition]:
        """Obtiene todas las transiciones de todos los breakers"""
        all_transitions = []
        for breaker in self.breakers.values():
            all_transitions.extend(breaker.get_transitions())
        return sorted(all_transitions, key=lambda t: t.timestamp)
